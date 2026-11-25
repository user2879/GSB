"""Parsers for the three historical quant receipt formats and their ledger."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import E_RECEIPT_MALFORMED, FailureList
from .hashing import fingerprint

BASE_FIELDS = ("layer", "version", "coverage_permille")
Q2_FIELDS = BASE_FIELDS + ("param_version", "elapsed_ms")
Q3_FIELDS = Q2_FIELDS + ("group_scale_summary", "override_refs", "checksum")


@dataclass
class ReceiptEntry:
    layer: str
    version: str
    source_file: str
    entry_index: int
    coverage_permille: int
    param_version: int | None = None
    elapsed_ms: int | None = None
    group_scale_summary: list[float] | None = None
    override_refs: list[str] | None = None
    checksum: str | None = None
    missing_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "version": self.version,
            "source_file": self.source_file,
            "entry_index": self.entry_index,
            "coverage_permille": self.coverage_permille,
            "param_version": self.param_version,
            "elapsed_ms": self.elapsed_ms,
            "group_scale_summary": self.group_scale_summary,
            "override_refs": self.override_refs,
            "checksum": self.checksum,
            "missing_fields": list(self.missing_fields),
        }


def _detect_version(record: dict[str, Any]) -> str:
    version = record.get("version")
    if version in ("q1", "q2", "q3"):
        return version
    if "checksum" in record or "override_refs" in record:
        return "q3"
    if "param_version" in record or "elapsed_ms" in record:
        return "q2"
    return "q1"


def _parse_entry(record: dict[str, Any], source_file: str,
                 entry_index: int, failures: FailureList) -> ReceiptEntry | None:
    target = f"{source_file}#entry{entry_index}"
    if not isinstance(record, dict):
        failures.add(E_RECEIPT_MALFORMED, "receipt", target,
                     "entry must be an object")
        return None
    layer = record.get("layer")
    if not isinstance(layer, str) or not layer:
        failures.add(E_RECEIPT_MALFORMED, "receipt", target,
                     "missing or invalid layer")
        return None
    coverage = record.get("coverage_permille")
    if not isinstance(coverage, int) or not 0 <= coverage <= 1000:
        failures.add(E_RECEIPT_MALFORMED, "receipt", target,
                     "coverage_permille must be an int in [0,1000]")
        return None
    version = _detect_version(record)
    missing = [name for name in BASE_FIELDS if name not in record]
    entry = ReceiptEntry(
        layer=layer,
        version=version,
        source_file=source_file,
        entry_index=entry_index,
        coverage_permille=coverage,
        missing_fields=missing,
    )
    if version in ("q2", "q3"):
        param_version = record.get("param_version")
        elapsed_ms = record.get("elapsed_ms")
        if param_version is None:
            entry.missing_fields.append("param_version")
        elif not isinstance(param_version, int) or param_version < 0:
            failures.add(E_RECEIPT_MALFORMED, "receipt", target,
                         "param_version must be a non-negative int")
            return None
        else:
            entry.param_version = param_version
        if elapsed_ms is None:
            entry.missing_fields.append("elapsed_ms")
        elif not isinstance(elapsed_ms, int) or elapsed_ms < 0:
            failures.add(E_RECEIPT_MALFORMED, "receipt", target,
                         "elapsed_ms must be a non-negative int")
            return None
        else:
            entry.elapsed_ms = elapsed_ms
    else:
        entry.missing_fields.extend(["param_version", "elapsed_ms"])
    if version == "q3":
        summary = record.get("group_scale_summary")
        refs = record.get("override_refs")
        checksum = record.get("checksum")
        if summary is None:
            entry.missing_fields.append("group_scale_summary")
        elif not isinstance(summary, list) or not all(
                isinstance(x, (int, float)) for x in summary):
            failures.add(E_RECEIPT_MALFORMED, "receipt", target,
                         "group_scale_summary must be a number list")
            return None
        else:
            entry.group_scale_summary = [float(x) for x in summary]
        if refs is None:
            entry.missing_fields.append("override_refs")
        elif not isinstance(refs, list) or not all(isinstance(x, str) for x in refs):
            failures.add(E_RECEIPT_MALFORMED, "receipt", target,
                         "override_refs must be a string list")
            return None
        else:
            entry.override_refs = list(refs)
        if checksum is None:
            entry.missing_fields.append("checksum")
        elif not isinstance(checksum, str):
            failures.add(E_RECEIPT_MALFORMED, "receipt", target,
                         "checksum must be a string")
            return None
        else:
            entry.checksum = checksum
    else:
        entry.missing_fields.extend(
            ["group_scale_summary", "override_refs", "checksum"])
    return entry


def parse_receipt_dir(directory: str | Path,
                      failures: FailureList) -> list[ReceiptEntry]:
    directory = Path(directory)
    entries: list[ReceiptEntry] = []
    if not directory.exists():
        failures.add(E_RECEIPT_MALFORMED, "receipt", directory.name,
                     "receipt directory does not exist")
        return entries
    for path in sorted(directory.glob("*.json"), key=lambda p: p.name):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.add(E_RECEIPT_MALFORMED, "receipt", path.name,
                         "malformed json", str(exc))
            continue
        if isinstance(document, dict):
            records = document.get("entries")
            if not isinstance(records, list):
                failures.add(E_RECEIPT_MALFORMED, "receipt", path.name,
                             "document must contain an entries list")
                continue
        elif isinstance(document, list):
            records = document
        else:
            failures.add(E_RECEIPT_MALFORMED, "receipt", path.name,
                         "document must be a list or object")
            continue
        for index, record in enumerate(records):
            entry = _parse_entry(record, path.name, index, failures)
            if entry is not None:
                entries.append(entry)
    return entries


def _adjudication_key(entry: ReceiptEntry) -> tuple[int, int, int, str, int]:
    # 1) q3 entry carrying override references wins first.
    has_override = 1 if entry.version == "q3" and entry.override_refs else 0
    # 2) larger elapsed ms wins; missing elapsed is treated as -1, never wins.
    elapsed = entry.elapsed_ms if entry.elapsed_ms is not None else -1
    # 3) larger param version wins; missing treated as -1.
    param = entry.param_version if entry.param_version is not None else -1
    # 4) receipt filename lexicographic ascending, then entry index.
    return (-has_override, -elapsed, -param, entry.source_file, entry.entry_index)


def build_receipt_ledger(entries: list[ReceiptEntry]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[ReceiptEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.layer, []).append(entry)
    ledger: dict[str, dict[str, Any]] = {}
    for layer in sorted(grouped):
        candidates = sorted(grouped[layer], key=_adjudication_key)
        winner = candidates[0]
        basis = _describe_basis(winner, candidates)
        ledger[layer] = {
            "winner": winner.to_dict(),
            "candidate_count": len(candidates),
            "candidates": [c.to_dict() for c in candidates],
            "adjudication": basis,
            "ledger_fingerprint": fingerprint(
                [c.to_dict() for c in candidates], prefix="rcp-"),
        }
    return ledger


def _describe_basis(winner: ReceiptEntry,
                    candidates: list[ReceiptEntry]) -> dict[str, Any]:
    if winner.version == "q3" and winner.override_refs:
        rule = "q3_override_ref"
        detail = f"override_refs={sorted(winner.override_refs)}"
    else:
        elapsed_values = [c.elapsed_ms if c.elapsed_ms is not None else -1
                          for c in candidates]
        param_values = [c.param_version if c.param_version is not None else -1
                        for c in candidates]
        max_elapsed = max(elapsed_values)
        elapsed_tied = sum(1 for value in elapsed_values
                           if value == max_elapsed) > 1
        if any(c.elapsed_ms is not None for c in candidates) and not elapsed_tied:
            rule = "elapsed_ms_desc"
            detail = f"elapsed_ms={winner.elapsed_ms}"
        elif any(c.param_version is not None for c in candidates):
            rule = "param_version_desc"
            detail = f"param_version={winner.param_version}"
        else:
            rule = "filename_asc"
            detail = winner.source_file
    return {
        "rule": rule,
        "detail": detail,
        "tie_breakers": ["override_ref", "elapsed_ms_desc",
                         "param_version_desc", "filename_asc", "entry_index_asc"],
    }
