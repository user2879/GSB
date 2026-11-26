"""Requirement 2: c1/c2/c3 receipt codec and cross-version normalization."""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .errors import FieldError, FormatError

KNOWN_VERSIONS = (1, 2, 3)
# Fixed-width fields: minimum accepted character width for string fields.
MIN_WIDTH = {
    "batch_id": 6,
    "shard_id": 4,
    "loss_summary": 16,
    "param_summary": 16,
    "lease_id": 6,
}
REQUIRED_BY_VERSION = {
    1: ("batch_id", "shard_id", "loss_summary"),
    2: ("batch_id", "shard_id", "loss_summary", "param_version",
        "generation", "elapsed_ms"),
    3: ("batch_id", "shard_id", "loss_summary", "param_version",
        "generation", "elapsed_ms", "sample_ids", "param_summary", "lease_id"),
}
# Explicit back-fill defaults for fields absent in older on-disk formats.
BACKFILL_RULES = {
    "generation": ("default_generation", 0),
    "param_version": ("default_param_version", 0),
    "generation": ("default_generation", 0),
    "elapsed_ms": ("default_elapsed_ms", 0),
    "sample_ids": ("empty_participant_set", ()),
    "param_summary": ("unknown_param_summary", "UNKNOWN"),
    "lease_id": ("no_lease_legacy", "NO_LEASE"),
}


@dataclass(frozen=True)
class NormalizedReceipt:
    source_format: int
    batch_id: str
    shard_id: str
    loss_summary: str
    param_version: int
    generation: int
    elapsed_ms: int
    sample_ids: tuple[str, ...]
    param_summary: str
    lease_id: str
    backfilled_fields: tuple[str, ...]
    backfill_basis: dict[str, str]
    rejected_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "source_format": self.source_format,
            "batch_id": self.batch_id,
            "shard_id": self.shard_id,
            "loss_summary": self.loss_summary,
            "param_version": self.param_version,
            "generation": self.generation,
            "elapsed_ms": self.elapsed_ms,
            "sample_ids": list(self.sample_ids),
            "param_summary": self.param_summary,
            "lease_id": self.lease_id,
            "backfilled_fields": list(self.backfilled_fields),
            "backfill_basis": dict(sorted(self.backfill_basis.items())),
            "rejected_reason": self.rejected_reason,
        }


def _check_widths(envelope: dict, version: int, location: str) -> None:
    for field_name, min_width in MIN_WIDTH.items():
        if field_name not in envelope or envelope[field_name] is None:
            continue
        value = envelope[field_name]
        if isinstance(value, str) and len(value) < min_width:
            raise FieldError(
                f"fixed-width field {field_name} width {len(value)} < {min_width}",
                location, field_name, detail={"version": version, "width": len(value)})


def decode_receipt(raw_line: str, location: str = "<receipt>") -> NormalizedReceipt:
    try:
        envelope = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise FormatError(f"receipt is not valid JSON: {exc.msg}", location, "<json>")
    if not isinstance(envelope, dict):
        raise FormatError("receipt envelope must be an object", location, "<root>")
    version = envelope.get("format_version")
    if not isinstance(version, int) or isinstance(version, bool) \
            or version not in KNOWN_VERSIONS:
        raise FormatError(f"unknown receipt format version {version!r}",
                          location, "format_version",
                          detail={"known": list(KNOWN_VERSIONS)})
    for required in REQUIRED_BY_VERSION[version]:
        if required not in envelope or envelope[required] is None:
            raise FieldError(f"required field {required} missing for c{version}",
                             location, required, detail={"version": version})
    if "elapsed_ms" in envelope and (
            not isinstance(envelope["elapsed_ms"], int)
            or isinstance(envelope["elapsed_ms"], bool)
            or envelope["elapsed_ms"] < 0):
        raise FieldError("elapsed_ms must be a non-negative integer",
                         location, "elapsed_ms", detail={"version": version})
    pv = envelope.get("param_version", 0)
    if not isinstance(pv, int) or isinstance(pv, bool) or pv < 0:
        raise FieldError("param_version must be a non-negative integer",
                         location, "param_version")
    _check_widths(envelope, version, location)

    backfilled: list[str] = []
    basis: dict[str, str] = {}
    values: dict[str, Any] = {}
    for field_name, (basis_name, default) in BACKFILL_RULES.items():
        if field_name in envelope and envelope[field_name] is not None:
            values[field_name] = envelope[field_name]
        else:
            values[field_name] = list(default) if isinstance(default, tuple) else default
            backfilled.append(field_name)
            basis[field_name] = basis_name
    sample_ids = values["sample_ids"]
    if not isinstance(sample_ids, list) or not all(isinstance(s, str) for s in sample_ids):
        raise FieldError("sample_ids must be a list of strings", location, "sample_ids")
    return NormalizedReceipt(
        source_format=version,
        batch_id=envelope["batch_id"],
        shard_id=envelope["shard_id"],
        loss_summary=envelope["loss_summary"],
        param_version=values["param_version"],
        generation=values["generation"],
        elapsed_ms=values["elapsed_ms"],
        sample_ids=tuple(sorted(sample_ids)),
        param_summary=values["param_summary"],
        lease_id=values["lease_id"],
        backfilled_fields=tuple(backfilled),
        backfill_basis=basis,
    )


def encode_receipt(version: int, *, batch_id: str, shard_id: str, loss_summary: str,
                   param_version: int = 0, generation: int = 0, elapsed_ms: int = 0,
                   sample_ids: list[str] | None = None, param_summary: str = "UNKNOWN",
                   lease_id: str = "NO_LEASE") -> str:
    if version not in KNOWN_VERSIONS:
        raise FormatError(f"unknown receipt format version {version}", "<encode>",
                          "format_version")
    envelope: dict[str, Any] = {
        "format_version": version,
        "batch_id": batch_id,
        "shard_id": shard_id,
        "loss_summary": loss_summary,
    }
    if version >= 2:
        envelope.update(param_version=param_version, generation=generation,
                        elapsed_ms=elapsed_ms)
    if version >= 3:
        envelope.update(sample_ids=sorted(sample_ids or []),
                        param_summary=param_summary, lease_id=lease_id)
    return json.dumps(envelope, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def load_receipt_file(path: str | Path) -> list[NormalizedReceipt]:
    receipts: list[NormalizedReceipt] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                receipts.append(decode_receipt(line, f"{path}:{line_no}"))
    return receipts


def normalize_mixed_batch(receipts: list[NormalizedReceipt]) -> tuple[
        NormalizedReceipt, list[NormalizedReceipt]]:
    """Pick one accepted receipt for a batch; keep rejected ones with reasons.

    Stable rule: higher param_version first, then higher source format version.
    Ties keep original arrival order (Python's sort is stable).
    """
    if not receipts:
        raise FieldError("no receipts supplied for batch", "<batch>", "receipts")
    ordered = sorted(receipts, key=lambda r: (r.param_version, r.source_format),
                     reverse=True)
    accepted = ordered[0]
    rejected: list[NormalizedReceipt] = []
    for receipt in ordered[1:]:
        reason = (
            f"superseded by c{accepted.source_format} param_version="
            f"{accepted.param_version}; self param_version={receipt.param_version} "
            f"format=c{receipt.source_format}"
        )
        rejected.append(replace(receipt, rejected_reason=reason))
    return accepted, rejected
