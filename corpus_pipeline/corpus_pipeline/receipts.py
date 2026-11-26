from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .errors import Code, fail
from .models import PipelineConfig, Receipt


def _receipt_items(value: Any, *, source_file: str) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, Mapping) and isinstance(value.get("receipts"), list):
        items = value["receipts"]
    elif isinstance(value, Mapping) and "receipt_id" in value:
        items = [value]
    else:
        fail(Code.INVALID_RECEIPT, "receipt file must contain a receipt list", file=source_file)
    normalized: list[Mapping[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            fail(Code.INVALID_RECEIPT, "receipt entries must be objects", file=source_file)
        normalized.append(item)
    return normalized


def _read_receipt_file(path: Path) -> list[Mapping[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            if path.suffix.lower() == ".json":
                value = json.load(handle)
            else:
                value = yaml.safe_load(handle)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        fail(Code.INVALID_RECEIPT, "receipt file could not be parsed", file=str(path), error=str(exc))
    return _receipt_items(value, source_file=str(path))


def load_receipts(
    source: str | Path | Iterable[str | Path] | None,
    config: PipelineConfig,
) -> tuple[list[Receipt], list[str]]:
    if source is None:
        return [], []
    if isinstance(source, (str, Path)):
        path = Path(source)
        paths = sorted(
            [item for item in path.iterdir() if item.suffix.lower() in {".yaml", ".yml", ".json"}]
        ) if path.is_dir() else [path]
    else:
        paths = sorted([Path(item) for item in source])
    receipts: list[Receipt] = []
    files: list[str] = []
    seen_ids: set[str] = set()
    known_domains = set(config.domain_ids)
    for path in paths:
        files.append(str(path))
        for raw in _read_receipt_file(path):
            receipt = Receipt.from_dict(raw, source_file=str(path), known_domains=known_domains)
            if receipt.receipt_id in seen_ids:
                fail(
                    Code.INVALID_RECEIPT,
                    "receipt identifiers must be unique across files",
                    receipt_id=receipt.receipt_id,
                    file=str(path),
                )
            seen_ids.add(receipt.receipt_id)
            receipts.append(receipt)
    return receipts, files


def arbitrate_receipts(receipts: Sequence[Receipt]) -> dict[str, Any]:
    """Resolve one authoritative receipt per ``(domain, shard_id)`` group.

    A valid coverage edge is strictly high-version to low-version.  Such an
    edge removes the lower receipt from the root candidate set.  This makes a
    live higher version impossible to overwrite with a lower one.  If any
    valid coverage edge exists in a group, ties use integer elapsed milliseconds
    before filename; otherwise the documented version/file-name order is used.
    """

    receipt_by_id: dict[str, Receipt] = {}
    groups: dict[tuple[str, str], list[Receipt]] = {}
    for receipt in receipts:
        if receipt.receipt_id in receipt_by_id:
            fail(Code.INVALID_RECEIPT, "duplicate receipt identifier", receipt_id=receipt.receipt_id)
        receipt_by_id[receipt.receipt_id] = receipt
        groups.setdefault((receipt.domain, receipt.shard_id), []).append(receipt)

    results: list[dict[str, Any]] = []
    for (domain, shard_id), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        group_ids = {receipt.receipt_id for receipt in group}
        valid_edges: list[tuple[str, str]] = []
        rejected_edges: list[dict[str, Any]] = []
        for receipt in sorted(group, key=lambda item: item.receipt_id):
            for covered_id in receipt.covers:
                if covered_id not in receipt_by_id:
                    fail(
                        Code.INVALID_RECEIPT,
                        "receipt covers an unknown receipt identifier",
                        receipt_id=receipt.receipt_id,
                        covered_id=covered_id,
                    )
                if covered_id not in group_ids:
                    fail(
                        Code.INVALID_RECEIPT,
                        "receipt coverage must stay within the same domain and shard",
                        receipt_id=receipt.receipt_id,
                        covered_id=covered_id,
                        shard_id=shard_id,
                    )
                covered = receipt_by_id[covered_id]
                if receipt.parameter_version <= covered.parameter_version:
                    rejected_edges.append(
                        {
                            "from": receipt.receipt_id,
                            "to": covered_id,
                            "code": int(Code.RECEIPT_OVERRIDE_REJECTED),
                            "code_name": Code.RECEIPT_OVERRIDE_REJECTED.name,
                            "reason": "effective receipt cannot be overwritten by an equal or lower parameter version",
                            "from_version": receipt.parameter_version,
                            "to_version": covered.parameter_version,
                        }
                    )
                    continue
                valid_edges.append((receipt.receipt_id, covered_id))

        dominated: set[str] = {covered for _, covered in valid_edges}
        candidates = [receipt for receipt in group if receipt.receipt_id not in dominated]
        if not candidates:
            # Strictly increasing versions make this unreachable, but keep a
            # defensive business error rather than an ambiguous ledger.
            fail(Code.INVALID_RECEIPT, "receipt coverage graph has no root", domain=domain, shard_id=shard_id)

        has_coverage = bool(valid_edges)
        if has_coverage:
            candidates.sort(
                key=lambda receipt: (
                    -receipt.parameter_version,
                    receipt.elapsed_ms,
                    receipt.source_file,
                    receipt.receipt_id,
                )
            )
            basis = ["highest_effective_parameter_version", "integer_elapsed_ms", "source_filename_asc"]
        else:
            candidates.sort(
                key=lambda receipt: (
                    -receipt.parameter_version,
                    receipt.source_file,
                    receipt.receipt_id,
                )
            )
            basis = ["parameter_version_desc", "source_filename_asc", "receipt_id_asc"]
        authority = candidates[0]

        covered_by_authority: list[str] = []
        queue = list(authority.covers)
        seen: set[str] = {authority.receipt_id}
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            covered_by_authority.append(current)
            current_receipt = receipt_by_id[current]
            queue.extend(current_receipt.covers)
        superseded = sorted(receipt.receipt_id for receipt in group if receipt.receipt_id != authority.receipt_id)
        results.append(
            {
                "domain": domain,
                "shard_id": shard_id,
                "authority_receipt_id": authority.receipt_id,
                "authority_source_file": authority.source_file,
                "authority_parameter_version": authority.parameter_version,
                "authority_elapsed_ms": authority.elapsed_ms,
                "authority_sample_count": authority.sample_count,
                "authority_sample_fingerprint": authority.sample_fingerprint,
                "decision_basis": basis if len(group) > 1 else ["only_receipt"],
                "covered_refs": sorted(covered_by_authority),
                "superseded_refs": superseded,
                "rejected_overrides": rejected_edges,
                "group_size": len(group),
                "receipts": [receipt.to_dict() for receipt in sorted(group, key=lambda item: item.receipt_id)],
            }
        )

    rejected_total = sum(len(result["rejected_overrides"]) for result in results)
    return {
        "groups": results,
        "group_count": len(results),
        "receipt_count": len(receipts),
        "rejected_override_count": rejected_total,
        "authority_count": len(results),
    }


def receipt_authority_diagnostics(
    authority: Mapping[str, Any],
    sample_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare authoritative receipt sample counts with the actual plan."""

    plans = {shard["shard_id"]: shard for shard in sample_plan.get("shards", [])}
    mismatches: list[dict[str, Any]] = []
    matched: list[str] = []
    for group in authority.get("groups", []):
        shard_id = group["shard_id"]
        plan = plans.get(shard_id)
        expected = group.get("authority_sample_count")
        if plan is None:
            mismatches.append(
                {
                    "shard_id": shard_id,
                    "reason": "authoritative receipt references a shard absent from the sample plan",
                }
            )
            continue
        actual = len(plan["documents"])
        if expected is not None and expected != actual:
            mismatches.append(
                {
                    "shard_id": shard_id,
                    "reason": "receipt sample_count differs from the sample plan",
                    "receipt_sample_count": expected,
                    "actual_sample_count": actual,
                }
            )
        else:
            matched.append(shard_id)
    return {
        "matched_shards": sorted(matched),
        "mismatches": sorted(mismatches, key=lambda item: item["shard_id"]),
        "authority_groups": authority.get("group_count", 0),
    }
