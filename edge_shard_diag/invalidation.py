from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .errors import Codes, DiagnosticError
from .spec import SpecBundle, stable_topological_order


def propagate_invalidation(
    bundle: SpecBundle,
    *,
    changed_samples: Iterable[str] = (),
    changed_nodes: Iterable[str] = (),
    changed_receipts: Iterable[str] = (),
    all_shards: bool = False,
    receipt_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    order = stable_topological_order(bundle.shards)
    shards_by_id = bundle.shards_by_id
    invalid_shards: set[str] = set(order) if all_shards else set()
    steps: list[dict[str, str]] = []

    def add_shard(shard_id: str, kind: str, source: str, basis: str) -> None:
        if shard_id not in invalid_shards:
            invalid_shards.add(shard_id)
            steps.append({"kind": kind, "source": source, "target": shard_id, "basis": basis})

    for sample_id in sorted(set(changed_samples)):
        for shard in bundle.shards:
            if sample_id in shard.sample_ids:
                add_shard(shard.id, "sample-seed", sample_id, f"shard directly references changed sample {sample_id}")
    for node_id in sorted(set(changed_nodes)):
        for shard in bundle.shards:
            if shard.node_id == node_id:
                add_shard(shard.id, "node-seed", node_id, f"shard is directly bound to changed node {node_id}")
    if all_shards:
        for shard_id in order:
            add_shard(shard_id, "generation-seed", "capacity-or-scale", "generation/session fixed-point change reaches every shard conclusion")

    changed = True
    while changed:
        changed = False
        for shard_id in order:
            shard = shards_by_id[shard_id]
            for dependency in sorted(shard.dependencies):
                if dependency in invalid_shards and shard_id not in invalid_shards:
                    add_shard(shard_id, "dependency-propagation", dependency, f"upstream invalidation follows {dependency} -> {shard_id}")
                    changed = True

    records = receipt_records or []
    receipts_by_id = {record["receipt_id"]: record for record in records}
    invalid_receipts: set[str] = set(changed_receipts)
    receipt_steps: list[dict[str, str]] = []

    def add_receipt(receipt_id: str, kind: str, source: str, basis: str) -> None:
        if receipt_id not in invalid_receipts:
            invalid_receipts.add(receipt_id)
            receipt_steps.append({"kind": kind, "source": source, "target": receipt_id, "basis": basis})

    for record in records:
        if record.get("shard_id") in invalid_shards:
            add_receipt(record["receipt_id"], "shard-receipt", record["shard_id"], "normalized receipt belongs to invalid shard")
        if record["receipt_id"] in set(changed_receipts):
            add_receipt(record["receipt_id"], "receipt-seed", record["receipt_id"], "receipt source was directly replaced")

    changed = True
    while changed:
        changed = False
        for record in sorted(records, key=lambda item: item["receipt_id"]):
            for ref in sorted(record.get("duplicate_refs", [])):
                if ref in invalid_receipts and record["receipt_id"] not in invalid_receipts and ref in receipts_by_id:
                    add_receipt(record["receipt_id"], "receipt-reference", ref, f"receipt normalization relies on referenced conclusion {ref}")
                    changed = True

    invalid_ordered = [shard_id for shard_id in order if shard_id in invalid_shards]
    invalid_receipt_ordered = sorted(invalid_receipts)
    return {
        "stable_topological_order": order,
        "invalid_shards": invalid_ordered,
        "invalid_receipts": invalid_receipt_ordered,
        "steps": steps + receipt_steps,
        "unchanged_shards": [shard_id for shard_id in order if shard_id not in invalid_shards],
    }


def json_canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def shard_record_path(directory: str | Path, shard_id: str) -> Path:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{shard_id}.json"


def write_shard_record(directory: str | Path, record: dict[str, Any]) -> Path:
    path = shard_record_path(directory, record["shard_id"])
    path.write_bytes(json_canonical_bytes(record))
    return path


def hash_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _first_difference(left: Any, right: Any, path: str = "$") -> str | None:
    if type(left) is not type(right):
        # ints and floats may compare equal but field type is still a divergence.
        if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool) and not isinstance(right, bool):
            if float(left) == float(right):
                return None
        return f"{path} (type {type(left).__name__}!={type(right).__name__})"
    if isinstance(left, dict):
        keys = sorted(set(left) | set(right))
        for key in keys:
            if key not in left or key not in right:
                return f"{path}.{key}"
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}.length"
        for index, (a, b) in enumerate(zip(left, right)):
            difference = _first_difference(a, b, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if left != right:
        return path
    return None


def compare_semantic_views(
    incremental: dict[str, dict[str, Any]],
    full: dict[str, dict[str, Any]],
    *,
    code: str = Codes.DIVERGE,
    order: list[str] | None = None,
) -> None:
    expected_order = order or sorted(set(incremental) | set(full))
    for shard_id in expected_order:
        if shard_id not in incremental:
            raise DiagnosticError(code, f"incremental view missing shard {shard_id}", field="$", details={"shard_id": shard_id})
        if shard_id not in full:
            raise DiagnosticError(code, f"full view missing shard {shard_id}", field="$", details={"shard_id": shard_id})
        field = _first_difference(incremental[shard_id], full[shard_id], "$")
        if field:
            field_name = field[2:] if field.startswith("$.") else field
            raise DiagnosticError(
                code,
                f"semantic views diverge at shard={shard_id} field={field_name}",
                field=field_name,
                details={"first_mismatch_shard": shard_id, "first_mismatch_field": field_name},
            )


def assert_byte_identical(path_before: dict[str, str], path_after: dict[str, str]) -> None:
    changed = [path for path, digest in path_before.items() if path_after.get(path) != digest]
    if changed:
        raise DiagnosticError(Codes.DIVERGE, f"unchanged shard artifact bytes changed: {changed[0]}", field=changed[0])
