"""Authoritative recompute, byte-level convergence checks and preview guard.

The authority always recomputes expected derived state and aggregate digests
from the compiled plan.  Receipts are compared using canonical JSON byte
equality; UI preview caches are never allowed to replace the authority.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .errors import BusinessError, ErrorCode
from .fixedpoint import Grid, derive_qvalue
from .models import ManifestVersion, Receipt
from .sample import aggregate_shards
from .util import canonical_json


def expected_node_state(plan_node: dict[str, Any], node_id: str,
                        shard_index: int, shard_count: int,
                        grid: Grid) -> dict[str, Any]:
    items = plan_node["items"]
    qvalues = {key: int(payload["qvalue"]) for key, payload in items.items()}
    derived_state: dict[str, int] = {}
    for key in sorted(items):
        payload = items[key]
        if not payload.get("is_derived"):
            continue
        coeffs = {dep: int(q) for dep, q in payload["derived_from_q"].items()}
        derived_state[key] = derive_qvalue(coeffs, qvalues, grid)
    aggregate = aggregate_shards(
        [qvalues[key] for key in sorted(qvalues)], node_id, shard_count,
    )
    aggregate["shard_index"] = shard_index
    aggregate_digest = hashlib.sha256(
        json.dumps(aggregate, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "derived_state": dict(sorted(derived_state.items())),
        "aggregate_digest": aggregate_digest,
    }


def _canonical_bytes(obj: Any) -> bytes:
    return canonical_json(obj)


def verify_convergence(version: ManifestVersion, plan: dict[str, Any],
                       receipts: list[Receipt], grid: Grid,
                       ) -> dict[str, Any]:
    """Compare each expected shard receipt against actual receipts."""
    expected_index: dict[tuple[str, int], dict[str, Any]] = {}
    for node_id in version.targets:
        plan_node = plan["nodes"][node_id]
        shard_count = int(plan_node["shard_count"])
        for shard_index in range(shard_count):
            expected_index[(node_id, shard_index)] = expected_node_state(
                plan_node, node_id, shard_index, shard_count, grid,
            )
    actual_index = {(r.node_id, r.shard_index): r for r in receipts}
    deviations: list[dict[str, Any]] = []
    for key in sorted(expected_index):
        node_id, shard_index = key
        expected = expected_index[key]
        actual = actual_index.get(key)
        if actual is None:
            deviations.append({
                "node_id": node_id,
                "shard_index": shard_index,
                "reason": "missing_receipt",
                "expected_derived_state": expected["derived_state"],
                "expected_aggregate_digest": expected["aggregate_digest"],
                "actual": None,
            })
            continue
        actual_state = dict(sorted(actual.derived_state.items()))
        if _canonical_bytes(actual_state) != _canonical_bytes(expected["derived_state"]):
            deviations.append({
                "node_id": node_id,
                "shard_index": shard_index,
                "reason": "derived_state_mismatch",
                "expected_derived_state": expected["derived_state"],
                "actual_derived_state": actual_state,
            })
        if actual.aggregate_digest != expected["aggregate_digest"]:
            deviations.append({
                "node_id": node_id,
                "shard_index": shard_index,
                "reason": "aggregate_digest_mismatch",
                "expected_aggregate_digest": expected["aggregate_digest"],
                "actual_aggregate_digest": actual.aggregate_digest,
            })
    if deviations:
        raise BusinessError(
            ErrorCode.E_CONVERGE,
            f"convergence failed for version {version.version}",
            deviations,
        )
    return {
        "version": version.version,
        "converged": True,
        "checked_shards": len(expected_index),
        "receipts": len(receipts),
    }


def verify_preview(version: ManifestVersion, plan: dict[str, Any],
                   preview_cache: dict[str, Any], grid: Grid) -> dict[str, Any]:
    """Detect the first node/field where the UI cache diverges from authority."""
    for node_id in version.targets:
        plan_node = plan["nodes"][node_id]
        shard_count = int(plan_node["shard_count"])
        cached_node = preview_cache.get(node_id)
        if cached_node is None:
            raise BusinessError(
                ErrorCode.E_PREVIEW,
                "preview cache missing node entry",
                [{"node_id": node_id, "field": "<node>", "reason": "missing_preview_node"}],
            )
        for shard_index in range(shard_count):
            expected = expected_node_state(plan_node, node_id, shard_index,
                                           shard_count, grid)
            cached_shard = cached_node.get("shards", {}).get(str(shard_index))
            if cached_shard is None:
                raise BusinessError(
                    ErrorCode.E_PREVIEW,
                    "preview cache missing shard entry",
                    [{"node_id": node_id, "field": f"shards.{shard_index}",
                      "reason": "missing_preview_shard"}],
                )
            if _canonical_bytes(cached_shard.get("derived_state")) != \
                    _canonical_bytes(expected["derived_state"]):
                first_field = _first_dict_diff(
                    cached_shard.get("derived_state", {}), expected["derived_state"])
                raise BusinessError(
                    ErrorCode.E_PREVIEW,
                    "preview derived state diverges from authoritative recompute",
                    [{"node_id": node_id,
                      "field": f"shards.{shard_index}.derived_state.{first_field}",
                      "expected": expected["derived_state"].get(first_field),
                      "actual": cached_shard.get("derived_state", {}).get(first_field)}],
                )
            if cached_shard.get("aggregate_digest") != expected["aggregate_digest"]:
                raise BusinessError(
                    ErrorCode.E_PREVIEW,
                    "preview aggregate digest diverges from authority",
                    [{"node_id": node_id,
                      "field": f"shards.{shard_index}.aggregate_digest",
                      "expected": expected["aggregate_digest"],
                      "actual": cached_shard.get("aggregate_digest")}],
                )
    return {"version": version.version, "preview_consistent": True}


def _first_dict_diff(actual: dict[str, Any], expected: dict[str, Any]) -> str:
    for key in sorted(set(actual) | set(expected)):
        if actual.get(key) != expected.get(key):
            return key
    return "<unknown>"


def build_authoritative_preview(version: ManifestVersion, plan: dict[str, Any],
                                grid: Grid) -> dict[str, Any]:
    preview: dict[str, Any] = {}
    for node_id in version.targets:
        plan_node = plan["nodes"][node_id]
        shard_count = int(plan_node["shard_count"])
        preview[node_id] = {"shards": {}}
        for shard_index in range(shard_count):
            preview[node_id]["shards"][str(shard_index)] = expected_node_state(
                plan_node, node_id, shard_index, shard_count, grid,
            )
    return preview
