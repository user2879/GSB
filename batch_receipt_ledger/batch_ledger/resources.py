from __future__ import annotations

from collections import defaultdict
from typing import Any

from .fixedpoint import Q, ceil_div, ratio_permille, ratio_q
from .model import MISSING, CanonicalReceipt, is_missing


def _positive_int_or_default(value: Any, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return int(value)
    return default


def _max_present_int(values: list[Any]) -> Any:
    present = [int(value) for value in values if not is_missing(value)]
    return max(present) if present else MISSING


def _single_value(values: list[Any]) -> Any:
    present = [value for value in values if not is_missing(value)]
    unique = set(present)
    if not unique:
        return MISSING
    return sorted(unique, key=lambda value: str(value))[0]


def build_resource_entry(
    batch_id: str,
    worker_id: str,
    items: list[CanonicalReceipt],
    config: dict[str, Any],
) -> dict[str, Any]:
    model_dim = _positive_int_or_default(config.get("model_dim"), 16)
    hidden_dim = _positive_int_or_default(config.get("hidden_dim"), 32)
    quant_bits = _positive_int_or_default(config.get("quant_bits", 8), 8)
    budget_ms = _positive_int_or_default(config.get("budget_ms", 100), 100)
    capacity_mac_per_ms = _positive_int_or_default(
        config.get("capacity_mac_per_ms", 100_000), 100_000
    )

    request_count = len({item.request_id for item in items})
    elapsed_ms = _max_present_int([item.elapsed_ms for item in items])
    param_version = _single_value([item.param_version for item in items])
    param_digest = _single_value([item.param_digest for item in items])
    param_count = model_dim * hidden_dim + hidden_dim + hidden_dim * model_dim + model_dim
    macs_per_request = model_dim * hidden_dim + hidden_dim * model_dim
    mac_count = request_count * macs_per_request
    param_bytes = ceil_div(param_count * quant_bits, 8)
    mac_fixed_q = mac_count * Q
    required_time_q = ceil_div(mac_count * Q, max(1, capacity_mac_per_ms))

    if is_missing(elapsed_ms) or elapsed_ms == 0:
        mac_per_ms_q = MISSING if is_missing(elapsed_ms) else 0
        utilization_q = MISSING if is_missing(elapsed_ms) else 0
        utilization_permille = MISSING if is_missing(elapsed_ms) else 0
    else:
        mac_per_ms_q = mac_count * Q // int(elapsed_ms)
        utilization_q = min(Q, int(elapsed_ms) * Q // max(1, budget_ms))
        utilization_permille = min(1000, int(elapsed_ms) * 1000 // max(1, budget_ms))

    return {
        "resource_key": f"resource:{batch_id}:{worker_id}",
        "batch_id": batch_id,
        "worker_id": worker_id,
        "request_count": request_count,
        "receipt_count": len(items),
        "elapsed_ms": elapsed_ms,
        "param_version": param_version,
        "param_digest": param_digest,
        "model_dim": model_dim,
        "hidden_dim": hidden_dim,
        "quant_bits": quant_bits,
        "fixed_point_scale": Q,
        "param_count": param_count,
        "quantized_param_bytes": param_bytes,
        "macs_per_request": macs_per_request,
        "mac_count": mac_count,
        "mac_fixed_q": mac_fixed_q,
        "modeled_service_time_q": required_time_q,
        "mac_per_ms_q": mac_per_ms_q,
        "budget_ms": budget_ms,
        "utilization_q": utilization_q,
        "utilization_permille": utilization_permille,
    }


def batch_worker_groups(
    receipts: list[CanonicalReceipt],
    batch_worker_assignment: dict[str, str] | None = None,
) -> list[tuple[str, str, list[CanonicalReceipt]]]:
    assignment = batch_worker_assignment or {}
    groups: dict[tuple[str, str], list[CanonicalReceipt]] = defaultdict(list)
    for receipt in receipts:
        worker_id = receipt.worker_id
        if worker_id is MISSING:
            worker_id = assignment.get(receipt.batch_id, MISSING)
        groups[(receipt.batch_id, str(worker_id))].append(receipt)
    return [(batch_id, worker_id, groups[(batch_id, worker_id)]) for batch_id, worker_id in sorted(groups)]


def build_resource_view(
    receipts: list[CanonicalReceipt],
    config: dict[str, Any],
    *,
    batch_worker_assignment: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    return [
        build_resource_entry(batch_id, worker_id, items, config)
        for batch_id, worker_id, items in batch_worker_groups(receipts, batch_worker_assignment)
    ]


def build_invalidation_view(
    *,
    workers: list[dict[str, Any]],
    baseline_worker_versions: dict[str, int],
    resource_entries: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    current_versions = {worker["worker_id"]: int(worker["param_version"]) for worker in workers}
    changes: list[dict[str, Any]] = []
    changed_workers: set[str] = set()
    for worker_id in sorted(set(current_versions) | set(baseline_worker_versions)):
        before = baseline_worker_versions.get(worker_id)
        after = current_versions.get(worker_id)
        if before != after:
            changed_workers.add(worker_id)
            changes.append(
                {
                    "worker_id": worker_id,
                    "from_version": before,
                    "to_version": after,
                    "dependency_edge": "worker_param_version->batch->request_attribution",
                }
            )

    invalid_resource_keys: set[str] = set()
    invalid_batch_ids: set[str] = set()
    invalid_reasons: dict[str, str] = {}
    for entry in resource_entries:
        worker_id = entry["worker_id"]
        version = entry["param_version"]
        if worker_id in changed_workers:
            change = next(item for item in changes if item["worker_id"] == worker_id)
            old_version = change["from_version"]
            if version == old_version:
                invalid_resource_keys.add(entry["resource_key"])
                invalid_batch_ids.add(entry["batch_id"])
                invalid_reasons[entry["resource_key"]] = (
                    f"references {worker_id} param_version {old_version}"
                )

    invalid_decision_keys: set[str] = set()
    invalid_request_ids: set[str] = set()
    for decision in decisions:
        if decision["authoritative_batch_id"] in invalid_batch_ids:
            invalid_decision_keys.add(decision["decision_key"])
            invalid_request_ids.add(decision["request_id"])

    reuse_entries: list[dict[str, Any]] = []
    for entry in resource_entries:
        if entry["resource_key"] in invalid_resource_keys:
            continue
        worker_id = entry["worker_id"]
        if worker_id not in current_versions:
            reason = "worker_reference_missing_but_entry_preserved"
        elif worker_id not in changed_workers:
            reason = "worker_param_version_unchanged"
        elif entry["param_version"] is MISSING:
            reason = "version_unrecorded_entry_not_version_referenced"
        else:
            reason = "entry_references_current_param_version"
        reuse_entries.append(
            {
                "reuse_key": f"reuse:{entry['resource_key']}",
                "resource_key": entry["resource_key"],
                "batch_id": entry["batch_id"],
                "worker_id": worker_id,
                "reason": reason,
            }
        )

    reusable_decision_keys = sorted(
        item["decision_key"]
        for item in decisions
        if item["decision_key"] not in invalid_decision_keys
    )
    return {
        "changes": changes,
        "invalid_resource_keys": sorted(invalid_resource_keys),
        "invalid_batch_ids": sorted(invalid_batch_ids),
        "invalid_decision_keys": sorted(invalid_decision_keys),
        "invalid_request_ids": sorted(invalid_request_ids),
        "reuse_entries": sorted(reuse_entries, key=lambda item: item["reuse_key"]),
        "reusable_decision_keys": reusable_decision_keys,
        "reuse_count": len(reuse_entries),
        "invalid_batch_count": len(invalid_batch_ids),
        "invalid_attribution_count": len(invalid_decision_keys),
    }


def summarize_resources(entries: list[dict[str, Any]]) -> dict[str, Any]:
    total_macs = sum(int(entry["mac_count"]) for entry in entries)
    total_params = sum(int(entry["param_count"]) for entry in entries)
    elapsed_values = [int(entry["elapsed_ms"]) for entry in entries if entry["elapsed_ms"] is not MISSING]
    return {
        "entry_count": len(entries),
        "total_mac_count": total_macs,
        "total_param_count_per_worker_batch": total_params,
        "known_elapsed_batch_count": len(elapsed_values),
        "missing_elapsed_batch_count": len(entries) - len(elapsed_values),
        "duplicate_rate_aggregate_permille_note": "see batch_stats",
        "resource_entry_order": [entry["resource_key"] for entry in entries],
        "aggregate_utilization_q": ratio_q(sum(elapsed_values), len(entries) * 100) if entries else 0,
    }
