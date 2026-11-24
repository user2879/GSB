from __future__ import annotations

from collections import defaultdict
from typing import Any

from .fixedpoint import ratio_permille, ratio_q
from .model import MISSING, CanonicalReceipt


def _missing_last_int(value: Any) -> tuple[int, int]:
    return (1, 0) if value is MISSING else (0, -int(value))


def _version_rank(receipt: CanonicalReceipt) -> tuple[Any, ...]:
    missing_flag, negative_version = _missing_last_int(receipt.param_version)
    return (
        missing_flag,
        negative_version,
        receipt.batch_id,
        receipt.receipt_ref,
        receipt.request_id,
    )


def _coverage_rank(receipt: CanonicalReceipt) -> tuple[Any, ...]:
    elapsed_missing = 1 if receipt.elapsed_ms is MISSING else 0
    elapsed_value = 0 if receipt.elapsed_ms is MISSING else int(receipt.elapsed_ms)
    missing_version, negative_version = _missing_last_int(receipt.param_version)
    return (
        elapsed_missing,
        elapsed_value,
        missing_version,
        negative_version,
        receipt.batch_id,
        receipt.receipt_ref,
    )


def adjudicate_requests(
    receipts: list[CanonicalReceipt],
    *,
    merge_rule: str = "coverage_then_version_v1",
) -> list[dict[str, Any]]:
    by_request: dict[str, list[CanonicalReceipt]] = defaultdict(list)
    for receipt in receipts:
        by_request[receipt.request_id].append(receipt)

    decisions: list[dict[str, Any]] = []
    for request_id in sorted(by_request):
        candidates = by_request[request_id]
        refs = {receipt.receipt_ref for receipt in candidates}
        covering = [
            receipt
            for receipt in candidates
            if receipt.supersedes_ref is not MISSING and receipt.supersedes_ref in refs
        ]

        coverage_used = bool(covering) and merge_rule == "coverage_then_version_v1"
        if coverage_used:
            ranked = sorted(covering, key=_coverage_rank)
            winner = ranked[0]
            basis = "coverage_ref+elapsed_ms_asc"
            coverage_ref = winner.supersedes_ref
            ranked_refs = [item.receipt_ref for item in sorted(candidates, key=_coverage_rank)]
        else:
            ranked = sorted(candidates, key=_version_rank)
            winner = ranked[0]
            if merge_rule != "coverage_then_version_v1":
                basis = f"merge_rule:{merge_rule}:param_version_desc:batch_id_asc"
            elif any(receipt.supersedes_ref is not MISSING for receipt in candidates):
                basis = "coverage_unresolved:param_version_desc:batch_id_asc"
            else:
                basis = "param_version_desc:batch_id_asc"
            coverage_ref = MISSING
            ranked_refs = [item.receipt_ref for item in ranked]

        rejected = [ref for ref in ranked_refs if ref != winner.receipt_ref]
        decisions.append(
            {
                "decision_key": f"decision:{request_id}",
                "request_id": request_id,
                "authoritative_ref": winner.receipt_ref,
                "authoritative_batch_id": winner.batch_id,
                "authoritative_worker_id": winner.worker_id,
                "authoritative_param_version": winner.param_version,
                "authoritative_elapsed_ms": winner.elapsed_ms,
                "authoritative_rank_in_batch": winner.rank_in_batch,
                "basis": basis,
                "merge_rule": merge_rule,
                "coverage_ref": coverage_ref,
                "ranked_refs": ranked_refs,
                "rejected_refs": rejected,
                "candidate_count": len(candidates),
                "is_duplicate": len(candidates) > 1,
            }
        )
    return decisions


def aggregate_batches(
    receipts: list[CanonicalReceipt],
    decisions: list[dict[str, Any]],
    *,
    batch_worker_assignment: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assignment = batch_worker_assignment or {}
    authoritative_ref_by_request = {
        item["request_id"]: item["authoritative_ref"] for item in decisions
    }
    duplicate_requests = {item["request_id"] for item in decisions if item["is_duplicate"]}

    worker_groups: dict[tuple[str, str], list[CanonicalReceipt]] = defaultdict(list)
    batch_receipts: dict[str, list[CanonicalReceipt]] = defaultdict(list)
    for receipt in receipts:
        worker_id = receipt.worker_id if receipt.worker_id is not MISSING else MISSING
        worker_groups[(receipt.batch_id, str(worker_id))].append(receipt)
        batch_receipts[receipt.batch_id].append(receipt)

    worker_aggregates: list[dict[str, Any]] = []
    for batch_id, worker_id in sorted(worker_groups):
        items = worker_groups[(batch_id, worker_id)]
        assigned_worker_id = assignment.get(batch_id, MISSING)
        worker_aggregates.append(
            {
                "aggregate_key": f"{batch_id}|{worker_id}",
                "batch_id": batch_id,
                "worker_id": worker_id,
                "assigned_worker_id": assigned_worker_id,
                "effective_worker_id": worker_id if worker_id != MISSING else assigned_worker_id,
                "receipt_count": len(items),
                "request_ids": sorted({item.request_id for item in items}),
                "receipt_refs": sorted({item.receipt_ref for item in items}),
            }
        )

    batch_stats: list[dict[str, Any]] = []
    for batch_id in sorted(batch_receipts):
        items = batch_receipts[batch_id]
        request_ids = {item.request_id for item in items}
        duplicate_in_batch = sorted(request_ids & duplicate_requests)
        resolved = sum(
            1
            for request_id in duplicate_in_batch
            if authoritative_ref_by_request.get(request_id)
        )
        authoritative_count = sum(
            1
            for item in items
            if authoritative_ref_by_request.get(item.request_id) == item.receipt_ref
        )
        rejected_count = len(items) - authoritative_count
        batch_stats.append(
            {
                "batch_id": batch_id,
                "receipt_count": len(items),
                "request_count": len(request_ids),
                "within_batch_redundant_count": len(items) - len(request_ids),
                "duplicate_request_count": len(duplicate_in_batch),
                "duplicate_rate_q": ratio_q(len(duplicate_in_batch), len(request_ids)),
                "duplicate_rate_permille": ratio_permille(len(duplicate_in_batch), len(request_ids)),
                "resolved_request_count": resolved,
                "resolution_rate_q": ratio_q(resolved, len(duplicate_in_batch)),
                "resolution_rate_permille": ratio_permille(resolved, len(duplicate_in_batch)),
                "authoritative_receipt_count": authoritative_count,
                "rejected_receipt_count": rejected_count,
            }
        )
    return worker_aggregates, batch_stats
