"""Diagnostic report: staleness distribution, tie adjudications, refusal
reason statistics and budget/cache diagnostics."""
from __future__ import annotations

from collections import Counter


def build_report(load_result, requests, authorities, rejections,
                 duplicate_delivery, resolutions, ranking_results, pending,
                 failures, ledger, cache_book, recovery: dict) -> dict:
    staleness_distribution = {}
    state_counter = Counter()
    for request in requests:
        request_id = request["request_id"]
        per_request = Counter()
        staleness_values = []
        for (res_request, item_id, slot_key), resolution in resolutions.items():
            if res_request != request_id:
                continue
            per_request[resolution.state] += 1
            state_counter[resolution.state] += 1
            if resolution.staleness_ms is not None:
                staleness_values.append(resolution.staleness_ms)
        staleness_distribution[request_id] = {
            "states": dict(sorted(per_request.items())),
            "staleness_ms": sorted(staleness_values),
            "candidate_count": len(request["candidates"])}

    reason_counter = Counter(item.reason for item in rejections)
    tie_records = [{
        "group_key": record.group_key, "slot_key": record.slot_key,
        "winner_digest": record.receipt.digest,
        "winner_rev_seq": record.receipt.rev_seq,
        "winner_batch_seq": record.receipt.batch_seq}
        for record in authorities.values() if record.tie_resolved]

    pruned = []
    for record in authorities.values():
        for item in record.pruned:
            pruned.append({"slot_key": record.slot_key,
                           "group_key": record.group_key, **item})

    return {
        "recovery": recovery,
        "requests_total": len(requests),
        "requests_ranked": len(ranking_results),
        "requests_pending": pending,
        "global_state_distribution": dict(sorted(state_counter.items())),
        "per_request_staleness": staleness_distribution,
        "tie_adjudications": tie_records,
        "tie_count": len(tie_records),
        "refusal_reason_counts": dict(sorted(reason_counter.items())),
        "refusals": [item.__dict__ for item in rejections],
        "duplicate_deliveries": duplicate_delivery,
        "pruned": pruned,
        "validation_failures": failures,
        "budget_ledger": ledger,
        "cache": getattr(cache_book, "report_summary",
                         cache_book.summary()),
    }
