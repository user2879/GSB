"""Two-level capacity adjudication and truncation audit."""

from __future__ import annotations

from typing import Any


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, int, str]:
    # Higher numeric priority wins; earlier admission ordinal wins next; the
    # specification then requires lexicographic log_id tie-breaking.
    return (
        -int(candidate["priority"]),
        int(candidate["arrival_ordinal"]),
        str(candidate["log_id"]),
    )


def apply_global_budget(
    buckets: dict[str, dict[str, Any]],
    global_budget: int,
) -> dict[str, Any]:
    if global_budget <= 0:
        raise ValueError("global budget must be positive")

    candidates = [
        {
            **slot,
            "bucket_id": bucket_id,
            "priority": int(bucket["priority"]),
        }
        for bucket_id, bucket in sorted(buckets.items())
        for slot in bucket["slots"]
    ]
    candidates.sort(key=_candidate_sort_key)
    ranked: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        ordered_candidate = {**candidate, "global_rank": rank}
        retained = rank <= global_budget
        ordered_candidate["rank_decision"] = (
            "retained_by_global_budget"
            if retained
            else "truncated_by_global_budget"
        )
        ordered_candidate["rank_basis"] = (
            f"priority={candidate['priority']};"
            f"admission_ordinal={candidate['arrival_ordinal']};"
            f"log_id={candidate['log_id']};rank={rank};budget={global_budget}"
        )
        ranked.append(ordered_candidate)

    retained_records = [candidate for candidate in ranked if candidate["global_rank"] <= global_budget]
    global_truncated = [
        candidate
        for candidate in ranked
        if candidate["global_rank"] > global_budget
    ]

    final_samples = [
        {
            "rank": candidate["global_rank"],
            "bucket_id": candidate["bucket_id"],
            "slot_no": candidate["slot_no"],
            "arrival_ordinal": candidate["arrival_ordinal"],
            "arrival_seq": candidate["arrival_seq"],
            "record_index": candidate["record_index"],
            "log_id": candidate["log_id"],
            "timestamp": candidate["timestamp"],
            "device_id": candidate["device_id"],
            "severity": candidate["severity"],
            "template_id": candidate["template_id"],
            "metric": candidate["metric"],
            "quantized_weight": candidate["quantized_weight"],
            "priority": candidate["priority"],
            "truncation_layer": None,
            "basis": candidate["rank_basis"],
        }
        for candidate in retained_records
    ]

    audit: list[dict[str, Any]] = []
    retained_ids = {sample["log_id"] for sample in final_samples}
    global_cut_ids = {candidate["log_id"] for candidate in global_truncated}
    replacement_context: dict[str, dict[str, Any]] = {}
    for bucket_id, bucket in sorted(buckets.items()):
        for event in bucket["history"]:
            if event.get("replaced_log_id"):
                replacement_context[event["replaced_log_id"]] = {
                    "bucket_id": bucket_id,
                    "replaced_by": event["log_id"],
                    "ordinal": event["arrival_ordinal"],
                    "slot_no": event["target_slot_no"],
                }

    for bucket_id, bucket in sorted(buckets.items()):
        for event in bucket["history"]:
            log_id = event["log_id"]
            if not event["admitted"]:
                layer = "bucket"
                decision = "bucket_rejected"
                basis = event["basis"]
            elif log_id in retained_ids:
                layer = None
                decision = "retained"
                basis = f"retained:slot={event['slot_no']};bucket_capacity={bucket['capacity']}"
            elif log_id in global_cut_ids:
                layer = "global"
                decision = "global_truncated"
                candidate = next(
                    item for item in global_truncated if item["log_id"] == log_id
                )
                basis = candidate["rank_basis"]
            elif log_id in replacement_context:
                layer = "bucket"
                decision = "bucket_replaced"
                context = replacement_context[log_id]
                basis = (
                    f"slot={context['slot_no']};replaced_by={context['replaced_by']};"
                    f"at_ordinal={context['ordinal']}"
                )
            else:
                layer = "bucket"
                decision = "bucket_rejected"
                basis = event["basis"]
            audit.append(
                {
                    "log_id": log_id,
                    "bucket_id": bucket_id,
                    "arrival_seq": event["arrival_seq"],
                    "record_index": event["record_index"],
                    "arrival_ordinal": event["arrival_ordinal"],
                    "slot_no": event["slot_no"],
                    "admission_decision": event["decision"],
                    "decision": decision,
                    "truncation_layer": layer,
                    "basis": basis,
                }
            )

    return {
        "global_budget": global_budget,
        "candidate_count": len(candidates),
        "consumed_budget": len(final_samples),
        "budget_exhausted": len(candidates) > global_budget,
        "ranking_rule": "priority_desc,admission_ordinal_asc,log_id_asc",
        "ranked_candidates": ranked,
        "global_truncated": global_truncated,
        "final_samples": final_samples,
        "audit": audit,
    }

