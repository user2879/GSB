"""Stage 4b: deterministic budget scheduling and concurrent execution.

Three constraints are enforced:
  * global score-point budget (per run, analogous to one backfill batch quota);
  * per-slot point quota;
  * per-slot history retention (applied during adjudication).

Budget is claimed in one stable order (request_id, then slot_key, then
item_id) before any worker runs, so a thread pool of any size produces the
exact same claim/refund sequence, pending queue and failure ordering as the
serial executor. Workers only compute immutable per-unit artifacts.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from . import errors
from .fingerprint import cache_key
from .fixedpoint import WIDTH_TIER
from .scoring import score_slot


def _unit_plan(config, requests, slots):
    plan = []
    for request in sorted(requests, key=lambda r: r["request_id"]):
        request_id = request["request_id"]
        for item_id in [None, *sorted(request["candidates"])]:
            for slot_key in config["_slot_order"]:
                slot = slots[slot_key]
                if item_id is None and slot.layer == "request":
                    plan.append((request_id, None, slot_key, slot.dim))
                elif item_id is not None and slot.layer != "request":
                    plan.append((request_id, item_id, slot_key, slot.dim))
    return plan


def run(config, requests, slots, resolutions, staleness_fp,
        cache_book, concurrency_override=None):
    budget = int(config["budget"]["score_point_budget"])
    slot_quota = int(config["budget"]["slot_point_quota"])
    max_concurrency = int(config["scheduler"]["max_concurrency"])
    if concurrency_override is not None:
        max_concurrency = min(max_concurrency, int(concurrency_override))
    max_concurrency = max(1, max_concurrency)

    plan = _unit_plan(config, requests, slots)
    slot_spent: dict[str, int] = {}
    aborted_slots: set[str] = set()
    budget_events: list[dict] = []
    failures: list[dict] = []
    claim_by_request: dict[str, list] = {}
    executable: list[dict] = []
    insufficient: dict[str, list[dict]] = {}

    remaining = budget
    for request_id, item_id, slot_key, points in plan:
        if slot_key in aborted_slots:
            failures.append({
                "code": errors.E_SLOT_QUOTA_EXCEEDED,
                "slot_key": slot_key, "request_id": request_id,
                "item_id": item_id, "points": points,
                "message": "slot aborted after quota exhaustion"})
            continue
        if slot_spent.get(slot_key, 0) + points > slot_quota:
            aborted_slots.add(slot_key)
            failures.append({
                "code": errors.E_SLOT_QUOTA_EXCEEDED,
                "slot_key": slot_key, "request_id": request_id,
                "item_id": item_id, "points": points,
                "slot_spent": slot_spent.get(slot_key, 0),
                "slot_quota": slot_quota,
                "message": "single-slot point quota exceeded"})
            continue
        if remaining < points:
            budget_events.append({
                "request_id": request_id, "item_id": item_id,
                "slot_key": slot_key, "points": points,
                "event": "insufficient_budget",
                "remaining_before": remaining})
            insufficient.setdefault(request_id, []).append({
                "item_id": item_id, "slot_key": slot_key, "points": points})
            continue
        remaining -= points
        slot_spent[slot_key] = slot_spent.get(slot_key, 0) + points
        budget_events.append({
            "request_id": request_id, "item_id": item_id,
            "slot_key": slot_key, "points": points, "event": "claim",
            "remaining_after": remaining})
        unit = {"request_id": request_id, "item_id": item_id,
                "slot_key": slot_key,
                "resolution_key": (request_id, item_id, slot_key),
                "points": points,
                "unit_key": (request_id, item_id, slot_key)}
        executable.append(unit)
        claim_by_request.setdefault(request_id, []).append(unit)

    required_by_request: dict[str, int] = {}
    for request_id, item_id, slot_key, points in plan:
        if slot_key not in aborted_slots:
            required_by_request[request_id] = \
                required_by_request.get(request_id, 0) + points
    funded_by_request: dict[str, int] = {}
    for unit in executable:
        funded_by_request[unit["request_id"]] = \
            funded_by_request.get(unit["request_id"], 0) + unit["points"]

    pending: list[dict] = []
    executable_by_request: dict[str, list] = {}
    pending_ids = set()
    for request in sorted(requests, key=lambda r: r["request_id"]):
        request_id = request["request_id"]
        funded = funded_by_request.get(request_id, 0)
        required = required_by_request.get(request_id, 0)
        if funded < required:
            pending_ids.add(request_id)
            pending.append({
                "request_id": request_id,
                "code": errors.E_SCORE_BUDGET_PENDING,
                "funded_points": funded, "required_points": required,
                "missing_points": required - funded,
                "blocked_units": insufficient.get(request_id, []),
                "message": "insufficient global score point budget"})
            continue
        executable_by_request[request_id] = claim_by_request.get(
            request_id, [])

    refunded = 0
    for request_id in sorted(pending_ids):
        for unit in claim_by_request.get(request_id, []):
            refunded += unit["points"]
            remaining += unit["points"]
            budget_events.append({
                "request_id": request_id, "item_id": unit["item_id"],
                "slot_key": unit["slot_key"], "points": unit["points"],
                "event": "refund", "remaining_after": remaining})
    executable = [unit for unit in executable
                  if unit["request_id"] not in pending_ids]

    def work(unit):
        resolution = resolutions[unit["resolution_key"]]
        slot = slots[unit["slot_key"]]
        entry = score_slot(resolution, slot, config, staleness_fp)
        # Each worker builds its own dict; nothing is mutated across threads.
        entry = dict(entry)
        entry["request_id"] = unit["request_id"]
        entry["item_id"] = unit["item_id"]
        entry["points"] = unit["points"]
        return cache_book.record_score(dict(entry))

    score_entries: list[dict] = []
    if executable:
        with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            score_entries = list(pool.map(work, executable))
    score_entries.sort(key=lambda entry: (entry["request_id"],
                                          entry["item_id"] or "",
                                          entry["slot_key"]))

    ledger = {
        "budget": budget,
        "remaining": remaining,
        "consumed_points": budget - remaining,
        "refunded_points": refunded,
        "slot_spent": dict(sorted(slot_spent.items())),
        "slot_quota": slot_quota,
        "aborted_slots": sorted(aborted_slots),
        "events": budget_events,
    }
    return score_entries, executable_by_request, pending, failures, ledger


def rank_requests(config, requests, score_entries, slots, resolutions,
                  cache_book):
    entries_by_request: dict[str, list[dict]] = {}
    for entry in score_entries:
        entries_by_request.setdefault(entry["request_id"], []).append(entry)

    ranking_results = []
    for request in sorted(requests, key=lambda r: r["request_id"]):
        request_id = request["request_id"]
        entries = entries_by_request.get(request_id)
        if not entries:
            continue
        per_item: dict[str, list[dict]] = {}
        request_level: list[dict] = []
        for entry in entries:
            if entry["item_id"] is None:
                request_level.append(entry)
            else:
                per_item.setdefault(entry["item_id"], []).append(entry)
        if not per_item:
            continue

        candidate_rows = []
        referenced = []
        for item_id in sorted(per_item):
            item_entries = per_item[item_id]
            total = 0
            breakdown = []
            for entry in [*request_level, *item_entries]:
                total += entry["contribution_q16"]
                breakdown.append({
                    "slot_key": entry["slot_key"],
                    "item_id": entry["item_id"],
                    "state": entry["state"],
                    "staleness_ms": entry["staleness_ms"],
                    "contribution_q16": entry["contribution_q16"],
                    "score_cache_key": entry["score_cache_key"],
                    "used_digest": entry["used_digest"],
                    "vector_source": entry["vector_source"]})
                referenced.append(entry["score_cache_key"])
            candidate_rows.append({
                "item_id": item_id, "total_score_q16": int(total),
                "slot_breakdown": breakdown})

        candidate_rows.sort(key=lambda row: (-row["total_score_q16"],
                                             row["item_id"]))
        # The ranking cache key is derived solely from the slot score keys
        # referenced by this request: an unaffected request keeps the exact
        # same key, while any invalidated slot changes the referenced set.
        ranking_key = cache_key(["ranking", request_id,
                                 *sorted(set(referenced)), WIDTH_TIER])
        cache_book.record_ranking(request_id, ranking_key, referenced)
        ranking_results.append({
            "request_id": request_id,
            "rank_ms": request["rank_ms"],
            "ranking_cache_key": ranking_key,
            "referenced_score_cache_keys": sorted(set(referenced)),
            "candidates": candidate_rows,
            "ordering_tie_break": ["total_score_q16 desc", "item_id asc",
                                   "request_id asc"]})
    return ranking_results
