from __future__ import annotations

from collections import defaultdict
from concurrent.futures import Executor
from typing import Any

from .config import RuntimeConfig
from .util import fixed_from_float, fixed_to_float, fixed_to_string, round_div, stable_hash


UNKNOWN_ATTRIBUTE = "__unknown__"


def _similarity_fixed(
    left: dict[str, Any],
    right: dict[str, Any],
    attributes: dict[str, dict[str, Any]],
    cfg: RuntimeConfig,
) -> tuple[int, list[str]]:
    left_attr = attributes.get(left["item_id"], {})
    right_attr = attributes.get(right["item_id"], {})
    left_category = left_attr.get("category", UNKNOWN_ATTRIBUTE)
    right_category = right_attr.get("category", UNKNOWN_ATTRIBUTE)
    left_price = left_attr.get("price_band", UNKNOWN_ATTRIBUTE)
    right_price = right_attr.get("price_band", UNKNOWN_ATTRIBUTE)
    weights = cfg.raw["similarity_weights"]
    similarity = 0
    matched = []
    if left_category == right_category:
        similarity += fixed_from_float(float(weights["category"]), cfg.scale)
        matched.append("category")
    if left_price == right_price:
        similarity += fixed_from_float(float(weights["price_band"]), cfg.scale)
        matched.append("price_band")
    return min(cfg.scale, similarity), matched


def _max_selected_similarity(
    candidate: dict[str, Any],
    selected: list[dict[str, Any]],
    attributes: dict[str, dict[str, Any]],
    cfg: RuntimeConfig,
) -> tuple[int, dict[str, Any] | None, list[str]]:
    best_similarity = -1
    best_reference = None
    best_matches: list[str] = []
    for reference in sorted(selected, key=lambda item: item["item_id"]):
        similarity, matches = _similarity_fixed(candidate, reference, attributes, cfg)
        if similarity > best_similarity:
            best_similarity = similarity
            best_reference = reference
            best_matches = matches
    return max(0, best_similarity), best_reference, best_matches


def _rerank_request(
    request_id: str,
    fused_records: list[dict[str, Any]],
    attributes: dict[str, dict[str, Any]],
    cfg: RuntimeConfig,
) -> dict[str, Any]:
    ordered = sorted(
        fused_records,
        key=lambda record: (-int(record["fusion_fixed"]), record["item_id"]),
    )
    before = [
        {
            "rank": index + 1,
            "item_id": record["item_id"],
            "fusion_fixed": record["fusion_fixed"],
            "fusion_score": record["fusion_score"],
        }
        for index, record in enumerate(ordered)
    ]
    window_size = int(cfg.raw["rerank_window_size"])
    threshold_fixed = fixed_from_float(float(cfg.raw["dedup_similarity_threshold"]), cfg.scale)
    factor_fixed = fixed_from_float(float(cfg.raw["diversity_discount_factor"]), cfg.scale)
    discount_records: list[dict[str, Any]] = []
    boundary_records: list[dict[str, Any]] = []
    after: list[dict[str, Any]] = []
    windows = []
    for window_index, start in enumerate(range(0, len(ordered), window_size)):
        window = ordered[start : start + window_size]
        window_before = [record["item_id"] for record in window]
        remaining = []
        for record in window:
            remaining.append(
                {
                    **record,
                    "current_fixed": int(record["fusion_fixed"]),
                    "pre_discount_fixed": int(record["fusion_fixed"]),
                    "discount_total_fixed": 0,
                    "discounted_refs": set(),
                    "edge_flags": list(record.get("edge_flags", [])),
                }
            )
        selected: list[dict[str, Any]] = []
        equality_recorded: set[tuple[str, str]] = set()
        while remaining:
            best = min(
                remaining,
                key=lambda record: (-int(record["current_fixed"]), record["item_id"]),
            )
            if selected:
                similarity, reference, matches = _max_selected_similarity(
                    best, selected, attributes, cfg
                )
                reference_id = reference["item_id"] if reference else None
                reference_pair = (best["item_id"], reference_id or "")
                if similarity > threshold_fixed and reference is not None:
                    if reference_pair not in best["discounted_refs"]:
                        before_fixed = int(best["current_fixed"])
                        after_fixed = round_div(before_fixed * factor_fixed, cfg.scale)
                        delta = before_fixed - after_fixed
                        best["current_fixed"] = after_fixed
                        best["discount_total_fixed"] += delta
                        best["discounted_refs"].add(reference_pair)
                        flag = "similarity_above_threshold_discount"
                        if flag not in best["edge_flags"]:
                            best["edge_flags"].append(flag)
                        discount_records.append(
                            {
                                "request_id": request_id,
                                "window_index": window_index,
                                "candidate_item_id": best["item_id"],
                                "reference_item_id": reference_id,
                                "similarity_fixed": similarity,
                                "similarity": fixed_to_float(similarity, cfg.places),
                                "similarity_fixed_string": fixed_to_string(similarity, cfg.places),
                                "matched_attributes": matches,
                                "before_fixed": before_fixed,
                                "after_fixed": after_fixed,
                                "discount_fixed": delta,
                                "discount": fixed_to_float(delta, cfg.places),
                                "discount_fixed_string": fixed_to_string(delta, cfg.places),
                                "criterion": flag,
                            }
                        )
                        continue
                elif similarity == threshold_fixed and reference is not None:
                    if reference_pair not in equality_recorded:
                        equality_recorded.add(reference_pair)
                        best["edge_flags"].append("similarity_equal_threshold_no_discount")
                        boundary_records.append(
                            {
                                "criterion": "similarity_equal_threshold_no_discount",
                                "request_id": request_id,
                                "window_index": window_index,
                                "candidate_item_id": best["item_id"],
                                "reference_item_id": reference_id,
                                "similarity_fixed": similarity,
                                "decision": "do_not_discount_because_rule_is_strictly_greater_than",
                            }
                        )
            selected.append(best)
            remaining.remove(best)

        window_after = sorted(
            selected,
            key=lambda record: (-int(record["current_fixed"]), record["item_id"]),
        )
        equal_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in window_after:
            equal_groups[int(record["current_fixed"])].append(record)
        for score, group in sorted(equal_groups.items()):
            if len(group) > 1:
                items = sorted(record["item_id"] for record in group)
                for record in group:
                    flag = "rerank_score_tie_item_id_order"
                    if flag not in record["edge_flags"]:
                        record["edge_flags"].append(flag)
                    boundary_records.append(
                        {
                            "criterion": flag,
                            "request_id": request_id,
                            "window_index": window_index,
                            "item_id": record["item_id"],
                            "tied_items": items,
                            "score_fixed": score,
                            "decision": "item_id_lexicographic",
                        }
                    )
        for record in window_after:
            final_rank = len(after) + 1
            after.append(
                {
                    "rank": final_rank,
                    "item_id": record["item_id"],
                    "fusion_fixed": record["fusion_fixed"],
                    "current_fixed": record["current_fixed"],
                    "current_score": fixed_to_float(record["current_fixed"], cfg.places),
                    "current_fixed_string": fixed_to_string(record["current_fixed"], cfg.places),
                    "discount_total_fixed": record["discount_total_fixed"],
                    "discount_total": fixed_to_float(record["discount_total_fixed"], cfg.places),
                    "fusion_cache_key": record["cache_key"],
                    "contributions": record["contributions"],
                    "references": record["references"],
                    "edge_flags": sorted(set(record["edge_flags"])),
                    "original_window": window_index,
                }
            )
        windows.append(
            {
                "window_index": window_index,
                "before_items": window_before,
                "after_items": [record["item_id"] for record in window_after],
                "discount_count": sum(
                    1
                    for record in discount_records
                    if record["request_id"] == request_id
                    and record["window_index"] == window_index
                ),
            }
        )
    after_ids = [record["item_id"] for record in after]
    before_ids = [record["item_id"] for record in before]
    result = {
        "request_id": request_id,
        "before": before,
        "after": after,
        "windows": windows,
        "discount_records": discount_records,
        "boundary_records": boundary_records,
        "order_changed": after_ids != before_ids,
        "before_items": before_ids,
        "after_items": after_ids,
    }
    result["cache_key"] = stable_hash(
        {
            "stage": "rerank",
            "request_id": request_id,
            "window_fingerprint": cfg.window_fingerprint,
            "attributes_fingerprint": "",
            "fusion_keys": [record["cache_key"] for record in ordered],
        }
    )
    return result


def rerank(
    fused_records: list[dict[str, Any]],
    attributes: dict[str, dict[str, Any]],
    attributes_fingerprint: str,
    cfg: RuntimeConfig,
    previous_cache: dict[str, dict[str, Any]] | None = None,
    pending_items: list[dict[str, str]] | None = None,
    executor: Executor | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    previous = previous_cache or {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in fused_records:
        grouped[record["request_id"]].append(record)
    pending_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in pending_items or []:
        pending_by_request[entry["request_id"]].append(entry)
    all_requests = sorted(set(grouped) | set(pending_by_request))
    tasks = []
    cache: dict[str, dict[str, Any]] = {}
    reused = 0
    for request_id in all_requests:
        records = sorted(
            grouped.get(request_id, []),
            key=lambda record: (-int(record["fusion_fixed"]), record["item_id"]),
        )
        key = stable_hash(
            {
                "stage": "rerank",
                "request_id": request_id,
                "window_fingerprint": cfg.window_fingerprint,
                "attributes_fingerprint": attributes_fingerprint,
                "fusion_keys": [record["cache_key"] for record in records],
            }
        )
        if key in previous and not pending_by_request.get(request_id):
            result = previous[key]
            result["cache_key"] = key
            cache[key] = result
            reused += 1
        else:
            tasks.append((request_id, records, key))

    def run(task: tuple[str, list[dict[str, Any]], str]) -> dict[str, Any]:
        request_id, records, key = task
        result = _rerank_request(request_id, records, attributes, cfg)
        result["cache_key"] = key
        return result

    if executor is not None and len(tasks) > 1:
        results = list(executor.map(run, tasks))
    else:
        results = [run(task) for task in tasks]
    for result in results:
        cache[result["cache_key"]] = result
    results_all = []
    for request_id in all_requests:
        matches = [result for result in cache.values() if result["request_id"] == request_id]
        if not matches:
            raise AssertionError(f"missing rerank result for {request_id}")
        result = matches[0]
        pending = sorted(
            pending_by_request.get(request_id, []),
            key=lambda item: item["item_id"],
        )
        results_all.append(
            {
                **result,
                "pending": [
                    {
                        "request_id": item["request_id"],
                        "item_id": item["item_id"],
                        "code": item.get("code", "E_RECOMPUTE_BUDGET_PENDING"),
                    }
                    for item in pending
                ],
                "complete": not pending,
            }
        )
    results_all.sort(key=lambda result: result["request_id"])
    discounts = [
        record
        for result in results_all
        for record in result["discount_records"]
    ]
    boundaries = [
        record
        for result in results_all
        for record in result["boundary_records"]
    ]
    telemetry = {
        "computed": len(tasks),
        "reused": reused,
        "active_keys": sorted(cache),
        "discount_records": discounts,
        "boundary_records": boundaries,
    }
    return results_all, cache, telemetry
