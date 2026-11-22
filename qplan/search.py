from __future__ import annotations

import itertools
from typing import Any, Iterable

from .canonical import content_digest
from .paths import AccessPath
from .spec import Spec


def _subset_key(total_cost: int, subset: Iterable[str]) -> tuple[int, int, tuple[str, ...]]:
    ordered = tuple(sorted(subset))
    return total_cost, len(ordered), ordered


def _evaluate(
    subset: Iterable[str],
    paths_by_query: dict[str, list[AccessPath]],
    estimates_by_query: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    subset_set = set(subset)
    total_cost = 0
    query_choices: dict[str, Any] = {}
    for query_id in sorted(paths_by_query):
        paths = paths_by_query[query_id]
        estimates = estimates_by_query[query_id]
        chosen_cost = None
        chosen_path = None
        eligible_summary: list[dict[str, Any]] = []
        rejected_summary: list[dict[str, Any]] = []
        for path in paths:
            estimate = estimates[path.id]
            eligible = path.index_name is None or path.index_name in subset_set
            if eligible:
                eligible_summary.append(
                    {
                        "path_id": path.id,
                        "kind": path.kind,
                        "index_name": path.index_name,
                        "estimated_cost": int(estimate["cost"]),
                    }
                )
                key = (int(estimate["cost"]), path.id)
                if chosen_path is None or key < (chosen_cost, chosen_path.id):
                    chosen_cost = int(estimate["cost"])
                    chosen_path = path
            else:
                rejected_summary.append(
                    {
                        "path_id": path.id,
                        "kind": path.kind,
                        "index_name": path.index_name,
                        "reason": "index_not_in_selected_set",
                    }
                )
        assert chosen_path is not None and chosen_cost is not None
        total_cost += chosen_cost
        query_choices[query_id] = {
            "selected_path_id": chosen_path.id,
            "selected_kind": chosen_path.kind,
            "selected_index": chosen_path.index_name,
            "estimated_cost": chosen_cost,
            "eligible_paths": sorted(eligible_summary, key=lambda item: item["path_id"]),
            "rejected_paths": sorted(rejected_summary, key=lambda item: item["path_id"]),
        }
    key = _subset_key(total_cost, subset_set)
    return {
        "key": key,
        "total_cost": total_cost,
        "selected_indexes": sorted(subset_set),
        "queries": query_choices,
    }


def _exhaustive(
    candidate_ids: tuple[str, ...],
    paths_by_query: dict[str, list[AccessPath]],
    estimates_by_query: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    best = None
    evaluations = 0
    for size in range(len(candidate_ids) + 1):
        for subset in itertools.combinations(candidate_ids, size):
            evaluations += 1
            result = _evaluate(subset, paths_by_query, estimates_by_query)
            if best is None or result["key"] < best["key"]:
                best = result
    assert best is not None
    best["evaluations"] = evaluations
    return best


def _greedy(
    candidate_ids: tuple[str, ...],
    paths_by_query: dict[str, list[AccessPath]],
    estimates_by_query: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    selected: list[str] = []
    current = _evaluate(selected, paths_by_query, estimates_by_query)
    trace = []
    evaluations = 1
    while True:
        best_next = None
        best_result = None
        for candidate in candidate_ids:
            if candidate in selected:
                continue
            evaluations += 1
            result = _evaluate(selected + [candidate], paths_by_query, estimates_by_query)
            key = (result["total_cost"], len(result["selected_indexes"]), tuple(result["selected_indexes"]))
            if best_next is None or key < best_next:
                best_next = key
                best_result = result
        if best_result is None:
            break
        if best_result["total_cost"] >= current["total_cost"]:
            break
        selected = list(best_result["selected_indexes"])
        trace.append(
            {
                "added": sorted(set(selected) - set(current["selected_indexes"])),
                "total_cost_before": current["total_cost"],
                "total_cost_after": best_result["total_cost"],
                "selected_indexes": sorted(selected),
            }
        )
        current = best_result
    current["evaluations"] = evaluations
    current["trace"] = trace
    return current


def search_indexes(
    spec: Spec,
    paths_by_query: dict[str, list[AccessPath]],
    estimates_by_query: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    candidate_ids = tuple(sorted(index.name for index in spec.candidate_indexes))
    candidate_by_id = {index.name: index for index in spec.candidate_indexes}
    mode = "exhaustive" if len(candidate_ids) <= spec.search.exhaustive_max_candidates else "greedy"
    exhaustive_result = None
    greedy_result = None
    if len(candidate_ids) <= spec.search.oracle_max_candidates:
        exhaustive_result = _exhaustive(candidate_ids, paths_by_query, estimates_by_query)
    if len(candidate_ids) <= spec.search.oracle_max_candidates:
        greedy_result = _greedy(candidate_ids, paths_by_query, estimates_by_query)
    elif mode == "greedy":
        greedy_result = _greedy(candidate_ids, paths_by_query, estimates_by_query)
    elif mode == "exhaustive":
        exhaustive_result = _exhaustive(candidate_ids, paths_by_query, estimates_by_query)

    diagnostics: list[dict[str, Any]] = []
    if mode == "exhaustive":
        applied = exhaustive_result
    else:
        applied = greedy_result
    truth_source = "greedy" if mode == "greedy" else "exhaustive"
    if exhaustive_result is not None and greedy_result is not None:
        exhaustive_set = set(exhaustive_result["selected_indexes"])
        greedy_set = set(greedy_result["selected_indexes"])
        if exhaustive_set != greedy_set:
            difference = sorted(exhaustive_set.symmetric_difference(greedy_set))
            diagnostics.append(
                {
                    "code": "E_SEARCH",
                    "message": "exhaustive and greedy index search produced different index sets",
                    "difference_indexes": difference,
                    "truth_source": "exhaustive",
                    "exhaustive_indexes": sorted(exhaustive_set),
                    "greedy_indexes": sorted(greedy_set),
                }
            )
            applied = exhaustive_result
            truth_source = "exhaustive"
        else:
            truth_source = "exhaustive" if exhaustive_result is not None else truth_source
            applied = exhaustive_result or applied
    assert applied is not None

    selected_set = set(applied["selected_indexes"])
    applied_objective = _evaluate(selected_set, paths_by_query, estimates_by_query)
    selected_candidates = []
    unselected_candidates = []
    for index_id in candidate_ids:
        index = candidate_by_id[index_id]
        item = {
            "index": index_id,
            "table": index.table,
            "columns": list(index.columns),
        }
        if index_id in selected_set:
            item["reason"] = "selected_by_search"
            selected_candidates.append(item)
            continue
        with_candidate = _evaluate(selected_set | {index_id}, paths_by_query, estimates_by_query)
        before_cost = applied_objective["total_cost"]
        after_cost = with_candidate["total_cost"]
        if after_cost > before_cost:
            reason = "not_selected:would_increase_cost"
        elif after_cost == before_cost and len(with_candidate["selected_indexes"]) > len(applied["selected_indexes"]):
            reason = "not_selected:tie_break_prefers_smaller_index_set"
        elif exhaustive_result is not None:
            reason = "not_selected:no_marginal_gain_under_exhaustive_truth"
        else:
            reason = "not_selected:greedy_stopped_at_local_optimum"
        item["reason"] = reason
        item["total_cost_without"] = before_cost
        item["total_cost_with"] = after_cost
        unselected_candidates.append(item)

    query_choices = applied_objective["queries"]
    return {
        "search_model_version": "qplan-index-search-v1",
        "candidate_count": len(candidate_ids),
        "candidate_enumeration": list(candidate_ids),
        "switch_rule": {
            "mode": mode,
            "exhaustive_max_candidates": spec.search.exhaustive_max_candidates,
            "oracle_max_candidates": spec.search.oracle_max_candidates,
            "rule": "exhaustive when candidate_count <= exhaustive_max_candidates; otherwise greedy",
        },
        "primary_mode": mode,
        "mode": mode,
        "search_mode": mode,
        "truth_source": truth_source,
        "applied_selected_indexes": sorted(selected_set),
        "selected_indexes": sorted(selected_set),
        "selected_candidates": selected_candidates,
        "unselected_candidates": unselected_candidates,
        "query_choices": query_choices,
        "total_estimated_cost": applied_objective["total_cost"],
        "exhaustive": exhaustive_result,
        "greedy": greedy_result,
        "diagnostics": diagnostics,
    }
