from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .canonical import canonical_bytes, ceil_div, digest_value
from .fixedpoint import (
    BP_SCALE,
    allocate_integer,
    counts_to_actual_bp,
    sat_i64,
    weights_to_target_bp,
)
from .specs import SpecBundle
from .synthetic import selection_rank

MICRO_BP = 1_000


def flatten_docs(bundle: SpecBundle | Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(bundle, SpecBundle):
        batches = bundle.batches
        domains = {d["domain_id"]: d for d in bundle.domains}
    else:
        batches = list(bundle)
        domains = {}
    docs: list[dict[str, Any]] = []
    for batch in batches:
        for doc in batch.get("docs", []):
            enriched = dict(doc)
            enriched.setdefault("domain_id", batch["domain_id"])
            enriched.setdefault("batch_seq", batch["batch_seq"])
            enriched["batch_id"] = batch.get("batch_id") or f"{batch['domain_id']}#b{int(batch['batch_seq']):04d}"
            docs.append(enriched)
    docs.sort(key=lambda doc: (str(doc["domain_id"]), int(doc["doc_seq"]), str(doc["doc_id"])))
    for doc in docs:
        if "selection_rank" not in doc and domains:
            doc["selection_rank"] = selection_rank(domains[doc["domain_id"]], doc)
    return docs


def weighted_sampling_stage(bundle: SpecBundle) -> dict[str, Any]:
    domain_ids = bundle.domain_ids
    config = bundle.config
    budget = int(config["sampling_budget_points"])
    weights = {domain: int(config["weights_bp"].get(domain, 0)) for domain in domain_ids}
    quotas = allocate_integer(budget, weights, domain_ids)
    docs = flatten_docs(bundle)
    by_domain: dict[str, list[dict[str, Any]]] = {domain: [] for domain in domain_ids}
    for doc in docs:
        by_domain[doc["domain_id"]].append(doc)

    selected: list[dict[str, Any]] = []
    domain_summary: dict[str, dict[str, int | bool]] = {}
    for domain in domain_ids:
        domain_docs = by_domain[domain]
        ranked = sorted(
            domain_docs,
            key=lambda doc: (
                selection_rank(bundle.domain(domain), doc),
                int(doc["doc_seq"]),
                str(doc["doc_id"]),
            ),
        )
        quota = int(quotas[domain])
        chosen = ranked[: min(quota, len(ranked))]
        saturated = quota > len(ranked)
        domain_summary[domain] = {
            "weight_bp": weights[domain],
            "quota_points": quota,
            "available_docs": len(ranked),
            "selected_count": len(chosen),
            "quota_saturated": saturated,
        }
        for doc in chosen:
            selected.append(
                {
                    "doc_id": doc["doc_id"],
                    "domain_id": domain,
                    "doc_seq": int(doc["doc_seq"]),
                    "batch_id": doc["batch_id"],
                    "content_digest": doc["content_digest"],
                    "rank": selection_rank(bundle.domain(domain), doc),
                }
            )

    # Domain-first ordering makes domain references explicit while the
    # content-derived rank inside a domain still propagates document edits
    # across fixed-size shard boundaries.
    selected.sort(key=lambda item: (str(item["domain_id"]), str(item["rank"]), str(item["doc_id"])))
    return {
        "stage": "weighted_sampling",
        "budget_points": budget,
        "rounding_rule": config["rounding_rule"],
        "saturation_rule": config["saturation_rule"],
        "quotas_bp_rule": "floor(budget * weight / sum(weights)); remainder seats by descending remainder then domain_id",
        "domains": domain_summary,
        "selected": selected,
        "selected_count": len(selected),
        "selected_digest": digest_value(selected),
    }


def plan_shards(selected: list[dict[str, Any]], shard_size: int) -> list[dict[str, Any]]:
    shards: list[dict[str, Any]] = []
    total = len(selected)
    for ordinal, start in enumerate(range(0, total, shard_size), start=1):
        members = selected[start : start + shard_size]
        domain_counts: dict[str, int] = defaultdict(int)
        for member in members:
            domain_counts[member["domain_id"]] += 1
        shard_id = f"S{ordinal:04d}"
        shards.append(
            {
                "shard_id": shard_id,
                "ordinal": ordinal,
                "start_index": start,
                "end_index_exclusive": start + len(members),
                "boundary": {
                    "start_index": start,
                    "end_index_exclusive": start + len(members),
                    "start_rank": members[0]["rank"] if members else None,
                    "end_rank_exclusive": None,
                    "first_doc_id": members[0]["doc_id"] if members else None,
                    "last_doc_id": members[-1]["doc_id"] if members else None,
                },
                "doc_count": len(members),
                "domains": sorted(domain_counts),
                "domain_counts": dict(sorted(domain_counts.items())),
                "member_ids": [member["doc_id"] for member in members],
                "members_digest": digest_value([member["doc_id"] for member in members]),
            }
        )
    return shards


def _similarity(left: dict[str, Any], right: dict[str, Any]) -> tuple[int, int, int]:
    left_counts = left["term_counts"]
    right_counts = right["term_counts"]
    size = max(len(left_counts), len(right_counts))
    intersection = 0
    union = 0
    shared_terms = 0
    for index in range(size):
        left_value = int(left_counts[index]) if index < len(left_counts) else 0
        right_value = int(right_counts[index]) if index < len(right_counts) else 0
        intersection += min(left_value, right_value)
        union += max(left_value, right_value)
        if left_value > 0 and right_value > 0:
            shared_terms += 1
    return intersection, union, shared_terms


def compute_domain_dedup(
    domain_id: str,
    docs: list[dict[str, Any]],
    threshold_bp: int,
    stage_params: dict[str, Any],
) -> dict[str, Any]:
    ordered = sorted(docs, key=lambda doc: (int(doc["doc_seq"]), str(doc["doc_id"])))
    min_shared_terms = int(stage_params.get("min_shared_terms", 1))
    pairs: list[dict[str, Any]] = []

    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            intersection, union, shared_terms = _similarity(left, right)
            if union <= 0:
                continue
            scaled_intersection = sat_i64(BP_SCALE * intersection)
            above = scaled_intersection >= threshold_bp * union
            reportable = (scaled_intersection >= 5_000 * union) or above
            if reportable and shared_terms >= min_shared_terms:
                pairs.append(
                    {
                        "a": left["doc_id"],
                        "b": right["doc_id"],
                        "intersection": intersection,
                        "union": union,
                        "shared_terms": shared_terms,
                        "similarity_bp_floor": (BP_SCALE * intersection) // union,
                        "scaled_intersection_num": scaled_intersection,
                        "union_denominator": union,
                        "above_threshold": above,
                        "near_duplicate": above,
                    }
                )

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for doc in ordered:
        matched_kept: dict[str, Any] | None = None
        for existing in kept:
            intersection, union, shared_terms = _similarity(existing, doc)
            if union > 0 and shared_terms >= min_shared_terms and BP_SCALE * intersection >= threshold_bp * union:
                matched_kept = existing
                break
        if matched_kept is None:
            kept.append(doc)
        else:
            dropped.append(
                {
                    "doc_id": doc["doc_id"],
                    "representative_id": matched_kept["doc_id"],
                    "intersection": intersection,
                    "union": union,
                    "shared_terms": shared_terms,
                }
            )

    pairs.sort(key=lambda pair: (str(pair["a"]), str(pair["b"])))
    kept_ids = [doc["doc_id"] for doc in kept]
    dropped_ids = [item["doc_id"] for item in dropped]
    result = {
        "domain_id": domain_id,
        "input_count": len(ordered),
        "kept_count": len(kept_ids),
        "dropped_count": len(dropped_ids),
        "kept_ids": kept_ids,
        "dropped": sorted(dropped, key=lambda item: item["doc_id"]),
        "pairs": pairs,
    }
    result["result_digest"] = digest_value({key: value for key, value in result.items() if key != "result_digest"})
    return result


def aggregate_shard_result(
    plan: dict[str, Any],
    domain_results: list[dict[str, Any]],
    *,
    threshold_bp: int,
) -> dict[str, Any]:
    by_domain = {result["domain_id"]: result for result in domain_results}
    if set(by_domain) != set(plan["domains"]):
        raise ValueError(f"shard {plan['shard_id']} domain result mismatch")
    pairs: list[dict[str, Any]] = []
    kept_ids: list[str] = []
    dropped_records: list[dict[str, Any]] = []
    domain_counts: dict[str, dict[str, int]] = {}
    for domain in plan["domains"]:
        result = by_domain[domain]
        pairs.extend(dict(pair, domain_id=domain) for pair in result["pairs"])
        kept_ids.extend(result["kept_ids"])
        for dropped in result["dropped"]:
            dropped_records.append(dict(dropped, domain_id=domain))
        domain_counts[domain] = {
            "input": int(result["input_count"]),
            "kept": int(result["kept_count"]),
            "dropped": int(result["dropped_count"]),
        }
    pairs.sort(key=lambda pair: (pair["domain_id"], pair["a"], pair["b"]))
    kept_ids.sort()
    dropped_records.sort(key=lambda item: (item["domain_id"], item["doc_id"]))
    result = {
        "shard_id": plan["shard_id"],
        "ordinal": plan["ordinal"],
        "start_index": plan["start_index"],
        "end_index_exclusive": plan["end_index_exclusive"],
        "boundary": plan["boundary"],
        "doc_count": plan["doc_count"],
        "domains": plan["domains"],
        "member_ids": plan["member_ids"],
        "members_digest": plan["members_digest"],
        "domain_counts": dict(sorted(domain_counts.items())),
        "pairs": pairs,
        "kept_ids": kept_ids,
        "dropped": dropped_records,
        "kept_count": len(kept_ids),
        "dropped_count": len(dropped_records),
    }
    result["shard_digest"] = digest_value({key: value for key, value in result.items() if key != "shard_digest"})
    return result


def compute_shard_locally(plan: dict[str, Any], docs_by_id: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc_id in plan["member_ids"]:
        doc = docs_by_id[doc_id]
        groups[doc["domain_id"]].append(doc)
    domain_results = [
        compute_domain_dedup(domain, groups[domain], int(config["similarity_threshold_bp"]), config["stage_params"]["dedup"])
        for domain in plan["domains"]
    ]
    return aggregate_shard_result(plan, domain_results, threshold_bp=int(config["similarity_threshold_bp"]))


def ratio_normalization_stage(bundle: SpecBundle, shard_results: list[dict[str, Any]]) -> dict[str, Any]:
    domain_ids = bundle.domain_ids
    kept_counts = {domain: 0 for domain in domain_ids}
    selected_counts = {domain: 0 for domain in domain_ids}
    dropped_counts = {domain: 0 for domain in domain_ids}
    for shard in shard_results:
        for domain, counts in shard["domain_counts"].items():
            kept_counts[domain] += int(counts["kept"])
            dropped_counts[domain] += int(counts["dropped"])
            selected_counts[domain] += int(counts["input"])

    total_kept = sum(kept_counts.values())
    actual_bp = counts_to_actual_bp(kept_counts, domain_ids)
    target_bp = weights_to_target_bp(bundle.config["weights_bp"], domain_ids)
    domains: list[dict[str, Any]] = []
    for domain in domain_ids:
        deviation = int(actual_bp[domain]) - int(target_bp[domain])
        domains.append(
            {
                "domain_id": domain,
                "selected_count": selected_counts[domain],
                "kept_count": kept_counts[domain],
                "dropped_count": dropped_counts[domain],
                "total_kept": total_kept,
                "actual_bp": actual_bp[domain],
                "target_bp": target_bp[domain],
                "deviation_bp": deviation,
                "abs_deviation_bp": abs(deviation),
                "deviation_micro_bp": deviation * MICRO_BP,
            }
        )
    result = {
        "stage": "ratio_normalization",
        "total_kept": total_kept,
        "actual_bp": actual_bp,
        "target_bp": target_bp,
        "rounding_rule": "floor(kept * 10000 / total); remainder seats by descending remainder then domain_id",
        "domains": domains,
        "max_abs_deviation_bp": max((item["abs_deviation_bp"] for item in domains), default=0),
        "l1_abs_deviation_bp": sum(item["abs_deviation_bp"] for item in domains),
    }
    result["ratio_digest"] = digest_value({key: value for key, value in result.items() if key != "ratio_digest"})
    return result


def error_attribution(bundle: SpecBundle, weighted: dict[str, Any], dedup_shards: list[dict[str, Any]], ratio: dict[str, Any]) -> dict[str, Any]:
    domain_ids = bundle.domain_ids
    budget = max(1, int(bundle.config["sampling_budget_points"]))
    sampling_u = ceil_div(MICRO_BP * BP_SCALE, budget)

    band = int(bundle.config["stage_params"]["dedup"].get("boundary_band_bp", 1))
    boundary_docs_by_domain = {domain: set() for domain in domain_ids}
    for shard in dedup_shards:
        for pair in shard.get("pairs", []):
            # A pair is boundary-attributable when its exact integer score is
            # within the declared stable reporting band of the threshold.
            approximate_distance_bp = abs(int(pair["similarity_bp_floor"]) - int(bundle.config["similarity_threshold_bp"]))
            if approximate_distance_bp <= band:
                domain = pair["domain_id"]
                boundary_docs_by_domain[domain].add(pair["a"])
                boundary_docs_by_domain[domain].add(pair["b"])

    dedup_u = 0
    boundary_counts: dict[str, int] = {}
    for domain in domain_ids:
        boundary_count = len(boundary_docs_by_domain[domain])
        boundary_counts[domain] = boundary_count
        denominator = max(1, int(weighted["domains"][domain]["selected_count"]))
        domain_bound = ceil_div(MICRO_BP * BP_SCALE * boundary_count, denominator)
        dedup_u = max(dedup_u, domain_bound)

    normalization_u = MICRO_BP
    stages = [
        {"stage": "weighted_sampling", "bound_micro_bp": sat_i64(sampling_u)},
        {"stage": "dedup_boundary", "bound_micro_bp": sat_i64(dedup_u)},
        {"stage": "ratio_normalization", "bound_micro_bp": sat_i64(normalization_u)},
    ]
    weights_for_share = {item["stage"]: max(1, int(item["bound_micro_bp"])) for item in stages}
    shares = allocate_integer(1_000, weights_for_share, [item["stage"] for item in stages])
    for item in stages:
        item["share_permille"] = shares[item["stage"]]
    total_u = sat_i64(sum(int(item["bound_micro_bp"]) for item in stages))
    dominant = max(stages, key=lambda item: (int(item["share_permille"]), -stages.index(item)))
    return {
        "scale": "micro_basis_point",
        "micro_bp_per_bp": MICRO_BP,
        "composition_order": ["weighted_sampling", "dedup_boundary", "ratio_normalization"],
        "composition_rule": "saturating_sum_in_stage_order",
        "dominance_threshold_permille": 600,
        "stages": stages,
        "boundary_doc_counts": boundary_counts,
        "total_bound_micro_bp": total_u,
        "total_bound_bp_integer_ceil": ceil_div(total_u, MICRO_BP),
        "dominant_stage": dominant["stage"],
        "dominant_share_permille": dominant["share_permille"],
        "dominant_above_threshold": dominant["share_permille"] >= 600,
    }


def adjudicate_candidates(ratio: dict[str, Any], error: dict[str, Any], candidates: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not candidates:
        return None
    actual = {item["domain_id"]: int(item["actual_bp"]) for item in ratio["domains"]}
    total_bound_u = int(error["total_bound_micro_bp"])
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        target = weights_to_target_bp(candidate["weights_bp"], sorted(actual))
        deviations = {domain: int(target[domain]) - int(actual[domain]) for domain in sorted(actual)}
        score_u = max(abs(value) for value in deviations.values()) * MICRO_BP
        scored.append(
            {
                "candidate_id": candidate_id,
                "target_bp": target,
                "deviation_bp": deviations,
                "max_abs_deviation_micro_bp": score_u,
                "within_error_bound": score_u <= total_bound_u,
            }
        )
    scored.sort(key=lambda item: str(item["candidate_id"]))
    if len(scored) != 2:
        selected = min(scored, key=lambda item: (int(item["max_abs_deviation_micro_bp"]), item["candidate_id"]))
        return {"rule": "lowest_fixed_deviation_then_candidate_id", "candidates": scored, "selected": selected["candidate_id"], "error_affected": selected["within_error_bound"]}

    left, right = scored
    gap = abs(int(left["max_abs_deviation_micro_bp"]) - int(right["max_abs_deviation_micro_bp"]))
    indistinguishable = gap <= total_bound_u
    if indistinguishable:
        selected_id = min(left["candidate_id"], right["candidate_id"])
        reason = "score_gap_within_composed_error_bound__lexicographic_candidate_id"
    else:
        selected_id = min(scored, key=lambda item: (int(item["max_abs_deviation_micro_bp"]), item["candidate_id"]))["candidate_id"]
        reason = "score_gap_exceeds_error_bound__lowest_fixed_deviation"
    selected = next(item for item in scored if item["candidate_id"] == selected_id)
    return {
        "rule": "if_abs_score_gap_le_error_bound_choose_lexicographic_candidate_id_else_lowest_score",
        "error_bound_micro_bp": total_bound_u,
        "score_gap_micro_bp": gap,
        "indistinguishable_under_error": indistinguishable,
        "decision_reason": reason,
        "candidates": scored,
        "selected_candidate_id": selected_id,
        "error_affected": bool(indistinguishable or selected["within_error_bound"]),
    }


def spec_digest(bundle: SpecBundle) -> str:
    return digest_value(
        {
            "domains": bundle.domains,
            "config_without_version": {k: v for k, v in bundle.config.items() if k != "config_version"},
            "batches_digest": digest_value([batch.get("batch_digest") for batch in bundle.batches]),
        }
    )


def build_conclusion(
    bundle: SpecBundle,
    weighted: dict[str, Any],
    shard_results: list[dict[str, Any]],
    candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ratio = ratio_normalization_stage(bundle, shard_results)
    error = error_attribution(bundle, weighted, shard_results, ratio)
    adjudication = adjudicate_candidates(ratio, error, candidates)
    conclusion = {
        "format_version": 1,
        "config_version": int(bundle.config["config_version"]),
        "spec_digest": spec_digest(bundle),
        "weighted_sampling": weighted,
        "shards": sorted(shard_results, key=lambda item: item["ordinal"]),
        "ratio_normalization": ratio,
        "error_attribution": error,
        "candidate_adjudication": adjudication,
    }
    conclusion["conclusion_digest"] = digest_value({k: v for k, v in conclusion.items() if k != "conclusion_digest"})
    return conclusion


def compute_conclusion(bundle: SpecBundle, candidates: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    weighted = weighted_sampling_stage(bundle)
    plans = plan_shards(weighted["selected"], int(bundle.config["shard_size"]))
    docs_by_id = {doc["doc_id"]: doc for doc in flatten_docs(bundle)}
    shard_results = [compute_shard_locally(plan, docs_by_id, bundle.config) for plan in plans]
    return build_conclusion(bundle, weighted, shard_results, candidates)


def conclusion_bytes(conclusion: dict[str, Any]) -> bytes:
    payload = {key: value for key, value in conclusion.items() if key != "conclusion_digest"}
    return canonical_bytes(payload)


def first_divergence(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first stable stage/domain/shard difference."""

    stage_order = ("weighted_sampling", "shards", "ratio_normalization", "error_attribution")
    for stage in stage_order:
        if stage == "shards":
            expected_shards = {item["shard_id"]: item for item in expected.get("shards", [])}
            actual_shards = {item["shard_id"]: item for item in actual.get("shards", [])}
            for shard_id in sorted(set(expected_shards) | set(actual_shards)):
                left = expected_shards.get(shard_id)
                right = actual_shards.get(shard_id)
                if left is None or right is None:
                    return {"stage": "near_duplicate_dedup", "shard_id": shard_id, "domain_id": None, "field": "shard_presence"}
                for field in ("members_digest", "domain_counts", "kept_count", "dropped_count", "kept_ids", "dropped", "pairs", "shard_digest"):
                    if canonical_bytes(left.get(field)) != canonical_bytes(right.get(field)):
                        domain_id = None
                        if field == "domain_counts":
                            for domain in sorted(set(left[field]) | set(right[field])):
                                if canonical_bytes(left[field].get(domain)) != canonical_bytes(right[field].get(domain)):
                                    domain_id = domain
                                    break
                        return {"stage": "near_duplicate_dedup", "shard_id": shard_id, "domain_id": domain_id, "field": field}
        elif stage == "ratio_normalization":
            left_domains = {item["domain_id"]: item for item in expected.get(stage, {}).get("domains", [])}
            right_domains = {item["domain_id"]: item for item in actual.get(stage, {}).get("domains", [])}
            for domain_id in sorted(set(left_domains) | set(right_domains)):
                left = left_domains.get(domain_id)
                right = right_domains.get(domain_id)
                if left is None or right is None:
                    return {"stage": stage, "domain_id": domain_id, "shard_id": None, "field": "domain_presence"}
                for field in ("actual_bp", "target_bp", "deviation_bp", "kept_count"):
                    if left.get(field) != right.get(field):
                        return {"stage": stage, "domain_id": domain_id, "shard_id": None, "field": field, "expected": left.get(field), "actual": right.get(field)}
        else:
            if digest_value(expected.get(stage)) != digest_value(actual.get(stage)):
                return {"stage": stage, "domain_id": None, "shard_id": None, "field": "stage_digest"}
    if conclusion_bytes(expected) != conclusion_bytes(actual):
        return {"stage": "conclusion", "domain_id": None, "shard_id": None, "field": "canonical_bytes"}
    return None
