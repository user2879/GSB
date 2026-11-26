from __future__ import annotations

from typing import Any

from .canonical import fingerprint
from .errors import PipelineError, E_SHARD_COMPARISON_LIMIT
from .features import jaccard_numerator_denominator, subword_set
from .fixedpoint import permille_to_fixed, quantize_ratio


DEDUP_ALGORITHM_VERSION = "dedup-v2"


def dedup_scope(config: dict[str, Any], domain_names: list[str] | None = None) -> dict[str, Any]:
    domain_map = config["domain_map"]
    names = domain_names if domain_names is not None else config["domain_names"]
    return {
        "algorithm": DEDUP_ALGORITHM_VERSION,
        "threshold_permille": config["dedup"]["similarity_threshold_permille"],
        "shard_size": config["dedup"]["shard_size"],
        "score_weights_permille": config["dedup"]["score_weights_permille"],
        "score_token_cap": config["dedup"]["score_token_cap"],
        "domain_weights_fp": {name: domain_map[name]["sampling_weight_fp"] for name in names},
        "fixed_point": config["fixed_point"],
    }


def plan_shards(cleaning_records: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    retained_by_domain: dict[str, list[str]] = {name: [] for name in config["domain_names"]}
    for record in cleaning_records:
        if record["status"] == "retained":
            retained_by_domain[record["domain"]].append(record["sample_id"])

    shard_size = config["dedup"]["shard_size"]
    shards: list[dict[str, Any]] = []
    for domain_index, domain in enumerate(config["domain_names"]):
        ids = sorted(retained_by_domain[domain])
        for shard_index, start in enumerate(range(0, len(ids), shard_size)):
            shard_ids = ids[start : start + shard_size]
            shards.append(
                {
                    "shard_id": f"shard-{domain}-{shard_index:04d}",
                    "domain": domain,
                    "domain_index": domain_index,
                    "shard_index": shard_index,
                    "sample_ids": shard_ids,
                    "pair_count": len(shard_ids) * (len(shard_ids) - 1) // 2,
                    "scheduling_cost": 1 + len(shard_ids) * (len(shard_ids) - 1) // 2,
                }
            )
    return shards


def composite_score_fp(sample: dict[str, Any], cleaning_record: dict[str, Any], config: dict[str, Any]) -> dict[str, int]:
    width = config["fixed_point"]["decimal_width"]
    tier = config["fixed_point"]["quantization_tier"]
    scale = 10**width
    domain = config["domain_map"][sample["domain"]]
    # The representative score references only this shard's domain weight.
    # Sampling itself still compares raw arbitrary weights across domains.
    weight_component_fp = min(domain["sampling_weight_fp"], scale)

    cap = config["dedup"]["score_token_cap"]
    token_component_fp = quantize_ratio(min(sample["term_count_total"], cap), cap, width, tier)
    missing_count = sum(sample["missing_bitmap"])
    completeness_fp = quantize_ratio(
        config["missing_bitmap_width"] - missing_count,
        config["missing_bitmap_width"],
        width,
        tier,
    )

    # Domain weight boosts the token-volume term, so a weight change can alter
    # representative ordering when token volume and completeness trade off.
    boosted_token_fp = token_component_fp * (scale + weight_component_fp) // scale
    weights = config["dedup"]["score_weights_permille"]
    composite_fp = (
        boosted_token_fp * weights["token_volume"]
        + completeness_fp * weights["completeness"]
        + weight_component_fp * weights["domain_weight"]
    ) // 1000
    return {
        "composite_score_fp": composite_fp,
        "token_component_fp": token_component_fp,
        "boosted_token_fp": boosted_token_fp,
        "completeness_fp": completeness_fp,
        "weight_component_fp": weight_component_fp,
    }


def shard_cache_key(
    shard: dict[str, Any],
    samples: dict[str, dict[str, Any]],
    cleaning_by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> str:
    upstream = []
    for sample_id in shard["sample_ids"]:
        upstream.append(
            {
                "sample_content_fingerprint": cleaning_by_id[sample_id]["content_fingerprint"],
                "cleaning_record_fingerprint": cleaning_by_id[sample_id]["record_fingerprint"],
            }
        )
    return fingerprint(
        {
            "stage": DEDUP_ALGORITHM_VERSION,
            "shard_id": shard["shard_id"],
            "shard_membership": shard["sample_ids"],
            "upstream_cleaning": upstream,
            "dedup_config_fingerprint": fingerprint(dedup_scope(config, [shard["domain"]])),
            "decimal_width": config["fixed_point"]["decimal_width"],
            "quantization_tier": config["fixed_point"]["quantization_tier"],
        }
    )


def deduplicate_shard(
    shard: dict[str, Any],
    samples: dict[str, dict[str, Any]],
    cleaning_by_id: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    comparison_limit = config["budgets"]["shard_comparison_limit"]
    if shard["pair_count"] > comparison_limit:
        raise PipelineError(
            E_SHARD_COMPARISON_LIMIT,
            "shard pair count exceeds configured comparison limit",
            field=f"dedup.shards.{shard['shard_id']}",
            details={
                "pair_count": shard["pair_count"],
                "comparison_limit": comparison_limit,
            },
        )

    ids = list(shard["sample_ids"])
    features = {sample_id: subword_set(samples[sample_id]) for sample_id in ids}
    scores = {
        sample_id: composite_score_fp(samples[sample_id], cleaning_by_id[sample_id], config)
        for sample_id in ids
    }
    threshold_permille = config["dedup"]["similarity_threshold_permille"]
    width = config["fixed_point"]["decimal_width"]
    tier = config["fixed_point"]["quantization_tier"]
    threshold_fp = permille_to_fixed(threshold_permille, width)
    shard_cache_key_value = shard_cache_key(shard, samples, cleaning_by_id, config)

    pair_reviews: list[dict[str, Any]] = []
    qualifying: dict[str, list[str]] = {sample_id: [] for sample_id in ids}
    comparisons = 0
    for left_index, left in enumerate(ids):
        for right in ids[left_index + 1 :]:
            comparisons += 1
            intersection, union = jaccard_numerator_denominator(features[left], features[right])
            if union == 0:
                similarity_fp = 10**width
                exact_boundary = threshold_permille == 1000
            else:
                similarity_fp = quantize_ratio(intersection, union, width, tier)
                exact_boundary = intersection * 1000 == threshold_permille * union
            qualifies = similarity_fp >= threshold_fp
            if qualifies:
                qualifying[left].append(right)
                qualifying[right].append(left)
            pair_reviews.append(
                {
                    "left_sample_id": left,
                    "right_sample_id": right,
                    "intersection_count": intersection,
                    "union_count": union,
                    "similarity_fp": similarity_fp,
                    "similarity_permille_floor": (intersection * 1000) // union if union else 1000,
                    "threshold_permille": threshold_permille,
                    "threshold_fp": threshold_fp,
                    "qualifies": qualifies,
                    "exact_threshold_boundary": exact_boundary,
                }
            )

    priority = sorted(ids, key=lambda sample_id: (-scores[sample_id]["composite_score_fp"], sample_id))
    representative_of: dict[str, str] = {}
    clusters: list[dict[str, Any]] = []
    assigned: set[str] = set()

    for representative in priority:
        if representative in assigned:
            continue
        members = [representative]
        boundary_ids: list[str] = []
        neighbor_sims = {
            other["right_sample_id"]
            if other["left_sample_id"] == representative
            else other["left_sample_id"]: other
            for other in pair_reviews
            if representative in (other["left_sample_id"], other["right_sample_id"])
        }
        for candidate in ids:
            if candidate == representative or candidate in assigned:
                continue
            review = neighbor_sims.get(candidate)
            if review is not None and review["qualifies"]:
                members.append(candidate)
                assigned.add(candidate)
                if review["exact_threshold_boundary"]:
                    boundary_ids.append(candidate)
        assigned.add(representative)
        members = sorted(members)
        for member in members:
            representative_of[member] = representative
        clusters.append(
            {
                "cluster_id": f"{shard['shard_id']}-cluster-{len(clusters):03d}",
                "cluster_cache_key": fingerprint(
                    {
                        "kind": "dedup_cluster",
                        "shard_cache_key": shard_cache_key_value,
                        "representative_sample_id": representative,
                        "member_ids": members,
                    }
                ),
                "representative_sample_id": representative,
                "member_ids": members,
                "boundary_sample_ids": sorted(boundary_ids),
                "representative_score_fp": scores[representative]["composite_score_fp"],
            }
        )

    decisions: list[dict[str, Any]] = []
    review_by_pair = {
        tuple(sorted((review["left_sample_id"], review["right_sample_id"]))): review
        for review in pair_reviews
    }
    for sample_id in sorted(ids):
        representative = representative_of[sample_id]
        score = scores[sample_id]
        if representative == sample_id:
            decisions.append(
                {
                    "sample_id": sample_id,
                    "domain": shard["domain"],
                    "shard_id": shard["shard_id"],
                    "decision": "representative",
                    "status": "kept",
                    "cluster_id": next(
                        cluster["cluster_id"]
                        for cluster in clusters
                        if cluster["representative_sample_id"] == sample_id
                    ),
                    "representative_sample_id": sample_id,
                    "similarity_fp": 10**width,
                    "similarity_permille_floor": 1000,
                    "exact_threshold_boundary": False,
                    "score_components_fp": score,
                }
            )
        else:
            review = review_by_pair[tuple(sorted((sample_id, representative)))]
            decisions.append(
                {
                    "sample_id": sample_id,
                    "domain": shard["domain"],
                    "shard_id": shard["shard_id"],
                    "decision": "duplicate_removed",
                    "status": "removed",
                    "cluster_id": representative_of and next(
                        cluster["cluster_id"]
                        for cluster in clusters
                        if cluster["representative_sample_id"] == representative
                    ),
                    "representative_sample_id": representative,
                    "similarity_fp": review["similarity_fp"],
                    "similarity_permille_floor": review["similarity_permille_floor"],
                    "exact_threshold_boundary": review["exact_threshold_boundary"],
                    "score_components_fp": score,
                }
            )

    artifact = {
        "shard_id": shard["shard_id"],
        "domain": shard["domain"],
        "shard_index": shard["shard_index"],
        "sample_ids": ids,
        "pairs_compared": comparisons,
        "comparison_limit": comparison_limit,
        "cache_key": shard_cache_key_value,
        "clusters": clusters,
        "decisions": decisions,
        "pair_reviews": pair_reviews,
        "score_components": {sample_id: scores[sample_id] for sample_id in sorted(scores)},
    }
    artifact["artifact_fingerprint"] = fingerprint(_artifact_hashable(artifact))
    return artifact


def _artifact_hashable(artifact: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in artifact.items() if key != "artifact_fingerprint"}


def dedup_artifact_fingerprint(shard_artifacts: list[dict[str, Any]]) -> str:
    return fingerprint([artifact["artifact_fingerprint"] for artifact in sorted(shard_artifacts, key=lambda item: item["shard_id"])])


def iter_dedup_survivors(shard_artifacts: list[dict[str, Any]]) -> list[str]:
    survivors: list[str] = []
    for artifact in shard_artifacts:
        for decision in artifact["decisions"]:
            if decision["status"] == "kept":
                survivors.append(decision["sample_id"])
    return sorted(survivors)
