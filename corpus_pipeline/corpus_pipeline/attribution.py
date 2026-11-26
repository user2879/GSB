from __future__ import annotations

from typing import Any, Mapping, Sequence

from .fixed import FixedDivision, fixed_div, rescale_fixed, to_permille
from .models import Document, PipelineConfig


def _fixed_or_zero(numerator: int, denominator: int, scale: int) -> FixedDivision:
    if denominator <= 0:
        return FixedDivision(
            quotient=0,
            residual_numerator=0,
            numerator=numerator,
            denominator=denominator,
            scale=scale,
        )
    return fixed_div(numerator, denominator, scale)


def build_composition_stats(
    config: PipelineConfig,
    plan: Mapping[str, Any],
    dedup_results: Sequence[Mapping[str, Any]],
    documents: Mapping[str, Sequence[Document]],
) -> dict[str, Any]:
    scale = config.precision.scale
    results_by_domain: dict[str, list[Mapping[str, Any]]] = {}
    for result in dedup_results:
        results_by_domain.setdefault(result["domain"], []).append(result)

    domain_stats: dict[str, dict[str, Any]] = {}
    for domain in sorted(config.domain_ids):
        doc_by_id = {doc.doc_id: doc for doc in documents.get(domain, ())}
        sampled_ids = list(plan.get("documents_by_domain", {}).get(domain, []))
        sampled_effective_terms = sum(doc_by_id[doc_id].effective_terms for doc_id in sampled_ids if doc_id in doc_by_id)
        kept_ids: set[str] = set()
        removed_ids: set[str] = set()
        for result in results_by_domain.get(domain, []):
            for decision in result.get("documents", []):
                if decision["kept"]:
                    kept_ids.add(decision["doc_id"])
                else:
                    removed_ids.add(decision["doc_id"])
        retained_effective_terms = sum(doc_by_id[doc_id].effective_terms for doc_id in kept_ids if doc_id in doc_by_id)
        domain_stats[domain] = {
            "sampled_count": len(sampled_ids),
            "retained_count": len(kept_ids),
            "removed_count": len(removed_ids),
            "sampled_effective_terms": sampled_effective_terms,
            "retained_effective_terms": retained_effective_terms,
            "dedup_ratio": _fixed_or_zero(len(kept_ids), len(sampled_ids), scale).to_dict(),
            "corpus_share": _fixed_or_zero(len(kept_ids), 0, scale).to_dict(),
            "effective_term_share": _fixed_or_zero(retained_effective_terms, 0, scale).to_dict(),
        }

    total_retained = sum(stats["retained_count"] for stats in domain_stats.values())
    total_sampled = sum(stats["sampled_count"] for stats in domain_stats.values())
    total_sampled_terms = sum(stats["sampled_effective_terms"] for stats in domain_stats.values())
    total_retained_terms = sum(stats["retained_effective_terms"] for stats in domain_stats.values())
    for domain, stats in domain_stats.items():
        stats["corpus_share"] = _fixed_or_zero(stats["retained_count"], total_retained, scale).to_dict()
        stats["effective_term_share"] = _fixed_or_zero(
            stats["retained_effective_terms"], total_retained_terms, scale
        ).to_dict()

    quantized_weights = plan.get("allocation_meta", {}).get("weights", {})
    return {
        "precision": config.precision.to_dict(),
        "quantization_step": f"1/{scale}",
        "fixed_scale": scale,
        "domain_weights": quantized_weights,
        "domains": domain_stats,
        "totals": {
            "sampled_count": total_sampled,
            "retained_count": total_retained,
            "removed_count": total_sampled - total_retained,
            "sampled_effective_terms": total_sampled_terms,
            "retained_effective_terms": total_retained_terms,
            "dedup_ratio": _fixed_or_zero(total_retained, total_sampled, scale).to_dict(),
        },
        "residual_units": {
            "dedup_ratio": "numerator at the fixed scale; a non-zero value records the exact rounding remainder",
            "corpus_share": "numerator at the fixed scale; a non-zero value records the exact rounding remainder",
            "effective_term_share": "numerator at the fixed scale; a non-zero value records the exact rounding remainder",
        },
    }


def _kept_ids(results: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
    kept: set[tuple[str, str]] = set()
    for result in results:
        for decision in result.get("documents", []):
            if decision["kept"]:
                kept.add((result["domain"], decision["doc_id"]))
    return kept


def _representatives(results: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], str]:
    representatives: dict[tuple[str, str], str] = {}
    for result in results:
        for decision in result.get("documents", []):
            representatives[(result["domain"], decision["doc_id"])] = decision["representative_doc_id"]
    return representatives


def _clusters(results: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], str]:
    clusters: dict[tuple[str, str], str] = {}
    for result in results:
        for decision in result.get("documents", []):
            clusters[(result["domain"], decision["doc_id"])] = decision["cluster_id"]
    return clusters


def _sample_documents(plan: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {
        (domain, doc_id)
        for domain, docs in plan.get("documents_by_domain", {}).items()
        for doc_id in docs
    }


def _pair_map(
    results: Sequence[Mapping[str, Any]],
    *,
    source_scale: int,
    target_scale: int,
) -> tuple[dict[tuple[str, str, str], int], set[tuple[str, str, str]]]:
    values: dict[tuple[str, str, str], int] = {}
    boundaries: set[tuple[str, str, str]] = set()
    for result in results:
        shard_id = result["shard_id"]
        for pair in result.get("pairs", []):
            key = (shard_id, pair["doc_a"], pair["doc_b"])
            values[key] = rescale_fixed(pair["similarity_fixed"], source_scale, target_scale)
            if pair["boundary"]:
                boundaries.add(key)
    return values, boundaries


def _stats_deviation(
    primary_stats: Mapping[str, Any],
    reference_stats: Mapping[str, Any],
    *,
    primary_scale: int,
    reference_scale: int,
) -> dict[str, Any]:
    ratio_delta = 0
    term_total_delta = 0
    per_domain: dict[str, dict[str, int]] = {}
    for domain in sorted(primary_stats["domains"]):
        primary = primary_stats["domains"][domain]
        reference = reference_stats["domains"].get(domain, {})
        dedup_delta = abs(
            primary["dedup_ratio"]["fixed"]
            - rescale_fixed(reference.get("dedup_ratio", {}).get("fixed", 0), reference_scale, primary_scale)
        )
        share_delta = abs(
            primary["corpus_share"]["fixed"]
            - rescale_fixed(reference.get("corpus_share", {}).get("fixed", 0), reference_scale, primary_scale)
        )
        term_share_delta = abs(
            primary["effective_term_share"]["fixed"]
            - rescale_fixed(reference.get("effective_term_share", {}).get("fixed", 0), reference_scale, primary_scale)
        )
        term_delta = abs(primary["retained_effective_terms"] - reference.get("retained_effective_terms", 0))
        ratio_delta += dedup_delta + share_delta + term_share_delta
        term_total_delta += term_delta
        per_domain[domain] = {
            "dedup_ratio_fixed_delta": dedup_delta,
            "corpus_share_fixed_delta": share_delta,
            "effective_term_share_fixed_delta": term_share_delta,
            "retained_effective_term_total_delta": term_delta,
        }
    return {
        "composition_fixed_delta": ratio_delta,
        "retained_effective_term_total_delta": term_total_delta,
        "per_domain": per_domain,
    }


def build_error_attribution(
    primary_config: PipelineConfig,
    primary_plan: Mapping[str, Any],
    primary_results: Sequence[Mapping[str, Any]],
    primary_stats: Mapping[str, Any],
    reference_config: PipelineConfig,
    reference_plan: Mapping[str, Any],
    reference_results: Sequence[Mapping[str, Any]],
    reference_stats: Mapping[str, Any],
) -> dict[str, Any]:
    primary_scale = primary_config.precision.scale
    reference_scale = reference_config.precision.scale

    primary_samples = _sample_documents(primary_plan)
    reference_samples = _sample_documents(reference_plan)
    symmetric_sample_difference = len(primary_samples.symmetric_difference(reference_samples))
    count_l1 = sum(
        abs(
            primary_plan.get("allocations", {}).get(domain, 0)
            - reference_plan.get("allocations", {}).get(domain, 0)
        )
        for domain in sorted(primary_config.domain_ids)
    )
    sample_detail = {
        "sample_set_symmetric_difference": symmetric_sample_difference,
        "per_domain_count_l1": count_l1,
        "primary_budget_consumed": primary_plan.get("budget_consumed", 0),
        "reference_budget_consumed": reference_plan.get("budget_consumed", 0),
    }

    primary_pairs, primary_boundaries = _pair_map(primary_results, source_scale=primary_scale, target_scale=primary_scale)
    reference_pairs, reference_boundaries = _pair_map(
        reference_results, source_scale=reference_scale, target_scale=primary_scale
    )
    pair_keys = set(primary_pairs) | set(reference_pairs)
    coverage_mismatches = 0
    absolute_pair_delta = 0
    for key in sorted(pair_keys):
        if key not in primary_pairs or key not in reference_pairs:
            coverage_mismatches += 1
        else:
            absolute_pair_delta += abs(primary_pairs[key] - reference_pairs[key])
    boundary_mismatches = len(primary_boundaries.symmetric_difference(reference_boundaries))
    similarity_permille_deviation = to_permille(absolute_pair_delta, primary_scale)
    similarity_permille_deviation += coverage_mismatches * 1000
    similarity_permille_deviation += boundary_mismatches * 1000
    similarity_detail = {
        "pair_coverage_mismatches": coverage_mismatches,
        "boundary_decision_mismatches": boundary_mismatches,
        "absolute_fixed_similarity_delta": absolute_pair_delta,
        "deviation_permille": similarity_permille_deviation,
        "primary_threshold_fixed": primary_results[0]["threshold_fixed"] if primary_results else None,
        "reference_threshold_fixed": reference_results[0]["threshold_fixed"] if reference_results else None,
    }

    primary_kept = _kept_ids(primary_results)
    reference_kept = _kept_ids(reference_results)
    kept_symmetric_difference = len(primary_kept.symmetric_difference(reference_kept))
    primary_rep = _representatives(primary_results)
    reference_rep = _representatives(reference_results)
    representative_differences = sum(
        1
        for key in sorted(set(primary_rep) & set(reference_rep))
        if primary_rep[key] != reference_rep[key]
    )
    primary_clusters = _clusters(primary_results)
    reference_clusters = _clusters(reference_results)
    cluster_differences = sum(
        1
        for key in sorted(set(primary_clusters) & set(reference_clusters))
        if primary_clusters[key] != reference_clusters[key]
    )
    dedup_decision_deviation = kept_symmetric_difference + representative_differences + cluster_differences
    dedup_detail = {
        "retained_set_symmetric_difference": kept_symmetric_difference,
        "representative_differences": representative_differences,
        "cluster_assignment_differences": cluster_differences,
        "decisions_deviation": dedup_decision_deviation,
    }

    composition_detail = _stats_deviation(
        primary_stats,
        reference_stats,
        primary_scale=primary_scale,
        reference_scale=reference_scale,
    )

    tolerance = primary_config.tolerance
    stages = [
        {
            "stage": "weighted_sampling",
            "deviation": symmetric_sample_difference + count_l1,
            "tolerance": tolerance.sampling_docs,
            "unit": "documents",
            "details": sample_detail,
        },
        {
            "stage": "similarity_quantization",
            "deviation": similarity_permille_deviation,
            "tolerance": tolerance.similarity_permille,
            "unit": "permille",
            "details": similarity_detail,
        },
        {
            "stage": "dedup_decision",
            "deviation": dedup_decision_deviation,
            "tolerance": tolerance.dedup_docs,
            "unit": "documents",
            "details": dedup_detail,
        },
        {
            "stage": "composition_summary",
            "deviation": composition_detail["composition_fixed_delta"],
            "tolerance": tolerance.composition_fixed,
            "unit": f"fixed units at scale {primary_scale}",
            "details": composition_detail,
        },
    ]
    first_exceeded: str | None = None
    for stage in stages:
        stage["exceeded"] = stage["deviation"] > stage["tolerance"]
        stage["residual"] = stage["deviation"] - stage["tolerance"]
        if first_exceeded is None and stage["exceeded"]:
            first_exceeded = stage["stage"]
    return {
        "reference_model": "primary decimal_places + 1, same quantization tier",
        "primary_precision": primary_config.precision.to_dict(),
        "reference_precision": reference_config.precision.to_dict(),
        "stages": stages,
        "first_exceeded_stage": first_exceeded,
        "conclusion": "within_tolerance" if first_exceeded is None else "tolerance_exceeded",
    }
