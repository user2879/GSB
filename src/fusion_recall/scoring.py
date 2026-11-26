from __future__ import annotations

from decimal import Decimal
from typing import Any

from .config import quantize_div, scale_for_bits
from .hashing import canonical_hash, composite_key
from .index import bucket_id
from .vocab import Vocabulary


def candidate_plan(
    query: dict[str, Any], index: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    referenced_buckets: set[str] = set()
    candidate_docs_by_view: dict[str, set[str]] = {
        view_name: set() for view_name in config["view_names"]
    }
    postings_by_bucket: dict[str, list[dict[str, Any]]] = {}
    for term_id in query["terms"]:
        for view_name in config["view_names"]:
            current_bucket_id = bucket_id(view_name, term_id)
            bucket = index["buckets"].get(current_bucket_id)
            if bucket is None:
                continue
            referenced_buckets.add(current_bucket_id)
            postings_by_bucket[current_bucket_id] = bucket["postings"]
            candidate_docs_by_view[view_name].update(bucket["doc_ids"])

    participating_docs = set().union(*candidate_docs_by_view.values()) if candidate_docs_by_view else set()
    cost_points = len(query["terms"]) * len(participating_docs)
    return {
        "query_id": query["query_id"],
        "referenced_buckets": sorted(referenced_buckets),
        "candidate_docs_by_view": {
            view_name: sorted(doc_ids)
            for view_name, doc_ids in candidate_docs_by_view.items()
        },
        "candidate_count_by_view": {
            view_name: len(doc_ids)
            for view_name, doc_ids in candidate_docs_by_view.items()
        },
        "participating_docs": sorted(participating_docs),
        "participating_document_count": len(participating_docs),
        "cost_points": cost_points,
    }


def _language_factor_q(
    query: dict[str, Any], view_name: str, config: dict[str, Any], scale: int, tier: str
) -> tuple[int, tuple[int, int, int, int]]:
    global_num, global_den = config["language_weight_rational"][view_name]
    pref_num, pref_den = query["language_preference_rational"][view_name]
    numerator = scale * global_num * pref_num
    denominator = global_den * pref_den
    language_q = quantize_div(numerator, denominator, tier)
    return language_q, (global_num, global_den, pref_num, pref_den)


def compute_fusion_cache_key(
    query: dict[str, Any],
    index: dict[str, Any],
    plan: dict[str, Any],
    config: dict[str, Any],
) -> str:
    referenced_fingerprints = [
        index["buckets"][item]["bucket_fingerprint"]
        for item in plan["referenced_buckets"]
        if item in index["buckets"]
    ]
    return composite_key(
        "fusion",
        query["query_fingerprint"],
        index["document_count"],
        plan["referenced_buckets"],
        referenced_fingerprints,
        config["config_fingerprint"],
        config["language_weights_fingerprint"],
        config["fixed_point_fingerprint"],
    )


def score_query(
    query: dict[str, Any],
    index: dict[str, Any],
    config: dict[str, Any],
    vocab: Vocabulary,
    *,
    fixed_override: dict[str, Any] | None = None,
    semantic_override: dict[str, Any] | None = None,
    language_override: dict[str, Any] | None = None,
    work_override: dict[str, Any] | None = None,
    include_boundary: bool = True,
) -> dict[str, Any]:
    semantic_fixed = semantic_override or fixed_override or config["fixed_point"]
    language_fixed = language_override or fixed_override or config["fixed_point"]
    work_fixed = work_override or semantic_fixed
    frac_bits = int(semantic_fixed["frac_bits"])
    tier = str(semantic_fixed["quant_tier"])
    scale = scale_for_bits(frac_bits)
    language_bits = int(language_fixed["frac_bits"])
    language_tier = str(language_fixed["quant_tier"])
    language_scale = scale_for_bits(language_bits)
    work_bits = int(work_fixed["frac_bits"])
    work_scale = scale_for_bits(work_bits)
    plan = candidate_plan(query, index, config)
    scores: dict[str, int] = {}
    evidence: list[dict[str, Any]] = []
    intermediate_fingerprint_rows: list[Any] = []
    document_count = index["document_count"]

    for term_order, term_id in enumerate(query["terms"]):
        for view_order, view_name in enumerate(config["view_names"]):
            current_bucket_id = bucket_id(view_name, term_id)
            bucket = index["buckets"].get(current_bucket_id)
            if bucket is None:
                continue
            language_q, language_rational = _language_factor_q(
                query, view_name, config, language_scale, language_tier
            )
            idf_q = quantize_div(
                scale * (document_count + 1),
                bucket["document_frequency"] + 1,
                tier,
            )
            for posting in sorted(bucket["postings"], key=lambda item: item["doc_id"]):
                doc_id = posting["doc_id"]
                semantic_q = 0
                field_components: list[dict[str, int]] = []
                for field_name in config["field_order"]:
                    field_posting = posting["fields"].get(field_name)
                    if field_posting is None:
                        continue
                    tf = int(field_posting["tf"])
                    saturation_q = quantize_div(
                        tf * scale, tf + int(config["saturation_k"]), tier
                    )
                    weighted_saturation_q = (
                        int(config["field_weights"][field_name]) * saturation_q
                    )
                    weighted_idf_product_q = weighted_saturation_q * idf_q
                    field_score_q = quantize_div(weighted_idf_product_q, scale, tier)
                    semantic_q += field_score_q
                    component = {
                        "field": field_name,
                        "tf": tf,
                        "saturation_q": saturation_q,
                        "weighted_saturation_q": weighted_saturation_q,
                        "idf_q": idf_q,
                        "field_score_q": field_score_q,
                    }
                    field_components.append(component)
                    intermediate_fingerprint_rows.append(
                        [query["query_id"], term_order, view_name, doc_id, field_name, component]
                    )
                before_language_semantic_q = semantic_q
                if work_scale >= scale:
                    semantic_work_q = semantic_q * (work_scale // scale)
                else:
                    semantic_work_q = quantize_div(semantic_q * work_scale, scale, "nearest")
                fusion_contribution_q = quantize_div(
                    semantic_work_q * language_q, language_scale, tier
                )
                scores[doc_id] = scores.get(doc_id, 0) + fusion_contribution_q
                evidence.append(
                    {
                        "term_order": term_order,
                        "term_id": term_id,
                        "view": view_name,
                        "view_order": view_order,
                        "doc_id": doc_id,
                        "idf_q": idf_q,
                        "semantic_q": before_language_semantic_q,
                        "semantic_work_q": semantic_work_q,
                        "language_q": language_q,
                        "language_rational": list(language_rational),
                        "contribution_q": fusion_contribution_q,
                        "fields": field_components,
                    }
                )

    rows: list[dict[str, Any]] = []
    for doc_id in plan["participating_docs"]:
        forward = index["forward"][doc_id]
        score_q = scores.get(doc_id, 0)
        rows.append(
            {
                "doc_id": doc_id,
                "version": forward["version"],
                "fusion_score_q": score_q,
                "fusion_score_decimal": str(Decimal(score_q) / Decimal(scale)),
            }
        )
    rows.sort(key=lambda row: (-row["fusion_score_q"], row["doc_id"], row["version"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    ranked = rank_candidates(rows, config["top_k"])
    if include_boundary:
        boundary = build_boundary_diagnostics(
            ranked["rows"], config["top_k"], scale, query, index, config, vocab
        )
    else:
        boundary = {
            "boundary_doc_ids": [],
            "boundary_count": 0,
            "boundary_fingerprint": canonical_hash([]),
        }
    fusion_cache_key = compute_fusion_cache_key(query, index, plan, config)
    ranking_cache_key = composite_key(
        "rank",
        fusion_cache_key,
        config["top_k"],
        canonical_hash(ranked["rows"]),
        boundary["boundary_fingerprint"],
    )
    result = {
        "query_id": query["query_id"],
        "terms": list(query["terms"]),
        "cost_points": plan["cost_points"],
        "referenced_buckets": plan["referenced_buckets"],
        "candidate_count_by_view": plan["candidate_count_by_view"],
        "candidate_docs_by_view": plan["candidate_docs_by_view"],
        "participating_docs": plan["participating_docs"],
        "participating_document_count": plan["participating_document_count"],
        "scale": work_scale,
        "frac_bits": work_bits,
        "semantic_frac_bits": frac_bits,
        "quant_tier": tier,
        "language_frac_bits": language_bits,
        "language_quant_tier": language_tier,
        "rows": ranked["rows"],
        "selected": ranked["selected"],
        "truncated": ranked["truncated"],
        "boundary": boundary,
        "evidence": evidence,
        "fusion_cache_key": fusion_cache_key,
        "ranking_cache_key": ranking_cache_key,
        "intermediate_fingerprint": canonical_hash(intermediate_fingerprint_rows),
    }
    return result


def rank_candidates(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (-row["fusion_score_q"], row["doc_id"], row["version"]))
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank
    selected = [row["doc_id"] for row in ordered[:top_k]]
    return {
        "rows": ordered,
        "selected": selected,
        "truncated": len(ordered) > top_k,
    }


def build_boundary_diagnostics(
    rows: list[dict[str, Any]],
    top_k: int,
    scale: int,
    query: dict[str, Any],
    index: dict[str, Any],
    config: dict[str, Any],
    vocab: Vocabulary,
) -> dict[str, Any]:
    if not rows:
        return {
            "cutoff_rank": None,
            "cutoff_score_q": None,
            "quantization_half_step_q": "1/2",
            "boundary_docs": [],
            "boundary_fingerprint": canonical_hash([]),
            "truncation_active": False,
        }
    cutoff_index = min(top_k, len(rows)) - 1
    cutoff = rows[cutoff_index]
    cutoff_score_q = cutoff["fusion_score_q"]
    boundary_docs: list[dict[str, Any]] = []
    reference_bits = next_reference_bits(
        int(config["fixed_point"]["frac_bits"]),
        tuple(config["fixed_point"].get("reference_bits_ladder", (16, 32, 40))),
        int(config["fixed_point"].get("reference_extra_bits", 8)),
    )
    reference_config = {
        **config,
        "fixed_point": {
            **config["fixed_point"],
            "frac_bits": reference_bits,
        },
    }
    reference_result = score_query(
        query,
        index,
        reference_config,
        vocab,
        fixed_override=reference_config["fixed_point"],
        include_boundary=False,
    )
    reference_by_doc = {row["doc_id"]: row for row in reference_result["rows"]}
    reference_scale = reference_result["scale"]
    reference_cutoff = reference_result["rows"][cutoff_index] if cutoff_index < len(reference_result["rows"]) else None
    reference_current_cutoff = reference_by_doc.get(cutoff["doc_id"])
    for row in rows:
        difference_q = abs(row["fusion_score_q"] - cutoff_score_q)
        reference_row = reference_by_doc.get(row["doc_id"])
        reference_score_q = reference_row["fusion_score_q"] if reference_row else None
        reference_distance_q = (
            abs(reference_score_q - reference_cutoff["fusion_score_q"])
            if reference_score_q is not None and reference_cutoff is not None
            else None
        )
        high_width_within_half_step = (
            reference_distance_q is not None
            and 2 * reference_distance_q * scale <= reference_scale
        )
        within_half_step = (2 * difference_q <= 1) or high_width_within_half_step
        if not within_half_step:
            continue
        reference_rescaled_q = (
            quantize_div(reference_score_q * scale, reference_scale, "nearest")
            if reference_score_q is not None
            else None
        )
        boundary_docs.append(
            {
                "doc_id": row["doc_id"],
                "version": row["version"],
                "rank": row["rank"],
                "fusion_score_q": row["fusion_score_q"],
                "cutoff_score_q": cutoff_score_q,
                "distance_to_cutoff_q": difference_q,
                "half_step_basis": (
                    "integer_tie" if 2 * difference_q <= 1 else "high_width_reference"
                ),
                "within_quantization_half_step": True,
                "high_width_bits": reference_bits,
                "high_width_score_q": reference_score_q,
                "high_width_score_current_scale_q": reference_rescaled_q,
                "high_width_minus_current_q": (
                    reference_rescaled_q - row["fusion_score_q"]
                    if reference_rescaled_q is not None
                    else None
                ),
                "high_width_rank": reference_row["rank"] if reference_row else None,
                "high_width_distance_to_reference_cutoff_q": (
                    reference_score_q - reference_cutoff["fusion_score_q"]
                    if reference_score_q is not None and reference_cutoff is not None
                    else None
                ),
                "high_width_distance_to_current_cutoff_doc_q": (
                    reference_score_q - reference_current_cutoff["fusion_score_q"]
                    if reference_score_q is not None and reference_current_cutoff is not None
                    else None
                ),
            }
        )
    fingerprint = canonical_hash(
        [
            {
                "doc_id": item["doc_id"],
                "score": item["fusion_score_q"],
                "ref": item["high_width_score_q"],
            }
            for item in boundary_docs
        ]
    )
    return {
        "cutoff_rank": cutoff["rank"],
        "cutoff_doc_id": cutoff["doc_id"],
        "cutoff_score_q": cutoff_score_q,
        "quantization_half_step_q": "1/2",
        "boundary_docs": boundary_docs,
        "boundary_doc_ids": [item["doc_id"] for item in boundary_docs],
        "boundary_count": len(boundary_docs),
        "boundary_fingerprint": fingerprint,
        "truncation_active": len(rows) > config["top_k"],
        "reference_frac_bits": reference_bits,
    }


def next_reference_bits(current_bits: int, ladder: tuple[int, ...], extra_bits: int) -> int:
    for bits in ladder:
        if bits > current_bits:
            return bits
    return current_bits + extra_bits


def scores_fingerprint(results: list[dict[str, Any]]) -> str:
    return canonical_hash(
        [
            {
                "query_id": result["query_id"],
                "rows": [
                    {
                        "doc_id": row["doc_id"],
                        "version": row["version"],
                        "score": row["fusion_score_q"],
                        "rank": row["rank"],
                    }
                    for row in result["rows"]
                ],
                "selected": result["selected"],
                "boundary_ids": result["boundary"]["boundary_doc_ids"],
                "fusion_cache_key": result["fusion_cache_key"],
                "ranking_cache_key": result["ranking_cache_key"],
            }
            for result in sorted(results, key=lambda item: item["query_id"])
        ]
    )
