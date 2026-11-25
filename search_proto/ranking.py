from __future__ import annotations

from typing import Any, Iterable

from .config import PrototypeConfig
from .model import QuerySpec


def _fixed_key(score: dict[str, Any]) -> tuple[int, str, int]:
    return (-int(score["fixed_score_units"]), score["doc_id"], int(score["doc_version"]))


def _float_key(score: dict[str, Any]) -> tuple[float, str, int]:
    return (-float(score["float_score"]), score["doc_id"], int(score["doc_version"]))


def _relation(
    left: dict[str, Any],
    right: dict[str, Any],
    score_key: str,
    *,
    fixed: bool,
) -> int:
    left_score = left[score_key]
    right_score = right[score_key]
    if fixed:
        left_score = int(left_score)
        right_score = int(right_score)
    else:
        left_score = float(left_score)
        right_score = float(right_score)
    if left_score > right_score:
        return -1
    if left_score < right_score:
        return 1
    left_tie = (left["doc_id"], int(left["doc_version"]))
    right_tie = (right["doc_id"], int(right["doc_version"]))
    if left_tie < right_tie:
        return -1
    if left_tie > right_tie:
        return 1
    return 0


def _variant_relation(
    left: dict[str, Any],
    right: dict[str, Any],
    variant_name: str,
) -> int:
    left_value = left["diagnostics"]["variant_scores"][variant_name]
    right_value = right["diagnostics"]["variant_scores"][variant_name]
    if left_value > right_value:
        return -1
    if left_value < right_value:
        return 1
    left_tie = (left["doc_id"], int(left["doc_version"]))
    right_tie = (right["doc_id"], int(right["doc_version"]))
    if left_tie < right_tie:
        return -1
    if left_tie > right_tie:
        return 1
    return 0


def _pair_flip_cause(left: dict[str, Any], right: dict[str, Any]) -> str:
    fixed_relation = _relation(left, right, "fixed_score_units", fixed=True)
    float_relation = _relation(left, right, "float_score", fixed=False)
    if fixed_relation == float_relation:
        return "none"
    if int(left["fixed_score_units"]) == int(right["fixed_score_units"]):
        return "tie_break_rule"
    candidates: list[str] = []
    for variant_name, cause in (
        ("saturation_quantized_only", "term_frequency_saturation_rounding"),
        ("length_quantized_only", "field_length_normalization_quantization"),
        ("idf_quantized_only", "inverse_document_frequency_quantization"),
        ("field_weight_quantized_only", "field_weight_quantization"),
    ):
        if _variant_relation(left, right, variant_name) == fixed_relation:
            candidates.append(cause)
    return candidates[0] if candidates else "accumulation_quantization"


def _all_pair_flips(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for left_index in range(len(scores)):
        for right_index in range(left_index + 1, len(scores)):
            left = scores[left_index]
            right = scores[right_index]
            fixed_relation = _relation(left, right, "fixed_score_units", fixed=True)
            float_relation = _relation(left, right, "float_score", fixed=False)
            if fixed_relation == float_relation:
                continue
            fixed_first, fixed_second = (
                (left, right) if fixed_relation < 0 else (right, left)
            )
            float_first, float_second = (
                (left, right) if float_relation < 0 else (right, left)
            )
            cause = _pair_flip_cause(left, right)
            evidence.append(
                {
                    "fixed_first": fixed_first["doc_id"],
                    "fixed_second": fixed_second["doc_id"],
                    "float_first": float_first["doc_id"],
                    "float_second": float_second["doc_id"],
                    "cause": cause,
                    "fixed_scores": {
                        left["doc_id"]: left["fixed_score_units"],
                        right["doc_id"]: right["fixed_score_units"],
                    },
                    "float_scores": {
                        left["doc_id"]: left["float_score"],
                        right["doc_id"]: right["float_score"],
                    },
                }
            )
    return evidence


def rank_query(
    query: QuerySpec,
    scores: Iterable[dict[str, Any]],
    config: PrototypeConfig,
) -> dict[str, Any]:
    score_list = sorted(scores, key=_fixed_key)
    fixed_order = [score["doc_id"] for score in score_list]
    float_order = [
        score["doc_id"] for score in sorted(score_list, key=_float_key)
    ]
    fixed_positions = {
        doc_id: index for index, doc_id in enumerate(fixed_order)
    }
    float_positions = {
        doc_id: index for index, doc_id in enumerate(float_order)
    }
    position_flip_count = sum(
        1 for doc_id in fixed_order if fixed_positions[doc_id] != float_positions[doc_id]
    )
    pair_flip_evidence = _all_pair_flips(score_list)

    top_k = score_list[: config.top_k]
    top_k_ids = [score["doc_id"] for score in top_k]
    float_top_k_ids = float_order[: config.top_k]
    top_k_set_diff = sorted(set(top_k_ids) ^ set(float_top_k_ids))

    boundary_units: int | None = None
    boundary_kind = "empty"
    if score_list:
        if len(score_list) >= config.top_k:
            boundary_units = int(score_list[config.top_k - 1]["fixed_score_units"])
            boundary_kind = "top_k_cutoff"
        else:
            boundary_units = int(score_list[-1]["fixed_score_units"])
            boundary_kind = "no_truncation_minimum"

    quantum = config.fixed.quantum
    boundary_scores: list[dict[str, Any]] = []
    if boundary_units is not None:
        for score in score_list:
            delta = int(score["fixed_score_units"]) - boundary_units
            if 2 * abs(delta) <= quantum:
                boundary_scores.append(score)

    boundary_doc_ids = {score["doc_id"] for score in boundary_scores}
    boundary_docs: list[dict[str, Any]] = []
    for score in boundary_scores:
        related_causes = sorted(
            {
                pair["cause"]
                for pair in pair_flip_evidence
                if pair["cause"] != "none"
                and (
                    pair["fixed_first"] in boundary_doc_ids
                    or pair["fixed_second"] in boundary_doc_ids
                )
                and (
                    pair["fixed_first"] == score["doc_id"]
                    or pair["fixed_second"] == score["doc_id"]
                )
            }
        )
        boundary_docs.append(
            {
                "doc_id": score["doc_id"],
                "doc_version": score["doc_version"],
                "in_top_k": score["doc_id"] in top_k_ids,
                "fixed_rank": fixed_positions[score["doc_id"]] + 1,
                "float_rank": float_positions[score["doc_id"]] + 1,
                "fixed_score_units": score["fixed_score_units"],
                "fixed_score": score["fixed_score"],
                "float_reference_score": score["float_score"],
                "boundary_delta_units": int(score["fixed_score_units"])
                - int(boundary_units),
                "fixed_minus_float_units": score["diagnostics"][
                    "fixed_minus_float_units"
                ],
                "within_half_quantum": True,
                "stage_abs_error_units": score["diagnostics"][
                    "stage_max_abs_error_units"
                ],
                "first_tolerance_exceeded_stage": score["diagnostics"][
                    "first_tolerance_exceeded_stage"
                ],
                "flip_causes": related_causes,
                "attribution": (
                    "tie_break_rule"
                    if "tie_break_rule" in related_causes
                    else (related_causes[0] if related_causes else "no_order_flip")
                ),
            }
        )

    return {
        "query_id": query.query_id,
        "query_terms": list(query.sorted_unique_terms()),
        "field_preferences": list(query.field_preferences),
        "top_k_limit": config.top_k,
        "top_k": [
            {
                "rank": index + 1,
                "doc_id": score["doc_id"],
                "doc_version": score["doc_version"],
                "fixed_score_units": score["fixed_score_units"],
                "fixed_score": score["fixed_score"],
                "float_reference_score": score["float_score"],
                "is_boundary": score["doc_id"] in boundary_doc_ids,
            }
            for index, score in enumerate(top_k)
        ],
        "all_fixed_order": fixed_order,
        "all_float_order": float_order,
        "order_flip": {
            "has_flip": bool(pair_flip_evidence),
            "pair_flip_count": len(pair_flip_evidence),
            "position_flip_count": position_flip_count,
            "top_k_symmetric_difference": top_k_set_diff,
            "evidence": pair_flip_evidence,
        },
        "boundary": {
            "kind": boundary_kind,
            "score_units": boundary_units,
            "half_quantum_units": quantum / 2,
            "documents": boundary_docs,
        },
    }


def ranking_cache_key(
    query: QuerySpec,
    scores: Iterable[dict[str, Any]],
    config: PrototypeConfig,
    score_cache_keys: dict[str, str],
) -> str:
    from .util import sha256_json

    ordered = sorted(scores, key=_fixed_key)
    return sha256_json(
        {
            "query_fingerprint": query.content_fingerprint,
            "top_k": config.top_k,
            "quantum": config.fixed.quantum,
            "decimal_places": config.fixed.decimal_places,
            "stage_tolerance_units": config.stage_tolerance_units,
            "scores": [
                {
                    "doc_id": score["doc_id"],
                    "doc_version": score["doc_version"],
                    "fixed_score_units": score["fixed_score_units"],
                    "float_score": score["float_score"],
                    "score_cache_key": score_cache_keys.get(score["doc_id"], ""),
                }
                for score in ordered
            ],
        }
    )
