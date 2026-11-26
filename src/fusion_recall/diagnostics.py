from __future__ import annotations

from copy import deepcopy
from typing import Any

from .scoring import score_query
from .vocab import Vocabulary


HIGH_PRECISION = {"frac_bits": 64, "quant_tier": "nearest"}


def _selected(result: dict[str, Any]) -> list[str]:
    return list(result["selected"])


def diagnose_query_flip(
    query: dict[str, Any],
    index: dict[str, Any],
    config: dict[str, Any],
    vocab: Vocabulary,
    profile_a: dict[str, Any] | None = None,
    profile_b: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile_a = profile_a or config["fixed_point"]
    profile_b = profile_b or {"frac_bits": 8, "quant_tier": "floor"}
    result_a = score_query(query, index, config, vocab, fixed_override=profile_a)
    result_b = score_query(query, index, config, vocab, fixed_override=profile_b)
    selected_a = _selected(result_a)
    selected_b = _selected(result_b)
    flipped = selected_a != selected_b

    diagnosis: dict[str, Any] = {
        "query_id": query["query_id"],
        "flipped": flipped,
        "selected_a": selected_a,
        "selected_b": selected_b,
        "profile_a": profile_a,
        "profile_b": profile_b,
        "causes": [],
        "primary_cause": "no_flip" if not flipped else "undetermined",
    }
    if not flipped:
        return diagnosis

    tf_a_lang_high = score_query(
        query,
        index,
        config,
        vocab,
        semantic_override=profile_a,
        language_override=HIGH_PRECISION,
        work_override=HIGH_PRECISION,
        include_boundary=False,
    )
    tf_b_lang_high = score_query(
        query,
        index,
        config,
        vocab,
        semantic_override=profile_b,
        language_override=HIGH_PRECISION,
        work_override=HIGH_PRECISION,
        include_boundary=False,
    )
    tf_delta = _selected(tf_a_lang_high) != _selected(tf_b_lang_high)

    tf_high_lang_a = score_query(
        query,
        index,
        config,
        vocab,
        semantic_override=HIGH_PRECISION,
        language_override=profile_a,
        work_override=HIGH_PRECISION,
        include_boundary=False,
    )
    tf_high_lang_b = score_query(
        query,
        index,
        config,
        vocab,
        semantic_override=HIGH_PRECISION,
        language_override=profile_b,
        work_override=HIGH_PRECISION,
        include_boundary=False,
    )
    language_delta = _selected(tf_high_lang_a) != _selected(tf_high_lang_b)

    boundary_a = result_a.get("boundary", {})
    boundary_b = result_b.get("boundary", {})
    boundary_ids_a = set(boundary_a.get("boundary_doc_ids", []))
    boundary_ids_b = set(boundary_b.get("boundary_doc_ids", []))
    symmetric_difference = set(selected_a) ^ set(selected_b)
    tie_at_cutoff = bool(
        symmetric_difference & (boundary_ids_a | boundary_ids_b)
        and (
            boundary_a.get("boundary_count", 0) >= 2
            or boundary_b.get("boundary_count", 0) >= 2
        )
    )

    diagnosis.update(
        {
            "tf_quantization_changes_hybrid_ranking": tf_delta,
            "language_rounding_changes_hybrid_ranking": language_delta,
            "tie_resolution_decides_boundary": tie_at_cutoff,
            "boundary_ids_a": sorted(boundary_ids_a),
            "boundary_ids_b": sorted(boundary_ids_b),
            "hybrid_selected": {
                "tf_a_lang_high": _selected(tf_a_lang_high),
                "tf_b_lang_high": _selected(tf_b_lang_high),
                "tf_high_lang_a": _selected(tf_high_lang_a),
                "tf_high_lang_b": _selected(tf_high_lang_b),
            },
        }
    )

    if tf_delta and language_delta:
        diagnosis["causes"] = ["tf_quantization", "language_weight_rounding"]
        diagnosis["primary_cause"] = "mixed_tf_and_language_rounding"
    elif tf_delta:
        diagnosis["causes"] = ["tf_quantization"]
        diagnosis["primary_cause"] = "tf_quantization"
    elif language_delta:
        diagnosis["causes"] = ["language_weight_rounding"]
        diagnosis["primary_cause"] = "language_weight_rounding"
    elif tie_at_cutoff:
        diagnosis["causes"] = ["truncation_tie"]
        diagnosis["primary_cause"] = "truncation_tie"
    else:
        diagnosis["causes"] = ["score_or_config_change"]
        diagnosis["primary_cause"] = "score_or_config_change"
    if tie_at_cutoff and "truncation_tie" not in diagnosis["causes"]:
        diagnosis["causes"].append("truncation_tie_context")
    return diagnosis


def diagnose_all_flips(
    queries: list[dict[str, Any]],
    index: dict[str, Any],
    config: dict[str, Any],
    vocab: Vocabulary,
    profile_a: dict[str, Any] | None = None,
    profile_b: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return [
        diagnose_query_flip(
            query,
            index,
            config,
            vocab,
            profile_a=deepcopy(profile_a) if profile_a else None,
            profile_b=deepcopy(profile_b) if profile_b else None,
        )
        for query in sorted(queries, key=lambda item: item["query_id"])
    ]
