"""Fixed-point initial emotion evidence aggregation."""

from __future__ import annotations

from typing import Any

from .constants import STAGE_ARTIFACT_VERSION
from .util import (
    decimal_to_fixed,
    json_hash,
    mul_fixed,
    normalize_fixed,
    quantize_levels,
    round_div,
    stable_softmax_fixed,
)

FEATURE_TO_AGGREGATE = {
    "f0_mean": "f0_mean_hz",
    "f0_std": "f0_std_hz",
    "energy_mean": "energy_mean_dbfs",
    "energy_dynamic": "energy_dynamic_db",
    "speech_rate": "speech_rate_sps",
    "rhythm_variability": "rhythm_variability",
}


def initial_cache_key(
    extract_key: str,
    config: dict[str, Any],
    injection_fingerprint: list[dict[str, Any]] | None = None,
) -> str:
    fixed = config["fixed_point"]
    return json_hash(
        {
            "stage": "initial",
            "artifact_version": STAGE_ARTIFACT_VERSION,
            "extract_key": extract_key,
            "evidence": config["evidence"],
            "decimal_places": fixed["decimal_places"],
            "quantization_levels": fixed["quantization_levels"],
            "injections": injection_fingerprint or [],
        }
    )


def build_initial_evidence(
    aggregate: dict[str, Any],
    extract_key: str,
    config: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    fixed = config["fixed_point"]
    scale = int(fixed["scale"])
    levels = int(fixed["quantization_levels"])
    cache_key = initial_cache_key(extract_key, config)
    normalized: dict[str, int] = {}
    for key in sorted(FEATURE_TO_AGGREGATE):
        low, high = config["evidence"]["feature_ranges"][key]
        normalized[key] = normalize_fixed(float(aggregate[FEATURE_TO_AGGREGATE[key]]), float(low), float(high), scale)

    scores: dict[str, int] = {}
    weight_details: dict[str, dict[str, int]] = {}
    for emotion in config["evidence"]["emotions"]:
        score = 0
        detail: dict[str, int] = {}
        for feature in sorted(FEATURE_TO_AGGREGATE):
            weight = decimal_to_fixed(config["evidence"]["initial_weights"][emotion][feature], int(fixed["decimal_places"]))
            term = round_div(weight * normalized[feature], scale)
            detail[feature] = term
            score += term
        scores[emotion] = int(score)
        weight_details[emotion] = detail
    intensities = stable_softmax_fixed(scores.items(), scale, levels)
    quantized = {emotion: quantize_levels(intensities[emotion], scale, levels) for emotion in sorted(intensities)}
    record = {
        "cache_key": cache_key,
        "status": "ok",
        "extract_key": extract_key,
        "algorithm": "weighted_feature_scores_then_stable_softmax",
        "fixed_point": {
            "decimal_places": fixed["decimal_places"],
            "bit_width": fixed["bit_width"],
            "scale": scale,
            "quantization_levels": levels,
        },
        "frames_used": aggregate["frames_used"],
        "feature_subset": aggregate["feature_subset"],
        "scoring_feature_subset": sorted(FEATURE_TO_AGGREGATE),
        "aggregate_features": aggregate,
        "normalized_features_fixed": normalized,
        "raw_scores_fixed": scores,
        "score_terms_fixed": weight_details,
        "intensities_fixed": quantized,
    }
    return cache_key, record
