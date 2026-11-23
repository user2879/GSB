"""Fixed-point feature fusion with an exact, ordered integer MAC sequence."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from .canonical import (
    decimal_text,
    fixed_from_real,
    round_half_away,
    score_scale_text,
    weight_fixed_from_real,
)
from .models import FeatureCatalog, LabConfig, RawCandidate


def fusion_parameters(config: LabConfig) -> dict[str, Any]:
    scale = 10**config.fixed_decimal_places
    weights_fixed = [weight_fixed_from_real(weight, config.fixed_decimal_places) for weight in config.feature_weights]
    temperature_fixed = weight_fixed_from_real(
        config.fusion_temperature, config.fixed_decimal_places
    )
    if temperature_fixed <= 0:
        raise ValueError("fusion temperature quantises to a non-positive integer")
    lower_fixed = fixed_from_real(
        config.score_min, config.fixed_decimal_places, config.quantization_step
    )
    upper_fixed = fixed_from_real(
        config.score_max, config.fixed_decimal_places, config.quantization_step
    )
    if lower_fixed > upper_fixed:
        raise ValueError("quantised score interval is empty")
    return {
        "scale": scale,
        "weights_fixed": tuple(weights_fixed),
        "temperature_fixed": temperature_fixed,
        "lower_fixed": lower_fixed,
        "upper_fixed": upper_fixed,
        "score_scale": score_scale_text(
            config.fixed_decimal_places, config.quantization_step
        ),
    }


def fuse_candidate(
    candidate: RawCandidate,
    normalized_vector: list[int],
    catalog: FeatureCatalog,
    config: LabConfig,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Fuse one candidate using integer multiply-add ordered by feature order."""

    if len(normalized_vector) != catalog.dimension:
        raise ValueError("normalized vector dimension mismatch")
    weights_fixed: tuple[int, ...] = params["weights_fixed"]
    temperature_fixed: int = params["temperature_fixed"]
    contributions: list[dict[str, Any]] = []
    raw_score = 0
    for index, feature_name in enumerate(catalog.feature_order):
        feature_fixed = normalized_vector[index]
        weight_fixed = weights_fixed[index]
        contribution = round_half_away(Fraction(feature_fixed * weight_fixed, temperature_fixed))
        raw_score += contribution
        contributions.append(
            {
                "feature": feature_name,
                "group": catalog.group_for(feature_name) or "",
                "normalized_fixed": feature_fixed,
                "weight_fixed": weight_fixed,
                "contribution_fixed": contribution,
            }
        )
    score_fixed = min(max(raw_score, params["lower_fixed"]), params["upper_fixed"])
    return {
        "item_id": candidate.item_id,
        "raw_score_fixed": raw_score,
        "score_fixed": score_fixed,
        "clamped": score_fixed != raw_score,
        "clamp_reason": (
            "lower_bound"
            if score_fixed == params["lower_fixed"] and raw_score < params["lower_fixed"]
            else (
                "upper_bound"
                if score_fixed == params["upper_fixed"] and raw_score > params["upper_fixed"]
                else None
            )
        ),
        "feature_dimension": catalog.dimension,
        "contributions": contributions,
    }


def fuse_request(
    candidates: tuple[RawCandidate, ...],
    normalization_entry: dict[str, Any],
    catalog: FeatureCatalog,
    config: LabConfig,
) -> dict[str, Any]:
    params = fusion_parameters(config)
    normalized = normalization_entry["normalized_fixed"]
    scored = [
        fuse_candidate(
            candidate,
            list(normalized[candidate.item_id]),
            catalog,
            config,
            params,
        )
        for candidate in sorted(candidates, key=lambda item: item.item_id)
    ]
    order = sorted(scored, key=lambda item: (-item["score_fixed"], item["item_id"]))
    return {
        "feature_dimension": catalog.dimension,
        "score_scale": params["score_scale"],
        "score_min_fixed": params["lower_fixed"],
        "score_max_fixed": params["upper_fixed"],
        "weights_fixed": list(params["weights_fixed"]),
        "temperature_fixed": params["temperature_fixed"],
        "candidates": {item["item_id"]: item for item in scored},
        "fusion_order": [item["item_id"] for item in order],
        "mac_units": catalog.dimension * len(scored),
    }


def fusion_summary(entry: dict[str, Any], candidate_ids: list[str]) -> dict[str, Any]:
    return {
        "feature_dimension": entry["feature_dimension"],
        "score_scale": entry["score_scale"],
        "score_min_fixed": entry["score_min_fixed"],
        "score_max_fixed": entry["score_max_fixed"],
        "scores": [
            {
                "item_id": item_id,
                "score_fixed": entry["candidates"][item_id]["score_fixed"],
                "raw_score_fixed": entry["candidates"][item_id]["raw_score_fixed"],
                "score_scale": entry["score_scale"],
            }
            for item_id in candidate_ids
        ],
    }


def score_decimal(score_fixed: int, config: LabConfig) -> str:
    return decimal_text(
        Fraction(score_fixed * config.quantization_step, 10**config.fixed_decimal_places),
        places=config.fixed_decimal_places + 4,
    )


def contribution_sum(score_entry: dict[str, Any]) -> int:
    return sum(item["contribution_fixed"] for item in score_entry["contributions"])



def assemble_fusion_from_entries(
    candidates: tuple[RawCandidate, ...],
    catalog: FeatureCatalog,
    config: LabConfig,
    entries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Assemble a fusion stage from cached/recomputed per-candidate entries."""

    params = fusion_parameters(config)
    ordered = [entries[candidate.item_id] for candidate in sorted(candidates, key=lambda item: item.item_id)]
    order = sorted(ordered, key=lambda item: (-item["score_fixed"], item["item_id"]))
    return {
        "feature_dimension": catalog.dimension,
        "score_scale": params["score_scale"],
        "score_min_fixed": params["lower_fixed"],
        "score_max_fixed": params["upper_fixed"],
        "weights_fixed": list(params["weights_fixed"]),
        "temperature_fixed": params["temperature_fixed"],
        "candidates": {item["item_id"]: item for item in ordered},
        "fusion_order": [item["item_id"] for item in order],
        "mac_units": catalog.dimension * len(ordered),
    }
