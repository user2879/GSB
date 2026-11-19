"""Fixed-point normalization, weighted fusion, feature stacking, and ranking."""

from __future__ import annotations

from typing import Any

import numpy as np

from .fixed import FixedFormat, round_ratio
from .models import AppConfig, CHANNEL_ORDER


def _normalize_channel(channel: dict[str, Any], config: AppConfig, fmt: FixedFormat) -> dict[str, Any]:
    candidates = channel["candidates"]
    if not candidates:
        return {
            **channel,
            "normalization_min": 0.0,
            "normalization_max": 0.0,
            "normalized_candidates": [],
        }
    raw_scores = np.asarray([candidate["raw_score"] for candidate in candidates], dtype=np.float64)
    raw_min = float(np.min(raw_scores))
    raw_max = float(np.max(raw_scores))
    if config.normalization == "min_max":
        denominator = raw_max - raw_min
        normalized = np.ones_like(raw_scores) if denominator == 0 else (raw_scores - raw_min) / denominator
    else:
        normalized = np.zeros_like(raw_scores) if raw_max == 0 else raw_scores / raw_max
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized_fixed = fmt.quantize_array(normalized).tolist()
    normalized_candidates = []
    for candidate, value, fixed_value in zip(candidates, normalized.tolist(), normalized_fixed):
        normalized_candidates.append(
            {
                "item_id": candidate["item_id"],
                "raw_score": candidate["raw_score"],
                "rank": candidate["rank"],
                "normalized_score": float(value),
                "normalized_fixed": int(fixed_value),
            }
        )
    return {
        **channel,
        "normalization_min": raw_min,
        "normalization_max": raw_max,
        "normalized_candidates": normalized_candidates,
    }


def _effective_weights(
    config: AppConfig, empty_channels: set[str], fmt: FixedFormat
) -> tuple[dict[str, int], list[dict[str, Any]], dict[str, Any]]:
    enabled = [name for name in CHANNEL_ORDER if config.channels[name].enabled]
    base_q = {name: fmt.quantize(config.channels[name].weight) for name in CHANNEL_ORDER}
    effective_q = {name: 0 for name in CHANNEL_ORDER}
    for name in enabled:
        effective_q[name] = base_q[name]

    active = [name for name in enabled if name not in empty_channels]
    missing = [name for name in enabled if name in empty_channels]
    for name in missing:
        effective_q[name] = 0
    reallocation: dict[str, Any] = {
        "policy": config.zero_channel_policy,
        "empty_channels": sorted(missing),
        "from_channels": [],
        "to_channels": [],
        "redistributed_fixed": 0,
    }

    if missing and config.zero_channel_policy == "none":
        for name in missing:
            effective_q[name] = 0
            reallocation["from_channels"].append(name)
    elif missing and active:
        redistributed = sum(base_q[name] for name in missing)
        active_base = sum(base_q[name] for name in active)
        reallocation["from_channels"] = sorted(missing)
        reallocation["to_channels"] = sorted(active)
        reallocation["redistributed_fixed"] = redistributed
        if active_base > 0:
            allocated = 0
            for name in active:
                addition = round_ratio(redistributed * base_q[name], active_base)
                effective_q[name] += addition
                allocated += addition
            effective_q[active[-1]] += redistributed - allocated
        else:
            quotient, remainder = divmod(redistributed, len(active))
            for name in active:
                effective_q[name] += quotient
            effective_q[active[-1]] += remainder

    details: list[dict[str, Any]] = []
    for name in CHANNEL_ORDER:
        if not config.channels[name].enabled:
            reason = "disabled"
        elif name in empty_channels:
            reason = "zero_candidates"
        elif name in reallocation["to_channels"] and reallocation["redistributed_fixed"] > 0:
            reason = "redistributed"
        else:
            reason = "active"
        details.append(
            {
                "channel": name,
                "enabled": config.channels[name].enabled,
                "base_weight": config.channels[name].weight,
                "base_weight_fixed": base_q[name],
                "effective_weight_fixed": effective_q[name],
                "effective_weight": effective_q[name] / fmt.scale,
                "reason": reason,
            }
        )
    reallocation["base_total_fixed"] = sum(base_q[name] for name in enabled)
    reallocation["effective_total_fixed"] = sum(effective_q[name] for name in CHANNEL_ORDER)
    return effective_q, details, reallocation


def _feature_values(
    item_id: str,
    recall_record: dict[str, Any],
    normalized_by_channel: dict[str, dict[str, dict[str, Any]]],
    fmt: FixedFormat,
) -> dict[str, tuple[float, int]]:
    values: dict[str, tuple[float, int]] = {}
    channel_value_names = {
        "lexical_norm": "lexical",
        "collaborative_norm": "collaborative",
        "popularity_norm": "popularity",
    }
    for feature_name, channel_name in channel_value_names.items():
        entry = normalized_by_channel[channel_name].get(item_id)
        if entry is None:
            values[feature_name] = (0.0, 0)
        else:
            values[feature_name] = (entry["normalized_score"], int(entry["normalized_fixed"]))
    inputs = recall_record["feature_inputs"].get(item_id, {})
    for feature_name in ("history_overlap", "freshness", "popularity_prior"):
        raw = min(1.0, max(0.0, float(inputs.get(feature_name, 0.0))))
        values[feature_name] = (raw, fmt.quantize(raw))
    return values


def run_fusion_for_user(recall_record: dict[str, Any], config: AppConfig, bits: int) -> dict[str, Any]:
    fmt = FixedFormat(bits=bits, rounding=config.rounding)
    user_id = recall_record["user_id"]
    normalized_channels: dict[str, Any] = {}
    normalized_by_channel: dict[str, dict[str, dict[str, Any]]] = {}
    empty_channels: set[str] = set()
    item_ids: set[str] = set()

    for name in CHANNEL_ORDER:
        channel = _normalize_channel(recall_record["channels"][name], config, fmt)
        normalized_channels[name] = channel
        normalized_by_channel[name] = {
            candidate["item_id"]: candidate for candidate in channel["normalized_candidates"]
        }
        if not channel["normalized_candidates"]:
            empty_channels.add(name)
        item_ids.update(candidate["item_id"] for candidate in channel["normalized_candidates"])

    effective_weights, weight_details, weight_reallocation = _effective_weights(config, empty_channels, fmt)
    temperature_fixed = fmt.quantize(config.temperature)
    if temperature_fixed <= 0:
        raise ValueError("fusion temperature is below the fixed-point resolution")
    clip_min_fixed = fmt.quantize(config.score_min)
    clip_max_fixed = fmt.quantize(config.score_max)

    candidates: list[dict[str, Any]] = []
    for item_id in sorted(item_ids):
        contribution_details: list[dict[str, Any]] = []
        rank_vector: list[int] = []
        fused_fixed = 0
        for name in CHANNEL_ORDER:
            channel = normalized_channels[name]
            normalized_entry = normalized_by_channel[name].get(item_id)
            rank = int(normalized_entry["rank"]) if normalized_entry is not None else int(channel["candidate_limit"]) + 1
            rank_vector.append(rank)
            normalized_fixed = int(normalized_entry["normalized_fixed"]) if normalized_entry is not None else 0
            weight_fixed = int(effective_weights[name])
            contribution_fixed = fmt.multiply(normalized_fixed, weight_fixed)
            contribution_with_temperature = fmt.divide(contribution_fixed, temperature_fixed)
            fused_fixed += contribution_with_temperature
            contribution_details.append(
                {
                    "channel": name,
                    "source_rank": None if normalized_entry is None else rank,
                    "normalized_score": None if normalized_entry is None else normalized_entry["normalized_score"],
                    "normalized_fixed": normalized_fixed,
                    "effective_weight": weight_fixed / fmt.scale,
                    "effective_weight_fixed": weight_fixed,
                    "pre_temperature_contribution_fixed": contribution_fixed,
                    "contribution_fixed": contribution_with_temperature,
                    "contribution": contribution_with_temperature / fmt.scale,
                }
            )

        feature_values = _feature_values(item_id, recall_record, normalized_by_channel, fmt)
        feature_details: list[dict[str, Any]] = []
        feature_total = 0
        for feature in config.ranking_features:
            raw_value, value_fixed = feature_values[feature.name]
            weight_fixed = fmt.quantize(feature.weight)
            contribution_fixed = fmt.multiply(value_fixed, weight_fixed)
            feature_total += contribution_fixed
            feature_details.append(
                {
                    "name": feature.name,
                    "value": raw_value,
                    "value_fixed": value_fixed,
                    "weight": feature.weight,
                    "weight_fixed": weight_fixed,
                    "contribution_fixed": contribution_fixed,
                    "contribution": contribution_fixed / fmt.scale,
                }
            )

        preclip_fixed = fused_fixed + feature_total
        final_fixed = fmt.clamp(preclip_fixed, clip_min_fixed, clip_max_fixed)
        candidates.append(
            {
                "item_id": item_id,
                "rank_vector": rank_vector,
                "channel_contributions": contribution_details,
                "fused_fixed": fused_fixed,
                "fused_score": fused_fixed / fmt.scale,
                "feature_contributions": feature_details,
                "feature_total_fixed": feature_total,
                "feature_total": feature_total / fmt.scale,
                "preclip_fixed": preclip_fixed,
                "preclip_score": preclip_fixed / fmt.scale,
                "final_fixed": final_fixed,
                "final_score": final_fixed / fmt.scale,
                "clipped": preclip_fixed != final_fixed,
            }
        )

    weighted_order = sorted(
        candidates,
        key=lambda candidate: (-candidate["fused_fixed"], tuple(candidate["rank_vector"]), candidate["item_id"]),
    )
    for rank, candidate in enumerate(weighted_order, start=1):
        candidate["weighted_rank"] = rank

    final_order = sorted(
        candidates,
        key=lambda candidate: (-candidate["final_fixed"], tuple(candidate["rank_vector"]), candidate["item_id"]),
    )
    for rank, candidate in enumerate(final_order, start=1):
        candidate["rank"] = rank

    return {
        "user_id": user_id,
        "bits": bits,
        "scale": fmt.scale,
        "quantum": fmt.quantum,
        "rounding": fmt.rounding,
        "rounding_metadata": {
            "fractional_bits": bits,
            "scale": fmt.scale,
            "quantization_step": fmt.quantum,
            "rounding": fmt.rounding,
            "operation_order": [
                "normalize_channel",
                "quantize_normalized",
                "multiply_effective_weight",
                "divide_by_temperature",
                "sum_channels_in_canonical_order",
                "quantize_feature_value_and_weight",
                "multiply_feature_contribution",
                "sum_features_in_config_order",
                "clip_to_configured_interval",
            ],
            "channel_order": list(CHANNEL_ORDER),
        },
        "temperature_fixed": temperature_fixed,
        "clip_min_fixed": clip_min_fixed,
        "clip_max_fixed": clip_max_fixed,
        "zero_channels": [name for name in CHANNEL_ORDER if name in empty_channels and config.channels[name].enabled],
        "weight_reallocation": weight_reallocation,
        "weight_decomposition": weight_details,
        "channels": normalized_channels,
        "candidates": candidates,
        "final_order": [candidate["item_id"] for candidate in final_order],
        "weighted_order": [candidate["item_id"] for candidate in weighted_order],
    }
