"""Independent float64 reference path for numerical observability."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import FusionConfig


def reference_normalize_channel(load_channel_state: dict[str, Any], normalization: str) -> dict[str, float]:
    candidates = load_channel_state.get("candidates", {})
    ordered_ids = sorted(candidates)
    if not ordered_ids:
        return {}
    values = np.asarray([float(candidates[candidate_id]["score"]) for candidate_id in ordered_ids], dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        if normalization == "minmax":
            minimum = np.min(values, initial=values[0])
            maximum = np.max(values, initial=values[0])
            denominator = float(maximum - minimum)
            if denominator == 0.0 or not math.isfinite(denominator):
                normalized = np.zeros_like(values)
            else:
                normalized = (values - minimum) / denominator
        else:
            mean = np.mean(values, dtype=np.float64)
            denominator = float(np.sqrt(np.mean(np.square(values - mean), dtype=np.float64)))
            if denominator == 0.0 or not math.isfinite(denominator):
                normalized = np.zeros_like(values)
            else:
                normalized = (values - mean) / denominator
    return {candidate_id: float(normalized[index]) for index, candidate_id in enumerate(ordered_ids)}


def reference_normalized_channels(load_state: dict[str, Any], config: FusionConfig) -> dict[str, dict[str, float]]:
    return {
        channel: reference_normalize_channel(load_state["channels"][channel], config.normalization)
        for channel in config.channel_names
    }


def reference_fused_score(
    contributions: list[dict[str, Any]],
    reference_channels: dict[str, dict[str, float]],
    config: FusionConfig,
) -> float:
    total = 0.0
    for contribution in sorted(contributions, key=lambda item: item["channel"]):
        channel = contribution["channel"]
        candidate_id = contribution["candidate_id"]
        total += float(reference_channels[channel][candidate_id]) * config.channels[channel].weight
    return total
