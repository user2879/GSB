"""Deterministic channel normalization and fixed-point quantization."""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP
from typing import Any

from .config import FusionConfig
from .hashing import candidate_fingerprint


ROUNDING_MODES = {
    "half_even": ROUND_HALF_EVEN,
    "half_up": ROUND_HALF_UP,
    "truncate": ROUND_DOWN,
}


def _channel_statistics(scores: list[tuple[str, float]], normalization: str) -> tuple[dict[str, Any], bool]:
    count = len(scores)
    if count == 0:
        return {
            "count": 0,
            "denominator": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "min": 0.0,
            "sum": 0.0,
            "sum_sq": 0.0,
        }, True

    # The input list is already sorted by candidate_id. Every accumulation is
    # performed explicitly in that order to avoid dictionary/hash ordering or
    # thread completion timing leaking into floating-point statistics.
    total = 0.0
    total_sq = 0.0
    minimum = scores[0][1]
    maximum = scores[0][1]
    for _, score in scores:
        total += score
        total_sq += score * score
        if score < minimum:
            minimum = score
        if score > maximum:
            maximum = score
    mean = total / count
    if normalization == "minmax":
        denominator = maximum - minimum
    else:
        variance = 0.0
        for _, score in scores:
            delta = score - mean
            variance += delta * delta
        variance /= count
        denominator = math.sqrt(max(variance, 0.0))
    denominator = denominator if math.isfinite(denominator) else 0.0
    stats = {
        "count": count,
        "denominator": denominator,
        "max": maximum,
        "mean": mean if math.isfinite(mean) else 0.0,
        "min": minimum,
        "sum": total if math.isfinite(total) else 0.0,
        "sum_sq": total_sq if math.isfinite(total_sq) else 0.0,
    }
    return stats, denominator == 0.0


def _normalized_value(score: float, stats: dict[str, Any], normalization: str, zero_denominator: bool) -> float:
    if zero_denominator:
        return 0.0
    if normalization == "minmax":
        return (score - float(stats["min"])) / float(stats["denominator"])
    return (score - float(stats["mean"])) / float(stats["denominator"])


def quantize_normalized(value: float, config: FusionConfig) -> tuple[int, float, float]:
    scale = Decimal(2) ** config.quant_bits
    decimal_value = Decimal(str(value)) * scale
    rounding = ROUNDING_MODES[config.rounding]
    fixed = int(decimal_value.to_integral_value(rounding=rounding))
    step = config.quant_step
    dequantized = float(fixed) * step
    residual = float(value) - dequantized
    return fixed, dequantized, residual


def normalize_channel(
    channel: str,
    load_channel_state: dict[str, Any],
    config: FusionConfig,
) -> dict[str, Any]:
    candidates = load_channel_state.get("candidates", {})
    ordered_ids = sorted(candidates)
    score_pairs = [(candidate_id, float(candidates[candidate_id]["score"])) for candidate_id in ordered_ids]
    stats, zero_denominator = _channel_statistics(score_pairs, config.normalization)
    events: list[dict[str, Any]] = []
    if zero_denominator:
        events.append(
            {
                "channel": channel,
                "code": "E_NORMALIZE_ZERO_DENOM",
                "candidate_id": None,
                "detail": (
                    f"{config.normalization} denominator is zero for channel {channel!r}; "
                    "all normalized scores are set to 0.0"
                ),
            }
        )

    normalized_candidates: dict[str, Any] = {}
    for candidate_id in ordered_ids:
        source = candidates[candidate_id]
        normalized = _normalized_value(float(source["score"]), stats, config.normalization, zero_denominator)
        fixed, dequantized, residual = quantize_normalized(normalized, config)
        fingerprint = candidate_fingerprint(
            channel=channel,
            candidate_id=candidate_id,
            fixed=fixed,
            feature_dim=config.feature_dim,
        )
        normalized_candidates[candidate_id] = {
            "candidate_id": candidate_id,
            "channel": channel,
            "dequantized": dequantized,
            "fingerprint": fingerprint,
            "fixed": fixed,
            "fragment_id": source.get("fragment_id"),
            "fragment_sha256": source.get("fragment_sha256"),
            "norm": normalized,
            "quant_step": config.quant_step,
            "residual": residual,
        }

    return {
        "candidates": normalized_candidates,
        "events": events,
        "fragment_set_hash": load_channel_state["fragment_set_hash"],
        "stats": stats,
    }


def normalize_all(load_state: dict[str, Any], config: FusionConfig) -> dict[str, Any]:
    channels: dict[str, Any] = {}
    for channel in config.channel_names:
        channels[channel] = normalize_channel(channel, load_state["channels"][channel], config)
    return {
        "channels": channels,
        "parameters": config.normalization_parameters(),
    }
