"""Per-dimension adaptive quantization with physically applied tier changes."""

from __future__ import annotations

import copy
import math
from typing import Any

from .util import cache_key, round_half_away_from_zero, sha256_obj, stable_float


def _feature_value(features: dict[str, Any], source: str) -> float:
    value = features.get(source, 0.0)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return numeric if math.isfinite(numeric) else 0.0


def _choose_tier(minimum: float, maximum: float, dimension: dict[str, Any], forced: int | None) -> tuple[int, str]:
    if forced is not None:
        return int(forced), "forced"
    dynamic_range = maximum - minimum
    candidates = [int(value) for value in dimension["candidate_tiers"]]
    max_step = float(dimension["max_step"])
    for bits in candidates:
        levels = 2**bits
        step = dynamic_range / max(1, levels - 1)
        if step <= max_step:
            return bits, "auto_dynamic_range"
    return candidates[-1], "auto_dynamic_range_ceiling"


def compute_quantization_plan(
    segment_rows: list[dict[str, Any]],
    config: dict[str, Any],
    tier_overrides: dict[str, int | None] | None = None,
) -> dict[str, Any]:
    qconfig = config["quantization"]
    overrides = tier_overrides or {}
    minimum_range = float(qconfig.get("minimum_range", 1e-9))
    dimensions: list[dict[str, Any]] = []
    for dimension in qconfig["dimensions"]:
        source = str(dimension["source"])
        values = [_feature_value(row["features"], source) for row in segment_rows]
        if values:
            minimum = min(values)
            maximum = max(values)
        else:
            minimum = 0.0
            maximum = 0.0
        if maximum - minimum < minimum_range:
            maximum = minimum + minimum_range
        forced = overrides.get(str(dimension["name"]))
        bits, mode = _choose_tier(minimum, maximum, dimension, forced)
        if bits <= 0 or bits > 24:
            raise ValueError(f"invalid quantization tier {bits} for {dimension['name']}")
        levels = 2**bits
        step = (maximum - minimum) / max(1, levels - 1)
        signature_payload = {
            "name": dimension["name"],
            "source": source,
            "minimum": stable_float(minimum, 12),
            "maximum": stable_float(maximum, 12),
            "bits": bits,
            "levels": levels,
            "step": stable_float(step, 15),
            "tier_mode": mode,
        }
        dimensions.append(
            {
                **signature_payload,
                "signature": sha256_obj(signature_payload),
                "candidate_tiers": [int(value) for value in dimension["candidate_tiers"]],
                "max_step": float(dimension["max_step"]),
            }
        )
    plan_payload = {"dimensions": [dimension["signature"] for dimension in dimensions], "version": 1}
    return {
        "dimensions": dimensions,
        "signature": sha256_obj(plan_payload),
        "dimension_count": len(dimensions),
    }


def quantize_segment(
    features: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    vector: list[float] = []
    codes: list[int] = []
    for dimension in plan["dimensions"]:
        source = dimension["source"]
        value = _feature_value(features, source)
        minimum = float(dimension["minimum"])
        maximum = float(dimension["maximum"])
        levels = int(dimension["levels"])
        step = float(dimension["step"])
        scaled = (value - minimum) / step if step > 0 else 0.0
        code = max(0, min(levels - 1, round_half_away_from_zero(scaled)))
        # Use the center of the selected quantization cell. Unlike code/(levels-1),
        # the cell-center representation changes when the bit width changes even for
        # values at a dynamic-range endpoint.
        normalized = (code + 0.5) / levels
        reconstructed = minimum + code * step
        residual = value - reconstructed
        records.append(
            {
                "dimension": dimension["name"],
                "source": source,
                "tier_bits": int(dimension["bits"]),
                "tier_mode": dimension["tier_mode"],
                "levels": levels,
                "step": stable_float(step, 15),
                "input_value": stable_float(value, 12),
                "quantized_integer": int(code),
                "normalized_value": stable_float(normalized, 12),
                "reconstructed_value": stable_float(reconstructed, 12),
                "residual": stable_float(residual, 12),
                "dynamic_range": stable_float(maximum - minimum, 12),
            }
        )
        vector.append(stable_float(normalized, 12))
        codes.append(int(code))
    return {
        "vector": vector,
        "codes": codes,
        "dimension_records": records,
    }


def description_cache_key(
    feature_key: str,
    plan: dict[str, Any],
    quantization_config: dict[str, Any],
) -> str:
    return cache_key(
        "segment_description",
        feature_key,
        plan["signature"],
        sha256_obj(quantization_config),
    )


def clone_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(plan)
