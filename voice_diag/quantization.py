"""Fixed-point template quantization with auditable per-dimension residuals."""

from __future__ import annotations

import math
from typing import Any

from .config import profile_bits
from .fingerprint import round_float, sha256_json


def quantization_cache_key(acoustic: dict[str, Any], profile: str, fractional_bits: int) -> str:
    return "quant:" + sha256_json(
        {
            "template_key": acoustic["template_key"],
            "path": acoustic.get("path"),
            "vector_fingerprint": acoustic["vector_fingerprint"],
            "content_sha256": acoustic["content_sha256"],
            "projection_fingerprint": acoustic["projection_fingerprint"],
            "profile": profile,
            "fractional_bits": fractional_bits,
        }
    )


def _round_half_away(value: float) -> int:
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def quantize_vector(vector: list[float], profile: str, fractional_bits: int) -> dict[str, Any]:
    bits = profile_bits(profile)
    scale = 1 << fractional_bits
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    integers: list[int] = []
    dimensions: list[dict[str, Any]] = []
    clamped_dimensions: list[int] = []
    for index, value in enumerate(vector):
        scaled = float(value) * scale
        rounded = _round_half_away(scaled)
        clamped = max(minimum, min(maximum, rounded))
        if clamped != rounded:
            clamped_dimensions.append(index)
        integers.append(clamped)
        dimensions.append(
            {
                "index": index,
                "source": round_float(float(value)),
                "scaled": round_float(scaled),
                "integer": clamped,
                "step": f"1/{scale}",
                "step_value": round_float(1.0 / scale),
                "residual": round_float(float(value) - clamped / scale),
                "clamped": clamped != rounded,
            }
        )
    integer_fingerprint = sha256_json({"integers": integers, "profile": profile, "fractional_bits": fractional_bits})
    return {
        "profile": profile,
        "storage_bits": bits,
        "fractional_bits": fractional_bits,
        "step": f"1/{scale}",
        "step_value": round_float(1.0 / scale),
        "minimum": minimum,
        "maximum": maximum,
        "integers": integers,
        "dimensions": dimensions,
        "clamped": bool(clamped_dimensions),
        "clamped_dimensions": clamped_dimensions,
        "integer_fingerprint": integer_fingerprint,
    }


def quantize_acoustic(acoustic: dict[str, Any], profile: str, fractional_bits: int) -> dict[str, Any]:
    result = quantize_vector(acoustic["vector"], profile, fractional_bits)
    result.update(
        {
            "template_key": acoustic["template_key"],
            "command_word": acoustic["command_word"],
            "template_id": acoustic["template_id"],
            "content_sha256": acoustic["content_sha256"],
            "acoustic_fingerprint": acoustic["vector_fingerprint"],
            "effective_frames": acoustic["effective_frames"],
            "cache_key": quantization_cache_key(acoustic, profile, fractional_bits),
        }
    )
    return result


def reference_profile(profile: str) -> str:
    width = profile_bits(profile)
    for candidate in ("q8", "q12", "q16", "q20", "q24"):
        if profile_bits(candidate) > width:
            return candidate
    return profile
