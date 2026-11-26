"""Fixed-point quantisation and accumulation primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canonical import content_digest, round_div, rounding_direction
from .errors import E_QUANT_PROFILE, fail


@dataclass(frozen=True)
class QuantProfile:
    name: str
    fraction_bits: int
    scale: int

    @property
    def descriptor(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fraction_bits": self.fraction_bits,
            "scale": self.scale,
        }


def make_profile(name: str, fraction_bits: int) -> QuantProfile:
    if fraction_bits < 0 or fraction_bits > 30:
        fail(E_QUANT_PROFILE, "quantisation fraction bits outside supported range", profile=name)
    return QuantProfile(name=name, fraction_bits=fraction_bits, scale=1 << fraction_bits)


def quantize_milli_vector(vector: list[int], profile: QuantProfile) -> dict[str, Any]:
    """Quantise milli-unit values into a profile's binary fixed-point domain."""

    quantized: list[int] = []
    residuals: list[int] = []
    directions: list[str] = []
    for value in vector:
        numerator = int(value) * profile.scale
        qvalue = round_div(numerator, 1000)
        quantized.append(qvalue)
        residuals.append(qvalue * 1000 - numerator)
        directions.append(rounding_direction(numerator, 1000))
    return {
        "profile": profile.name,
        "scale": profile.scale,
        "fraction_bits": profile.fraction_bits,
        "input_scale": 1000,
        "values": quantized,
        "residual_num": residuals,
        "round_directions": directions,
        "digest": content_digest(
            {
                "profile": profile.name,
                "fraction_bits": profile.fraction_bits,
                "values": quantized,
            }
        ),
    }


def weighted_accumulate(
    quantized: list[int],
    type_weights_permille: dict[str, int],
    ordered_types: list[str],
    profile: QuantProfile,
) -> dict[str, Any]:
    """Fuse vectors in fixed-point integer space.

    Contributions are consumed by condition type name and then by dimension.
    Every multiplication and residual is represented by integers; no floating
    point operation participates in the result.
    """

    if sorted(type_weights_permille) != sorted(ordered_types):
        fail(
            E_QUANT_PROFILE,
            "fusion weight map and ordered condition types differ",
            weights=sorted(type_weights_permille),
            types=sorted(ordered_types),
        )
    accumulator = [0] * len(quantized[ordered_types[0]])
    stages: list[dict[str, Any]] = []
    for condition_type in sorted(ordered_types):
        values = quantized[condition_type]
        weight = int(type_weights_permille[condition_type])
        if len(values) != len(accumulator):
            fail(
                "E_DIMENSION",
                "condition vectors have inconsistent dimensions during fusion",
                condition_type=condition_type,
                expected=len(accumulator),
                actual=len(values),
            )
        for dimension, value in enumerate(values):
            numerator = int(value) * weight
            delta = round_div(numerator, 1000)
            residual = numerator - delta * 1000
            accumulator[dimension] += delta
            stages.append(
                {
                    "order": len(stages),
                    "condition_type": condition_type,
                    "dimension": dimension,
                    "quantization_step_scale": profile.scale,
                    "weight_permille": weight,
                    "input": int(value),
                    "numerator": numerator,
                    "delta": delta,
                    "residual_num": residual,
                    "round_direction": rounding_direction(numerator, 1000),
                    "accumulator_after": accumulator[dimension],
                }
            )
    digest = content_digest(
        {
            "profile": profile.name,
            "scale": profile.scale,
            "weights": {key: int(type_weights_permille[key]) for key in sorted(type_weights_permille)},
            "vector": accumulator,
        }
    )
    return {
        "profile": profile.name,
        "fraction_bits": profile.fraction_bits,
        "scale": profile.scale,
        "vector": accumulator,
        "stages": stages,
        "residual_total": sum(abs(item["residual_num"]) for item in stages),
        "digest": digest,
    }
