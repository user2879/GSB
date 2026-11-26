"""Unified multi-scheme RoPE encoding and fixed-point rotation kernels."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .fixedpoint import (
    dequantize,
    in_signed_range,
    quantize_real,
    requantize_product,
    wrap_signed,
)


@dataclass(frozen=True)
class RoPEScheme:
    """All named schemes use this one interface and return structure."""

    name: str
    base_frequency: float
    angle_scale: float
    block_size: int
    frequency_correction: int

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> "RoPEScheme":
        return cls(
            name=str(raw["name"]),
            base_frequency=float(raw["base_frequency"]),
            angle_scale=float(raw["angle_scale"]),
            block_size=int(raw["block_size"]),
            frequency_correction=int(raw["frequency_correction"]),
        )

    def angle_for_dimension(self, position: int, dimension_index: int, dimension_count: int) -> dict[str, Any]:
        pair_first = dimension_index - (dimension_index % 2)
        block_index = pair_first // self.block_size
        block_start = block_index * self.block_size
        numerator = block_start + self.frequency_correction
        inverse_frequency = self.base_frequency ** (-numerator / dimension_count)
        angle = float(position) * self.angle_scale * inverse_frequency
        return {
            "scheme": self.name,
            "position": int(position),
            "dimension_index": int(dimension_index),
            "pair_index": pair_first // 2,
            "block_index": int(block_index),
            "block_start": int(block_start),
            "inverse_frequency": inverse_frequency,
            "angle": angle,
            "cos": math.cos(angle),
            "sin": math.sin(angle),
        }

    def encode_position(self, position: int, dimension_count: int) -> list[dict[str, Any]]:
        return [
            self.angle_for_dimension(position, dimension_index, dimension_count)
            for dimension_index in range(dimension_count)
        ]


def scheme_map(config: dict[str, Any]) -> dict[str, RoPEScheme]:
    return {raw["name"]: RoPEScheme.from_config(raw) for raw in config["schemes"]}


def quantized_encoding(encoding: dict[str, Any], fixed: dict[str, Any]) -> dict[str, Any]:
    bits = fixed["fraction_bits"]
    level = fixed["quant_level"]
    signed_bits = fixed["signed_bits"]
    result = dict(encoding)
    for field_name in ("angle", "cos", "sin"):
        real_value = float(encoding[field_name])
        integer_value = quantize_real(real_value, bits, level)
        reconstructed = dequantize(integer_value, bits)
        result[f"q_{field_name}"] = integer_value
        result[f"{field_name}_step"] = dequantize(1, bits)
        result[f"{field_name}_residual"] = reconstructed - real_value
        result[f"{field_name}_fixed"] = reconstructed
        result[f"{field_name}_range_overflow"] = not in_signed_range(integer_value, signed_bits)
    return result


def float_rotation_for_pair(
    x0: float,
    x1: float,
    cosine: float,
    sine: float,
) -> tuple[float, float]:
    # Explicit float64 operations; no NumPy reduction or implicit pairwise kernel.
    out0 = float(x0) * float(cosine) - float(x1) * float(sine)
    out1 = float(x0) * float(sine) + float(x1) * float(cosine)
    return out0, out1


def rotate_vector_fixed(
    vector: list[float],
    position: int,
    scheme: RoPEScheme,
    dimension_count: int,
    fixed: dict[str, Any],
) -> list[dict[str, Any]]:
    """Rotate one vector and return one comparable record per dimension."""
    bits = fixed["fraction_bits"]
    level = fixed["quant_level"]
    signed_bits = fixed["signed_bits"]
    records: list[dict[str, Any]] = []

    for dimension_index in range(0, dimension_count, 2):
        x0 = float(vector[dimension_index])
        x1 = float(vector[dimension_index + 1])
        encoding = scheme.angle_for_dimension(position, dimension_index, dimension_count)
        qenc = quantized_encoding(encoding, fixed)
        cosine = encoding["cos"]
        sine = encoding["sin"]

        truth0, truth1 = float_rotation_for_pair(x0, x1, cosine, sine)

        qx0 = quantize_real(x0, bits, level)
        qx1 = quantize_real(x1, bits, level)
        qcos = qenc["q_cos"]
        qsin = qenc["q_sin"]

        raw0 = qx0 * qcos - qx1 * qsin
        raw1 = qx0 * qsin + qx1 * qcos
        qrot0 = requantize_product(raw0, bits, level)
        qrot1 = requantize_product(raw1, bits, level)

        qrot0_wrapped, wrap0 = wrap_signed(qrot0, signed_bits)
        qrot1_wrapped, wrap1 = wrap_signed(qrot1, signed_bits)
        if wrap0:
            qrot0 = qrot0_wrapped
        if wrap1:
            qrot1 = qrot1_wrapped

        for offset, (x, qx, raw, qrot, truth, wrapped) in enumerate(
            (
                (x0, qx0, raw0, qrot0, truth0, wrap0),
                (x1, qx1, raw1, qrot1, truth1, wrap1),
            )
        ):
            d = dimension_index + offset
            fixed_value = dequantize(qrot, bits)
            records.append(
                {
                    "dimension_index": d,
                    "pair_index": dimension_index // 2,
                    "position": int(position),
                    "input_value": x,
                    "q_input": qx,
                    "angle": encoding["angle"],
                    "q_angle": qenc["q_angle"],
                    "cos": cosine,
                    "sin": sine,
                    "q_cos": qcos,
                    "q_sin": qsin,
                    "raw_product_sum": raw,
                    "q_rotated": qrot,
                    "fixed_value": fixed_value,
                    "truth_value": truth,
                    "error": fixed_value - truth,
                    "signed_wrap": wrapped,
                    "angle_residual": qenc["angle_residual"],
                    "cos_residual": qenc["cos_residual"],
                    "sin_residual": qenc["sin_residual"],
                }
            )
    return records


def rotate_single_dimension(
    vector: list[float],
    position: int,
    dimension_index: int,
    scheme: RoPEScheme,
    dimension_count: int,
    fixed: dict[str, Any],
) -> dict[str, Any]:
    """Rotate one output dimension. Its adjacent pair is its dependency unit."""
    bits = fixed["fraction_bits"]
    level = fixed["quant_level"]
    signed_bits = fixed["signed_bits"]
    d0 = int(dimension_index) - (int(dimension_index) % 2)
    d1 = d0 + 1
    x0 = float(vector[d0])
    x1 = float(vector[d1])
    encoding = scheme.angle_for_dimension(position, d0, dimension_count)
    qenc = quantized_encoding(encoding, fixed)
    truth0, truth1 = float_rotation_for_pair(x0, x1, encoding["cos"], encoding["sin"])

    qx0 = quantize_real(x0, bits, level)
    qx1 = quantize_real(x1, bits, level)
    raw0 = qx0 * qenc["q_cos"] - qx1 * qenc["q_sin"]
    raw1 = qx0 * qenc["q_sin"] + qx1 * qenc["q_cos"]
    qrot0 = requantize_product(raw0, bits, level)
    qrot1 = requantize_product(raw1, bits, level)
    qrot0, wrap0 = wrap_signed(qrot0, signed_bits)
    qrot1, wrap1 = wrap_signed(qrot1, signed_bits)

    offset = int(dimension_index) - d0
    x, qx, raw, qrot, truth, wrapped = (
        (x0, qx0, raw0, qrot0, truth0, wrap0)
        if offset == 0
        else (x1, qx1, raw1, qrot1, truth1, wrap1)
    )
    return {
        "dimension_index": int(dimension_index),
        "pair_index": d0 // 2,
        "position": int(position),
        "input_value": x,
        "q_input": qx,
        "angle": encoding["angle"],
        "q_angle": qenc["q_angle"],
        "cos": encoding["cos"],
        "sin": encoding["sin"],
        "q_cos": qenc["q_cos"],
        "q_sin": qenc["q_sin"],
        "raw_product_sum": raw,
        "q_rotated": qrot,
        "fixed_value": dequantize(qrot, bits),
        "truth_value": truth,
        "error": dequantize(qrot, bits) - truth,
        "signed_wrap": wrapped,
        "angle_residual": qenc["angle_residual"],
        "cos_residual": qenc["cos_residual"],
        "sin_residual": qenc["sin_residual"],
    }


def rotate_vector_angle_quantized(
    vector: list[float],
    position: int,
    scheme: RoPEScheme,
    dimension_count: int,
    fixed: dict[str, Any],
) -> list[float]:
    """Counterfactual: quantize only the angle, then stay in float64."""
    bits = fixed["fraction_bits"]
    level = fixed["quant_level"]
    output: list[float] = []
    for dimension_index in range(0, dimension_count, 2):
        encoding = scheme.angle_for_dimension(position, dimension_index, dimension_count)
        q_angle = quantize_real(encoding["angle"], bits, level)
        angle = dequantize(q_angle, bits)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        out0, out1 = float_rotation_for_pair(
            float(vector[dimension_index]),
            float(vector[dimension_index + 1]),
            cosine,
            sine,
        )
        output.extend((out0, out1))
    return output
