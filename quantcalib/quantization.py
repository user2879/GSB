"""Fixed-point group quantization and saturation accounting."""

from __future__ import annotations

from typing import Any

from .model import Layer, LayerQuantConfig
from .utils import clean_float, round_half_away_from_zero, stable_hash


def fixed_point_round(value: float, fractional_bits: int) -> float:
    quantum = 2.0 ** (-fractional_bits)
    rounded = round_half_away_from_zero(float(value) / quantum) * quantum
    return clean_float(rounded, 17)


def affine_raw(
    value: float,
    *,
    scale: float,
    zero: int,
    saturate: bool,
    qmin: int,
    qmax: int,
) -> tuple[float, int, int]:
    real = float(value) / float(scale) + float(zero)
    rounded = round_half_away_from_zero(real)
    if saturate:
        integer = min(qmax, max(qmin, rounded))
    else:
        integer = rounded
    return clean_float(real, 17), rounded, integer


def dequantize_integer(integer: int, *, scale: float, zero: int) -> float:
    return clean_float((float(integer) - float(zero)) * float(scale), 17)


def quantize_group(
    layer: Layer,
    cfg: LayerQuantConfig,
    calibration_group: dict[str, Any],
    *,
    batch_point_budget: int,
    calibration_sample_ids: list[str],
) -> dict[str, Any]:
    start = int(calibration_group["start"])
    end = int(calibration_group["end"])
    scale = float(calibration_group["scale"])
    zero = int(calibration_group["zero"])
    qmin = int(calibration_group["qmin"])
    qmax = int(calibration_group["qmax"])
    integers: list[int] = []
    dequantized: list[float] = []
    truncations: list[dict[str, Any]] = []

    for offset, value in enumerate(layer.weights[start:end]):
        pre_real, rounded, integer = affine_raw(
            value,
            scale=scale,
            zero=zero,
            saturate=True,
            qmin=qmin,
            qmax=qmax,
        )
        dequant_pre = dequantize_integer(integer, scale=scale, zero=zero)
        dequant = fixed_point_round(dequant_pre, cfg.fractional_bits)
        global_index = start + offset
        integers.append(integer)
        dequantized.append(dequant)
        if integer != rounded:
            truncations.append(
                {
                    "element_index": offset,
                    "global_index": global_index,
                    "weight": clean_float(value),
                    "direction": "high" if rounded > qmax else "low",
                    "pre_saturation_fixed": pre_real,
                    "pre_saturation_integer": rounded,
                    "saturated_fixed": integer,
                    "dequantized_before_rounding": dequant_pre,
                    "dequantized": dequant,
                }
            )

    chunks = [
        {"start": cursor, "end": min(cursor + batch_point_budget, end)}
        for cursor in range(start, end, batch_point_budget)
    ]
    role = {
        "scheme": cfg.scheme,
        "bit_width": cfg.bit_width,
        "fractional_bits": cfg.fractional_bits,
        "quantization_levels": cfg.quantization_levels,
        "scale_effective_digits": cfg.scale_effective_digits,
        "scale": clean_float(scale, cfg.scale_effective_digits),
        "zero": zero,
        "qmin": qmin,
        "qmax": qmax,
        "retained_fractional_bits": cfg.fractional_bits,
    }
    return {
        "kind": "group_quantization",
        "layer": layer.name,
        "content_version": layer.content_version,
        "content_fingerprint": layer.content_fingerprint,
        "group_index": int(calibration_group["group_index"]),
        "start": start,
        "end": end,
        "num_points": end - start,
        "scale": clean_float(scale, cfg.scale_effective_digits),
        "zero": zero,
        "qmin": qmin,
        "qmax": qmax,
        "fixed_integers": integers,
        "dequantized": dequantized,
        "saturation_count": len(truncations),
        "truncations": truncations,
        "batch_chunks": chunks,
        "calibration_sample_ids": list(calibration_sample_ids),
        "role_fingerprint": stable_hash(
            {
                "layer": layer.content_fingerprint,
                "group_index": int(calibration_group["group_index"]),
                "role": role,
                "calibration": calibration_group.get("role_fingerprint"),
            }
        ),
        "role": role,
    }


def quantize_unsat(
    value: float,
    *,
    scale: float,
    zero: int,
    dequant_fixed_bits: int | None = None,
) -> tuple[int, float]:
    real = float(value) / float(scale) + float(zero)
    integer = round_half_away_from_zero(real)
    dequant = (float(integer) - float(zero)) * float(scale)
    if dequant_fixed_bits is not None:
        dequant = fixed_point_round(dequant, dequant_fixed_bits)
    return integer, clean_float(dequant, 17)


def quantize_pre_saturation(
    value: float,
    *,
    scale: float,
    zero: int,
) -> tuple[float, int, float]:
    real, rounded, _ = affine_raw(
        value,
        scale=scale,
        zero=zero,
        saturate=False,
        qmin=0,
        qmax=0,
    )
    dequant = dequantize_integer(rounded, scale=scale, zero=zero)
    return real, rounded, dequant
