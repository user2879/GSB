"""Fixed-point re-quantization, saturation and error attribution."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from quantcalib.calib.quantiles import round_to_integer

# Fixed priority used when component shares tie.
_SOURCE_PRIORITY = ["saturation", "quantile_discretization",
                    "scale_quantization", "dequant_rounding"]


@dataclass
class GroupQuantization:
    layer: str
    group_index: int
    integer_codes: list[int]
    dequantized: list[float]
    clipped: list[dict[str, Any]]
    abs_errors: list[float]
    rel_errors: list[float]
    over_tolerance: list[bool]
    error_sources: list[str]
    source_evidence: list[dict[str, float]]
    saturation_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def quantize_dequantize(values: np.ndarray, scale: float, zero_point: int,
                        bits: int, rounding_mode: str,
                        saturation_policy: str) -> dict[str, np.ndarray]:
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))

    projected = values / scale + zero_point
    pre_clip = round_to_integer(projected, rounding_mode).astype(np.int64)

    over = pre_clip > qmax
    under = pre_clip < qmin
    if saturation_policy == "saturate":
        codes = np.clip(pre_clip, qmin, qmax)
    else:  # wrap: two's-complement style wrap-around
        span = qmax - qmin + 1
        codes = ((pre_clip - qmin) % span) + qmin

    dequant = (codes.astype(np.float64) - zero_point) * scale
    return {
        "pre_clip": pre_clip,
        "codes": codes,
        "dequant": dequant,
        "over": over,
        "under": under,
    }


def _attribute(original: float, dequant: float, pre_clip: int, code: int,
               scale: float, zero_point: int, q_lo: float, q_hi: float,
               raw_step: float) -> tuple[str, dict[str, float]]:
    if code != pre_clip:
        return "saturation", {
            "saturation": abs(dequant - original),
            "quantile_discretization": 0.0,
            "scale_quantization": 0.0,
            "dequant_rounding": 0.0,
        }

    # Component decomposition of |original - dequant|.
    clipped_grid = np.floor(original / scale + zero_point)
    dequant_grid_value = (clipped_grid - zero_point) * scale
    scale_share = abs(scale - raw_step)
    quantile_share = max(0.0, abs(original) - max(abs(q_lo), abs(q_hi)))
    scale_quant_share = abs(original - dequant_grid_value)
    dequant_round_share = abs(dequant - dequant_grid_value)

    shares = {
        "saturation": 0.0,
        "quantile_discretization": float(quantile_share),
        "scale_quantization": float(max(scale_quant_share, scale_share)),
        "dequant_rounding": float(dequant_round_share),
    }
    best_value = -1.0
    best_source = _SOURCE_PRIORITY[-1]
    for source in _SOURCE_PRIORITY:  # priority order; first max wins on ties
        if shares[source] > best_value:
            best_value = shares[source]
            best_source = source
    return best_source, shares


def quantize_group(layer: str, group_index: int, values: np.ndarray,
                   calibration, bits: int, tolerance: float,
                   rounding_mode: str, saturation_policy: str) -> GroupQuantization:
    result = quantize_dequantize(
        values, calibration.scale, calibration.zero_point, bits,
        rounding_mode, saturation_policy,
    )
    pre_clip = result["pre_clip"]
    codes = result["codes"]
    dequant = result["dequant"]

    abs_err = np.abs(dequant - values)
    denom = np.maximum(np.abs(values), 1e-12)
    rel_err = abs_err / denom
    over_tol = abs_err > tolerance

    clipped: list[dict[str, Any]] = []
    sources: list[str] = []
    evidence: list[dict[str, float]] = []
    for idx in range(values.size):
        if bool(result["over"][idx]) or bool(result["under"][idx]):
            clipped.append({
                "index": int(idx),
                "direction": "up" if result["over"][idx] else "down",
                "pre_clip_value": int(pre_clip[idx]),
                "clipped_value": int(codes[idx]),
            })
        source, shares = _attribute(
            float(values[idx]), float(dequant[idx]), int(pre_clip[idx]),
            int(codes[idx]), calibration.scale, calibration.zero_point,
            calibration.q_lo, calibration.q_hi, calibration.raw_step,
        )
        sources.append(source)
        evidence.append(shares)

    return GroupQuantization(
        layer=layer,
        group_index=group_index,
        integer_codes=[int(x) for x in codes.tolist()],
        dequantized=[float(x) for x in dequant.tolist()],
        clipped=clipped,
        abs_errors=[float(x) for x in abs_err.tolist()],
        rel_errors=[float(x) for x in rel_err.tolist()],
        over_tolerance=[bool(x) for x in over_tol.tolist()],
        error_sources=sources,
        source_evidence=evidence,
        saturation_count=len(clipped),
    )
