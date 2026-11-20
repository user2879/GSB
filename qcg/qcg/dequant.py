"""Dequantization reduction with deterministic rounding semantics.

Pipeline: real = integer_accum * (weight_scale * activation_scale);
the real value is converted to a fixed-point number with ``cfg.frac_bits``
fractional bits, rounded according to the node's declared mode, clipped to
the configured output interval, then converted back to float. Values that
land exactly on a rounding midpoint are flagged as boundary samples.
"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .config import Config
from .models import Node

DEQUANT_IMPL_VERSION = "dequant-1.0.0"
MID_TOL = 1e-12


@dataclass
class DequantResult:
    values: np.ndarray
    boundary_positions: List[List[int]]
    scale: float
    frac_bits: int
    rounding: str


def _round_fixed_point(x: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """x is the fixed-point-unscaled real value * 2**frac_bits."""
    floor = np.floor(x)
    diff = x - floor
    is_mid = np.abs(diff - 0.5) <= MID_TOL
    if mode == "ROUND_HALF_UP":
        out = np.where(diff >= 0.5, np.ceil(x), floor)
    elif mode == "ROUND_HALF_DOWN":
        out = np.where(diff > 0.5, np.ceil(x), floor)
    elif mode == "ROUND_HALF_EVEN":
        up = np.ceil(x)
        out = np.where(is_mid,
                       np.where(np.mod(floor, 2) == 0, floor, up),
                       np.where(diff > 0.5, up, floor))
    elif mode == "ROUND_TRUNCATE":
        out = np.trunc(x)
    elif mode == "ROUND_FLOOR":
        out = floor
    else:
        raise ValueError("unknown rounding mode %r" % mode)
    return out, is_mid


def dequantize(acc: np.ndarray, node: Node, activation_scale: float,
               cfg: Config) -> DequantResult:
    weight_scale = node.scale_float if node.scale_float > 0 \
        else node.scale_permille / 1000.0
    scale = float(weight_scale) * float(activation_scale)
    real = acc.astype(np.float64) * scale
    fixed = real * (2.0 ** cfg.frac_bits)
    rounded, is_mid = _round_fixed_point(fixed, node.rounding)
    clipped = np.clip(rounded,
                      cfg.output_min * (2.0 ** cfg.frac_bits),
                      cfg.output_max * (2.0 ** cfg.frac_bits))
    values = clipped / (2.0 ** cfg.frac_bits)
    boundary = [list(map(int, p)) for p in np.argwhere(is_mid)]
    return DequantResult(values=values, boundary_positions=boundary,
                         scale=scale, frac_bits=cfg.frac_bits,
                         rounding=node.rounding)
