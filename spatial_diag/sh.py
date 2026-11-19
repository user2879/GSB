"""Real spherical-harmonic basis and deterministic weighted fitting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.special import gammaln, lpmv


def component_index(order_l: int, order_m: int) -> int:
    """Ambisonic Channel Number (ACN): ``l*l + l + m``."""

    return int(order_l) * int(order_l) + int(order_l) + int(order_m)


def component_descriptor(index: int) -> dict[str, object]:
    q = int(index)
    order_l = int(math.isqrt(q))
    order_m = q - order_l * order_l - order_l
    return {
        "index": q,
        "l": order_l,
        "m": order_m,
        "name": f"Y_{order_l}^{order_m:+d}",
    }


def sh_component_count(max_order: int) -> int:
    return (int(max_order) + 1) ** 2


def _normalization(order_l: int, order_m: int) -> float:
    log_ratio = gammaln(order_l - order_m + 1) - gammaln(order_l + order_m + 1)
    return math.sqrt((2.0 * order_l + 1.0) / (4.0 * math.pi) * math.exp(log_ratio))


def real_sh(directions: np.ndarray, max_order: int) -> np.ndarray:
    """Orthonormal real SH basis with ``integral(Y_q*Y_r dOmega)=delta_qr``.

    ``directions`` is ``(n, 3)``.  Columns use ACN ordering and the
    Condon-Shortley associated Legendre convention.
    """

    directions = np.asarray(directions, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("directions must have shape (n, 3)")
    max_order = int(max_order)
    if max_order < 0:
        raise ValueError("max_order must be non-negative")
    x = np.clip(directions[:, 2], -1.0, 1.0)
    phi = np.arctan2(directions[:, 1], directions[:, 0])
    basis = np.empty((len(directions), sh_component_count(max_order)), dtype=np.float64)
    for order_l in range(max_order + 1):
        for order_m in range(-order_l, order_l + 1):
            abs_m = abs(order_m)
            legendre = lpmv(abs_m, order_l, x)
            scale = _normalization(order_l, abs_m)
            if order_m == 0:
                values = scale * legendre
            elif order_m > 0:
                values = math.sqrt(2.0) * scale * legendre * np.cos(abs_m * phi)
            else:
                values = math.sqrt(2.0) * scale * legendre * np.sin(abs_m * phi)
            basis[:, component_index(order_l, order_m)] = values
    return basis


@dataclass(frozen=True)
class SHFit:
    coefficients: np.ndarray
    max_order: int
    rank: int
    residual_rms: float
    ridge: float


def fit_weighted_sh(
    directions: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    max_order: int,
    ridge: float = 1.0e-8,
) -> SHFit:
    """Weighted least-squares real SH fit.

    This is an explicit least-squares fit on the supplied rows.  Rows marked
    low confidence must be removed by the caller; this function does not
    inspect confidence metadata.
    """

    directions = np.asarray(directions, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("directions must have shape (n, 3)")
    if values.shape != (len(directions),) or weights.shape != (len(directions),):
        raise ValueError("values and weights must match directions")
    basis = real_sh(directions, max_order)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    basis = basis[keep]
    values = values[keep]
    weights = weights[keep]
    if len(values) == 0:
        raise ValueError("no rows available for SH fit")
    root_weight = np.sqrt(weights)
    design = basis * root_weight[:, None]
    target = values * root_weight
    if ridge > 0.0:
        design = np.vstack((design, math.sqrt(float(ridge)) * np.eye(design.shape[1])))
        target = np.concatenate((target, np.zeros(design.shape[1], dtype=np.float64)))
    coefficients, residuals, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    prediction = basis @ coefficients
    if len(values):
        residual = prediction - values
        denominator = float(np.sum(weights))
        rms = math.sqrt(float(np.sum(weights * residual * residual)) / denominator) if denominator > 0 else 0.0
    else:
        rms = 0.0
    return SHFit(
        coefficients=np.asarray(coefficients, dtype=np.float64),
        max_order=int(max_order),
        rank=int(rank),
        residual_rms=float(rms),
        ridge=float(ridge),
    )


def synth_energy_field(coefficients: np.ndarray, directions: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.size == 0:
        return np.zeros(len(directions), dtype=np.float64)
    max_order = math.isqrt(coefficients.size) - 1
    expected = sh_component_count(max_order)
    if expected != coefficients.size:
        raise ValueError(f"coefficient length {coefficients.size} is not an SH pyramid")
    return real_sh(directions, max_order) @ coefficients


def shared_component_count(left_order: int, right_order: int) -> int:
    return sh_component_count(min(int(left_order), int(right_order)))


def iter_component_indices(max_order: int) -> Iterable[int]:
    return range(sh_component_count(max_order))

