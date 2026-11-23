"""Window feature extraction in pure deterministic fixed-point integer math.

For every window we take the band window *centred* on its centre region (edge
windows use replication padding), compute the four features once and record
the quantization residual (micro-units) and rounding direction produced by the
configured fractional bit width / quantization gear.  No float is used on the
hot path; a feature describes the whole centre region owned by the window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .fixedpoint import (
    FixedConfig,
    from_int,
    isqrt,
    q_div,
)


@dataclass
class FeatureRecord:
    ndvi: int
    ndwi: int
    texture: int
    slope: int
    residuals: Dict[str, int]
    rounding: Dict[str, int]


def _padded_band(arr: np.ndarray, half: int) -> np.ndarray:
    return np.pad(arr.astype(np.int64), half, mode="edge")


def _window_values(padded: np.ndarray, r: int, c: int, size: int) -> np.ndarray:
    return padded[r : r + size, c : c + size]


def _mean_int(values: np.ndarray) -> int:
    return int(values.sum()) // int(values.size)


def extract_window_features(
    bands: Dict[str, np.ndarray],
    padded: Dict[str, np.ndarray],
    r: int,
    c: int,
    size: int,
    cfg: FixedConfig,
) -> FeatureRecord:
    red = _window_values(padded["R"], r, c, size)
    green = _window_values(padded["G"], r, c, size)
    blue = _window_values(padded["B"], r, c, size)
    nir_vals = _window_values(padded["NIR"], r, c, size)
    dem = _window_values(padded["DEM"], r, c, size)

    r_mean = _mean_int(red)
    g_mean = _mean_int(green)
    b_mean = _mean_int(blue)
    nir_mean = _mean_int(nir_vals)
    _ = b_mean

    residuals: Dict[str, int] = {}
    rounding: Dict[str, int] = {}

    # --- NDVI = (NIR - R) / (NIR + R), fixed order: subtract, add, divide ---
    ndvi_num = nir_mean - r_mean
    ndvi_den = nir_mean + r_mean
    ndvi, ndvi_rem, ndvi_dir = q_div(ndvi_num, ndvi_den, cfg)
    residuals["ndvi"] = cfg.residual_micro(ndvi_rem)
    rounding["ndvi"] = ndvi_dir

    # --- NDWI = (G - NIR) / (G + NIR) ---
    ndwi_num = g_mean - nir_mean
    ndwi_den = g_mean + nir_mean
    ndwi, ndwi_rem, ndwi_dir = q_div(ndwi_num, ndwi_den, cfg)
    residuals["ndwi"] = cfg.residual_micro(ndwi_rem)
    rounding["ndwi"] = ndwi_dir

    # --- texture gradient: sqrt(mean(|dx|)+mean(|dy|)) over the R band ---
    dx = np.abs(np.diff(red, axis=1)).mean()
    dy = np.abs(np.diff(red, axis=0)).mean()
    grad_raw = int(dx + dy)
    grad_sqrt = isqrt(grad_raw)
    texture, tex_rem, tex_dir = from_int(grad_sqrt, cfg)
    residuals["texture"] = cfg.residual_micro(tex_rem)
    rounding["texture"] = tex_dir

    # --- elevation slope: sqrt(dem_dx^2 + dem_dy^2), integer differences ---
    dem_dx = int(np.abs(np.diff(dem, axis=1)).mean())
    dem_dy = int(np.abs(np.diff(dem, axis=0)).mean())
    slope_raw = isqrt(dem_dx * dem_dx + dem_dy * dem_dy)
    slope, sl_rem, sl_dir = from_int(slope_raw, cfg)
    residuals["slope"] = cfg.residual_micro(sl_rem)
    rounding["slope"] = sl_dir

    return FeatureRecord(
        ndvi=ndvi,
        ndwi=ndwi,
        texture=texture,
        slope=slope,
        residuals=residuals,
        rounding=rounding,
    )


def pad_bands(bands: Dict[str, np.ndarray], half: int) -> Dict[str, np.ndarray]:
    return {name: _padded_band(arr, half) for name, arr in bands.items()}
