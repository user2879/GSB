"""Terrain shading: piecewise elevation bands + slope override.

Parameters are plain data, serialized with the block manifest, so shading is
recomputable for any historical frame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass(frozen=True)
class ShadingParams:
    # (upper elevation bound in [0,1], (r, g, b)) evaluated in order
    bands: tuple = (
        (0.25, (64, 96, 160)),    # water / lowland
        (0.45, (96, 144, 80)),    # grass
        (0.65, (128, 112, 88)),   # rock
        (0.85, (150, 150, 150)),  # high rock
        (1.01, (240, 240, 245)),  # snow
    )
    slope_threshold: float = 0.55   # 1 - normal.z above this blends cliff color
    slope_color: tuple = (110, 100, 95)
    slope_blend: float = 0.7

    def to_dict(self):
        return {
            "bands": [[b, list(c)] for b, c in self.bands],
            "slope_threshold": self.slope_threshold,
            "slope_color": list(self.slope_color),
            "slope_blend": self.slope_blend,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            bands=tuple((b, tuple(c)) for b, c in d["bands"]),
            slope_threshold=d["slope_threshold"],
            slope_color=tuple(d["slope_color"]),
            slope_blend=d["slope_blend"],
        )


DEFAULT_SHADING = ShadingParams()


def shade(elev_norm: np.ndarray, slope: np.ndarray,
          params: ShadingParams = DEFAULT_SHADING) -> np.ndarray:
    """Piecewise elevation-band color with slope override. uint8 (..., 3)."""
    elev_norm = np.asarray(elev_norm, dtype=np.float64)
    slope = np.asarray(slope, dtype=np.float64)
    out = np.zeros(elev_norm.shape + (3,), dtype=np.float64)
    remaining = np.ones(elev_norm.shape, dtype=bool)
    for bound, color in params.bands:
        mask = remaining & (elev_norm < bound)
        out[mask] = color
        remaining &= ~mask
    out[remaining] = params.bands[-1][1]
    steep = slope > params.slope_threshold
    blend = params.slope_blend
    out[steep] = out[steep] * (1.0 - blend) + np.asarray(params.slope_color) * blend
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
