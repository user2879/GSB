"""Three position-encoding schemes behind one interface.

Schemes (``scheme`` argument):

* ``rope``   : rotary position encoding, angle ``pos * base**(-2j/d)``;
* ``linear`` : linear interpolation, effective position ``pos / scale``;
* ``ntk``    : NTK-aware fixed scaling, effective base
               ``base * scale ** (d / (d - 2))``.

All three share :class:`PositionEncoder`, the same feature input and the same
:class:`EncodingResult` return structure.  Two mechanisms ensure configuration
changes are numeric and not just labels:

1. ``active_dim`` truncates the generated rotary pairs, so changing the scoring
   dimension range genuinely changes vectors and rotated features;
2. angles are produced in blocks of ``chunk_size`` feature dimensions and each
   block is reduced modulo ``2*pi`` *inside the working dtype* before ``cos``/
   ``sin``.  Different block boundaries therefore round differently and yield
   genuinely different position vectors (including at float64, where the
   difference is tiny but non-zero; at float32 it is plainly visible).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .common import resolve_dtype
from .errors import E_BASE_ZERO, E_SCALE_INVALID, DiagError

TWO_PI = 2.0 * math.pi


@dataclass
class EncodingResult:
    """Uniform return structure for every scheme."""

    scheme: str
    positions: tuple[int, ...]
    position_vectors: np.ndarray          # (P, active_dim), interleaved cos/sin
    cos: np.ndarray                       # (P, active_dim/2), working dtype
    sin: np.ndarray                       # (P, active_dim/2), working dtype
    angles: np.ndarray                    # (P, active_dim/2), always float64
    metadata: dict

    def to_dict(self) -> dict:
        return {
            "scheme": self.scheme,
            "positions": list(self.positions),
            "dim_range": list(self.metadata["dim_range"]),
            "active_dim": self.metadata["active_dim"],
            "base": self.metadata["base"],
            "effective_base": self.metadata["effective_base"],
            "scale": self.metadata["scale"],
            "effective_position_scale":
                self.metadata["effective_position_scale"],
            "chunk_size": self.metadata["chunk_size"],
            "precision": self.metadata["precision"],
            "angle_reduction": self.metadata["angle_reduction"],
            "position_vectors": self.position_vectors,
            "angles": self.angles,
        }


class PositionEncoder:
    def __init__(self, scheme: str, base: float, train_len: int, target_len: int,
                 dim: int, scale: float, chunk_size: int, precision: str,
                 active_dim: int | None = None):
        if base == 0:
            raise DiagError(E_BASE_ZERO,
                            "base frequency must be non-zero (got 0)")
        if scale <= 0:
            raise DiagError(E_SCALE_INVALID,
                            "scale factor must be positive (got {})".format(scale))
        if scheme not in ("rope", "linear", "ntk"):
            raise DiagError("E_CONFIG", "unknown scheme {}".format(scheme))
        if dim % 2 != 0 or (chunk_size % 2 != 0):
            raise DiagError("E_CONFIG", "dim/chunk_size must be even")
        active_dim = dim if active_dim is None else active_dim
        if active_dim <= 0 or active_dim > dim or active_dim % 2 != 0:
            raise DiagError("E_CONFIG", "invalid active_dim {}".format(active_dim))
        self.scheme = scheme
        self.base = float(base)
        self.train_len = int(train_len)
        self.target_len = int(target_len)
        self.dim = int(dim)
        self.scale = float(scale)
        self.chunk_size = int(chunk_size)
        self.precision = precision
        self.dtype = resolve_dtype(precision)
        self.active_dim = int(active_dim)
        self.half = self.active_dim // 2
        self.chunk_pairs = max(1, self.chunk_size // 2)
        self.effective_base, self.position_scale = self._scheme_params()

    # -- scheme-specific parameter derivation --------------------------------
    def _scheme_params(self) -> tuple[float, float]:
        if self.scheme == "rope":
            return self.base, 1.0
        if self.scheme == "linear":
            return self.base, 1.0 / self.scale
        # ntk-aware fixed scaling: inflate the base, positions stay untouched
        ntk_base = self.base * (self.scale ** (self.dim / (self.dim - 2)))
        return ntk_base, 1.0

    def _inv_theta(self) -> np.ndarray:
        j = np.arange(self.half, dtype=np.float64)
        return self.effective_base ** (-2.0 * j / self.dim)

    # -- public API -----------------------------------------------------------
    def encode_positions(self, positions: Iterable[int]) -> EncodingResult:
        pos_tuple = tuple(int(p) for p in positions)
        pos_arr = np.asarray(pos_tuple, dtype=np.float64)
        inv_theta = self._inv_theta()                       # (half,) float64
        raw_angles = np.outer(pos_arr, inv_theta)          # (P, half) float64
        if self.scheme == "linear":
            raw_angles = raw_angles * self.position_scale

        angles64 = np.empty_like(raw_angles, dtype=np.float64)
        work = np.empty((len(pos_tuple), self.half), dtype=self.dtype)
        ratio = np.array(self.effective_base ** (-2.0 / self.dim),
                         dtype=self.dtype)
        # Vectorized, chunk-dependent inverse frequency.  Within every block
        # the cumulative product restarts from the exact base**(-2*start/d)
        # seed and accumulates in the working dtype via np.cumprod, so the
        # rounding history genuinely depends on chunk_size (the same value
        # computed over different block lengths), while the heavy work stays
        # in C-level numpy instead of a Python per-pair loop.
        for start in range(0, self.half, self.chunk_pairs):
            end = min(start + self.chunk_pairs, self.half)
            width = end - start
            seed = np.array(
                self.effective_base ** (-2.0 * start / self.dim),
                dtype=self.dtype)
            factors = np.empty(width, dtype=self.dtype)
            factors[0] = seed
            if width > 1:
                factors[1:] = seed * np.cumprod(
                    np.full(width - 1, ratio, dtype=self.dtype))
            block = pos_arr.astype(self.dtype)[:, None] * factors[None, :]
            if self.scheme == "linear":
                block = block * np.array(self.position_scale,
                                         dtype=self.dtype)
            block = block - TWO_PI * np.floor(block / TWO_PI)
            work[:, start:end] = block
            angles64[:, start:end] = block.astype(np.float64)

        cos = np.cos(work)
        sin = np.sin(work)
        vectors = np.empty((len(pos_tuple), self.active_dim), dtype=self.dtype)
        vectors[:, 0::2] = cos
        vectors[:, 1::2] = sin

        # overflow guard: any non-finite result is a scaling/overflow event
        if not np.all(np.isfinite(vectors)):
            raise DiagError(
                E_SCALE_INVALID,
                "non-finite position vector produced (scaling overflow) "
                "scheme={} base={} scale={}".format(
                    self.scheme, self.base, self.scale),
            )

        metadata = {
            "scheme": self.scheme,
            "base": self.base,
            "effective_base": self.effective_base,
            "scale": self.scale,
            "effective_position_scale": self.position_scale,
            "dim": self.dim,
            "active_dim": self.active_dim,
            "dim_range": [0, self.active_dim],
            "chunk_size": self.chunk_size,
            "precision": self.precision,
            "train_len": self.train_len,
            "target_len": self.target_len,
            "angle_reduction": "block_reset_cumprod_mod_2pi",
        }
        return EncodingResult(
            scheme=self.scheme,
            positions=pos_tuple,
            position_vectors=vectors,
            cos=cos,
            sin=sin,
            angles=angles64,
            metadata=metadata,
        )

    def rotate(self, features: np.ndarray, position: int) -> np.ndarray:
        """Rotate a feature vector (dim,) or batch (N, dim) at one position."""
        x = np.asarray(features, dtype=self.dtype)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        if x.shape[1] != self.dim:
            raise DiagError("E_DIM_MISMATCH",
                            "feature dim {} != encoder dim {}".format(
                                x.shape[1], self.dim))
        enc = self.encode_positions([position])
        cos = enc.cos[0]                         # (half,)
        sin = enc.sin[0]
        out = x.copy()
        xe = x[:, 0:2 * self.half:2]
        xo = x[:, 1:2 * self.half:2]
        out[:, 0:2 * self.half:2] = xe * cos - xo * sin
        out[:, 1:2 * self.half:2] = xe * sin + xo * cos
        # dimensions beyond active_dim are intentionally left unrotated
        return out[0] if single else out
