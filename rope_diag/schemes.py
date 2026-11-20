"""Three RoPE position-encoding schemes behind one unified operator interface.

Schemes
-------
- ``standard``: angles from raw position index, theta_i = base^(-2i/d).
- ``linear``  : linear interpolation, position index compressed by 1/scale.
- ``ntk``     : NTK-aware scaling, base' = base * scale^(d/(d-2)).

All schemes share the same query/key feature input interface and return the
same :class:`SchemeOutput` structure, including the scheme name and a
parameter fingerprint. Outputs are element-wise reproducible for a fixed
configuration and input.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fingerprints import scheme_param_fingerprint

SCHEMES = ("standard", "linear", "ntk")

# Angle increments below this magnitude at the extrapolation target are
# treated as a dimension-level truncation (the rotation is indistinguishable
# from identity at that dimension pair).
TRUNC_EPS = 1e-9


class SchemeFault(Exception):
    """Raised when scheme parameters are physically invalid."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SchemeParams:
    scheme: str
    base: float
    scale: float
    dim: int
    train_len: int
    target_len: int

    def as_dict(self) -> dict:
        return {
            "scheme": self.scheme,
            "base": self.base,
            "scale": self.scale,
            "dim": self.dim,
            "train_len": self.train_len,
            "target_len": self.target_len,
        }

    @property
    def fingerprint(self) -> str:
        return scheme_param_fingerprint(self.as_dict())


@dataclass
class SchemeOutput:
    scheme: str
    param_fingerprint: str
    rotated_query: np.ndarray
    rotated_keys: np.ndarray
    query_embedding: np.ndarray
    key_embeddings: np.ndarray
    boundary_flags: dict = field(default_factory=dict)


def effective_base(params: SchemeParams) -> float:
    if params.scheme == "ntk":
        exponent = params.dim / (params.dim - 2.0)
        return params.base * params.scale ** exponent
    return params.base


def effective_positions(positions: np.ndarray, params: SchemeParams) -> np.ndarray:
    if params.scheme == "linear":
        return positions / params.scale
    return positions


def inv_freq(params: SchemeParams) -> np.ndarray:
    """Per-pair inverse frequencies theta_i = base_eff^(-2i/dim)."""
    if params.scale <= 0:
        raise SchemeFault(
            "E_SCALE_NONPOSITIVE", f"scale must be positive, got {params.scale!r}"
        )
    base = effective_base(params)
    if base <= 0:
        raise SchemeFault("E_BASE_ZERO", f"base must be positive, got {base!r}")
    half = params.dim // 2
    idx = np.arange(half, dtype=np.float64)
    return np.power(base, -2.0 * idx / params.dim)


def position_angles(positions: np.ndarray, params: SchemeParams) -> np.ndarray:
    """Angles matrix of shape (len(positions), dim//2)."""
    pos = effective_positions(np.asarray(positions, dtype=np.float64), params)
    return pos[:, None] * inv_freq(params)[None, :]


def position_embedding(positions: np.ndarray, params: SchemeParams) -> np.ndarray:
    """Interleaved [cos, sin] embedding of shape (len(positions), dim)."""
    angles = position_angles(positions, params)
    emb = np.empty((angles.shape[0], params.dim), dtype=np.float64)
    emb[:, 0::2] = np.cos(angles)
    emb[:, 1::2] = np.sin(angles)
    return emb


def _rotate(vec: np.ndarray, angles: np.ndarray) -> np.ndarray:
    pairs = vec.reshape(vec.shape[0], vec.shape[1] // 2, 2)
    cos = np.cos(angles)
    sin = np.sin(angles)
    x = pairs[..., 0]
    y = pairs[..., 1]
    out = np.empty_like(pairs)
    out[..., 0] = x * cos - y * sin
    out[..., 1] = x * sin + y * cos
    return out.reshape(vec.shape)


def rotate_features(features: np.ndarray, positions: np.ndarray, params: SchemeParams) -> np.ndarray:
    """Rotate feature rows (n, dim) at the given positions."""
    angles = position_angles(positions, params)
    return _rotate(np.asarray(features, dtype=np.float64), angles)


def boundary_scan(params: SchemeParams) -> dict:
    """Detect dimension-level truncation / scaling overflow at target length."""
    flags = {"truncated_dims": [], "overflow": False, "unusable": False, "reasons": []}
    try:
        freq = inv_freq(params)
    except SchemeFault as exc:
        flags["unusable"] = True
        flags["reasons"].append(exc.code)
        return flags
    pos = effective_positions(np.array([params.target_len], dtype=np.float64), params)
    increments = pos[0] * freq
    if not np.all(np.isfinite(increments)):
        flags["overflow"] = True
        flags["unusable"] = True
        flags["reasons"].append("E_SCALE_OVERFLOW")
        return flags
    truncated = [int(i) for i in np.nonzero(increments < TRUNC_EPS)[0]]
    if truncated:
        flags["truncated_dims"] = truncated
        flags["unusable"] = True
        flags["reasons"].append("E_DIM_TRUNCATION")
    return flags


def apply_scheme(
    params: SchemeParams,
    query: np.ndarray,
    keys: np.ndarray,
    query_pos: int,
    key_positions: np.ndarray,
) -> SchemeOutput:
    """Unified operator: rotate query/key features and return embeddings.

    Raises SchemeFault for non-positive base/scale so the pipeline can map
    the failure to a business code.
    """
    if params.scheme not in SCHEMES:
        raise SchemeFault("E_UNKNOWN_SCHEME", f"unknown scheme {params.scheme!r}")
    query = np.asarray(query, dtype=np.float64).reshape(1, -1)
    keys = np.asarray(keys, dtype=np.float64)
    key_positions = np.asarray(key_positions, dtype=np.float64)
    rotated_query = rotate_features(query, np.array([query_pos], dtype=np.float64), params)
    rotated_keys = rotate_features(keys, key_positions, params)
    return SchemeOutput(
        scheme=params.scheme,
        param_fingerprint=params.fingerprint,
        rotated_query=rotated_query[0],
        rotated_keys=rotated_keys,
        query_embedding=position_embedding(
            np.array([query_pos], dtype=np.float64), params
        )[0],
        key_embeddings=position_embedding(key_positions, params),
        boundary_flags=boundary_scan(params),
    )
