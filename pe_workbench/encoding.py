"""Three positional-encoding schemes with one common interface."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .config import SCHEME_DISPLAY, WorkbenchConfig
from .hashing import stable_float, stable_vector

PrecisionLevel = Literal["fp16", "fp32", "fp64"]
SchemeName = Literal["rope", "linear", "ntk"]

PRECISION_DTYPES: dict[str, Any] = {
    "fp16": np.float16,
    "fp32": np.float32,
    "fp64": np.float64,
}


@dataclass(frozen=True)
class EncodedVector:
    position: int
    effective_position: float
    vector: tuple[float, ...]
    angles: tuple[float, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "effective_position": stable_float(self.effective_position),
            "vector": stable_vector(self.vector),
            "angles": stable_vector(self.angles),
            "metadata": self.metadata,
        }


def scheme_fingerprint(
    scheme: str,
    base_frequency: float,
    dimension: int,
    scale_factor: float,
    precision_level: str,
) -> dict[str, Any]:
    """Only include parameters that can actually affect the chosen scheme."""
    fingerprint: dict[str, Any] = {
        "formula": "pe-formula-v1",
        "scheme": scheme,
        "base_frequency": float(base_frequency),
        "dimension": dimension,
        "precision_level": precision_level,
    }
    if scheme == "linear":
        fingerprint["scale_factor"] = float(scale_factor)
    elif scheme == "ntk":
        fingerprint["scale_factor"] = float(scale_factor)
    return fingerprint


def effective_parameters(
    scheme: str,
    position: int,
    base_frequency: float,
    dimension: int,
    scale_factor: float,
) -> tuple[float, float, float]:
    """Return (effective position, effective base, NTK exponent)."""
    effective_position = float(position)
    effective_base = float(base_frequency)
    exponent = 1.0
    if scheme == "linear":
        effective_position = float(position) / float(scale_factor)
    elif scheme == "ntk":
        if dimension > 2:
            exponent = dimension / (dimension - 2)
        else:
            exponent = 1.0
        effective_base = float(base_frequency) * (float(scale_factor) ** exponent)
    return effective_position, effective_base, exponent


def encode_vector(
    features: tuple[float, ...] | list[float],
    position: int,
    *,
    scheme: str,
    base_frequency: float,
    dimension: int,
    scale_factor: float,
    precision_level: str = "fp64",
) -> EncodedVector:
    """Encode one query/key vector using the selected scheme.

    RoPE rotates adjacent coordinate pairs. Linear interpolation divides the
    position index by ``scale_factor``. NTK-aware scaling changes the frequency
    base by ``scale_factor ** (dim / (dim - 2))`` while leaving the index intact.
    All three paths share the same input/output interface.
    """
    if length := len(features):
        if length != dimension:
            raise ValueError(f"feature length {length} != dimension {dimension}")
    else:
        raise ValueError("feature vector must not be empty")
    if dimension % 2:
        raise ValueError("dimension must be even")
    if precision_level not in PRECISION_DTYPES:
        raise ValueError(f"unsupported precision level {precision_level}")
    if scheme not in SCHEME_DISPLAY:
        raise ValueError(f"unsupported scheme {scheme}")
    if base_frequency <= 0 or not math.isfinite(base_frequency):
        raise ValueError("base_frequency must be positive and finite")
    if scale_factor <= 0 or not math.isfinite(scale_factor):
        raise ValueError("scale_factor must be positive and finite")

    dtype = PRECISION_DTYPES[precision_level]
    effective_position, effective_base, exponent = effective_parameters(
        scheme, position, base_frequency, dimension, scale_factor
    )
    effective_position_dt = dtype(effective_position)
    base_dt = dtype(effective_base)
    inv_freq = [
        dtype(base_dt ** dtype(-2.0 * pair_index / dimension))
        for pair_index in range(dimension // 2)
    ]
    angles = [dtype(effective_position_dt * inv_freq[pair_index]) for pair_index in range(dimension // 2)]

    output: list[float] = [0.0] * dimension
    for pair_index, angle in enumerate(angles):
        cosine = dtype(np.cos(angle))
        sine = dtype(np.sin(angle))
        even_index = 2 * pair_index
        odd_index = even_index + 1
        even_value = dtype(features[even_index])
        odd_value = dtype(features[odd_index])
        output[even_index] = float(dtype(even_value * cosine - odd_value * sine))
        output[odd_index] = float(dtype(even_value * sine + odd_value * cosine))

    metadata = {
        "formula_version": "pe-formula-v1",
        "scheme": scheme,
        "scheme_display": SCHEME_DISPLAY[scheme],
        "base_frequency": stable_float(base_frequency),
        "effective_base_frequency": stable_float(effective_base),
        "scale_factor": stable_float(scale_factor),
        "precision_level": precision_level,
        "dimension": dimension,
        "position": position,
        "effective_position": stable_float(effective_position),
        "ntk_exponent": stable_float(exponent),
    }
    return EncodedVector(
        position=position,
        effective_position=effective_position,
        vector=tuple(stable_vector(output)),
        angles=tuple(stable_float(float(angle)) for angle in angles),
        metadata=metadata,
    )


def vector_l2_norm(vector: tuple[float, ...] | list[float]) -> float:
    return stable_float(math.sqrt(math.fsum(float(value) * float(value) for value in vector)))


def position_summary(encoded: EncodedVector, prefix: int = 4) -> dict[str, Any]:
    return {
        "position": encoded.position,
        "effective_position": stable_float(encoded.effective_position),
        "l2_norm": vector_l2_norm(encoded.vector),
        "prefix": stable_vector(encoded.vector[:prefix]),
        "angle_prefix": stable_vector(encoded.angles[:prefix]),
    }


def encode_pair(
    sample_id: str,
    query_position: int,
    key_positions: tuple[int, ...],
    query_vector: tuple[float, ...],
    key_vectors: tuple[tuple[float, ...], ...],
    config: WorkbenchConfig,
    *,
    precision_level: str | None = None,
    scheme: str | None = None,
) -> dict[str, Any]:
    precision = precision_level or config.precision_level
    active_scheme = scheme or config.scheme
    query = encode_vector(
        query_vector,
        query_position,
        scheme=active_scheme,
        base_frequency=config.base_frequency,
        dimension=config.dimension,
        scale_factor=config.scale_factor,
        precision_level=precision,
    )
    keys = [
        encode_vector(
            key_vector,
            key_position,
            scheme=active_scheme,
            base_frequency=config.base_frequency,
            dimension=config.dimension,
            scale_factor=config.scale_factor,
            precision_level=precision,
        )
        for key_position, key_vector in zip(key_positions, key_vectors)
    ]
    return {
        "sample_id": sample_id,
        "scheme": active_scheme,
        "scheme_display": SCHEME_DISPLAY[active_scheme],
        "precision_level": precision,
        "metadata": {
            "scheme": active_scheme,
            "scheme_display": SCHEME_DISPLAY[active_scheme],
            "base_frequency": stable_float(config.base_frequency),
            "scale_factor": stable_float(config.scale_factor),
            "precision_level": precision,
            "dimension": config.dimension,
            "formula_version": "pe-formula-v1",
            "effective_base_frequency": query.metadata["effective_base_frequency"],
        },
        "query": query.to_dict(),
        "keys": [key.to_dict() for key in keys],
        "summary": {
            "query": position_summary(query),
            "keys": [position_summary(key) for key in keys],
        },
    }
