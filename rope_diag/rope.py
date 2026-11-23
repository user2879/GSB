from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .config import DiagnosticConfig, SCHEMES
from .errors import DiagnosticError, EncodingBoundaryError, ErrorCode


@dataclass(frozen=True)
class RotationPair:
    pair_index: int
    dim_a: int
    dim_b: int
    theta: float
    cos: float
    sin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_index": self.pair_index,
            "dim_a": self.dim_a,
            "dim_b": self.dim_b,
            "theta": self.theta,
            "cos": self.cos,
            "sin": self.sin,
        }


@dataclass(frozen=True)
class PositionEncoding:
    scheme: str
    position: int
    dimension: int
    vector: np.ndarray
    pairs: tuple[RotationPair, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "position": self.position,
            "dimension": self.dimension,
            "vector": [float(value) for value in self.vector.tolist()],
            "pairs": [pair.to_dict() for pair in self.pairs],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PositionEncoding":
        pairs = tuple(
            RotationPair(
                pair_index=int(item["pair_index"]),
                dim_a=int(item["dim_a"]),
                dim_b=int(item["dim_b"]),
                theta=float(item["theta"]),
                cos=float(item["cos"]),
                sin=float(item["sin"]),
            )
            for item in payload["pairs"]
        )
        return cls(
            scheme=str(payload["scheme"]),
            position=int(payload["position"]),
            dimension=int(payload["dimension"]),
            vector=np.asarray(payload["vector"], dtype=np.float64),
            pairs=pairs,
            metadata=dict(payload["metadata"]),
        )


def validate_scheme(scheme: str) -> None:
    if scheme not in SCHEMES:
        raise DiagnosticError(
            ErrorCode.CONFIG_INVALID,
            f"未知位置编码方案: {scheme}",
            stage="position",
            details={"allowed": list(SCHEMES)},
        )


def ntk_adjusted_base(base: float, scaling_factor: float, dimension: int) -> float:
    if dimension <= 2:
        exponent = 1.0
    else:
        exponent = dimension / (dimension - 2)
    try:
        return float(base * math.pow(scaling_factor, exponent))
    except OverflowError:
        return math.inf


def _boundary(
    condition: str,
    *,
    scheme: str,
    base: float,
    scaling: float,
    details: dict[str, Any] | None = None,
) -> EncodingBoundaryError:
    payload = {
        "condition": condition,
        "scheme": scheme,
        "base": base,
        "scaling_factor": scaling,
    }
    payload.update(details or {})
    return EncodingBoundaryError(
        ErrorCode.CONFIG_INVALID,
        f"位置编码边界触发: {condition}",
        stage="position",
        details=payload,
    )


def generate_position_encoding(
    scheme: str,
    position: int,
    config: DiagnosticConfig,
    *,
    base_override: float | None = None,
    scaling_override: float | None = None,
) -> PositionEncoding:
    """Generate one deterministic RoPE encoding.

    All trigonometric and frequency calculations are performed in float64.
    The three schemes share the same output shape but differ in their actual
    angle computation:

    * ``rope``: ``theta_i = p * base^(-2i/d)``
    * ``linear_interpolation``: ``theta_i = (p / scale) * base^(-2i/d)``
    * ``ntk_aware``: ``theta_i = p * (base * scale^(d/(d-2)))^(-2i/d)``
    """

    validate_scheme(scheme)
    if not isinstance(position, int) or isinstance(position, bool) or position < 0:
        raise DiagnosticError(
            ErrorCode.SAMPLE_INVALID,
            "位置索引必须为非负整数",
            stage="position",
            details={"position": position},
        )
    base = float(config.base if base_override is None else base_override)
    scaling = float(config.scaling_factor if scaling_override is None else scaling_override)
    if not math.isfinite(base) or base <= 0.0:
        raise _boundary("base_non_positive_or_non_finite", scheme=scheme, base=base, scaling=scaling)
    if not math.isfinite(scaling) or scaling <= 0.0:
        raise _boundary(
            "scaling_non_positive_or_non_finite", scheme=scheme, base=base, scaling=scaling
        )

    dimension = config.dimension
    pair_count = dimension // 2
    effective_base = base
    effective_position = float(position)
    try:
        if scheme == "linear_interpolation":
            effective_position = float(position) / scaling
        elif scheme == "ntk_aware":
            with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
                effective_base = ntk_adjusted_base(base, scaling, dimension)
            if not math.isfinite(effective_base) or effective_base <= 0.0:
                raise _boundary(
                    "scaling_overflow:adjusted_base",
                    scheme=scheme,
                    base=base,
                    scaling=scaling,
                    details={"adjusted_base": str(effective_base)},
                )

        exponents = np.arange(pair_count, dtype=np.float64) * (2.0 / float(dimension))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            inverse_frequencies = np.power(effective_base, -exponents).astype(np.float64)
            theta = effective_position * inverse_frequencies
            cosine = np.cos(theta).astype(np.float64)
            sine = np.sin(theta).astype(np.float64)
    except EncodingBoundaryError:
        raise
    except (OverflowError, FloatingPointError, ValueError) as exc:
        raise _boundary(
            "angle_overflow",
            scheme=scheme,
            base=base,
            scaling=scaling,
            details={"error": str(exc)},
        ) from exc

    truncated_pairs = [
        pair_index
        for pair_index, frequency in enumerate(inverse_frequencies.tolist())
        if not math.isfinite(frequency) or frequency < np.finfo(np.float64).tiny
    ]
    if truncated_pairs:
        raise _boundary(
            "dimension_truncation:inverse_frequency_underflow",
            scheme=scheme,
            base=base,
            scaling=scaling,
            details={
                "pair_indices": truncated_pairs,
                "dimensions": [value for pair in truncated_pairs for value in (pair * 2, pair * 2 + 1)],
            },
        )
    non_finite_angles = [
        pair_index
        for pair_index, angle in enumerate(theta.tolist())
        if not math.isfinite(angle)
    ]
    if non_finite_angles:
        raise _boundary(
            "angle_overflow",
            scheme=scheme,
            base=base,
            scaling=scaling,
            details={"pair_indices": non_finite_angles},
        )

    vector = np.empty(dimension, dtype=np.float64)
    pairs: list[RotationPair] = []
    for pair_index in range(pair_count):
        dim_a = pair_index * 2
        dim_b = dim_a + 1
        cos_value = float(cosine[pair_index])
        sin_value = float(sine[pair_index])
        vector[dim_a] = cos_value
        vector[dim_b] = sin_value
        pairs.append(
            RotationPair(
                pair_index=pair_index,
                dim_a=dim_a,
                dim_b=dim_b,
                theta=float(theta[pair_index]),
                cos=cos_value,
                sin=sin_value,
            )
        )

    metadata = {
        "scheme": scheme,
        "base": base,
        "scaling_factor": scaling,
        "effective_base": effective_base,
        "effective_position": effective_position,
        "dimension": dimension,
        "precision": config.precision,
        "quantization": config.quantization,
        "training_context_length": config.training_context_length,
        "extrapolation_target_length": config.extrapolation_target_length,
        "angle_rule": (
            "position/base_frequency"
            if scheme == "rope"
            else "position_scaled/base_frequency"
            if scheme == "linear_interpolation"
            else "position/ntk_adjusted_base_frequency"
        ),
    }
    return PositionEncoding(
        scheme=scheme,
        position=position,
        dimension=dimension,
        vector=vector,
        pairs=tuple(pairs),
        metadata=metadata,
    )


def apply_rotation(features: np.ndarray, encoding: PositionEncoding) -> np.ndarray:
    features_array = np.asarray(features, dtype=np.float64)
    if features_array.ndim != 1 or features_array.shape[0] != encoding.dimension:
        raise DiagnosticError(
            ErrorCode.SAMPLE_INVALID,
            "特征向量维度与位置编码维度不一致",
            stage="position",
            details={
                "feature_shape": list(features_array.shape),
                "encoding_dimension": encoding.dimension,
            },
        )
    rotated = features_array.copy()
    for pair in encoding.pairs:
        value_a = features_array[pair.dim_a]
        value_b = features_array[pair.dim_b]
        rotated[pair.dim_a] = value_a * pair.cos - value_b * pair.sin
        rotated[pair.dim_b] = value_a * pair.sin + value_b * pair.cos
    return rotated.astype(np.float64, copy=False)


def apply_rotation_batch(features: np.ndarray, encodings: tuple[PositionEncoding, ...]) -> np.ndarray:
    features_array = np.asarray(features, dtype=np.float64)
    if features_array.ndim != 2 or features_array.shape[0] != len(encodings):
        raise DiagnosticError(
            ErrorCode.INTERNAL_ERROR,
            "批量旋转输入形状不一致",
            stage="position",
            details={"shape": list(features_array.shape), "encoding_count": len(encodings)},
        )
    if not encodings:
        return np.empty((0, features_array.shape[1] if features_array.ndim == 2 else 0), dtype=np.float64)
    return np.vstack([apply_rotation(row, encoding) for row, encoding in zip(features_array, encodings)])


def apply_input_precision_and_quantization(
    features: np.ndarray, config: DiagnosticConfig
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64).copy()
    if config.precision == "float32":
        values = values.astype(np.float32).astype(np.float64)
    if config.quantization == "int8_symmetric":
        maximum = float(np.max(np.abs(values))) if values.size else 0.0
        if maximum > 0.0:
            scale = maximum / 127.0
            values = np.rint(values / scale).clip(-127, 127).astype(np.float64) * scale
    return values.astype(np.float64, copy=False)


def rotation_pair_payload(encoding: PositionEncoding) -> list[dict[str, Any]]:
    return [pair.to_dict() for pair in encoding.pairs]
