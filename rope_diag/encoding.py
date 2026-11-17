"""Unified RoPE, linear-interpolation and NTK-aware position encoders."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import (
    BoundaryError,
    ConfigError,
    E_BASE_ZERO,
    E_CONFIG_INVALID,
    E_NUMERIC_OVERFLOW,
    E_SCALE_INVALID,
    WorkbenchError,
)
from .injection import Injection
from .samples import Sample
from .util import clean_float


PRECISION_DTYPES = {"float64": np.float64, "float32": np.float32}


def _as_float_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def validate_scheme_numeric_inputs(
    scheme: str,
    base: float,
    dimension: int,
    scaling_factor: float,
    precision: str,
    *,
    allow_boundary: bool = False,
) -> dict[str, Any] | None:
    if scheme not in ("RoPE", "LinearInterpolation", "NTK-aware"):
        raise ConfigError("unsupported position encoding scheme", {"scheme": scheme})
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ConfigError("dimension must be a positive integer", {"dimension": dimension})
    if dimension % 2 != 0:
        raise ConfigError("dimension must be even", {"dimension": dimension})
    if precision not in PRECISION_DTYPES:
        raise ConfigError("unsupported precision tier", {"precision": precision})
    if not math.isfinite(base):
        raise ConfigError("base must be finite", {"base": base})
    if base == 0.0:
        info = {
            "parameter": "base",
            "criteria": "base_zero",
            "reason": "base must be greater than zero",
            "trigger_dimensions": [],
        }
        if allow_boundary:
            return info
        raise WorkbenchError(E_BASE_ZERO, "base must be greater than zero", details=info)
    if base < 0.0:
        raise ConfigError("base must be greater than zero", {"base": base})
    if not math.isfinite(scaling_factor):
        raise ConfigError("scaling_factor must be finite", {"scaling_factor": scaling_factor})
    if scaling_factor <= 0.0:
        info = {
            "parameter": "scaling_factor",
            "criteria": "scale_nonpositive",
            "reason": "scaling factor must be greater than zero",
            "trigger_dimensions": [],
        }
        if allow_boundary:
            return info
        raise WorkbenchError(
            E_SCALE_INVALID,
            "scaling factor must be greater than zero",
            details=info,
        )
    return None


def _effective_parameters(
    scheme: str, base: float, dimension: int, scaling_factor: float
) -> tuple[float, float]:
    """Return (effective_base, position_divisor)."""

    if scheme == "RoPE":
        return base, 1.0
    if scheme == "LinearInterpolation":
        return base, scaling_factor
    exponent = 1.0 if dimension <= 2 else dimension / (dimension - 2.0)
    try:
        scaled_base = float(scaling_factor**exponent)
    except OverflowError:
        scaled_base = float("inf")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        effective_base = float(base * scaled_base)
    return effective_base, 1.0


def _boundary_info(
    *,
    positions: np.ndarray,
    effective_positions: np.ndarray,
    effective_base: float,
    inv_freq: np.ndarray,
    angles: np.ndarray,
    cos_values: np.ndarray,
    sin_values: np.ndarray,
    resolution_floor: float,
) -> dict[str, Any] | None:
    trigger_dimensions: list[int] = []
    criteria = ""
    reason = ""

    if not math.isfinite(effective_base) or not np.all(np.isfinite(inv_freq)):
        criteria = "scaling_overflow"
        reason = "effective base or inverse frequency is non-finite"
        trigger_dimensions = list(range(len(inv_freq)))
    elif not np.all(np.isfinite(angles)):
        criteria = "scaling_overflow"
        reason = "rotation angle overflowed"
        bad_pairs = np.where(~np.all(np.isfinite(angles), axis=0))[0]
        trigger_dimensions = [int(index) for index in bad_pairs]
    elif not np.all(np.isfinite(cos_values)) or not np.all(np.isfinite(sin_values)):
        criteria = "scaling_overflow"
        reason = "rotation vector overflowed"
        trigger_dimensions = list(range(len(inv_freq)))
    else:
        underflow_pairs = np.where(inv_freq == 0.0)[0]
        zero_angle_pairs = np.where(
            np.any(
                (effective_positions[:, None] != 0.0)
                & (inv_freq[None, :] > 0.0)
                & (np.abs(angles) < resolution_floor),
                axis=0,
            )
        )[0]
        truncated = sorted({int(index) for index in underflow_pairs} | {int(index) for index in zero_angle_pairs})
        if truncated:
            criteria = "dimension_truncation"
            reason = "one or more rotation dimensions fell below the configured precision resolution"
            trigger_dimensions = truncated

    if not criteria:
        return None
    return {
        "criteria": criteria,
        "reason": reason,
        "trigger_dimensions": trigger_dimensions,
        "positions": [int(item) for item in positions.tolist()],
    }


def _encode_positions_impl(
    scheme: str,
    positions: Any,
    dimension: int,
    base: float,
    scaling_factor: float,
    precision: str,
    *,
    allow_boundary: bool,
) -> tuple["PositionEncoding | None", dict[str, Any] | None]:
    early = validate_scheme_numeric_inputs(
        scheme,
        base,
        dimension,
        scaling_factor,
        precision,
        allow_boundary=allow_boundary,
    )
    if early is not None:
        return None, early

    position_array = np.asarray(positions, dtype=np.int64)
    if position_array.ndim != 1:
        raise ConfigError("positions must be one-dimensional", {"shape": list(position_array.shape)})
    position_float = position_array.astype(np.float64)
    effective_base, divisor = _effective_parameters(scheme, base, dimension, scaling_factor)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        effective_positions = position_float / divisor

    exponents = np.arange(dimension // 2, dtype=np.float64) * (2.0 / float(dimension))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        inv_freq = np.power(np.float64(effective_base), -exponents)
        angles64 = effective_positions[:, None] * inv_freq[None, :]
    work_dtype = PRECISION_DTYPES[precision]
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        angles = angles64.astype(work_dtype)
        cos_values = np.cos(angles).astype(np.float64)
        sin_values = np.sin(angles).astype(np.float64)

    info = _boundary_info(
        positions=position_array,
        effective_positions=effective_positions,
        effective_base=effective_base,
        inv_freq=inv_freq,
        angles=np.asarray(angles, dtype=np.float64),
        cos_values=cos_values,
        sin_values=sin_values,
        resolution_floor=float(np.finfo(work_dtype).eps),
    )
    if info is not None and not allow_boundary:
        raise BoundaryError("position encoding reached a numeric boundary", info)
    if info is not None:
        return None, info

    return (
        PositionEncoding(
            positions=position_array,
            effective_positions=effective_positions,
            effective_base=effective_base,
            inv_freq=inv_freq.astype(np.float64),
            cos_values=cos_values,
            sin_values=sin_values,
            precision=precision,
        ),
        None,
    )


@dataclass
class PositionEncoding:
    positions: np.ndarray
    effective_positions: np.ndarray
    effective_base: float
    inv_freq: np.ndarray
    cos_values: np.ndarray
    sin_values: np.ndarray
    precision: str


def encode_positions(
    scheme: str,
    positions: Any,
    dimension: int,
    base: float,
    scaling_factor: float,
    precision: str,
) -> PositionEncoding:
    encoding, _ = _encode_positions_impl(
        scheme,
        positions,
        dimension,
        base,
        scaling_factor,
        precision,
        allow_boundary=False,
    )
    assert encoding is not None
    return encoding


def probe_positions(
    scheme: str,
    positions: Any,
    dimension: int,
    base: float,
    scaling_factor: float,
    precision: str,
) -> tuple[PositionEncoding | None, dict[str, Any] | None]:
    """Encoding probe used by sensitivity scans; boundaries return metadata."""

    return _encode_positions_impl(
        scheme,
        positions,
        dimension,
        base,
        scaling_factor,
        precision,
        allow_boundary=True,
    )


def rotate_features(features: np.ndarray, encoding: PositionEncoding) -> np.ndarray:
    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2:
        raise ConfigError("features must be a two-dimensional array", {"shape": list(array.shape)})
    if array.shape[0] != len(encoding.positions):
        raise ConfigError(
            "feature rows and position count differ",
            {"rows": int(array.shape[0]), "positions": int(len(encoding.positions))},
        )
    dimension = array.shape[1]
    if dimension % 2 != 0:
        raise ConfigError("feature dimension must be even", {"dimension": dimension})
    if encoding.cos_values.shape[1] * 2 != dimension:
        raise ConfigError(
            "feature dimension and rotation pairs differ",
            {"dimension": dimension, "rotation_pairs": int(encoding.cos_values.shape[1])},
        )
    even = array[:, 0::2]
    odd = array[:, 1::2]
    result = np.empty_like(array, dtype=np.float64)
    result[:, 0::2] = even * encoding.cos_values - odd * encoding.sin_values
    result[:, 1::2] = even * encoding.sin_values + odd * encoding.cos_values
    if not np.all(np.isfinite(result)):
        raise WorkbenchError(E_NUMERIC_OVERFLOW, "rotated feature vector became non-finite")
    return result


def encoding_parameters(scheme: str, base: float, dimension: int, scaling_factor: float) -> dict[str, Any]:
    params: dict[str, Any] = {"scheme": scheme, "base": base, "dimension": dimension}
    if scheme != "RoPE":
        params["scaling_factor"] = scaling_factor
    if scheme == "NTK-aware":
        params["ntk_exponent"] = 1.0 if dimension <= 2 else dimension / (dimension - 2.0)
    return params


@dataclass
class SampleEncoding:
    scheme: str
    sample_id: str
    base: float
    scaling_factor: float
    dimension: int
    precision: str
    positions: np.ndarray
    effective_positions: np.ndarray
    effective_base: float
    inv_freq: np.ndarray
    cos_values: np.ndarray
    sin_values: np.ndarray
    query_encoded: np.ndarray
    key_encoded: np.ndarray

    def to_payload(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "sample_id": self.sample_id,
            "base": self.base,
            "scaling_factor": self.scaling_factor,
            "dimension": self.dimension,
            "precision": self.precision,
            "positions": self.positions.tolist(),
            "effective_positions": self.effective_positions.tolist(),
            "effective_base": self.effective_base,
            "inv_freq": self.inv_freq.tolist(),
            "cos_values": self.cos_values.tolist(),
            "sin_values": self.sin_values.tolist(),
            "query_encoded": self.query_encoded.tolist(),
            "key_encoded": self.key_encoded.tolist(),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SampleEncoding":
        return cls(
            scheme=str(payload["scheme"]),
            sample_id=str(payload["sample_id"]),
            base=float(payload["base"]),
            scaling_factor=float(payload["scaling_factor"]),
            dimension=int(payload["dimension"]),
            precision=str(payload["precision"]),
            positions=np.asarray(payload["positions"], dtype=np.int64),
            effective_positions=np.asarray(payload["effective_positions"], dtype=np.float64),
            effective_base=float(payload["effective_base"]),
            inv_freq=np.asarray(payload["inv_freq"], dtype=np.float64),
            cos_values=np.asarray(payload["cos_values"], dtype=np.float64),
            sin_values=np.asarray(payload["sin_values"], dtype=np.float64),
            query_encoded=np.asarray(payload["query_encoded"], dtype=np.float64),
            key_encoded=np.asarray(payload["key_encoded"], dtype=np.float64),
        )


def encode_sample(
    scheme: str,
    sample: Sample,
    dimension: int,
    base: float,
    scaling_factor: float,
    precision: str,
) -> SampleEncoding:
    query = sample.query_array
    keys = sample.key_array
    if query.shape != (dimension,):
        raise ConfigError(
            "query feature dimension does not match configuration",
            {
                "sample_id": sample.sample_id,
                "expected": dimension,
                "actual": int(query.size),
            },
        )
    if keys.ndim != 2 or keys.shape[1] != dimension:
        raise ConfigError(
            "key feature dimension does not match configuration",
            {
                "sample_id": sample.sample_id,
                "expected": dimension,
                "actual": list(keys.shape),
            },
        )
    if not np.all(np.isfinite(query)) or not np.all(np.isfinite(keys)):
        raise WorkbenchError(
            E_CONFIG_INVALID,
            "non-finite features must be reported as E_NONFINITE_INPUT",
            stage="encoding",
            details={"sample_id": sample.sample_id},
        )
    positions = np.asarray([sample.query_position, *sample.key_positions], dtype=np.int64)
    encoding = encode_positions(scheme, positions, dimension, base, scaling_factor, precision)
    encoded = rotate_features(np.vstack([query[None, :], keys]), encoding)
    return SampleEncoding(
        scheme=scheme,
        sample_id=sample.sample_id,
        base=base,
        scaling_factor=scaling_factor,
        dimension=dimension,
        precision=precision,
        positions=positions,
        effective_positions=encoding.effective_positions,
        effective_base=encoding.effective_base,
        inv_freq=encoding.inv_freq,
        cos_values=encoding.cos_values,
        sin_values=encoding.sin_values,
        query_encoded=encoded[0],
        key_encoded=encoded[1:],
    )


def reference_encoding(sample: Sample, dimension: int, base: float, precision: str) -> SampleEncoding:
    """Analytic in-train RoPE extension used as the extrapolation reference."""

    return encode_sample("RoPE", sample, dimension, base, 1.0, precision)
