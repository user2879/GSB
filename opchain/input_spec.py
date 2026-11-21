"""Generate deterministic synthetic input tensors from a JSON description."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import E_INPUT_INVALID, BusinessError
from .util import sha256_json, to_jsonable


@dataclass
class InputSpec:
    name: str
    seed: int
    shape: tuple[int, ...]
    dtype: np.dtype
    distribution: dict[str, Any]
    special_values: list[dict[str, Any]]
    raw: dict[str, Any]

    def canonical(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "shape": list(self.shape),
            "dtype": self.dtype.str,
            "distribution": self.distribution,
            "special_values": self.special_values,
        }

    def fingerprint(self) -> str:
        return sha256_json(self.canonical())


def _parse_special_value(value: Any) -> float:
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"nan", "+nan", "-nan"}:
            return float("nan")
        if lowered in {"inf", "+inf", "infinity", "+infinity"}:
            return float("inf")
        if lowered in {"-inf", "-infinity"}:
            return float("-inf")
        try:
            return float(value)
        except ValueError as exc:
            raise BusinessError(E_INPUT_INVALID, f"invalid special value: {value!r}", stage="input") from exc
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise BusinessError(E_INPUT_INVALID, f"invalid special value: {value!r}", stage="input") from exc


def parse_input_spec(raw: dict[str, Any] | InputSpec) -> InputSpec:
    if isinstance(raw, InputSpec):
        return raw
    source = dict(raw)
    if "tensor" in source and isinstance(source["tensor"], dict):
        source = dict(source["tensor"])
    name = str(source.get("name", "x"))
    seed = source.get("seed", 0)
    if isinstance(seed, bool):
        raise BusinessError(E_INPUT_INVALID, "seed must be an integer", stage="input")
    try:
        seed = int(seed)
    except (TypeError, ValueError) as exc:
        raise BusinessError(E_INPUT_INVALID, "seed must be an integer", stage="input") from exc
    if seed < 0:
        raise BusinessError(E_INPUT_INVALID, "seed must be non-negative", stage="input")

    shape_raw = source.get("shape")
    if not isinstance(shape_raw, (list, tuple)) or not shape_raw:
        raise BusinessError(E_INPUT_INVALID, "shape must be a non-empty integer list", stage="input")
    shape: list[int] = []
    for dim in shape_raw:
        if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)) or int(dim) <= 0:
            raise BusinessError(E_INPUT_INVALID, "shape dimensions must be positive integers", stage="input")
        shape.append(int(dim))

    dtype_raw = str(source.get("dtype", "float64"))
    try:
        dtype = np.dtype(dtype_raw)
    except TypeError as exc:
        raise BusinessError(E_INPUT_INVALID, f"invalid input dtype: {dtype_raw!r}", stage="input") from exc
    if dtype.kind != "f":
        raise BusinessError(E_INPUT_INVALID, "input dtype must be floating point", stage="input")

    distribution_raw = source.get("distribution", {})
    if isinstance(distribution_raw, str):
        distribution = {"kind": distribution_raw}
    elif isinstance(distribution_raw, dict):
        distribution = dict(distribution_raw)
    else:
        raise BusinessError(E_INPUT_INVALID, "distribution must be a string or object", stage="input")
    kind = str(distribution.get("kind", "uniform")).lower()
    if kind not in {"uniform", "normal", "integer", "zeros", "ones"}:
        raise BusinessError(E_INPUT_INVALID, f"unsupported distribution: {kind!r}", stage="input")

    special_values = source.get("special_values", source.get("specials", []))
    if not isinstance(special_values, list):
        raise BusinessError(E_INPUT_INVALID, "special_values must be a list", stage="input")
    parsed_special_values: list[dict[str, Any]] = []
    for item in special_values:
        if not isinstance(item, dict) or "value" not in item:
            raise BusinessError(E_INPUT_INVALID, "each special value needs index and value", stage="input")
        index = item.get("index", item.get("position", item.get("indices")))
        if not isinstance(index, (list, tuple)) or len(index) != len(shape):
            raise BusinessError(E_INPUT_INVALID, "special value index rank does not match shape", stage="input")
        checked_index: list[int] = []
        for position, dim in zip(index, shape):
            if isinstance(position, bool) or not isinstance(position, (int, np.integer)):
                raise BusinessError(E_INPUT_INVALID, "special value index must contain integers", stage="input")
            if int(position) < 0 or int(position) >= dim:
                raise BusinessError(E_INPUT_INVALID, "special value index is out of bounds", stage="input")
            checked_index.append(int(position))
        parsed_special_values.append({"index": checked_index, "value": item["value"]})

    normalized_distribution = _normalize_distribution(kind, distribution)
    return InputSpec(
        name=name,
        seed=seed,
        shape=tuple(shape),
        dtype=dtype,
        distribution=normalized_distribution,
        special_values=parsed_special_values,
        raw=source,
    )


def _normalize_distribution(kind: str, distribution: dict[str, Any]) -> dict[str, Any]:
    try:
        if kind == "uniform":
            low = float(distribution.get("low", -1.0))
            high = float(distribution.get("high", 1.0))
            if not low < high:
                raise ValueError("uniform low must be less than high")
            return {"kind": kind, "low": low, "high": high}
        if kind == "normal":
            mean = float(distribution.get("mean", 0.0))
            std = float(distribution.get("std", 1.0))
            if std < 0:
                raise ValueError("normal std must be non-negative")
            return {"kind": kind, "mean": mean, "std": std}
        if kind == "integer":
            low = int(distribution.get("low", -8))
            high = int(distribution.get("high", 8))
            if not low <= high:
                raise ValueError("integer low must be <= high")
            return {"kind": kind, "low": low, "high": high}
        return {"kind": kind}
    except (TypeError, ValueError) as exc:
        raise BusinessError(E_INPUT_INVALID, f"invalid distribution parameters: {exc}", stage="input") from exc


def generate_input(spec: InputSpec) -> np.ndarray:
    rng = np.random.default_rng(spec.seed)
    shape = spec.shape
    kind = spec.distribution["kind"]
    if kind == "uniform":
        values = rng.uniform(spec.distribution["low"], spec.distribution["high"], size=shape)
    elif kind == "normal":
        values = rng.normal(spec.distribution["mean"], spec.distribution["std"], size=shape)
    elif kind == "integer":
        values = rng.integers(spec.distribution["low"], spec.distribution["high"], size=shape, endpoint=True)
    elif kind == "zeros":
        values = np.zeros(shape, dtype=np.float64)
    else:
        values = np.ones(shape, dtype=np.float64)

    values = np.asarray(values, dtype=spec.dtype).copy(order="C")
    for item in spec.special_values:
        values[tuple(item["index"])] = _parse_special_value(item["value"])
    return values


def input_description(raw: dict[str, Any] | InputSpec) -> dict[str, Any]:
    spec = parse_input_spec(raw)
    return to_jsonable(spec.canonical())
