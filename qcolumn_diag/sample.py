"""Deterministic sample-description loading and fixed-seed materialization."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .errors import make_failure, sort_failures
from .fingerprint import sha256_json
from .models import TensorConfig


@dataclass
class TensorSample:
    tensor_id: str
    shape: tuple[int, ...]
    column_axis: int
    matrix: np.ndarray
    missing_mask: np.ndarray
    injection_ids: list[list[list[str]]]
    original_dtype: str
    generator: dict[str, Any]
    normalized_spec: dict[str, Any]
    spec_hash: str
    seed: int
    applied_injections: list[dict[str, Any]]

    @property
    def row_count(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def column_count(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def other_shape(self) -> tuple[int, ...]:
        return tuple(
            dimension
            for index, dimension in enumerate(self.shape)
            if index != self.column_axis
        )

    def element_index(self, row: int, column: int) -> list[int]:
        other_shape = self.other_shape
        other_coordinates = np.unravel_index(row, other_shape) if other_shape else ()
        index = list(int(value) for value in other_coordinates)
        index.insert(self.column_axis, int(column))
        return index

    def element_ref(self, row: int, column: int) -> dict[str, Any]:
        return {
            "row": int(row),
            "index": self.element_index(row, column),
            "injection_ids": list(self.injection_ids[row][column]),
        }


@dataclass
class SampleLoadResult:
    sample_id: str | None
    tensors: dict[str, TensorSample]
    failures: list[dict[str, Any]]
    sample_hash: str
    normalized: dict[str, Any] | None


_MISSING = object()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _add_failure(
    failures: list[dict[str, Any]],
    field: str,
    message: str,
    *,
    tensor_id: str | None = None,
) -> None:
    failures.append(
        make_failure(
            "E_SAMPLE_INVALID",
            field,
            message,
            tensor_id=tensor_id,
        )
    )


def _locate_spec(raw: dict[str, Any], tensor_id: str) -> tuple[dict[str, Any] | None, str]:
    tensors_raw = raw.get("tensors", _MISSING)
    if tensors_raw is _MISSING:
        if "generator" in raw or "seed" in raw:
            return raw, "sample"
        return None, "sample"
    if isinstance(tensors_raw, dict):
        spec = tensors_raw.get(tensor_id, _MISSING)
        if spec is _MISSING:
            return None, f"sample.tensors.{tensor_id}"
        if not isinstance(spec, dict):
            return {}, f"sample.tensors.{tensor_id}"
        return spec, f"sample.tensors.{tensor_id}"
    if isinstance(tensors_raw, list):
        matches = [
            item
            for item in tensors_raw
            if isinstance(item, dict)
            and str(item.get("tensor_id", item.get("id", ""))) == tensor_id
        ]
        if len(matches) != 1:
            return None, f"sample.tensors[{tensor_id}]"
        return matches[0], f"sample.tensors[{tensor_id}]"
    return {}, "sample.tensors"


def _parse_coordinate(
    raw: Any,
    *,
    shape: tuple[int, ...],
    column_axis: int,
    field: str,
    failures: list[dict[str, Any]],
    tensor_id: str,
) -> tuple[list[int] | None, int | None, int | None]:
    index: list[int] | None = None
    if isinstance(raw, (list, tuple)):
        index = [int(value) if _is_int(value) else value for value in raw]
    elif isinstance(raw, dict):
        if "index" in raw:
            index_raw = raw["index"]
            if isinstance(index_raw, (list, tuple)):
                index = [
                    int(value) if _is_int(value) else value for value in index_raw
                ]
        else:
            row_raw = raw.get("row", _MISSING)
            column_raw = raw.get("column", _MISSING)
            if _is_int(row_raw) and _is_int(column_raw) and len(shape) == 2:
                index = [int(row_raw), int(column_raw)]
            elif (
                "other_index" in raw
                and isinstance(raw["other_index"], list)
                and _is_int(column_raw)
            ):
                other = raw["other_index"]
                index = [
                    int(value) if _is_int(value) else value for value in other
                ]
                index.insert(column_axis, int(column_raw))
    if index is None or len(index) != len(shape) or any(not _is_int(v) for v in index):
        _add_failure(
            failures,
            field,
            f"coordinate must contain {len(shape)} integer indices",
            tensor_id=tensor_id,
        )
        return None, None, None
    for dimension, value in zip(shape, index):
        if not (0 <= int(value) < dimension):
            _add_failure(
                failures,
                field,
                f"coordinate {index} is outside shape {list(shape)}",
                tensor_id=tensor_id,
            )
            return None, None, None
    column = int(index[column_axis])
    other_index = tuple(int(value) for i, value in enumerate(index) if i != column_axis)
    other_shape = tuple(
        dimension for i, dimension in enumerate(shape) if i != column_axis
    )
    row = int(np.ravel_multi_index(other_index, other_shape)) if other_shape else 0
    return [int(value) for value in index], row, column


def _extract_seed(
    spec: dict[str, Any],
    *,
    field: str,
    failures: list[dict[str, Any]],
    tensor_id: str,
) -> tuple[int | None, dict[str, Any] | None]:
    generator_raw = spec.get("generator", _MISSING)
    if generator_raw is _MISSING:
        if "seed" in spec:
            generator_raw = {
                "kind": "normal",
                "seed": spec.get("seed"),
                "loc": spec.get("loc", 0.0),
                "scale": spec.get("scale", 1.0),
            }
        else:
            _add_failure(
                failures,
                f"{field}.generator",
                "generator or seed is required",
                tensor_id=tensor_id,
            )
            return None, None
    if not isinstance(generator_raw, dict):
        _add_failure(
            failures,
            f"{field}.generator",
            "generator must be an object",
            tensor_id=tensor_id,
        )
        return None, None
    seed = generator_raw.get("seed")
    if not _is_int(seed) or seed < 0:
        _add_failure(
            failures,
            f"{field}.generator.seed",
            "seed must be a non-negative integer",
            tensor_id=tensor_id,
        )
        return None, None
    kind = generator_raw.get("kind", "normal")
    normalized: dict[str, Any] = {"kind": kind, "seed": int(seed)}
    if kind == "normal":
        loc = generator_raw.get("loc", generator_raw.get("mean", 0.0))
        scale = generator_raw.get("scale", generator_raw.get("std", 1.0))
        if not _is_finite_number(loc):
            _add_failure(
                failures,
                f"{field}.generator.loc",
                "loc must be a finite number",
                tensor_id=tensor_id,
            )
            return None, None
        if not _is_finite_number(scale) or float(scale) < 0:
            _add_failure(
                failures,
                f"{field}.generator.scale",
                "scale must be a finite non-negative number",
                tensor_id=tensor_id,
            )
            return None, None
        normalized.update({"loc": float(loc), "scale": float(scale)})
    elif kind == "uniform":
        low = generator_raw.get("low")
        high = generator_raw.get("high")
        if not _is_finite_number(low) or not _is_finite_number(high):
            _add_failure(
                failures,
                f"{field}.generator",
                "uniform generator requires finite low and high",
                tensor_id=tensor_id,
            )
            return None, None
        if not (float(low) < float(high)):
            _add_failure(
                failures,
                f"{field}.generator",
                "uniform generator requires low < high",
                tensor_id=tensor_id,
            )
            return None, None
        normalized.update({"low": float(low), "high": float(high)})
    else:
        _add_failure(
            failures,
            f"{field}.generator.kind",
            "generator kind must be 'normal' or 'uniform'",
            tensor_id=tensor_id,
        )
        return None, None
    return int(seed), normalized


def _parse_vector(
    spec: dict[str, Any],
    name: str,
    *,
    length: int,
    field: str,
    failures: list[dict[str, Any]],
    tensor_id: str,
    default: float,
) -> np.ndarray | None:
    raw = spec.get(name, _MISSING)
    if raw is _MISSING:
        return np.full(length, default, dtype=np.float64)
    if not isinstance(raw, list) or len(raw) != length:
        _add_failure(
            failures,
            f"{field}.{name}",
            f"must be a list of length {length}",
            tensor_id=tensor_id,
        )
        return None
    if any(not _is_finite_number(value) for value in raw):
        _add_failure(
            failures,
            f"{field}.{name}",
            "all values must be finite numbers",
            tensor_id=tensor_id,
        )
        return None
    return np.asarray([float(value) for value in raw], dtype=np.float64)


def _materialize_tensor(
    tensor: TensorConfig,
    spec: dict[str, Any],
    *,
    field: str,
    raw: dict[str, Any],
    failures: list[dict[str, Any]],
) -> TensorSample | None:
    tensor_id = tensor.tensor_id
    error_start = len(failures)
    if "shape" in spec:
        shape_raw = spec["shape"]
        if (
            not isinstance(shape_raw, list)
            or any(not _is_int(value) or value <= 0 for value in shape_raw)
            or tuple(shape_raw) != tensor.shape
        ):
            _add_failure(
                failures,
                f"{field}.shape",
                f"sample shape must equal configuration shape {list(tensor.shape)}",
                tensor_id=tensor_id,
            )
            shape = tensor.shape
        else:
            shape = tuple(int(value) for value in shape_raw)
    else:
        shape = tensor.shape

    dtype_raw = spec.get("dtype", raw.get("default_dtype", "float64"))
    if dtype_raw not in ("float32", "float64"):
        _add_failure(
            failures,
            f"{field}.dtype",
            "dtype must be 'float32' or 'float64'",
            tensor_id=tensor_id,
        )
        original_dtype = "float64"
    else:
        original_dtype = str(dtype_raw)

    seed, generator = _extract_seed(
        spec, field=field, failures=failures, tensor_id=tensor_id
    )
    if seed is None or generator is None:
        return None

    rng = np.random.Generator(np.random.PCG64(seed))
    if generator["kind"] == "normal":
        generated = rng.standard_normal(size=shape)
        generated = generator["loc"] + generator["scale"] * generated
    else:
        generated = rng.uniform(generator["low"], generator["high"], size=shape)
    generated = np.asarray(generated, dtype=np.float64)

    column_count = tensor.column_count
    offsets = _parse_vector(
        spec,
        "column_offsets",
        length=column_count,
        field=field,
        failures=failures,
        tensor_id=tensor_id,
        default=0.0,
    )
    scales = _parse_vector(
        spec,
        "column_scales",
        length=column_count,
        field=field,
        failures=failures,
        tensor_id=tensor_id,
        default=1.0,
    )
    if offsets is None or scales is None:
        return None
    if np.any(scales < 0):
        _add_failure(
            failures,
            f"{field}.column_scales",
            "column scales must be non-negative",
            tensor_id=tensor_id,
        )
        return None
    moved = np.moveaxis(generated, tensor.column_axis, -1)
    moved = moved * scales.reshape((1,) * (moved.ndim - 1) + (-1,))
    moved = moved + offsets.reshape((1,) * (moved.ndim - 1) + (-1,))
    generated = np.moveaxis(moved, -1, tensor.column_axis)

    constant_raw = spec.get("constant_columns", {})
    if constant_raw is not _MISSING:
        if not isinstance(constant_raw, dict):
            _add_failure(
                failures,
                f"{field}.constant_columns",
                "constant_columns must be an object mapping column to value",
                tensor_id=tensor_id,
            )
            return None
        for column_text, value in constant_raw.items():
            try:
                column = int(column_text)
            except (TypeError, ValueError):
                _add_failure(
                    failures,
                    f"{field}.constant_columns",
                    f"invalid column key {column_text!r}",
                    tensor_id=tensor_id,
                )
                return None
            if not (0 <= column < column_count):
                _add_failure(
                    failures,
                    f"{field}.constant_columns",
                    f"column {column} is outside [0, {column_count - 1}]",
                    tensor_id=tensor_id,
                )
                return None
            if not _is_finite_number(value):
                _add_failure(
                    failures,
                    f"{field}.constant_columns",
                    f"value for column {column} must be finite",
                    tensor_id=tensor_id,
                )
                return None
            moved = np.moveaxis(generated, tensor.column_axis, -1)
            moved[..., column] = float(value)
            generated = np.moveaxis(moved, -1, tensor.column_axis)

    missing_raw = spec.get("missing_values", [])
    if not isinstance(missing_raw, list):
        _add_failure(
            failures,
            f"{field}.missing_values",
            "missing_values must be a list",
            tensor_id=tensor_id,
        )
        return None
    missing_coordinates: set[tuple[int, ...]] = set()
    for position, coordinate_raw in enumerate(missing_raw):
        index, _, _ = _parse_coordinate(
            coordinate_raw,
            shape=shape,
            column_axis=tensor.column_axis,
            field=f"{field}.missing_values[{position}]",
            failures=failures,
            tensor_id=tensor_id,
        )
        if index is None:
            continue
        coordinate = tuple(index)
        if coordinate in missing_coordinates:
            _add_failure(
                failures,
                f"{field}.missing_values[{position}]",
                "duplicate missing coordinate",
                tensor_id=tensor_id,
            )
            continue
        missing_coordinates.add(coordinate)
        generated[coordinate] = np.nan

    generated = generated.astype(original_dtype, copy=False).astype(
        np.float64, copy=False
    )
    if np.any(np.isinf(generated)):
        _add_failure(
            failures,
            f"{field}",
            "generated sample contains infinity after dtype conversion",
            tensor_id=tensor_id,
        )
        return None

    moved = np.moveaxis(generated, tensor.column_axis, -1)
    matrix = np.asarray(moved.reshape(-1, column_count), dtype=np.float64)
    missing_mask = np.isnan(matrix)
    injection_ids: list[list[list[str]]] = [
        [[] for _ in range(column_count)] for _ in range(matrix.shape[0])
    ]
    applied_injections: list[dict[str, Any]] = []
    injections_raw = spec.get("injections", [])
    if not isinstance(injections_raw, list):
        _add_failure(
            failures,
            f"{field}.injections",
            "injections must be a list",
            tensor_id=tensor_id,
        )
        return None
    seen_injection_ids: set[str] = set()
    for position, injection_raw in enumerate(injections_raw):
        injection_field = f"{field}.injections[{position}]"
        if not isinstance(injection_raw, dict):
            _add_failure(
                failures,
                injection_field,
                "injection must be an object",
                tensor_id=tensor_id,
            )
            continue
        injection_id_raw = injection_raw.get("id")
        if (
            not isinstance(injection_id_raw, (str, int))
            or isinstance(injection_id_raw, bool)
            or not str(injection_id_raw).strip()
        ):
            _add_failure(
                failures,
                f"{injection_field}.id",
                "id must be a string or integer",
                tensor_id=tensor_id,
            )
            continue
        injection_id = str(injection_id_raw).strip()
        if injection_id in seen_injection_ids:
            _add_failure(
                failures,
                f"{injection_field}.id",
                f"duplicate injection id {injection_id!r}",
                tensor_id=tensor_id,
            )
            continue
        seen_injection_ids.add(injection_id)
        coordinate_raw = injection_raw.get("coordinate", _MISSING)
        if coordinate_raw is _MISSING:
            coordinate_raw = {
                key: injection_raw[key]
                for key in ("row", "column", "index", "other_index")
                if key in injection_raw
            }
        index, row, column = _parse_coordinate(
            coordinate_raw,
            shape=shape,
            column_axis=tensor.column_axis,
            field=f"{injection_field}.coordinate",
            failures=failures,
            tensor_id=tensor_id,
        )
        if index is None or row is None or column is None:
            continue
        operation = injection_raw.get("operation", injection_raw.get("op", "set"))
        if operation not in ("set", "add", "multiply"):
            _add_failure(
                failures,
                f"{injection_field}.operation",
                "operation must be set, add or multiply",
                tensor_id=tensor_id,
            )
            continue
        value = injection_raw.get("value")
        if not _is_finite_number(value):
            _add_failure(
                failures,
                f"{injection_field}.value",
                "value must be a finite number",
                tensor_id=tensor_id,
            )
            continue
        value_dtype = np.asarray(float(value), dtype=original_dtype).astype(np.float64)
        before = float(matrix[row, column])
        if operation == "set":
            matrix[row, column] = float(value_dtype)
        elif operation == "add":
            matrix[row, column] = matrix[row, column] + float(value_dtype)
        else:
            matrix[row, column] = matrix[row, column] * float(value_dtype)
        after = float(matrix[row, column])
        hit = bool(math.isfinite(after))
        if hit:
            injection_ids[row][column].append(injection_id)
        missing_mask[row, column] = not hit
        applied_injections.append(
            {
                "id": injection_id,
                "index": list(index),
                "row": int(row),
                "column": int(column),
                "operation": operation,
                "requested_value": float(value),
                "before": before if math.isfinite(before) else None,
                "after": after if math.isfinite(after) else None,
                "hit": hit,
            }
        )
    if np.any(np.isinf(matrix)):
        _add_failure(
            failures,
            field,
            "sample contains infinity after injections",
            tensor_id=tensor_id,
        )
        return None

    if len(failures) != error_start:
        return None

    normalized_spec = {
        "dtype": original_dtype,
        "shape": list(shape),
        "generator": generator,
        "column_offsets": offsets.tolist(),
        "column_scales": scales.tolist(),
        "constant_columns": {
            str(key): float(value) for key, value in sorted(constant_raw.items())
        }
        if isinstance(constant_raw, dict)
        else {},
        "missing_values": sorted([list(value) for value in missing_coordinates]),
        "injections": sorted(
            [
                {
                    "id": str(item.get("id")),
                    "index": list(
                        _parse_coordinate(
                            item.get("coordinate", item),
                            shape=shape,
                            column_axis=tensor.column_axis,
                            field="normalization",
                            failures=[],
                            tensor_id=tensor_id,
                        )[0]
                        or []
                    ),
                    "operation": item.get(
                        "operation", item.get("op", "set")
                    ),
                    "value": float(item.get("value")),
                }
                for item in injections_raw
                if isinstance(item, dict) and item.get("id") is not None
            ],
            key=lambda item: (item["id"], item["index"]),
        ),
    }
    spec_hash = sha256_json(normalized_spec)
    return TensorSample(
        tensor_id=tensor_id,
        shape=shape,
        column_axis=tensor.column_axis,
        matrix=matrix,
        missing_mask=missing_mask,
        injection_ids=injection_ids,
        original_dtype=original_dtype,
        generator=generator,
        normalized_spec=normalized_spec,
        spec_hash=spec_hash,
        seed=int(seed),
        applied_injections=applied_injections,
    )


def load_sample(path: str | Path, tensors: Iterable[TensorConfig]) -> SampleLoadResult:
    failures: list[dict[str, Any]] = []
    sample_path = Path(path)
    try:
        raw = json.loads(sample_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        failures.append(
            make_failure(
                "E_SAMPLE_INVALID",
                "sample_path",
                f"sample file not found: {sample_path}",
            )
        )
        return SampleLoadResult(None, {}, sort_failures(failures), sha256_json({}), None)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        failures.append(
            make_failure(
                "E_SAMPLE_INVALID",
                "sample_json",
                f"cannot read sample JSON: {exc}",
            )
        )
        return SampleLoadResult(None, {}, sort_failures(failures), sha256_json({}), None)
    if not isinstance(raw, dict):
        failures.append(
            make_failure(
                "E_SAMPLE_INVALID",
                "root",
                "sample description root must be a JSON object",
            )
        )
        return SampleLoadResult(None, {}, sort_failures(failures), sha256_json({}), None)

    sample_id_raw = raw.get("sample_id", raw.get("id", "fixed-seed-sample"))
    if not isinstance(sample_id_raw, str) or not sample_id_raw.strip():
        _add_failure(
            failures,
            "sample_id",
            "sample_id must be a non-empty string",
        )
        sample_id = "fixed-seed-sample"
    else:
        sample_id = sample_id_raw.strip()

    tensor_list = sorted(tensors, key=lambda item: item.tensor_id)
    sample_tensors: dict[str, TensorSample] = {}
    normalized_tensors: dict[str, Any] = {}
    for tensor in tensor_list:
        spec, field = _locate_spec(raw, tensor.tensor_id)
        if spec is None:
            _add_failure(
                failures,
                field,
                f"no sample description for tensor {tensor.tensor_id!r}",
                tensor_id=tensor.tensor_id,
            )
            continue
        if not isinstance(spec, dict):
            _add_failure(
                failures,
                field,
                "sample tensor spec must be an object",
                tensor_id=tensor.tensor_id,
            )
            continue
        sample_tensor = _materialize_tensor(
            tensor,
            spec,
            field=field,
            raw=raw,
            failures=failures,
        )
        if sample_tensor is not None:
            sample_tensors[tensor.tensor_id] = sample_tensor
            normalized_tensors[tensor.tensor_id] = sample_tensor.normalized_spec

    normalized = {
        "schema_version": "1.0",
        "sample_id": sample_id,
        "dtype": raw.get("default_dtype", "float64"),
        "tensors": {
            key: normalized_tensors[key] for key in sorted(normalized_tensors)
        },
    }
    sample_hash = sha256_json(
        {
            "normalized_sample": normalized,
            "sample_failures": sort_failures(failures),
        }
    )
    return SampleLoadResult(
        sample_id,
        sample_tensors,
        sort_failures(failures),
        sample_hash,
        normalized,
    )
