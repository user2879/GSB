"""Weight-pack, calibration-set, and fault-injection loading/validation."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .config import PipelineConfig
from .errors import (
    E_CALIBRATION_INVALID,
    E_CONFIG_INVALID,
    E_INJECTION_INVALID,
    E_WEIGHT_PACK_INVALID,
    CalibrationError,
    ConfigError,
    FailureRecord,
    WeightPackError,
    failure,
)
from .util import finite_float, is_int, read_json, sha256_bytes, sha256_json
from .yaml_lite import load_yaml


INJECTION_TYPES = {
    "non_finite_weight": E_WEIGHT_PACK_INVALID,
    "group_indivisible": E_WEIGHT_PACK_INVALID,
    "bit_width_out_of_range": E_CONFIG_INVALID,
    "calibration_dimension_mismatch": E_CALIBRATION_INVALID,
    "zero_temperature": E_CONFIG_INVALID,
    "duplicate_layer_name": E_WEIGHT_PACK_INVALID,
}


@dataclass(frozen=True)
class WeightLayer:
    name: str
    shape: tuple[int, ...]
    weights: np.ndarray
    version: int
    marker_group_size: int
    line_no: int
    content_fingerprint: str

    @property
    def numel(self) -> int:
        return int(self.weights.size)

    @property
    def input_dim(self) -> int:
        return int(self.shape[-1])

    @property
    def output_dim(self) -> int:
        if len(self.shape) == 1:
            return int(self.shape[0])
        return int(np.prod(self.shape[:-1], dtype=np.int64))

    def apply(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64)
        if vector.ndim != 1 or vector.size != self.input_dim:
            raise CalibrationError(
                E_CALIBRATION_INVALID,
                f"input vector dimension {vector.size} does not match layer input dimension {self.input_dim}",
                stage="calibration",
                layer=self.name,
            )
        if len(self.shape) == 1:
            return self.weights * vector
        matrix = self.weights.reshape((-1, self.input_dim))
        return matrix @ vector


@dataclass
class RawWeightRecord:
    line_no: int
    data: dict[str, Any] | None = None
    parse_error: str | None = None
    faults: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CalibrationSample:
    sample_id: str
    inputs: dict[str, tuple[float, ...]]
    reference_outputs: dict[str, tuple[float, ...]]


@dataclass
class CalibrationSet:
    samples: tuple[CalibrationSample, ...]
    raw_hash: str
    sample_ids: tuple[str, ...]
    failures: list[FailureRecord] = field(default_factory=list)

    def for_layer(self, layer_name: str) -> tuple[list[tuple[float, ...]], list[tuple[float, ...]]]:
        inputs: list[tuple[float, ...]] = []
        references: list[tuple[float, ...]] = []
        for sample in self.samples:
            input_key = layer_name if layer_name in sample.inputs else ("*" if "*" in sample.inputs else None)
            reference_key = (
                layer_name
                if layer_name in sample.reference_outputs
                else ("*" if "*" in sample.reference_outputs else None)
            )
            if input_key is None or reference_key is None:
                raise CalibrationError(
                    E_CALIBRATION_INVALID,
                    f"calibration sample {sample.sample_id!r} has no input/reference for layer {layer_name!r}",
                    stage="calibration",
                    layer=layer_name,
                    details={"sample_id": sample.sample_id},
                )
            inputs.append(sample.inputs[input_key])
            references.append(sample.reference_outputs[reference_key])
        return inputs, references


@dataclass(frozen=True)
class LayerCalibrationView:
    layer: str
    sample_ids: tuple[str, ...]
    inputs: tuple[np.ndarray, ...]
    references: tuple[np.ndarray, ...]

    def as_meta(self) -> dict[str, Any]:
        return {
            "sample_ids": list(self.sample_ids),
            "sample_count": len(self.sample_ids),
            "input_dim": int(self.inputs[0].size) if self.inputs else 0,
            "reference_dim": int(self.references[0].size) if self.references else 0,
        }

    def fingerprint(self) -> str:
        return sha256_json(
            {
                "layer": self.layer,
                "sample_ids": list(self.sample_ids),
                "inputs": [value.tolist() for value in self.inputs],
                "references": [value.tolist() for value in self.references],
            }
        )


@dataclass(frozen=True)
class InjectionItem:
    injection_id: str
    fault_type: str
    layer: str | None = None
    stage: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    expected_code: str = E_INJECTION_INVALID

    def as_meta(self) -> dict[str, Any]:
        return {
            "injection_id": self.injection_id,
            "type": self.fault_type,
            "layer": self.layer,
            "stage": self.stage,
            "expected_business_code": self.expected_code,
        }


@dataclass
class InjectionReport:
    item: InjectionItem
    applied: bool = False
    hit: bool = False
    business_code: str | None = None
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        body = self.item.as_meta()
        body.update(
            {
                "applied": self.applied,
                "hit": self.hit,
                "business_code": self.business_code,
                "message": self.message,
            }
        )
        return body


@dataclass
class InjectionEngine:
    items: list[InjectionItem] = field(default_factory=list)
    reports: dict[str, InjectionReport] = field(default_factory=dict)
    failures: list[FailureRecord] = field(default_factory=list)
    weight_marks: dict[int, list[str]] = field(default_factory=dict)
    config_marks: dict[str, list[str]] = field(default_factory=dict)
    calibration_marks: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for item in self.items:
            self.reports[item.injection_id] = InjectionReport(item=item)

    def mark_weight(self, line_no: int, injection_id: str) -> None:
        self.weight_marks.setdefault(line_no, []).append(injection_id)

    def mark_config(self, scope: str, injection_id: str) -> None:
        self.config_marks.setdefault(scope, []).append(injection_id)

    def mark_calibration(self, layer: str, injection_id: str) -> None:
        self.calibration_marks.setdefault(layer, []).append(injection_id)

    def mark_applied(self, injection_id: str, message: str) -> None:
        report = self.reports[injection_id]
        report.applied = True
        report.message = message

    def mark_hit(self, injection_id: str, code: str, message: str | None = None) -> None:
        report = self.reports[injection_id]
        report.hit = True
        report.business_code = code
        if message:
            report.message = message

    def injection_ids_for_config_layer(self, layer: str) -> list[str]:
        return list(self.config_marks.get(f"layer:{layer}", []))

    def injection_ids_for_global_distillation(self) -> list[str]:
        return list(self.config_marks.get("distillation", []))

    def injection_ids_for_weight_line(self, line_no: int) -> list[str]:
        return list(self.weight_marks.get(line_no, []))

    def injection_ids_for_calibration(self, layer: str) -> list[str]:
        return list(self.calibration_marks.get(layer, []))

    def reports_sorted(self) -> list[dict[str, Any]]:
        return [self.reports[key].as_dict() for key in sorted(self.reports)]

    def finalize_hits(self, failures: Iterable[FailureRecord]) -> None:
        by_injection: dict[str, list[FailureRecord]] = {}
        for record in failures:
            if record.injection_id:
                by_injection.setdefault(record.injection_id, []).append(record)
        for injection_id, records in by_injection.items():
            if injection_id in self.reports:
                first = sorted(
                    records,
                    key=lambda item: (item.stage or "", item.layer or "", item.code),
                )[0]
                self.mark_hit(injection_id, first.code, first.message)


def load_weight_records(path: str | Path) -> list[RawWeightRecord]:
    records: list[RawWeightRecord] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = __import__("json").loads(line)
            except Exception as exc:
                records.append(RawWeightRecord(line_no, parse_error=f"invalid JSON: {exc}"))
                continue
            if not isinstance(value, dict):
                records.append(RawWeightRecord(line_no, parse_error="weight-pack line must be a JSON object"))
                continue
            records.append(RawWeightRecord(line_no, data=value))
    return records


def _parse_weights_array(value: Any) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except Exception as exc:
        raise WeightPackError(
            E_WEIGHT_PACK_INVALID,
            f"weights are not a numeric array: {exc}",
            stage="validation",
        ) from exc
    return array


def _number_list_shape(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, list):
        return None
    if not value:
        return (0,)
    if all(not isinstance(item, list) for item in value):
        return (len(value),)
    child_shapes = [_number_list_shape(item) for item in value]
    if any(shape is None for shape in child_shapes):
        return None
    first = child_shapes[0]
    if any(shape != first for shape in child_shapes):
        return None
    return (len(value),) + first  # type: ignore[operator]


def _content_fingerprint(
    name: str,
    shape: tuple[int, ...],
    weights: np.ndarray,
    version: int,
    marker_group_size: int,
) -> str:
    header = sha256_json(
        {
            "name": name,
            "shape": list(shape),
            "version": version,
            "marker_group_size": marker_group_size,
            "dtype": "float64-le",
        }
    )
    payload = header.encode("ascii") + np.asarray(weights, dtype="<f8").tobytes(order="C")
    return sha256_bytes(payload)


def validate_weight_pack(
    records: list[RawWeightRecord],
    config: PipelineConfig,
    injection: InjectionEngine,
) -> tuple[dict[str, WeightLayer], list[FailureRecord]]:
    failures: list[FailureRecord] = []
    layers: dict[str, WeightLayer] = {}
    seen_names: set[str] = set()
    last_version: int | None = None

    for record in records:
        injected_ids = list(injection.injection_ids_for_weight_line(record.line_no))
        candidate_name_value = (
            record.data.get("layer", record.data.get("name"))
            if isinstance(record.data, dict)
            else None
        )
        if isinstance(candidate_name_value, str):
            for injection_id in injection.injection_ids_for_config_layer(candidate_name_value):
                if injection_id not in injected_ids:
                    injected_ids.append(injection_id)
        injection_id = injected_ids[0] if injected_ids else None
        if record.parse_error is not None:
            failures.append(
                failure(
                    E_WEIGHT_PACK_INVALID,
                    record.parse_error,
                    stage="validation",
                    exception_type="JSONDecodeError" if "JSON" in record.parse_error else "WeightPackError",
                    injection_id=injection_id,
                    details={"line": record.line_no},
                )
            )
            continue
        assert record.data is not None
        data = record.data
        name_value = data.get("layer", data.get("name"))
        name = str(name_value) if name_value is not None else None
        try:
            if not isinstance(name_value, str) or not name_value.strip():
                raise WeightPackError(
                    E_WEIGHT_PACK_INVALID,
                    f"line {record.line_no}: layer name must be a non-empty string",
                    stage="validation",
                    layer=name,
                )
            if name in seen_names:
                raise WeightPackError(
                    E_WEIGHT_PACK_INVALID,
                    f"line {record.line_no}: duplicate layer name {name!r}",
                    stage="validation",
                    layer=name,
                )

            shape_raw = data.get("shape")
            if not isinstance(shape_raw, list) or not shape_raw:
                raise WeightPackError(
                    E_WEIGHT_PACK_INVALID,
                    f"line {record.line_no}: shape must be a non-empty integer array",
                    stage="validation",
                    layer=name,
                )
            if any(not is_int(dim) or dim <= 0 for dim in shape_raw):
                raise WeightPackError(
                    E_WEIGHT_PACK_INVALID,
                    f"line {record.line_no}: every shape dimension must be a positive integer",
                    stage="validation",
                    layer=name,
                    details={"shape": shape_raw},
                )
            shape = tuple(int(dim) for dim in shape_raw)
            numel = int(np.prod(shape, dtype=np.int64))

            version = data.get("version", data.get("layer_version", data.get("content_version")))
            if not is_int(version) or version < 0:
                raise WeightPackError(
                    E_WEIGHT_PACK_INVALID,
                    f"line {record.line_no}: version must be a non-negative integer",
                    stage="validation",
                    layer=name,
                )
            if last_version is not None and version < last_version:
                raise WeightPackError(
                    E_WEIGHT_PACK_INVALID,
                    f"line {record.line_no}: version {version} is lower than previous version {last_version}",
                    stage="validation",
                    layer=name,
                    details={"version": version, "previous_version": last_version},
                )
            last_version = version

            marker_group = data.get("group_size", data.get("group_granularity"))
            if not is_int(marker_group) or marker_group <= 0:
                raise WeightPackError(
                    E_WEIGHT_PACK_INVALID,
                    f"line {record.line_no}: group_size marker must be a positive integer",
                    stage="validation",
                    layer=name,
                )
            if numel % marker_group != 0:
                raise WeightPackError(
                    E_WEIGHT_PACK_INVALID,
                    f"line {record.line_no}: group_size {marker_group} does not divide {numel} elements",
                    stage="validation",
                    layer=name,
                    details={"group_size": marker_group, "numel": numel},
                )

            weights_raw = data.get("weights")
            if not isinstance(weights_raw, list):
                raise WeightPackError(
                    E_WEIGHT_PACK_INVALID,
                    f"line {record.line_no}: weights must be an array",
                    stage="validation",
                    layer=name,
                )
            array = _parse_weights_array(weights_raw)
            if array.size != numel:
                raise WeightPackError(
                    E_WEIGHT_PACK_INVALID,
                    f"line {record.line_no}: weight element count {array.size} does not match shape {shape} ({numel})",
                    stage="validation",
                    layer=name,
                    details={"expected": numel, "actual": int(array.size), "shape": list(shape)},
                )
            if array.ndim > 1:
                nested_shape = array.shape
                if nested_shape != shape:
                    raise WeightPackError(
                        E_WEIGHT_PACK_INVALID,
                        f"line {record.line_no}: nested weight shape {nested_shape} does not match declared shape {shape}",
                        stage="validation",
                        layer=name,
                        details={"declared_shape": list(shape), "actual_shape": list(nested_shape)},
                    )
            flat = np.asarray(array, dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(flat)):
                bad_indices = np.flatnonzero(~np.isfinite(flat)).astype(int).tolist()
                raise WeightPackError(
                    E_WEIGHT_PACK_INVALID,
                    f"line {record.line_no}: weights contain non-finite values",
                    stage="validation",
                    layer=name,
                    details={"non_finite_indices": bad_indices[:32]},
                )

            try:
                config.resolve_layer(name, numel, marker_group_size=marker_group)
            except ConfigError as exc:
                raise ConfigError(
                    exc.code,
                    exc.message,
                    stage="config",
                    layer=name,
                    exception_type=exc.exception_type,
                    injection_id=injection_id,
                    details=exc.details,
                ) from exc

            seen_names.add(name)
            layers[name] = WeightLayer(
                name=name,
                shape=shape,
                weights=flat,
                version=version,
                marker_group_size=marker_group,
                line_no=record.line_no,
                content_fingerprint=_content_fingerprint(name, shape, flat, version, marker_group),
            )
        except (WeightPackError, ConfigError) as exc:
            code = exc.code
            stage = exc.stage or "validation"
            failures.append(
                failure(
                    code,
                    exc.message,
                    stage=stage,
                    layer=exc.layer or name,
                    exception_type=exc.exception_type,
                    injection_id=injection_id,
                    details=exc.details,
                )
            )
    return layers, failures


def _number_sequence(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    result: list[float] = []
    for item in value:
        if not finite_float(item):
            return None
        result.append(float(item))
    return tuple(result)


def _find_layer_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def load_calibration(path: str | Path) -> tuple[CalibrationSet | None, list[FailureRecord]]:
    failures: list[FailureRecord] = []
    try:
        raw = read_json(path)
    except Exception as exc:
        return None, [
            failure(
                E_CALIBRATION_INVALID,
                f"cannot read calibration JSON: {exc}",
                stage="validation",
                exception_type=type(exc).__name__,
            )
        ]
    try:
        return normalize_calibration(raw), failures
    except CalibrationError as exc:
        failures.append(FailureRecord.from_exception(exc))
        return None, failures


def normalize_calibration(raw: Any) -> CalibrationSet:
    if not isinstance(raw, Mapping):
        raise CalibrationError(
            E_CALIBRATION_INVALID,
            "calibration root must be a JSON object",
            stage="validation",
        )
    sample_values = raw.get("samples")
    top_layer_inputs = _find_layer_mapping(
        raw.get("inputs_by_layer", raw.get("layer_inputs", raw.get("inputs", {})))
    )
    top_layer_refs = _find_layer_mapping(
        raw.get(
            "reference_outputs_by_layer",
            raw.get("layer_reference_outputs", raw.get("reference_outputs", raw.get("outputs", {}))),
        )
    )
    if sample_values is None and top_layer_inputs:
        lengths = {len(value) for value in top_layer_inputs.values() if isinstance(value, list)}
        lengths.update(
            len(value) for value in top_layer_refs.values() if isinstance(value, list)
        )
        if len(lengths) != 1:
            raise CalibrationError(
                E_CALIBRATION_INVALID,
                "top-level per-layer calibration arrays must contain the same number of samples",
                stage="validation",
            )
        count = lengths.pop() if lengths else 0
        sample_values = []
        for index in range(count):
            sample: dict[str, Any] = {
                "id": f"s{index:04d}",
                "inputs": {
                    layer: values[index]
                    for layer, values in top_layer_inputs.items()
                    if isinstance(values, list) and index < len(values)
                },
                "reference_outputs": {
                    layer: values[index]
                    for layer, values in top_layer_refs.items()
                    if isinstance(values, list) and index < len(values)
                },
            }
            sample_values.append(sample)
    if not isinstance(sample_values, list) or not sample_values:
        raise CalibrationError(
            E_CALIBRATION_INVALID,
            "calibration samples must be a non-empty JSON array",
            stage="validation",
        )

    samples: list[CalibrationSample] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(sample_values):
        if not isinstance(value, Mapping):
            raise CalibrationError(
                E_CALIBRATION_INVALID,
                f"calibration sample {index} must be an object",
                stage="validation",
            )
        sample_id = str(value.get("id", value.get("sample_id", f"s{index:04d}")))
        if sample_id in seen_ids:
            raise CalibrationError(
                E_CALIBRATION_INVALID,
                f"duplicate calibration sample id {sample_id!r}",
                stage="validation",
            )
        seen_ids.add(sample_id)

        input_map = _find_layer_mapping(
            value.get("inputs_by_layer", value.get("inputs", value.get("input_by_layer", {})))
        )
        ref_map = _find_layer_mapping(
            value.get(
                "reference_outputs_by_layer",
                value.get(
                    "reference_outputs",
                    value.get("reference_output", value.get("outputs", value.get("output", {}))),
                ),
            )
        )
        fallback_input = _number_sequence(value.get("input"))
        fallback_ref = _number_sequence(
            value.get("reference_output", value.get("reference", value.get("output")))
        )
        if isinstance(value.get("input"), Mapping):
            input_map = _find_layer_mapping(value.get("input"))
        if isinstance(value.get("reference_output"), Mapping):
            ref_map = _find_layer_mapping(value.get("reference_output"))

        layer_names = sorted(set(input_map) | set(ref_map))
        if not layer_names and fallback_input is not None:
            layer_names = ["*"]
            input_map = {"*": fallback_input}
            if fallback_ref is not None:
                ref_map = {"*": fallback_ref}

        normalized_inputs: dict[str, tuple[float, ...]] = {}
        normalized_refs: dict[str, tuple[float, ...]] = {}
        for layer in layer_names:
            input_value = input_map.get(layer, fallback_input)
            ref_value = ref_map.get(layer, fallback_ref)
            input_seq = _number_sequence(input_value)
            ref_seq = _number_sequence(ref_value)
            if input_seq is None:
                raise CalibrationError(
                    E_CALIBRATION_INVALID,
                    f"calibration sample {sample_id!r} layer {layer!r} input must contain finite numbers",
                    stage="validation",
                    details={"sample_id": sample_id, "layer": layer},
                )
            if ref_seq is None:
                raise CalibrationError(
                    E_CALIBRATION_INVALID,
                    f"calibration sample {sample_id!r} layer {layer!r} reference output must contain finite numbers",
                    stage="validation",
                    details={"sample_id": sample_id, "layer": layer},
                )
            normalized_inputs[layer] = input_seq
            normalized_refs[layer] = ref_seq
        samples.append(
            CalibrationSample(
                sample_id=sample_id,
                inputs=normalized_inputs,
                reference_outputs=normalized_refs,
            )
        )

    sample_ids = tuple(sample.sample_id for sample in samples)
    return CalibrationSet(
        samples=tuple(samples),
        raw_hash=sha256_json(_jsonable_calibration(raw)),
        sample_ids=sample_ids,
    )


def _jsonable_calibration(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable_calibration(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_calibration(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def build_layer_calibration_views(
    layers: Mapping[str, WeightLayer],
    calibration: CalibrationSet,
    injection: InjectionEngine,
) -> tuple[dict[str, LayerCalibrationView], list[FailureRecord]]:
    views: dict[str, LayerCalibrationView] = {}
    failures: list[FailureRecord] = []
    for layer_name in sorted(layers):
        layer = layers[layer_name]
        injection_ids = injection.injection_ids_for_calibration(layer_name)
        injection_id = injection_ids[0] if injection_ids else None
        try:
            inputs_raw, refs_raw = calibration.for_layer(layer_name)
            if not inputs_raw:
                raise CalibrationError(
                    E_CALIBRATION_INVALID,
                    f"layer {layer_name!r} has no calibration samples",
                    stage="calibration",
                    layer=layer_name,
                )
            input_arrays: list[np.ndarray] = []
            ref_arrays: list[np.ndarray] = []
            for sample_id, input_raw, ref_raw in zip(
                calibration.sample_ids, inputs_raw, refs_raw, strict=True
            ):
                input_array = np.asarray(input_raw, dtype=np.float64)
                ref_array = np.asarray(ref_raw, dtype=np.float64)
                if input_array.size != layer.input_dim:
                    raise CalibrationError(
                        E_CALIBRATION_INVALID,
                        f"calibration sample {sample_id!r} input dimension {input_array.size} does not match layer input dimension {layer.input_dim}",
                        stage="calibration",
                        layer=layer_name,
                        details={"sample_id": sample_id, "expected": layer.input_dim, "actual": int(input_array.size)},
                    )
                if ref_array.size != layer.output_dim:
                    raise CalibrationError(
                        E_CALIBRATION_INVALID,
                        f"calibration sample {sample_id!r} reference dimension {ref_array.size} does not match layer output dimension {layer.output_dim}",
                        stage="calibration",
                        layer=layer_name,
                        details={"sample_id": sample_id, "expected": layer.output_dim, "actual": int(ref_array.size)},
                    )
                if not np.all(np.isfinite(input_array)) or not np.all(np.isfinite(ref_array)):
                    raise CalibrationError(
                        E_CALIBRATION_INVALID,
                        f"calibration sample {sample_id!r} contains non-finite values for layer {layer_name!r}",
                        stage="calibration",
                        layer=layer_name,
                    )
                input_arrays.append(input_array)
                ref_arrays.append(ref_array)
            views[layer_name] = LayerCalibrationView(
                layer=layer_name,
                sample_ids=calibration.sample_ids,
                inputs=tuple(input_arrays),
                references=tuple(ref_arrays),
            )
        except CalibrationError as exc:
            failures.append(
                failure(
                    exc.code,
                    exc.message,
                    stage=exc.stage or "calibration",
                    layer=layer_name,
                    exception_type=exc.exception_type,
                    injection_id=injection_id,
                    details=exc.details,
                )
            )
    return views, failures


def load_injections(path: str | Path | None) -> tuple[list[InjectionItem], list[FailureRecord]]:
    if path is None:
        return [], []
    path = Path(path)
    if not path.exists():
        return [], [
            failure(
                E_INJECTION_INVALID,
                f"injection configuration does not exist: {path}",
                stage="injection",
                exception_type="FileNotFoundError",
            )
        ]
    try:
        raw = load_yaml(path)
    except Exception as exc:
        return [], [
            failure(
                E_INJECTION_INVALID,
                f"cannot load injection configuration: {exc}",
                stage="injection",
                exception_type=type(exc).__name__,
            )
        ]
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        return [], [
            failure(
                E_INJECTION_INVALID,
                "injection configuration root must be a mapping",
                stage="injection",
            )
        ]
    values = raw.get("injections", raw.get("items", []))
    if values is None:
        values = []
    if not isinstance(values, list):
        return [], [
            failure(
                E_INJECTION_INVALID,
                "injections must be a sequence",
                stage="injection",
            )
        ]
    items: list[InjectionItem] = []
    failures: list[FailureRecord] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            failures.append(
                failure(
                    E_INJECTION_INVALID,
                    f"injection at index {index} must be a mapping",
                    stage="injection",
                )
            )
            continue
        injection_id = str(value.get("id", value.get("injection_id", f"INJ-{index + 1:03d}")))
        fault_type = value.get("type", value.get("kind"))
        if injection_id in seen_ids:
            failures.append(
                failure(
                    E_INJECTION_INVALID,
                    f"duplicate injection id {injection_id!r}",
                    stage="injection",
                    injection_id=injection_id,
                )
            )
            continue
        seen_ids.add(injection_id)
        if not isinstance(fault_type, str) or fault_type not in INJECTION_TYPES:
            failures.append(
                failure(
                    E_INJECTION_INVALID,
                    f"unsupported injection type for {injection_id!r}: {fault_type!r}",
                    stage="injection",
                    injection_id=injection_id,
                    details={"supported_types": sorted(INJECTION_TYPES)},
                )
            )
            continue
        layer_value = value.get("layer")
        layer = str(layer_value) if layer_value is not None else None
        if fault_type in {
            "non_finite_weight",
            "group_indivisible",
            "bit_width_out_of_range",
            "calibration_dimension_mismatch",
            "duplicate_layer_name",
        } and not layer:
            failures.append(
                failure(
                    E_INJECTION_INVALID,
                    f"injection {injection_id!r} requires a layer",
                    stage="injection",
                    injection_id=injection_id,
                )
            )
            continue
        stage = value.get("stage")
        params = value.get("params", {})
        if not isinstance(params, Mapping):
            failures.append(
                failure(
                    E_INJECTION_INVALID,
                    f"injection {injection_id!r} params must be a mapping",
                    stage="injection",
                    injection_id=injection_id,
                )
            )
            continue
        items.append(
            InjectionItem(
                injection_id=injection_id,
                fault_type=fault_type,
                layer=layer,
                stage=str(stage) if stage is not None else None,
                params=dict(params),
                expected_code=INJECTION_TYPES[fault_type],
            )
        )
    return items, failures


def _set_first_number_nonfinite(value: Any) -> bool:
    if isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                value[index] = math.nan
                return True
            if _set_first_number_nonfinite(item):
                return True
    return False


def apply_injections(
    items: list[InjectionItem],
    records: list[RawWeightRecord],
    config_raw: dict[str, Any],
    calibration_raw: Any,
) -> InjectionEngine:
    engine = InjectionEngine(items=list(items))
    by_name: dict[str, RawWeightRecord] = {}
    for record in records:
        if record.data and isinstance(record.data.get("layer", record.data.get("name")), str):
            by_name[str(record.data.get("layer", record.data.get("name")))] = record

    def layer_config_for(name: str) -> dict[str, Any]:
        layers_value = config_raw.get("layers")
        if isinstance(layers_value, dict):
            body = layers_value.setdefault(name, {})
            if not isinstance(body, dict):
                body = {}
                layers_value[name] = body
            return body
        if isinstance(layers_value, list):
            for item in layers_value:
                if isinstance(item, dict) and str(item.get("name", item.get("layer"))) == name:
                    return item
            body: dict[str, Any] = {"name": name}
            layers_value.append(body)
            return body
        body = {}
        config_raw["layers"] = {name: body}
        return body

    for item in items:
        if item.layer and item.layer not in by_name:
            engine.failures.append(
                failure(
                    E_INJECTION_INVALID,
                    f"injection {item.injection_id!r} targets missing layer {item.layer!r}",
                    stage="injection",
                    layer=item.layer,
                    injection_id=item.injection_id,
                )
            )
            continue
        if item.fault_type == "non_finite_weight":
            record = by_name[item.layer]  # type: ignore[index]
            assert record.data is not None
            weights = record.data.get("weights")
            changed = _set_first_number_nonfinite(weights)
            if not changed:
                engine.failures.append(
                    failure(
                        E_INJECTION_INVALID,
                        f"injection {item.injection_id!r} could not find a numeric weight",
                        stage="injection",
                        layer=item.layer,
                        injection_id=item.injection_id,
                    )
                )
                continue
            engine.mark_weight(record.line_no, item.injection_id)
            engine.mark_applied(item.injection_id, f"replaced first weight of {item.layer} with NaN")
        elif item.fault_type == "duplicate_layer_name":
            record = by_name[item.layer]  # type: ignore[index]
            assert record.data is not None
            duplicate = RawWeightRecord(
                line_no=max((entry.line_no for entry in records), default=0) + 1,
                data=copy.deepcopy(record.data),
                faults=[item.injection_id],
            )
            records.append(duplicate)
            engine.mark_weight(duplicate.line_no, item.injection_id)
            engine.mark_applied(item.injection_id, f"appended a duplicate line for {item.layer}")
        elif item.fault_type == "group_indivisible":
            record = by_name[item.layer]  # type: ignore[index]
            assert record.data is not None
            shape = record.data.get("shape", [])
            try:
                numel = int(np.prod([int(dim) for dim in shape], dtype=np.int64))
            except Exception:
                numel = 1
            bad_group = int(item.params.get("group_size", max(1, numel - 1)))
            if bad_group <= 0 or numel % bad_group == 0:
                bad_group = numel + 1
            record.data["group_size"] = bad_group
            layer_cfg = layer_config_for(item.layer)  # type: ignore[arg-type]
            layer_cfg["group_size"] = bad_group
            engine.mark_weight(record.line_no, item.injection_id)
            engine.mark_config(f"layer:{item.layer}", item.injection_id)
            engine.mark_applied(item.injection_id, f"set group_size={bad_group} for {item.layer}")
        elif item.fault_type == "bit_width_out_of_range":
            bad_bits = int(item.params.get("target_bits", 99))
            layer_cfg = layer_config_for(item.layer)  # type: ignore[arg-type]
            layer_cfg["target_bits"] = bad_bits
            engine.mark_config(f"layer:{item.layer}", item.injection_id)
            engine.mark_applied(item.injection_id, f"set target_bits={bad_bits} for {item.layer}")
        elif item.fault_type == "calibration_dimension_mismatch":
            changed = _mutate_calibration_dimension(calibration_raw, item.layer)  # type: ignore[arg-type]
            if not changed:
                engine.failures.append(
                    failure(
                        E_INJECTION_INVALID,
                        f"injection {item.injection_id!r} could not locate calibration input for {item.layer!r}",
                        stage="injection",
                        layer=item.layer,
                        injection_id=item.injection_id,
                    )
                )
                continue
            engine.mark_calibration(item.layer, item.injection_id)  # type: ignore[arg-type]
            engine.mark_applied(item.injection_id, f"appended one dimension to calibration input for {item.layer}")
        elif item.fault_type == "zero_temperature":
            distill = config_raw.get("distillation", config_raw.get("distill"))
            if not isinstance(distill, dict):
                distill = config_raw.setdefault("distillation", {})
            if not isinstance(distill, dict):
                continue
            distill["temperature"] = 0
            engine.mark_config("distillation", item.injection_id)
            engine.mark_applied(item.injection_id, "set distillation temperature to zero")
    return engine


def _mutate_calibration_dimension(calibration_raw: Any, layer: str) -> bool:
    if not isinstance(calibration_raw, Mapping):
        return False
    samples = calibration_raw.get("samples")
    if isinstance(samples, list):
        for sample in samples:
            if not isinstance(sample, Mapping):
                continue
            inputs = sample.get("inputs", sample.get("inputs_by_layer"))
            if isinstance(inputs, Mapping) and layer in inputs and isinstance(inputs[layer], list):
                inputs[layer].append(1.0)
                return True
            input_value = sample.get("input")
            if isinstance(input_value, Mapping) and layer in input_value and isinstance(input_value[layer], list):
                input_value[layer].append(1.0)
                return True
            if isinstance(input_value, list) and len(samples) == 1:
                input_value.append(1.0)
                return True
    top_inputs = calibration_raw.get("inputs_by_layer", calibration_raw.get("layer_inputs"))
    if isinstance(top_inputs, Mapping) and layer in top_inputs and isinstance(top_inputs[layer], list):
        values = top_inputs[layer]
        if values and isinstance(values[0], list):
            values[0].append(1.0)
            return True
    return False


def weight_pack_fingerprint(layers: Mapping[str, WeightLayer]) -> str:
    return sha256_json(
        {
            name: {
                "content": layers[name].content_fingerprint,
                "shape": list(layers[name].shape),
                "version": layers[name].version,
            }
            for name in sorted(layers)
        }
    )
