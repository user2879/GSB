"""Weight package, calibration set and configuration validation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import (
    CompressionConfig,
    load_config,
    validate_config_against_layers,
)
from .errors import (
    E_CALIB_MISSING,
    E_NONFINITE_WEIGHT,
    E_WEIGHT_SHAPE_MISMATCH,
    Failure,
    failure,
    sort_failures,
)
from .fingerprint import hash_file, sha256_array, sha256_json
from .model import CalibrationSample, CalibrationSet, Layer, LoadedState


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _layer_fingerprint(
    name: str, shape: tuple[int, ...], version: int, weights: np.ndarray
) -> str:
    return sha256_json(
        {
            "kind": "layer-content-v1",
            "name": name,
            "shape": list(shape),
            "content_version": version,
            "weights_sha256": sha256_array(weights, dtype="<f8"),
        }
    )


def make_layer(
    name: str,
    shape: Iterable[int],
    weights: Any,
    content_version: int,
    *,
    source_index: int = 0,
) -> Layer:
    """Build a validated in-memory layer, used by interactive replacement."""

    normalized_shape = tuple(int(item) for item in shape)
    if not normalized_shape or any(item <= 0 for item in normalized_shape):
        raise ValueError("shape must contain positive integers")
    array = np.asarray(weights, dtype=np.float64)
    if array.size != int(math.prod(normalized_shape)):
        raise ValueError("weight count does not match shape")
    if not np.all(np.isfinite(array)):
        raise ValueError("weights must be finite")
    array = np.ascontiguousarray(array.reshape(normalized_shape), dtype=np.float64)
    return Layer(
        name=name,
        shape=normalized_shape,
        weights=array,
        content_version=int(content_version),
        content_fingerprint=_layer_fingerprint(
            name, normalized_shape, int(content_version), array
        ),
        source_index=source_index,
    )


def load_weight_package(path: str | Path) -> tuple[dict[str, Layer], list[Failure]]:
    """Parse JSONL and return every independently valid layer plus failures."""

    file_path = Path(path)
    layers: dict[str, Layer] = {}
    issues: list[Failure] = []
    if not file_path.is_file():
        issues.append(
            failure(
                E_WEIGHT_SHAPE_MISMATCH,
                f"weight package does not exist: {file_path}",
                detail={"path": str(file_path)},
            )
        )
        return layers, issues

    previous_version: int | None = None
    record_count = 0
    try:
        stream = file_path.open("r", encoding="utf-8")
    except OSError as exc:
        issues.append(
            failure(
                E_WEIGHT_SHAPE_MISMATCH,
                f"weight package cannot be opened: {exc}",
                detail={"path": str(file_path)},
            )
        )
        return layers, issues

    with stream:
        for line_number, raw_line in enumerate(stream, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            record_count += 1
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                issues.append(
                    failure(
                        E_WEIGHT_SHAPE_MISMATCH,
                        f"invalid JSON on weight line {line_number}: {exc}",
                        detail={"line": line_number},
                    )
                )
                continue
            if not isinstance(record, dict):
                issues.append(
                    failure(
                        E_WEIGHT_SHAPE_MISMATCH,
                        f"weight line {line_number} must be a JSON object",
                        detail={"line": line_number},
                    )
                )
                continue

            name = record.get("name", record.get("layer", record.get("layer_name")))
            if not isinstance(name, str) or not name:
                issues.append(
                    failure(
                        E_WEIGHT_SHAPE_MISMATCH,
                        f"weight line {line_number} has no non-empty layer name",
                        detail={"line": line_number},
                    )
                )
                continue
            if name in layers:
                issues.append(
                    failure(
                        E_WEIGHT_SHAPE_MISMATCH,
                        f"duplicate layer name: {name}",
                        layer=name,
                        detail={"line": line_number},
                    )
                )
                continue

            shape_value = record.get("shape", record.get("dims", record.get("dimensions")))
            if (
                not isinstance(shape_value, list)
                or not shape_value
                or any(
                    not isinstance(dimension, int)
                    or isinstance(dimension, bool)
                    or dimension <= 0
                    for dimension in shape_value
                )
            ):
                issues.append(
                    failure(
                        E_WEIGHT_SHAPE_MISMATCH,
                        f"layer {name} shape must be a non-empty list of positive integers",
                        layer=name,
                        detail={"line": line_number, "shape": shape_value},
                    )
                )
                continue
            shape = tuple(int(dimension) for dimension in shape_value)

            version = record.get(
                "content_version",
                record.get(
                    "version",
                    record.get(
                        "layer_content_version", record.get("layer_version")
                    ),
                ),
            )
            if not isinstance(version, int) or isinstance(version, bool) or version < 0:
                issues.append(
                    failure(
                        E_WEIGHT_SHAPE_MISMATCH,
                        f"layer {name} content version must be a non-negative integer",
                        layer=name,
                        detail={"line": line_number},
                    )
                )
                continue
            if previous_version is not None and version < previous_version:
                issues.append(
                    failure(
                        E_WEIGHT_SHAPE_MISMATCH,
                        f"layer {name} content version {version} is not monotonic "
                        f"after {previous_version}",
                        layer=name,
                        detail={"line": line_number},
                    )
                )
                continue

            values = record.get(
                "weights", record.get("weight", record.get("values", record.get("data")))
            )
            if values is None:
                issues.append(
                    failure(
                        E_WEIGHT_SHAPE_MISMATCH,
                        f"layer {name} has no weights array",
                        layer=name,
                        detail={"line": line_number},
                    )
                )
                continue
            try:
                array = np.asarray(values, dtype=np.float64)
            except (TypeError, ValueError):
                issues.append(
                    failure(
                        E_WEIGHT_SHAPE_MISMATCH,
                        f"layer {name} weights are not a rectangular numeric array",
                        layer=name,
                        detail={"line": line_number},
                    )
                )
                continue

            if not np.all(np.isfinite(array)):
                issues.append(
                    failure(
                        E_NONFINITE_WEIGHT,
                        f"layer {name} contains a non-finite weight",
                        layer=name,
                        detail={"line": line_number},
                    )
                )
                continue

            expected_size = int(math.prod(shape))
            if array.size != expected_size:
                issues.append(
                    failure(
                        E_WEIGHT_SHAPE_MISMATCH,
                        f"layer {name} shape {list(shape)} requires {expected_size} values "
                        f"but received {array.size}",
                        layer=name,
                        detail={"line": line_number},
                    )
                )
                continue
            if array.shape != shape and array.shape != (expected_size,):
                issues.append(
                    failure(
                        E_WEIGHT_SHAPE_MISMATCH,
                        f"layer {name} nested weight shape {list(array.shape)} does not "
                        f"match declared shape {list(shape)}",
                        layer=name,
                        detail={"line": line_number},
                    )
                )
                continue
            array = np.ascontiguousarray(array.reshape(shape), dtype=np.float64)
            fingerprint = _layer_fingerprint(name, shape, version, array)
            layers[name] = Layer(
                name=name,
                shape=shape,
                weights=array,
                content_version=version,
                content_fingerprint=fingerprint,
                source_index=line_number,
            )
            previous_version = version

    if not layers and record_count == 0:
        issues.append(
            failure(
                E_WEIGHT_SHAPE_MISMATCH,
                "weight package contains no records",
                detail={"path": str(file_path)},
            )
        )

    return layers, issues


def _numeric_vector(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, list) or not value:
        return None
    if any(not _is_number(item) for item in value):
        return None
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in result):
        return None
    return result


def load_calibration_set(path: str | Path) -> CalibrationSet:
    """Load calibration samples. Any unusable input yields E_CALIB_MISSING."""

    file_path = Path(path)
    issues: list[Failure] = []
    if not file_path.is_file():
        issues.append(
            failure(
                E_CALIB_MISSING,
                f"calibration set does not exist: {file_path}",
                detail={"path": str(file_path)},
            )
        )
        return CalibrationSet([], sha256_json({"missing": str(file_path)}), issues)
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(
            failure(
                E_CALIB_MISSING,
                f"calibration set cannot be parsed: {exc}",
                detail={"path": str(file_path)},
            )
        )
        return CalibrationSet([], sha256_json({"invalid": str(file_path)}), issues)

    records = (
        raw.get("samples", raw.get("calibration_samples", raw.get("data")))
        if isinstance(raw, dict)
        else raw
    )
    if not isinstance(records, list) or not records:
        issues.append(
            failure(
                E_CALIB_MISSING,
                "calibration set must contain at least one sample",
                detail={"path": str(file_path)},
            )
        )
        return CalibrationSet([], sha256_json({"empty": str(file_path)}), issues)

    samples: list[CalibrationSample] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            issues.append(
                failure(
                    E_CALIB_MISSING,
                    f"calibration sample {index} must be an object",
                    detail={"index": index},
                )
            )
            continue
        sample_id = record.get("id", record.get("sample_id", record.get("name")))
        if not isinstance(sample_id, str) or not sample_id:
            issues.append(
                failure(
                    E_CALIB_MISSING,
                    f"calibration sample {index} has no non-empty id",
                    detail={"index": index},
                )
            )
            continue
        if sample_id in seen_ids:
            issues.append(
                failure(
                    E_CALIB_MISSING,
                    f"duplicate calibration sample id: {sample_id}",
                    detail={"sample_id": sample_id},
                )
            )
            continue
        inputs = _numeric_vector(
            record.get("input", record.get("inputs", record.get("input_vector")))
        )
        if inputs is None:
            issues.append(
                failure(
                    E_CALIB_MISSING,
                    f"calibration sample {sample_id} input must be a finite numeric vector",
                    detail={"sample_id": sample_id},
                )
            )
            continue
        reference = _numeric_vector(
            record.get(
                "reference_output",
                record.get(
                    "reference",
                    record.get("output", record.get("target")),
                ),
            )
        )
        if reference is None:
            issues.append(
                failure(
                    E_CALIB_MISSING,
                    f"calibration sample {sample_id} reference_output must be a finite numeric vector",
                    detail={"sample_id": sample_id},
                )
            )
            continue
        layers_value = record.get("layers", record.get("target_layers", []))
        if layers_value in (None, "all"):
            layers_value = [] if layers_value is None else ["all"]
        if not isinstance(layers_value, list) or any(
            not isinstance(item, str) or not item for item in layers_value
        ):
            issues.append(
                failure(
                    E_CALIB_MISSING,
                    f"calibration sample {sample_id} layers must be a string list",
                    detail={"sample_id": sample_id},
                )
            )
            continue
        samples.append(
            CalibrationSample(
                sample_id=sample_id,
                inputs=inputs,
                reference_output=reference,
                layers=tuple(layers_value),
                source_index=index,
            )
        )
        seen_ids.add(sample_id)

    if not samples and not issues:
        issues.append(
            failure(
                E_CALIB_MISSING,
                "calibration set has no valid samples",
                detail={"path": str(file_path)},
            )
        )
    ordered_samples = sorted(samples, key=lambda sample: sample.sample_id)
    fingerprint_payload = [sample.to_input_dict() for sample in ordered_samples]
    return CalibrationSet(
        samples=ordered_samples,
        fingerprint=sha256_json(
            {"kind": "calibration-set-v1", "samples": fingerprint_payload}
        ),
        issues=issues,
    )


def validate_calibration_against_layers(
    calibration: CalibrationSet, layers: dict[str, Layer]
) -> list[Failure]:
    issues: list[Failure] = list(calibration.issues)
    for sample in calibration.samples:
        if not sample.layers:
            targets = sorted(layers)
        else:
            targets = [name for name in sample.layers if name != "all"]
            if "all" in sample.layers:
                targets = sorted(layers)
            for name in targets:
                if name not in layers:
                    issues.append(
                        failure(
                            E_CALIB_MISSING,
                            f"calibration sample {sample.sample_id} references unknown layer {name}",
                            layer=name,
                            detail={"sample_id": sample.sample_id},
                        )
                    )
        for name in targets:
            layer = layers.get(name)
            if layer is None:
                continue
            expected_input = layer.shape[-1]
            if len(sample.inputs) != expected_input:
                issues.append(
                    failure(
                        E_CALIB_MISSING,
                        f"calibration sample {sample.sample_id} input length "
                        f"{len(sample.inputs)} does not match layer {name} last dimension "
                        f"{expected_input}",
                        layer=name,
                        detail={"sample_id": sample.sample_id},
                    )
                )
    return issues


def load_inputs(
    weights_path: str | Path,
    calibration_path: str | Path,
    config_path: str | Path,
) -> tuple[LoadedState, CompressionConfig]:
    """Load and cross-validate all three deterministic inputs."""

    config = load_config(config_path)
    layers, weight_issues = load_weight_package(weights_path)
    calibration = load_calibration_set(calibration_path)
    failures: list[Failure] = []
    failures.extend(weight_issues)
    failures.extend(calibration.issues)
    failures.extend(validate_config_against_layers(config, layers))
    failures.extend(validate_calibration_against_layers(calibration, layers))

    weights_hash = hash_file(str(weights_path)) if Path(weights_path).is_file() else ""
    calibration_hash = calibration.fingerprint
    return (
        LoadedState(
            layers=layers,
            calibration=calibration,
            failures=sort_failures(failures),
            weights_hash=weights_hash,
            calibration_hash=calibration_hash,
        ),
        config,
    )


def write_weight_package(path: str | Path, layers: Iterable[Layer]) -> None:
    """Write a normalized JSONL package, used by interactive session export."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for layer in layers:
            stream.write(json.dumps(layer.to_input_dict(), ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def write_calibration_set(path: str | Path, calibration: CalibrationSet) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "samples": [sample.to_input_dict() for sample in calibration.samples],
    }
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
