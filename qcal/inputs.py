"""Deterministic input loading and config normalization."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .codes import E_SAMPLE_PARSE, E_WEIGHT_NONFINITE, E_WEIGHT_SCHEMA
from .io_utils import (
    array_fingerprint,
    combined_fingerprint,
    load_json,
    load_jsonl,
    load_yaml,
    sha256_json,
)
from .models import FailureRecord, WeightPack


def _as_shape(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, list) or not value:
        return None
    shape: list[int] = []
    for item in value:
        try:
            dimension = int(item)
        except (TypeError, ValueError):
            return None
        if dimension <= 0:
            return None
        shape.append(dimension)
    return tuple(shape)


def _nested_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _weight_injection(
    injections: Iterable[dict[str, Any]], layer: str
) -> dict[str, Any] | None:
    for item in injections:
        if item.get("type") == "non_finite_weight" and item.get("layer") == layer:
            return item
    return None


def load_weights(
    path: str | Path,
    injections: Iterable[dict[str, Any]] = (),
) -> tuple[dict[str, WeightPack], list[FailureRecord], set[str]]:
    injections = list(injections)
    failures: list[FailureRecord] = []
    packs: dict[str, WeightPack] = {}
    seen_layers: set[str] = set()
    try:
        rows = load_jsonl(path)
    except (OSError, ValueError) as exc:
        failure = FailureRecord(
            layer="<weights>",
            stage="parse_validation",
            code=E_WEIGHT_SCHEMA,
            message=f"unable to read weight JSONL: {exc}",
        )
        return {}, [failure], set()

    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            failures.append(
                FailureRecord(
                    layer=f"<weights-row-{row_index}>",
                    stage="parse_validation",
                    code=E_WEIGHT_SCHEMA,
                    message="weight row must be an object",
                )
            )
            continue
        layer = str(row.get("layer", row.get("name", ""))).strip()
        if not layer:
            failures.append(
                FailureRecord(
                    layer=f"<weights-row-{row_index}>",
                    stage="parse_validation",
                    code=E_WEIGHT_SCHEMA,
                    message="weight layer name is missing",
                )
            )
            continue
        seen_layers.add(layer)
        if layer in packs:
            failures.append(
                FailureRecord(
                    layer=layer,
                    stage="parse_validation",
                    code=E_WEIGHT_SCHEMA,
                    message="duplicate weight layer name",
                )
            )
            continue
        shape = _as_shape(row.get("shape"))
        if shape is None:
            failures.append(
                FailureRecord(
                    layer=layer,
                    stage="parse_validation",
                    code=E_WEIGHT_SCHEMA,
                    message="shape must be a non-empty array of positive integers",
                )
            )
            continue
        try:
            weights = _nested_array(row["weights"])
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(
                FailureRecord(
                    layer=layer,
                    stage="parse_validation",
                    code=E_WEIGHT_SCHEMA,
                    message=f"weights are not a rectangular float array: {exc}",
                )
            )
            continue
        if weights.size != int(np.prod(shape)):
            failures.append(
                FailureRecord(
                    layer=layer,
                    stage="parse_validation",
                    code=E_WEIGHT_SCHEMA,
                    message=f"shape {shape} requires {int(np.prod(shape))} values, got {weights.size}",
                )
            )
            continue
        weights = weights.reshape(shape)
        content_version = str(row.get("content_version", row.get("version", "1")))
        injection = _weight_injection(injections, layer)
        injection_id = None
        if injection:
            injection_id = str(injection.get("id", "INJ-UNKNOWN"))
            index = injection.get("index")
            value = float(injection.get("value", "nan"))
            if index is None:
                flat_index = int(injection.get("flat_index", 0))
            else:
                flat_index = int(np.ravel_multi_index(tuple(int(i) for i in index), shape))
            flat = weights.reshape(-1)
            if flat_index < 0 or flat_index >= flat.size:
                failures.append(
                    FailureRecord(
                        layer=layer,
                        stage="parse_validation",
                        code=E_WEIGHT_SCHEMA,
                        message="injection index is out of bounds",
                        injection_id=injection_id,
                    )
                )
                continue
            flat[flat_index] = value
        if not np.all(np.isfinite(weights)):
            failures.append(
                FailureRecord(
                    layer=layer,
                    stage="parse_validation",
                    code=E_WEIGHT_NONFINITE,
                    message="weight package contains NaN or infinity",
                    injection_id=injection_id,
                )
            )
            continue
        packs[layer] = WeightPack(
            layer=layer,
            shape=shape,
            weights=np.ascontiguousarray(weights, dtype=np.float64),
            content_version=content_version,
            content_fingerprint=combined_fingerprint(array_fingerprint(weights), content_version),
        )
    return packs, failures, seen_layers


def load_samples(path: str | Path) -> tuple[list[dict[str, Any]], str, list[FailureRecord]]:
    failures: list[FailureRecord] = []
    try:
        data = load_json(path)
    except (OSError, ValueError) as exc:
        return [], "<missing>", [
            FailureRecord(
                layer="<calibration-samples>",
                stage="parse_validation",
                code=E_SAMPLE_PARSE,
                message=f"unable to read sample set: {exc}",
            )
        ]
    rows = data.get("samples") if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        return [], "<invalid>", [
            FailureRecord(
                layer="<calibration-samples>",
                stage="parse_validation",
                code=E_SAMPLE_PARSE,
                message="sample set must contain a non-empty samples array",
            )
        ]
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            failures.append(
                FailureRecord(
                    layer=f"<sample-{index}>",
                    stage="parse_validation",
                    code=E_SAMPLE_PARSE,
                    message="sample must be an object",
                )
            )
            continue
        sample_id = str(row.get("id", row.get("sample_id", ""))).strip()
        if not sample_id or sample_id in seen:
            failures.append(
                FailureRecord(
                    layer=sample_id or f"<sample-{index}>",
                    stage="parse_validation",
                    code=E_SAMPLE_PARSE,
                    message="sample ID is missing or duplicated",
                )
            )
            continue
        try:
            input_vector = np.asarray(row["input"], dtype=np.float64).reshape(-1)
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(
                FailureRecord(
                    layer=sample_id,
                    stage="parse_validation",
                    code=E_SAMPLE_PARSE,
                    message=f"sample input is invalid: {exc}",
                )
            )
            continue
        if input_vector.size == 0 or not np.all(np.isfinite(input_vector)):
            failures.append(
                FailureRecord(
                    layer=sample_id,
                    stage="parse_validation",
                    code=E_SAMPLE_PARSE,
                    message="sample input must be non-empty and finite",
                )
            )
            continue
        reference = row.get("reference", row.get("reference_output", {}))
        if not isinstance(reference, dict):
            reference = {}
        seen.add(sample_id)
        samples.append(
            {
                "id": sample_id,
                "input": [float(value) for value in input_vector.tolist()],
                "reference": reference,
            }
        )
    samples.sort(key=lambda item: item["id"])
    fingerprint = sha256_json(
        {
            "schema_version": "qcal-samples/1.0",
            "samples": samples,
        }
    )
    return samples, fingerprint, failures


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": "qcal-config/1.0",
    "exchange_version": "v1",
    "calibration": {
        "quantile_lower": 0.02,
        "quantile_upper": 0.98,
        "interpolation": "linear",
        "tie_break": "lower_index",
        "scheme": "asymmetric",
        "per_channel": False,
        "group_size": 0,
    },
    "fixed_point": {
        "fraction_bits": 10,
        "rounding_mode": "nearest_even",
        "saturation": "saturate",
    },
    "distillation": {"temperature": 1.5, "soft_label_weight": 0.35},
    "scheduler": {"workers": 4, "quant_points_budget": 0},
    "snapshot": {"commit_every_stages": 1},
    "layers": {},
    "full_precision_layers": [],
}


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be an object")
    config = _deep_merge(DEFAULT_CONFIG, raw)
    config["exchange_version"] = str(config.get("exchange_version", "v1")).lower()
    if config["exchange_version"] not in ("v1", "v2"):
        raise ValueError("exchange_version must be v1 or v2")
    calibration = config["calibration"]
    lower = float(calibration["quantile_lower"])
    upper = float(calibration["quantile_upper"])
    if not (math.isfinite(lower) and math.isfinite(upper) and 0.0 <= lower <= upper <= 1.0):
        raise ValueError("quantile bounds must be finite and satisfy 0 <= lower <= upper <= 1")
    calibration["quantile_lower"] = lower
    calibration["quantile_upper"] = upper
    config["full_precision_layers"] = sorted(set(str(v) for v in config.get("full_precision_layers", [])))
    layers: dict[str, Any] = {}
    default_bits = int(config.get("default_bit_width", 8))
    for layer, value in sorted(dict(config.get("layers", {})).items()):
        layer_config = copy.deepcopy(value) if isinstance(value, dict) else {}
        layer_config.setdefault("bit_width", default_bits)
        layer_config.setdefault(
            "group_size", int(calibration.get("group_size", 0) or 0)
        )
        layer_config.setdefault(
            "group_size_candidates", [int(layer_config["group_size"])]
        )
        layer_config.setdefault("scheme", calibration.get("scheme", "asymmetric"))
        layer_config.setdefault("per_channel", bool(calibration.get("per_channel", False)))
        layer_config["bit_width"] = int(layer_config["bit_width"])
        layer_config["group_size"] = int(layer_config["group_size"])
        layer_config["group_size_candidates"] = sorted(
            set(int(item) for item in layer_config["group_size_candidates"])
        )
        layers[str(layer)] = layer_config
    config["layers"] = layers
    config["fixed_point"]["fraction_bits"] = int(config["fixed_point"]["fraction_bits"])
    config["scheduler"]["workers"] = max(1, int(config["scheduler"].get("workers", 1)))
    config["scheduler"]["quant_points_budget"] = max(
        0, int(config["scheduler"].get("quant_points_budget", 0))
    )
    return config


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> tuple[dict[str, Any], str]:
    raw = load_yaml(path)
    if overrides:
        raw = _deep_merge(raw, overrides)
    config = normalize_config(raw)
    return config, sha256_json(config)


def layer_config(config: dict[str, Any], layer: str) -> dict[str, Any]:
    if layer in config.get("layers", {}):
        return config["layers"][layer]
    default_bits = int(config.get("default_bit_width", 8))
    calibration = config["calibration"]
    return {
        "bit_width": default_bits,
        "group_size": int(calibration.get("group_size", 0) or 0),
        "group_size_candidates": [int(calibration.get("group_size", 0) or 0)],
        "scheme": calibration.get("scheme", "asymmetric"),
        "per_channel": bool(calibration.get("per_channel", False)),
    }


def operation_apply(
    weights: dict[str, WeightPack],
    config: dict[str, Any],
    operation: dict[str, Any],
    replacement_generator: Any,
) -> tuple[dict[str, WeightPack], dict[str, Any], list[str]]:
    """Apply one declarative edit without touching source files."""

    new_weights = copy.deepcopy(weights)
    new_config = copy.deepcopy(config)
    changed_layers: list[str] = []
    set_values = operation.get("set", {})
    if isinstance(set_values, dict):
        for dotted, value in set_values.items():
            cursor = new_config
            parts = dotted.split(".")
            for part in parts[:-1]:
                if part not in cursor or not isinstance(cursor[part], dict):
                    cursor[part] = {}
                cursor = cursor[part]
            cursor[parts[-1]] = copy.deepcopy(value)
    layer_set = operation.get("layer_set", {})
    if isinstance(layer_set, dict):
        for layer, values in layer_set.items():
            if not isinstance(values, dict):
                continue
            target = new_config.setdefault("layers", {}).setdefault(str(layer), {})
            target.update(copy.deepcopy(values))
            changed_layers.append(str(layer))
    replace = operation.get("replace_weight")
    if isinstance(replace, dict):
        layer = str(replace["layer"])
        if layer in new_weights:
            pack = new_weights[layer]
            replacement = replacement_generator(
                layer=layer,
                shape=pack.shape,
                seed=int(replace.get("seed", 0)),
                scale=float(replace.get("scale", 0.3)),
            )
            version = str(replace.get("content_version", "edit-replacement"))
            new_weights[layer] = WeightPack(
                layer=layer,
                shape=pack.shape,
                weights=np.ascontiguousarray(replacement, dtype=np.float64),
                content_version=version,
                content_fingerprint=combined_fingerprint(array_fingerprint(replacement), version),
            )
            changed_layers.append(layer)
    normalized = normalize_config(new_config)
    return new_weights, normalized, sorted(set(changed_layers))
