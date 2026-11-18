from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

from .util import (
    ENGINE_VERSION,
    SCHEME_ORDER,
    decimal_to_fixed,
    failure,
    file_sha256,
    fingerprint,
    is_int,
    is_number,
    load_yaml,
    sort_failures,
)


class InputLoadError(Exception):
    def __init__(self, failures: list[dict[str, str]]):
        super().__init__("input loading failed")
        self.failures = sort_failures(failures)


def _yaml_load_with_failure(path: Path, stage: str, field: str) -> Any:
    if not path.exists():
        raise InputLoadError(
            [failure("E_CONFIG_INVALID", stage, field, f"file does not exist: {path}")]
        )
    try:
        value = load_yaml(path)
    except (OSError, yaml.YAMLError) as exc:
        raise InputLoadError(
            [failure("E_CONFIG_INVALID", stage, field, f"cannot parse YAML: {exc}")]
        ) from exc
    if value is None:
        raise InputLoadError(
            [failure("E_CONFIG_INVALID", stage, field, "YAML document is empty")]
        )
    return value


def resolve_inputs(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config_hash = file_sha256_with_failure(config_path, "config")
    config = _yaml_load_with_failure(config_path, "config", "config")
    if not isinstance(config, dict):
        raise InputLoadError(
            [failure("E_CONFIG_INVALID", "config", "config", "root must be a mapping")]
        )

    data_config = config.get("data")
    if not isinstance(data_config, dict):
        raise InputLoadError(
            [failure("E_CONFIG_INVALID", "config", "data", "data must be a mapping")]
        )
    spec_name = data_config.get("spec_file")
    if not isinstance(spec_name, str) or not spec_name:
        raise InputLoadError(
            [
                failure(
                    "E_CONFIG_INVALID",
                    "config",
                    "data.spec_file",
                    "a non-empty local path is required",
                )
            ]
        )
    spec_path = (config_path.parent / spec_name).resolve()
    spec_hash = file_sha256_with_failure(spec_path, "data")
    spec = _yaml_load_with_failure(spec_path, "data", "data.spec_file")
    if not isinstance(spec, dict):
        raise InputLoadError(
            [failure("E_CONFIG_INVALID", "data", "data.spec_file", "root must be a mapping")]
        )

    overrides_name = data_config.get("overrides_file")
    if not isinstance(overrides_name, str) or not overrides_name:
        raise InputLoadError(
            [
                failure(
                    "E_CONFIG_INVALID",
                    "config",
                    "data.overrides_file",
                    "a non-empty local path is required",
                )
            ]
        )
    overrides_path = (config_path.parent / overrides_name).resolve()
    overrides_hash = file_sha256_with_failure(overrides_path, "data")
    overrides = _yaml_load_with_failure(overrides_path, "data", "data.overrides_file")
    if not isinstance(overrides, dict):
        raise InputLoadError(
            [
                failure(
                    "E_CONFIG_INVALID",
                    "data",
                    "data.overrides_file",
                    "root must be a mapping",
                )
            ]
        )

    return {
        "config_path": config_path,
        "config": config,
        "config_hash": config_hash,
        "spec_path": spec_path,
        "spec": spec,
        "spec_hash": spec_hash,
        "overrides_path": overrides_path,
        "overrides": overrides,
        "overrides_hash": overrides_hash,
    }


def file_sha256_with_failure(path: Path, stage: str) -> str:
    try:
        return file_sha256(path)
    except OSError as exc:
        raise InputLoadError(
            [failure("E_CONFIG_INVALID", stage, str(path), f"cannot read file: {exc}")]
        ) from exc


def validate_inputs(bundle: dict[str, Any]) -> list[dict[str, str]]:
    config = bundle["config"]
    spec = bundle["spec"]
    overrides = bundle["overrides"]
    failures: list[dict[str, str]] = []

    def add(field: str, message: str, stage: str = "config") -> None:
        failures.append(failure("E_CONFIG_INVALID", stage, field, message))

    for key in ("data", "features", "target", "split", "models", "calibrators", "evaluation", "concurrency", "snapshot"):
        if key not in config:
            add(key, "required configuration section is missing")

    features = config.get("features")
    numeric_features: list[str] = []
    categorical_features: list[str] = []
    if not isinstance(features, dict):
        add("features", "features must be a mapping")
    else:
        for kind, target_list in (("numeric", numeric_features), ("categorical", categorical_features)):
            values = features.get(kind)
            if not isinstance(values, list):
                add(f"features.{kind}", "must be a list of column names")
                continue
            seen: set[str] = set()
            for index, name in enumerate(values):
                if not isinstance(name, str) or not name:
                    add(f"features.{kind}[{index}]", "column name must be a non-empty string")
                    continue
                if name in seen:
                    add(f"features.{kind}[{index}]", f"duplicate feature column: {name}")
                seen.add(name)
                target_list.append(name)

    target_name = config.get("target")
    if not isinstance(target_name, str) or not target_name:
        add("target", "target column must be a non-empty string")
        target_name = ""

    split = config.get("split")
    if not isinstance(split, dict):
        add("split", "split must be a mapping")
    else:
        ratio = split.get("train_ratio")
        if not is_number(ratio) or not (0.0 < float(ratio) < 1.0):
            add("split.train_ratio", "must be a number strictly between 0 and 1")

    models = config.get("models")
    model_names: set[str] = set()
    if not isinstance(models, list) or not models:
        add("models", "must be a non-empty list")
    else:
        for index, model in enumerate(models):
            prefix = f"models[{index}]"
            if not isinstance(model, dict):
                add(prefix, "model entry must be a mapping")
                continue
            name = model.get("name")
            if not isinstance(name, str) or not name:
                add(f"{prefix}.name", "non-empty model name required")
            elif name in model_names:
                add(f"{prefix}.name", f"duplicate model name: {name}")
            else:
                model_names.add(name)
            model_type = model.get("type")
            if model_type not in ("logistic_regression", "decision_tree"):
                add(f"{prefix}.type", "must be logistic_regression or decision_tree")
            candidates = model.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                add(f"{prefix}.candidates", "must be a non-empty list")
                continue
            for candidate_index, params in enumerate(candidates):
                candidate_prefix = f"{prefix}.candidates[{candidate_index}]"
                if not isinstance(params, dict):
                    add(candidate_prefix, "hyperparameters must be a mapping")
                    continue
                if model_type == "logistic_regression":
                    learning_rate = params.get("learning_rate")
                    if not is_number(learning_rate) or float(learning_rate) <= 0:
                        add(f"{candidate_prefix}.learning_rate", "must be positive")
                    epochs = params.get("epochs")
                    if not is_int(epochs) or epochs <= 0:
                        add(f"{candidate_prefix}.epochs", "must be a positive integer")
                    batch_size = params.get("batch_size")
                    if not is_int(batch_size) or batch_size <= 0:
                        add(f"{candidate_prefix}.batch_size", "must be a positive integer")
                    l2 = params.get("l2", 0.0)
                    if not is_number(l2) or float(l2) < 0:
                        add(f"{candidate_prefix}.l2", "must be non-negative")
                elif model_type == "decision_tree":
                    max_depth = params.get("max_depth")
                    if not is_int(max_depth) or max_depth < 0:
                        add(f"{candidate_prefix}.max_depth", "must be a non-negative integer")
                    min_leaf = params.get("min_samples_leaf")
                    if not is_int(min_leaf) or min_leaf <= 0:
                        add(f"{candidate_prefix}.min_samples_leaf", "must be a positive integer")
                    min_gain = params.get("min_gain", 0.0)
                    if not is_number(min_gain) or float(min_gain) < 0:
                        add(f"{candidate_prefix}.min_gain", "must be non-negative")

    calibrators = config.get("calibrators")
    if not isinstance(calibrators, dict):
        add("calibrators", "must be a mapping keyed by scheme name")
    else:
        for scheme in SCHEME_ORDER:
            scheme_config = calibrators.get(scheme)
            prefix = f"calibrators.{scheme}"
            if not isinstance(scheme_config, dict):
                add(prefix, "scheme configuration must be a mapping")
                continue
            enabled = scheme_config.get("enabled")
            if not isinstance(enabled, bool):
                add(f"{prefix}.enabled", "must be boolean")
            if "bins" in scheme_config:
                bins = scheme_config.get("bins")
                if not is_int(bins) or bins <= 0:
                    add(f"{prefix}.bins", "must be a positive integer")
            regularization = scheme_config.get("regularization", "0")
            if not is_number(regularization) and not isinstance(regularization, str):
                add(f"{prefix}.regularization", "must be a non-negative number")
            else:
                try:
                    if Decimal(str(regularization)) < 0:
                        add(f"{prefix}.regularization", "must be non-negative")
                except InvalidOperation:
                    add(f"{prefix}.regularization", "must be a number")
            if scheme in ("isotonic", "histogram") and "bins" not in scheme_config:
                add(f"{prefix}.bins", "is required for binned calibration")
            if scheme == "platt":
                learning_rate = scheme_config.get("learning_rate", "0.2")
                if not is_number(learning_rate) and not isinstance(learning_rate, str):
                    add(f"{prefix}.learning_rate", "must be positive")
                else:
                    try:
                        if Decimal(str(learning_rate)) <= 0:
                            add(f"{prefix}.learning_rate", "must be positive")
                    except InvalidOperation:
                        add(f"{prefix}.learning_rate", "must be a positive number")
                iterations = scheme_config.get("iterations", 200)
                if not is_int(iterations) or iterations <= 0:
                    add(f"{prefix}.iterations", "must be a positive integer")
        for key in calibrators:
            if key not in SCHEME_ORDER:
                add(f"calibrators.{key}", "unknown calibration scheme")

    evaluation = config.get("evaluation")
    calibration_bins = None
    threshold_step = None
    fixed = None
    if not isinstance(evaluation, dict):
        add("evaluation", "must be a mapping")
    else:
        calibration_bins = evaluation.get("calibration_bins")
        if not is_int(calibration_bins) or calibration_bins <= 0:
            add("evaluation.calibration_bins", "must be a positive integer")
        threshold_step = evaluation.get("threshold_step")
        try:
            parsed_step = Decimal(str(threshold_step))
            if not parsed_step.is_finite() or parsed_step <= 0:
                raise InvalidOperation
            if parsed_step > 1:
                add("evaluation.threshold_step", "must be at most 1")
        except (InvalidOperation, ValueError):
            add("evaluation.threshold_step", "must be a finite positive decimal")
        fixed = evaluation.get("fixed_point")
        if not isinstance(fixed, dict):
            add("evaluation.fixed_point", "must be a mapping")
        else:
            decimal_places = fixed.get("decimal_places")
            if not is_int(decimal_places) or not (0 <= decimal_places <= 9):
                add("evaluation.fixed_point.decimal_places", "must be an integer from 0 to 9")
            levels = fixed.get("quantization_levels")
            if not is_int(levels) or levels <= 0:
                add("evaluation.fixed_point.quantization_levels", "must be a positive integer")

    concurrency = config.get("concurrency")
    if not isinstance(concurrency, dict):
        add("concurrency", "must be a mapping")
    else:
        workers = concurrency.get("max_workers")
        if not is_int(workers) or workers <= 0:
            add("concurrency.max_workers", "must be a positive integer")

    snapshot = config.get("snapshot")
    if not isinstance(snapshot, dict):
        add("snapshot", "must be a mapping")
    else:
        interval = snapshot.get("commit_interval")
        if not is_int(interval) or interval <= 0:
            add("snapshot.commit_interval", "must be a positive integer")

    descriptor_map, spec_failures = _validate_spec(spec)
    failures.extend(spec_failures)

    if isinstance(features, dict):
        for kind, names in (("numeric", numeric_features), ("categorical", categorical_features)):
            for name in names:
                descriptor = descriptor_map.get(name)
                if descriptor is None:
                    add(f"features.{kind}.{name}", f"unknown column: {name}")
                elif descriptor.get("type") != kind:
                    add(
                        f"features.{kind}.{name}",
                        f"type mismatch: declared {kind}, schema says {descriptor.get('type')}",
                    )
    all_declared = set(numeric_features) | set(categorical_features)
    if target_name and target_name in all_declared:
        add("target", "target column cannot also be a feature")
    target_descriptor = descriptor_map.get(target_name) if target_name else None
    if target_name and target_descriptor is None:
        add("target", f"target column is missing from data description: {target_name}")
    elif target_descriptor is not None and target_descriptor.get("type") != "target":
        add("target", "target descriptor type must be target")

    _validate_overrides(overrides, descriptor_map, spec, add)

    if threshold_step is not None and isinstance(fixed, dict):
        decimal_places = fixed.get("decimal_places")
        if is_int(decimal_places):
            try:
                step_fixed = decimal_to_fixed(threshold_step, 10**decimal_places)
                if step_fixed <= 0:
                    add(
                        "evaluation.threshold_step",
                        "rounds to zero at the configured fixed-point precision",
                    )
            except ValueError:
                pass

    return sort_failures(failures)


def _validate_spec(spec: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    failures: list[dict[str, str]] = []

    def add(field: str, message: str) -> None:
        failures.append(failure("E_CONFIG_INVALID", "data", field, message))

    seed = spec.get("seed")
    if not is_int(seed):
        add("data.seed", "must be an integer")
    n_samples = spec.get("n_samples")
    if not is_int(n_samples) or n_samples <= 1:
        add("data.n_samples", "must be an integer greater than 1")
    columns = spec.get("columns")
    descriptors: dict[str, dict[str, Any]] = {}
    if not isinstance(columns, list) or not columns:
        add("data.columns", "must be a non-empty list")
        return descriptors, failures
    target_count = 0
    for index, descriptor in enumerate(columns):
        prefix = f"data.columns[{index}]"
        if not isinstance(descriptor, dict):
            add(prefix, "column descriptor must be a mapping")
            continue
        name = descriptor.get("name")
        if not isinstance(name, str) or not name:
            add(f"{prefix}.name", "non-empty column name required")
            continue
        if name in descriptors:
            add(f"{prefix}.name", f"duplicate column: {name}")
            continue
        descriptors[name] = descriptor
        kind = descriptor.get("type")
        if kind not in ("identifier", "numeric", "categorical", "target"):
            add(f"{prefix}.type", f"unsupported column type: {kind}")
            continue
        missing_rate = descriptor.get("missing_rate", 0.0)
        if not is_number(missing_rate) or not (0.0 <= float(missing_rate) < 1.0):
            add(f"{prefix}.missing_rate", "must be in [0, 1)")
        if kind == "numeric":
            generator = descriptor.get("generator")
            if generator not in ("normal", "uniform", "lognormal"):
                add(f"{prefix}.generator", "must be normal, uniform, or lognormal")
            elif generator == "normal":
                for key in ("mean", "std"):
                    if not is_number(descriptor.get(key)):
                        add(f"{prefix}.{key}", "must be numeric")
                if is_number(descriptor.get("std")) and float(descriptor.get("std")) < 0:
                    add(f"{prefix}.std", "must be non-negative")
            elif generator == "uniform":
                low, high = descriptor.get("low"), descriptor.get("high")
                if not is_number(low) or not is_number(high) or float(low) >= float(high):
                    add(f"{prefix}.low/high", "must define a non-empty numeric interval")
            elif generator == "lognormal":
                for key in ("mean", "sigma"):
                    if not is_number(descriptor.get(key)):
                        add(f"{prefix}.{key}", "must be numeric")
                if is_number(descriptor.get("sigma")) and float(descriptor.get("sigma")) < 0:
                    add(f"{prefix}.sigma", "must be non-negative")
            digits = descriptor.get("digits", 6)
            if not is_int(digits) or not (0 <= digits <= 12):
                add(f"{prefix}.digits", "must be an integer from 0 to 12")
            if "redundant_of" in descriptor:
                source = descriptor.get("redundant_of")
                if not isinstance(source, str) or source not in descriptors:
                    add(f"{prefix}.redundant_of", "must reference an earlier known column")
                elif descriptors[source].get("type") != "numeric":
                    add(f"{prefix}.redundant_of", "numeric redundant source must be numeric")
                else:
                    for key in ("scale", "offset", "noise_std"):
                        if key in descriptor and not is_number(descriptor.get(key)):
                            add(f"{prefix}.{key}", "must be numeric")
        elif kind == "categorical":
            values = descriptor.get("values")
            weights = descriptor.get("weights")
            if not isinstance(values, list) or not values or not all(isinstance(v, str) for v in values):
                add(f"{prefix}.values", "must be a non-empty list of strings")
            elif len(set(values)) != len(values):
                add(f"{prefix}.values", "must not contain duplicates")
            if weights is not None:
                if not isinstance(weights, list) or len(weights) != len(values or []):
                    add(f"{prefix}.weights", "must match values length")
                elif any(not is_number(w) or float(w) < 0 for w in weights):
                    add(f"{prefix}.weights", "must be non-negative numbers")
                elif sum(float(w) for w in weights) <= 0:
                    add(f"{prefix}.weights", "must have a positive sum")
            if "redundant_of" in descriptor:
                source = descriptor.get("redundant_of")
                if not isinstance(source, str) or source not in descriptors:
                    add(f"{prefix}.redundant_of", "must reference an earlier known column")
                elif descriptors[source].get("type") != "categorical":
                    add(f"{prefix}.redundant_of", "categorical redundant source must be categorical")
        elif kind == "identifier":
            if descriptor.get("name") != "sample_id":
                add(f"{prefix}.name", "identifier column must be named sample_id")
        elif kind == "target":
            target_count += 1
            positive_rate = descriptor.get("positive_rate")
            if not is_number(positive_rate) or not (0.0 < float(positive_rate) < 1.0):
                add(f"{prefix}.positive_rate", "must be between 0 and 1")
            noise_std = descriptor.get("noise_std", 1.0)
            if not is_number(noise_std) or float(noise_std) < 0:
                add(f"{prefix}.noise_std", "must be non-negative")
            drivers = descriptor.get("driver_weights", {})
            if not isinstance(drivers, dict):
                add(f"{prefix}.driver_weights", "must be a mapping")
            else:
                for column, weight in drivers.items():
                    if column not in descriptors or descriptors[column].get("type") not in ("numeric", "categorical"):
                        add(f"{prefix}.driver_weights.{column}", "must reference a known feature column")
                    if not is_number(weight):
                        add(f"{prefix}.driver_weights.{column}", "weight must be numeric")
    if target_count != 1:
        add("data.columns", "exactly one target descriptor is required")
    return descriptors, failures


def _validate_overrides(
    overrides: dict[str, Any],
    descriptor_map: dict[str, dict[str, Any]],
    spec: dict[str, Any],
    add: Any,
) -> None:
    samples = overrides.get("samples", {})
    if not isinstance(samples, dict):
        add("data.overrides_file.samples", "must be a mapping keyed by sample ID")
        return
    n_samples = spec.get("n_samples")
    for key, override in samples.items():
        raw_id = str(key)
        if not raw_id.isdigit() or (is_int(n_samples) and not (1 <= int(raw_id) <= n_samples)):
            add(f"data.overrides_file.samples.{key}", "sample ID is outside the generated ID range")
            continue
        prefix = f"data.overrides_file.samples.{key}"
        if not isinstance(override, dict):
            add(prefix, "sample override must be a mapping")
            continue
        values = override.get("values", {})
        if not isinstance(values, dict):
            add(f"{prefix}.values", "must be a mapping")
            continue
        for column, value in values.items():
            descriptor = descriptor_map.get(column)
            if descriptor is None:
                add(f"{prefix}.values.{column}", f"unknown column: {column}")
                continue
            kind = descriptor.get("type")
            if kind == "numeric":
                if value is not None and not is_number(value):
                    add(f"{prefix}.values.{column}", "numeric override must be a number or null")
            elif kind == "categorical":
                if value is not None and not isinstance(value, str):
                    add(f"{prefix}.values.{column}", "categorical override must be a string or null")
            elif kind == "target":
                if value not in (0, 1) or isinstance(value, bool):
                    add(f"{prefix}.values.{column}", "target override must be 0 or 1")
            elif kind == "identifier":
                add(f"{prefix}.values.{column}", "sample ID cannot be replaced")
        if "target" in override and (override["target"] not in (0, 1) or isinstance(override["target"], bool)):
            add(f"{prefix}.target", "target override must be 0 or 1")


def stage_inputs(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    config = bundle["config"]
    spec = bundle["spec"]
    features = config["features"]
    fixed = config["evaluation"]["fixed_point"]
    fixed_scale = 10 ** int(fixed["decimal_places"])
    fixed_key = fingerprint(
        {
            "decimal_places": fixed["decimal_places"],
            "quantization_levels": fixed["quantization_levels"],
            "scale": fixed_scale,
        }
    )
    data_inputs = {
        "engine_version": ENGINE_VERSION,
        "spec_hash": bundle["spec_hash"],
        "overrides_hash": bundle["overrides_hash"],
        "seed": spec.get("seed"),
        "n_samples": spec.get("n_samples"),
        "features": {
            "numeric": list(features.get("numeric", [])),
            "categorical": list(features.get("categorical", [])),
        },
        "target": config.get("target"),
        "split": config.get("split"),
    }
    data_key = fingerprint(data_inputs)
    model_inputs = {
        "engine_version": ENGINE_VERSION,
        "data_key": data_key,
        "models": config.get("models"),
        "fixed_point": fixed_key,
    }
    model_key = fingerprint(model_inputs)
    calibration_inputs = {
        "engine_version": ENGINE_VERSION,
        "model_key": model_key,
        "calibrators": effective_calibrator_inputs(config.get("calibrators", {})),
        "fixed_point": fixed_key,
    }
    calibration_key = fingerprint(calibration_inputs)
    evaluation = config["evaluation"]
    metrics_inputs = {
        "engine_version": ENGINE_VERSION,
        "calibration_key": calibration_key,
        "data_key": data_key,
        "calibration_bins": evaluation.get("calibration_bins"),
        "threshold_step": evaluation.get("threshold_step"),
        "fixed_point": fixed_key,
    }
    metrics_key = fingerprint(metrics_inputs)
    return {
        "data": {"key": data_key, "inputs": data_inputs},
        "model": {"key": model_key, "inputs": model_inputs},
        "calibration": {"key": calibration_key, "inputs": calibration_inputs},
        "metrics": {"key": metrics_key, "inputs": metrics_inputs},
    }


def fixed_point(config: dict[str, Any]) -> dict[str, int]:
    raw = config["evaluation"]["fixed_point"]
    decimals = int(raw["decimal_places"])
    return {
        "decimal_places": decimals,
        "quantization_levels": int(raw["quantization_levels"]),
        "scale": 10**decimals,
        "metric_scale": 10**decimals,
    }


def effective_calibrator_inputs(calibrators: dict[str, Any]) -> dict[str, Any]:
    effective: dict[str, Any] = {}
    for scheme in SCHEME_ORDER:
        scheme_config = calibrators.get(scheme)
        if not isinstance(scheme_config, dict) or not scheme_config.get("enabled"):
            continue
        if scheme == "raw":
            effective[scheme] = {"enabled": True}
        elif scheme == "platt":
            effective[scheme] = {
                "enabled": True,
                "regularization": scheme_config.get("regularization", "0"),
                "learning_rate": scheme_config.get("learning_rate", "0.2"),
                "iterations": scheme_config.get("iterations", 200),
            }
        elif scheme in ("isotonic", "histogram"):
            effective[scheme] = {
                "enabled": True,
                "bins": scheme_config.get("bins"),
                "regularization": scheme_config.get("regularization", "0"),
            }
    return effective
