"""Configuration loading, normalization and validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import canonical_json, content_key, sha256_text
from .errors import ConfigValidationError, Failure, config_failure, sort_failures
from . import yamlite


ALGORITHM_NAMES = (
    "histogram_density",
    "random_partition_depth",
    "robust_mahalanobis",
)
ALGORITHM_VERSIONS = {
    "histogram_density": "hist-v1",
    "random_partition_depth": "rpd-v1",
    "robust_mahalanobis": "rmd-v1",
}
PREPROCESS_VERSION = "prep-v1"
DECISION_VERSION = "decision-v1"

PREPROCESS_MODES = ("raw", "robust_standardize", "standardize")
NUMERIC_FILLS = ("mean", "median", "zero")
CATEGORICAL_FILLS = ("constant", "mode")

DEFAULT_STORAGE = {
    "snapshot_dir": ".workbench/snapshots",
    "cache_dir": ".workbench/cache",
    "report_path": ".workbench/report.json",
    "injection_report_path": ".workbench/injection_report.json",
}


@dataclass
class ConfigBundle:
    config_path: Path
    root: Path
    config: dict[str, Any]
    description_path: Path
    description: dict[str, Any]
    failures: list[dict[str, Any]]

    @property
    def valid(self) -> bool:
        return not self.failures

    @property
    def target_field(self) -> str:
        return str(self.config["data"]["target_field"])

    @property
    def feature_columns(self) -> list[str]:
        return list(self.config["data"]["feature_columns"])


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_mapping(value: Any, field: str, failures: list[Failure]) -> dict[str, Any]:
    if not isinstance(value, dict):
        failures.append(config_failure(field, "expected a mapping"))
        return {}
    return value


def _require_list(value: Any, field: str, failures: list[Failure]) -> list[Any]:
    if not isinstance(value, list):
        failures.append(config_failure(field, "expected a list"))
        return []
    return value


def _load_yaml(path: Path, field: str, failures: list[Failure]) -> Any:
    try:
        return yamlite.load(path)
    except FileNotFoundError:
        failures.append(config_failure(field, f"file not found: {path}"))
    except OSError as exc:
        failures.append(config_failure(field, f"cannot read {path}: {exc}"))
    except yamlite.YamlLiteError as exc:
        failures.append(config_failure(field, f"invalid YAML: {exc}"))
    return None


def _validate_description(description: Any, failures: list[Failure]) -> dict[str, Any]:
    root = _require_mapping(description, "data_description", failures)
    dataset = _require_mapping(root.get("dataset"), "data_description.dataset", failures)
    if not dataset:
        failures.append(config_failure("data_description.dataset", "target structure is missing"))
        return root

    target = dataset.get("target")
    if not isinstance(target, dict):
        failures.append(config_failure("data_description.dataset.target", "target structure is missing"))
        target = {}
    else:
        if not isinstance(target.get("field"), str) or not target.get("field"):
            failures.append(config_failure("data_description.dataset.target.field", "target field is missing"))
        ratio = target.get("anomaly_ratio")
        if not _is_number(ratio):
            failures.append(config_failure("data_description.dataset.target.anomaly_ratio", "anomaly ratio must be numeric"))
        elif not (0.0 <= float(ratio) < 1.0):
            failures.append(
                config_failure(
                    "data_description.dataset.target.anomaly_ratio",
                    "anomaly ratio must satisfy 0 <= ratio < 1",
                )
            )

    sample_count = dataset.get("sample_count")
    if not _is_int(sample_count) or sample_count < 1:
        failures.append(config_failure("data_description.dataset.sample_count", "sample count must be a positive integer"))

    missing_ratio = dataset.get("missing_ratio", 0.0)
    if not _is_number(missing_ratio) or not (0.0 <= float(missing_ratio) < 1.0):
        failures.append(config_failure("data_description.dataset.missing_ratio", "missing ratio must satisfy 0 <= ratio < 1"))

    seed = dataset.get("seed")
    if not _is_int(seed):
        failures.append(config_failure("data_description.dataset.seed", "seed must be an integer"))

    columns = _require_list(dataset.get("columns"), "data_description.dataset.columns", failures)
    if not columns:
        failures.append(config_failure("data_description.dataset.columns", "at least one column is required"))

    names: set[str] = set()
    for index, item in enumerate(columns):
        field = f"data_description.dataset.columns[{index}]"
        column = _require_mapping(item, field, failures)
        name = column.get("name")
        ctype = column.get("type")
        if not isinstance(name, str) or not name:
            failures.append(config_failure(f"{field}.name", "column name must be a non-empty string"))
            continue
        if name in names:
            failures.append(config_failure(f"{field}.name", f"duplicate column name: {name}"))
        names.add(name)
        if ctype not in ("numeric", "categorical"):
            failures.append(config_failure(f"{field}.type", "column type must be numeric or categorical"))
            continue
        if ctype == "numeric":
            distribution = column.get("distribution", "normal")
            if distribution not in ("normal", "redundant"):
                failures.append(config_failure(f"{field}.distribution", "numeric distribution must be normal or redundant"))
            if distribution == "normal":
                if not _is_number(column.get("mean")):
                    failures.append(config_failure(f"{field}.mean", "numeric mean must be finite"))
                if not _is_number(column.get("std")) or float(column.get("std")) <= 0.0:
                    failures.append(config_failure(f"{field}.std", "numeric std must be positive"))
        else:
            categories = _require_list(column.get("categories"), f"{field}.categories", failures)
            if not categories or not all(isinstance(x, str) and x for x in categories):
                failures.append(config_failure(f"{field}.categories", "categories must be a non-empty list of strings"))
            elif len(set(categories)) != len(categories):
                failures.append(config_failure(f"{field}.categories", "categories must be unique"))
    return root


def _check_column_refs(config: dict[str, Any], description: dict[str, Any], failures: list[Failure]) -> None:
    dataset = description.get("dataset", {}) if isinstance(description, dict) else {}
    raw_columns = dataset.get("columns", []) if isinstance(dataset, dict) else []
    columns: dict[str, dict[str, Any]] = {}
    if isinstance(raw_columns, list):
        for item in raw_columns:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                columns[item["name"]] = item

    data = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    feature_columns = data.get("feature_columns", [])
    if not isinstance(feature_columns, list) or not feature_columns:
        failures.append(config_failure("data.feature_columns", "feature columns must be a non-empty list"))
        feature_columns = []
    if len(feature_columns) != len(set(map(str, feature_columns))):
        failures.append(config_failure("data.feature_columns", "feature columns contain duplicates"))
    for index, name in enumerate(feature_columns):
        if not isinstance(name, str) or name not in columns:
            failures.append(config_failure(f"data.feature_columns[{index}]", f"unknown column: {name!r}"))

    target_field = data.get("target_field")
    target = dataset.get("target", {}) if isinstance(dataset, dict) else {}
    expected_target = target.get("field") if isinstance(target, dict) else None
    if not isinstance(target_field, str) or not target_field:
        failures.append(config_failure("data.target_field", "target field is missing"))
    elif isinstance(expected_target, str) and target_field != expected_target:
        failures.append(
            config_failure(
                "data.target_field",
                f"target field {target_field!r} does not match data description {expected_target!r}",
            )
        )
    if isinstance(target_field, str) and target_field in feature_columns:
        failures.append(config_failure("data.target_field", "target field must not be a feature column"))

    pre = config.get("preprocessing", {}) if isinstance(config.get("preprocessing"), dict) else {}
    numeric_columns = pre.get("numeric_columns", [])
    categorical_columns = pre.get("categorical_columns", [])
    if not isinstance(numeric_columns, list):
        failures.append(config_failure("preprocessing.numeric_columns", "expected a list"))
        numeric_columns = []
    if not isinstance(categorical_columns, list):
        failures.append(config_failure("preprocessing.categorical_columns", "expected a list"))
        categorical_columns = []
    for field, refs, expected_type in (
        ("preprocessing.numeric_columns", numeric_columns, "numeric"),
        ("preprocessing.categorical_columns", categorical_columns, "categorical"),
    ):
        for index, name in enumerate(refs):
            if not isinstance(name, str) or name not in columns:
                failures.append(config_failure(f"{field}[{index}]", f"unknown column: {name!r}"))
            elif columns[name].get("type") != expected_type:
                failures.append(
                    config_failure(
                        f"{field}[{index}]",
                        f"type mismatch: column {name!r} is {columns[name].get('type')}, expected {expected_type}",
                    )
                )
    declared = list(numeric_columns) + list(categorical_columns)
    if len(declared) != len(set(map(str, declared))):
        failures.append(config_failure("preprocessing.columns", "a column cannot be both numeric and categorical"))
    if feature_columns and set(map(str, feature_columns)) != set(map(str, declared)):
        failures.append(
            config_failure(
                "preprocessing.columns",
                "numeric_columns and categorical_columns must exactly cover feature_columns",
            )
        )
    for name in numeric_columns:
        col = columns.get(name, {})
        if col.get("distribution") == "redundant":
            source = col.get("source")
            if not isinstance(source, str) or source not in columns:
                failures.append(config_failure(f"column[{name}].source", f"unknown redundant source: {source!r}"))
            elif columns[source].get("type") != "numeric":
                failures.append(config_failure(f"column[{name}].source", "redundant source must be numeric"))


def _validate_hyperparameters(name: str, params: Any, failures: list[Failure]) -> None:
    field = f"algorithms.{name}.params"
    p = _require_mapping(params, field, failures)
    if name == "robust_mahalanobis":
        value = p.get("regularization")
        if not _is_number(value) or float(value) <= 0.0:
            failures.append(config_failure(f"{field}.regularization", "regularization must be positive"))
    elif name == "histogram_density":
        bins = p.get("bins")
        neighborhood = p.get("neighborhood")
        if not _is_int(bins) or bins < 2:
            failures.append(config_failure(f"{field}.bins", "bins must be an integer >= 2"))
        if not _is_int(neighborhood) or neighborhood < 0:
            failures.append(config_failure(f"{field}.neighborhood", "neighborhood must be a non-negative integer"))
        if _is_int(bins) and _is_int(neighborhood) and neighborhood >= bins:
            failures.append(config_failure(f"{field}.neighborhood", "neighborhood must be smaller than bins"))
    elif name == "random_partition_depth":
        trees = p.get("trees")
        sample_size = p.get("sample_size")
        max_depth = p.get("max_depth")
        if not _is_int(trees) or trees < 1:
            failures.append(config_failure(f"{field}.trees", "trees must be a positive integer"))
        if not _is_int(sample_size) or sample_size < 2:
            failures.append(config_failure(f"{field}.sample_size", "sample_size must be an integer >= 2"))
        if not _is_int(max_depth) or max_depth < 1:
            failures.append(config_failure(f"{field}.max_depth", "max_depth must be a positive integer"))


def _validate_algorithms(config: dict[str, Any], failures: list[Failure]) -> None:
    algorithms = _require_mapping(config.get("algorithms"), "algorithms", failures)
    if not algorithms:
        failures.append(config_failure("algorithms", "at least one algorithm configuration is required"))
        return
    for name in algorithms:
        if name not in ALGORITHM_NAMES:
            failures.append(config_failure(f"algorithms.{name}", "unknown detection algorithm"))
    enabled_count = 0
    for name in ALGORITHM_NAMES:
        if name not in algorithms:
            failures.append(config_failure(f"algorithms.{name}", "algorithm configuration is missing"))
            continue
        item = _require_mapping(algorithms.get(name), f"algorithms.{name}", failures)
        enabled = item.get("enabled", True)
        if not isinstance(enabled, bool):
            failures.append(config_failure(f"algorithms.{name}.enabled", "enabled must be boolean"))
        elif enabled:
            enabled_count += 1
        weight = item.get("weight", 1.0)
        if not _is_number(weight) or float(weight) <= 0.0:
            failures.append(config_failure(f"algorithms.{name}.weight", "weight must be positive"))
        _validate_hyperparameters(name, item.get("params"), failures)
        candidates = _require_mapping(item.get("candidates"), f"algorithms.{name}.candidates", failures)
        p = item.get("params") if isinstance(item.get("params"), dict) else {}
        for key, current in p.items():
            values = candidates.get(key)
            if not isinstance(values, list) or not values:
                failures.append(config_failure(f"algorithms.{name}.candidates.{key}", "candidate list is missing"))
                continue
            if not any(type(x) is type(current) and x == current for x in values):
                failures.append(
                    config_failure(
                        f"algorithms.{name}.params.{key}",
                        "selected hyperparameter is not one of the configured candidates",
                    )
                )
            for index, value in enumerate(values):
                probe = {name: {"params": {key: value}}}
                local: list[Failure] = []
                # Reuse the range checks without recursively validating candidates.
                if name == "robust_mahalanobis" and key == "regularization":
                    if not _is_number(value) or float(value) <= 0.0:
                        local.append(config_failure(f"algorithms.{name}.candidates.{key}[{index}]", "candidate must be positive"))
                elif name == "histogram_density" and key == "bins":
                    if not _is_int(value) or value < 2:
                        local.append(config_failure(f"algorithms.{name}.candidates.{key}[{index}]", "candidate must be an integer >= 2"))
                elif name == "histogram_density" and key == "neighborhood":
                    if not _is_int(value) or value < 0:
                        local.append(config_failure(f"algorithms.{name}.candidates.{key}[{index}]", "candidate must be a non-negative integer"))
                elif name == "random_partition_depth" and key in ("trees", "sample_size", "max_depth"):
                    if not _is_int(value) or value < 1:
                        local.append(config_failure(f"algorithms.{name}.candidates.{key}[{index}]", "candidate must be a positive integer"))
                failures.extend(local)
    if enabled_count == 0:
        failures.append(config_failure("algorithms", "at least one algorithm must be enabled"))


def _validate_preprocessing(config: dict[str, Any], failures: list[Failure]) -> None:
    pre = _require_mapping(config.get("preprocessing"), "preprocessing", failures)
    mode = pre.get("mode")
    numeric_fill = pre.get("numeric_fill")
    categorical_fill = pre.get("categorical_fill")
    if mode not in PREPROCESS_MODES:
        failures.append(config_failure("preprocessing.mode", f"mode must be one of {list(PREPROCESS_MODES)}"))
    if numeric_fill not in NUMERIC_FILLS:
        failures.append(config_failure("preprocessing.numeric_fill", f"numeric_fill must be one of {list(NUMERIC_FILLS)}"))
    if categorical_fill not in CATEGORICAL_FILLS:
        failures.append(config_failure("preprocessing.categorical_fill", f"categorical_fill must be one of {list(CATEGORICAL_FILLS)}"))
    if not isinstance(pre.get("constant_value", "__MISSING__"), str):
        failures.append(config_failure("preprocessing.constant_value", "constant_value must be a string"))

    candidates = _require_mapping(pre.get("candidates"), "preprocessing.candidates", failures)
    for key, current in (("mode", mode), ("numeric_fill", numeric_fill), ("categorical_fill", categorical_fill)):
        values = candidates.get(key)
        if not isinstance(values, list) or not values:
            failures.append(config_failure(f"preprocessing.candidates.{key}", "candidate list is missing"))
        elif current not in values:
            failures.append(
                config_failure(
                    f"preprocessing.{key}",
                    "selected preprocessing value is not one of the configured candidates",
                )
            )


def _validate_interaction(config: dict[str, Any], failures: list[Failure]) -> None:
    interaction = _require_mapping(config.get("interaction"), "interaction", failures)
    budget = interaction.get("latency_budget_ms")
    concurrency = interaction.get("concurrency")
    snapshot_interval = interaction.get("snapshot_interval", 1)
    if not _is_number(budget) or float(budget) <= 0.0:
        failures.append(config_failure("interaction.latency_budget_ms", "latency budget must be positive"))
    if not _is_int(concurrency) or concurrency <= 0:
        failures.append(config_failure("interaction.concurrency", "concurrency must be a positive integer"))
    if not _is_int(snapshot_interval) or snapshot_interval <= 0:
        failures.append(config_failure("interaction.snapshot_interval", "snapshot interval must be a positive integer"))


def validate_config(
    config: dict[str, Any],
    description: dict[str, Any],
    config_path: Path | str = "config.yaml",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failures: list[Failure] = []
    if not isinstance(config, dict):
        failures.append(config_failure("config", "configuration root must be a mapping"))
        return {}, sort_failures(failures)
    if config.get("version", 1) != 1:
        failures.append(config_failure("version", "unsupported configuration version"))
    _validate_description(description, failures)
    data = _require_mapping(config.get("data"), "data", failures)
    if not isinstance(data.get("description"), str) or not data.get("description"):
        failures.append(config_failure("data.description", "data description path is missing"))
    _validate_preprocessing(config, failures)
    _validate_algorithms(config, failures)
    threshold = config.get("threshold")
    if not _is_number(threshold) or not (0.0 <= float(threshold) <= 1.0):
        failures.append(config_failure("threshold", "threshold must be a finite number in [0, 1]"))
    _validate_interaction(config, failures)
    if not isinstance(config.get("storage", {}), dict):
        failures.append(config_failure("storage", "storage must be a mapping"))
    _check_column_refs(config, description, failures)
    return config, sort_failures(failures)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def load_config_bundle(config_path: str | Path, root: str | Path | None = None) -> ConfigBundle:
    path = Path(config_path).resolve()
    config_root = Path(root).resolve() if root is not None else path.parent
    failures: list[Failure] = []
    config = _load_yaml(path, "config", failures)
    if not isinstance(config, dict):
        if not failures:
            failures.append(config_failure("config", "configuration root must be a mapping"))
        config = {}
    description_path = config_root / "_missing_data_description.yaml"
    description: Any = {}
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    desc_value = data.get("description") if isinstance(data, dict) else None
    if isinstance(desc_value, str) and desc_value:
        description_path = _resolve(config_root, desc_value)
        description = _load_yaml(description_path, "data.description", failures)
    else:
        failures.append(config_failure("data.description", "data description path is missing"))
    if not isinstance(description, dict):
        description = {}
    normalized, validation_failures = validate_config(config, description, path)
    failures.extend(Failure(**x) for x in validation_failures)
    config = _normalize(config)
    return ConfigBundle(
        config_path=path,
        root=config_root,
        config=config,
        description_path=description_path,
        description=description,
        failures=sort_failures(failures),
    )


def require_valid_bundle(bundle: ConfigBundle) -> ConfigBundle:
    if bundle.failures:
        raise ConfigValidationError([Failure(**x) for x in bundle.failures])
    return bundle


def _normalize(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize numeric scalar types; validation has already established shape."""

    if not isinstance(config, dict):
        return config
    cfg = dict(config)
    if _is_number(cfg.get("threshold")):
        cfg["threshold"] = float(cfg["threshold"])
    interaction = cfg.get("interaction")
    if isinstance(interaction, dict):
        interaction = dict(interaction)
        if _is_number(interaction.get("latency_budget_ms")):
            interaction["latency_budget_ms"] = float(interaction["latency_budget_ms"])
        cfg["interaction"] = interaction
    storage = dict(DEFAULT_STORAGE)
    if isinstance(cfg.get("storage"), dict):
        storage.update({k: v for k, v in cfg["storage"].items() if isinstance(v, str)})
    cfg["storage"] = storage
    algorithms = cfg.get("algorithms")
    if isinstance(algorithms, dict):
        normalized_algorithms: dict[str, Any] = {}
        for name, item in algorithms.items():
            if not isinstance(item, dict):
                normalized_algorithms[name] = item
                continue
            entry = dict(item)
            params = dict(entry.get("params", {})) if isinstance(entry.get("params"), dict) else entry.get("params")
            entry["params"] = params
            entry.setdefault("enabled", True)
            entry.setdefault("weight", 1.0)
            if _is_number(entry.get("weight")):
                entry["weight"] = float(entry["weight"])
            normalized_algorithms[name] = entry
        cfg["algorithms"] = normalized_algorithms
    return cfg


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    return _normalize(config)


def semantic_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    cfg.pop("storage", None)
    return cfg


def config_hash(config: dict[str, Any]) -> str:
    return sha256_text("config\n" + canonical_json(semantic_config(config)))


def enabled_algorithms(config: dict[str, Any]) -> list[str]:
    algorithms = config.get("algorithms", {})
    return [name for name in ALGORITHM_NAMES if isinstance(algorithms.get(name), dict) and algorithms[name].get("enabled", True)]


def preprocess_fingerprint(config: dict[str, Any]) -> str:
    pre = config.get("preprocessing", {})
    payload = {
        "version": PREPROCESS_VERSION,
        "mode": pre.get("mode"),
        "numeric_fill": pre.get("numeric_fill"),
        "categorical_fill": pre.get("categorical_fill"),
        "constant_value": pre.get("constant_value", "__MISSING__"),
        "numeric_columns": pre.get("numeric_columns", []),
        "categorical_columns": pre.get("categorical_columns", []),
        "feature_columns": config.get("data", {}).get("feature_columns", []),
    }
    return content_key("preprocess-fingerprint", payload)


def algorithm_fingerprint(config: dict[str, Any], name: str) -> str:
    item = config.get("algorithms", {}).get(name, {})
    payload = {
        "version": ALGORITHM_VERSIONS[name],
        "name": name,
        "params": item.get("params", {}),
    }
    return content_key("algorithm-fingerprint", payload)


def algorithm_weight(config: dict[str, Any], name: str) -> float:
    return float(config.get("algorithms", {}).get(name, {}).get("weight", 1.0))
