from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import E_CONFIG_INVALID, Failure
from .fingerprint import is_finite_number


VALID_COLUMN_TYPES = {"numeric", "categorical"}
VALID_DIRECTIONS = {
    "ascending": "ascending",
    "increasing": "ascending",
    "up": "ascending",
    "descending": "descending",
    "decreasing": "descending",
    "down": "descending",
    "none": "none",
    "unordered": "none",
}
VALID_CONFLICT_RULES = {"left", "right", "max_gap", "min_count", "max_count"}


@dataclass(frozen=True)
class EffectiveColumn:
    name: str
    column_type: str
    max_bins: int
    min_bin_ratio: float
    direction: str
    conflict_priority: list[str]
    max_rounds: int


@dataclass(frozen=True)
class ValidatedConfig:
    experiment: dict[str, Any]
    schema: dict[str, Any]
    features: list[str]
    target_name: str
    positive_value: Any
    negative_value: Any
    max_workers: int
    prebins: int
    smoothing: float
    folds: int
    fold_seed: int
    expected_global_version: int | None
    effective_columns: dict[str, EffectiveColumn]

    @property
    def directions(self) -> dict[str, str]:
        return {name: col.direction for name, col in self.effective_columns.items()}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _failure(column: str, message: str) -> Failure:
    return Failure(
        column=column,
        stage="config",
        code=E_CONFIG_INVALID,
        exception_type=None,
        message=message,
    )


def _int_value(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: Any) -> bool:
    return _int_value(value) and value > 0


def _scalar_value(value: Any) -> bool:
    if value is None or isinstance(value, (dict, list)):
        return False
    if isinstance(value, float) and not is_finite_number(value):
        return False
    return isinstance(value, (str, int, float, bool))


def normalize_direction(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return VALID_DIRECTIONS.get(value.strip().lower())


def validate_inputs(experiment: dict[str, Any], schema: dict[str, Any]) -> tuple[ValidatedConfig | None, list[Failure]]:
    failures: list[Failure] = []

    if not isinstance(experiment, dict):
        return None, [_failure("", "experiment YAML root must be a mapping")]
    if not isinstance(schema, dict):
        return None, [_failure("", "synthetic data YAML root must be a mapping")]

    if experiment.get("schema_version", 1) != 1:
        failures.append(_failure("", "experiment schema_version must be 1"))
    if schema.get("schema_version", 1) != 1:
        failures.append(_failure("", "synthetic schema_version must be 1"))

    rows = schema.get("rows")
    if not _positive_int(rows):
        failures.append(_failure("", "synthetic.rows must be a positive integer"))
    seed = schema.get("seed")
    if not _int_value(seed):
        failures.append(_failure("", "synthetic.seed must be an integer"))

    runtime = experiment.get("runtime", {})
    if not isinstance(runtime, dict):
        failures.append(_failure("", "runtime must be a mapping"))
        runtime = {}
    max_workers = runtime.get("max_workers")
    if not _positive_int(max_workers):
        failures.append(_failure("", "runtime.max_workers must be a positive integer"))

    binning = experiment.get("binning", {})
    if not isinstance(binning, dict):
        failures.append(_failure("", "binning must be a mapping"))
        binning = {}
    global_max_bins = binning.get("max_bins")
    if not _positive_int(global_max_bins):
        failures.append(_failure("", "binning.max_bins must be a positive integer"))
    min_ratio = binning.get("min_bin_ratio")
    if not is_finite_number(min_ratio) or not (0.0 <= float(min_ratio) <= 1.0):
        failures.append(_failure("", "binning.min_bin_ratio must be in [0, 1]"))
    prebins = binning.get("prebins", 64)
    if not _positive_int(prebins):
        failures.append(_failure("", "binning.prebins must be a positive integer"))
    smoothing = binning.get("smoothing", 1.0)
    if not is_finite_number(smoothing) or float(smoothing) <= 0.0:
        failures.append(_failure("", "binning.smoothing must be a positive number"))

    monotonic = experiment.get("monotonic", {})
    if not isinstance(monotonic, dict):
        failures.append(_failure("", "monotonic must be a mapping"))
        monotonic = {}
    max_rounds = monotonic.get("max_rounds", 20)
    if not _int_value(max_rounds) or max_rounds < 0:
        failures.append(_failure("", "monotonic.max_rounds must be a non-negative integer"))
    priority = monotonic.get("conflict_priority", ["max_gap", "left", "right"])
    if (
        not isinstance(priority, list)
        or not priority
        or any(not isinstance(item, str) or item not in VALID_CONFLICT_RULES for item in priority)
    ):
        failures.append(
            _failure(
                "",
                "monotonic.conflict_priority must be a non-empty subset of "
                + ", ".join(sorted(VALID_CONFLICT_RULES)),
            )
        )
        priority = ["max_gap", "left", "right"]

    evaluation = experiment.get("evaluation", {})
    if not isinstance(evaluation, dict):
        failures.append(_failure("", "evaluation must be a mapping"))
        evaluation = {}
    folds = evaluation.get("folds", 5)
    if not _positive_int(folds) or folds < 2:
        failures.append(_failure("", "evaluation.folds must be an integer >= 2"))
    fold_seed = evaluation.get("seed", schema.get("seed", 0))
    if not _int_value(fold_seed):
        failures.append(_failure("", "evaluation.seed must be an integer"))

    snapshot_conf = experiment.get("snapshot", {})
    if not isinstance(snapshot_conf, dict):
        failures.append(_failure("", "snapshot must be a mapping"))
        snapshot_conf = {}
    expected_version = snapshot_conf.get("expected_global_version")
    if expected_version is not None and (not _int_value(expected_version) or expected_version < 0):
        failures.append(_failure("", "snapshot.expected_global_version must be null or a non-negative integer"))

    target = schema.get("target")
    target_name = ""
    positive_value: Any = None
    negative_value: Any = None
    if not isinstance(target, dict):
        failures.append(_failure("target", "target column description is missing"))
    else:
        target_name = target.get("name")
        if not isinstance(target_name, str) or not target_name.strip():
            failures.append(_failure("target", "target.name is required"))
            target_name = ""
        if "positive_value" not in target or "negative_value" not in target:
            failures.append(_failure(target_name or "target", "target positive_value and negative_value are required"))
            positive_value = target.get("positive_value")
            negative_value = target.get("negative_value")
        else:
            positive_value = target["positive_value"]
            negative_value = target["negative_value"]
            if not _scalar_value(positive_value) or not _scalar_value(negative_value):
                failures.append(_failure(target_name or "target", "target enum values must be non-null scalars"))
            if positive_value == negative_value:
                failures.append(_failure(target_name or "target", "target positive_value and negative_value must differ"))
        missing_rate = target.get("missing_rate", 0.0)
        if not is_finite_number(missing_rate) or not (0.0 <= float(missing_rate) < 1.0):
            failures.append(_failure(target_name or "target", "target.missing_rate must be in [0, 1)"))

    raw_columns = schema.get("columns")
    schema_columns: dict[str, dict[str, Any]] = {}
    if not isinstance(raw_columns, list) or not raw_columns:
        failures.append(_failure("", "synthetic.columns must be a non-empty list"))
    else:
        for index, column in enumerate(raw_columns):
            if not isinstance(column, dict):
                failures.append(_failure("", f"synthetic.columns[{index}] must be a mapping"))
                continue
            name = column.get("name")
            if not isinstance(name, str) or not name.strip():
                failures.append(_failure("", f"synthetic.columns[{index}].name is required"))
                continue
            if name in schema_columns:
                failures.append(_failure(name, "duplicate column name in synthetic data description"))
                continue
            column_type = column.get("type")
            if column_type not in VALID_COLUMN_TYPES:
                failures.append(_failure(name, f"unsupported synthetic column type: {column_type!r}"))
            missing_rate = column.get("missing_rate", 0.0)
            if not is_finite_number(missing_rate) or not (0.0 <= float(missing_rate) < 1.0):
                failures.append(_failure(name, "column missing_rate must be in [0, 1)"))
            if "seed_offset" in column and not _int_value(column["seed_offset"]):
                failures.append(_failure(name, "column seed_offset must be an integer"))
            if column_type == "numeric":
                distribution = column.get("distribution", "normal")
                if distribution not in {"normal", "lognormal", "uniform"}:
                    failures.append(_failure(name, f"unsupported numeric distribution: {distribution!r}"))
                outlier_rate = column.get("outlier_rate", 0.0)
                if not is_finite_number(outlier_rate) or not (0.0 <= float(outlier_rate) < 1.0):
                    failures.append(_failure(name, "column outlier_rate must be in [0, 1)"))
                outlier_scale = column.get("outlier_scale", 0.0)
                if not is_finite_number(outlier_scale) or float(outlier_scale) < 0.0:
                    failures.append(_failure(name, "column outlier_scale must be non-negative"))
            elif column_type == "categorical":
                cardinality = column.get("cardinality")
                if not _positive_int(cardinality):
                    failures.append(_failure(name, "categorical cardinality must be a positive integer"))
                elif _positive_int(rows) and cardinality > rows:
                    failures.append(_failure(name, "categorical cardinality cannot exceed synthetic.rows"))
            if name == target_name:
                failures.append(_failure(name, "target column must be declared under target, not columns"))
            schema_columns[name] = column

    features_raw = experiment.get("features")
    if features_raw is None:
        features = sorted(schema_columns)
    elif not isinstance(features_raw, list):
        failures.append(_failure("", "features must be a list"))
        features = []
    else:
        features = []
        seen_features: set[str] = set()
        for item in features_raw:
            if not isinstance(item, str) or not item:
                failures.append(_failure("", "features entries must be non-empty strings"))
                continue
            if item in seen_features:
                failures.append(_failure(item, "duplicate feature name"))
                continue
            seen_features.add(item)
            if item not in schema_columns:
                failures.append(_failure(item, "unknown feature column"))
                continue
            features.append(item)
        features = sorted(features)

    settings_raw = experiment.get("columns", {})
    if not isinstance(settings_raw, dict):
        failures.append(_failure("", "columns must be a mapping keyed by column name"))
        settings_raw = {}
    for name, setting in settings_raw.items():
        if not isinstance(name, str) or name not in schema_columns:
            failures.append(_failure(str(name), "unknown column in experiment column settings"))
            continue
        if not isinstance(setting, dict):
            failures.append(_failure(name, "column settings must be a mapping"))
            continue
        if "type" in setting and setting["type"] != schema_columns[name].get("type"):
            failures.append(
                _failure(
                    name,
                    f"column type mismatch: experiment={setting['type']!r}, schema={schema_columns[name].get('type')!r}",
                )
            )
        if "max_bins" in setting and not _positive_int(setting["max_bins"]):
            failures.append(_failure(name, "column max_bins must be a positive integer"))
        if "min_bin_ratio" in setting and (
            not is_finite_number(setting["min_bin_ratio"])
            or not (0.0 <= float(setting["min_bin_ratio"]) <= 1.0)
        ):
            failures.append(_failure(name, "column min_bin_ratio must be in [0, 1]"))
        direction = setting.get("monotonic_direction")
        if direction is not None:
            normalized = normalize_direction(direction)
            if normalized is None:
                failures.append(_failure(name, f"unsupported monotonic direction: {direction!r}"))
            elif schema_columns[name].get("type") != "numeric" and normalized != "none":
                failures.append(_failure(name, "monotonic_direction is only valid for numeric columns"))

    if target_name and target_name in features:
        failures.append(_failure(target_name, "target column cannot be a feature"))

    effective_columns: dict[str, EffectiveColumn] = {}
    if not failures:
        for name in features:
            setting = settings_raw.get(name, {})
            column_type = schema_columns[name]["type"]
            direction_raw = setting.get("monotonic_direction", "none")
            direction = normalize_direction(direction_raw)
            if direction is None:
                direction = "none"
            effective_columns[name] = EffectiveColumn(
                name=name,
                column_type=column_type,
                max_bins=int(setting.get("max_bins", global_max_bins)),
                min_bin_ratio=float(setting.get("min_bin_ratio", min_ratio)),
                direction=direction,
                conflict_priority=list(priority),
                max_rounds=int(max_rounds),
            )

    if failures:
        return None, sorted(failures, key=lambda f: (f.column, f.stage, f.code, f.message))

    validated = ValidatedConfig(
        experiment=experiment,
        schema=schema,
        features=features,
        target_name=target_name,
        positive_value=positive_value,
        negative_value=negative_value,
        max_workers=int(max_workers),
        prebins=int(prebins),
        smoothing=float(smoothing),
        folds=int(folds),
        fold_seed=int(fold_seed),
        expected_global_version=expected_version,
        effective_columns=effective_columns,
    )
    return validated, []


def validate_generated_table(config: ValidatedConfig, columns: dict[str, list[Any]], target: list[Any]) -> list[Failure]:
    failures: list[Failure] = []
    if config.target_name not in columns:
        failures.append(_failure(config.target_name, "target column is missing from generated table"))
    if not isinstance(target, list):
        failures.append(_failure(config.target_name, "target values are missing"))
        return failures

    allowed = {config.negative_value, config.positive_value}
    for index, value in enumerate(target):
        if value not in allowed:
            failures.append(
                _failure(
                    config.target_name,
                    f"target value at row {index} is outside binary enum: {value!r}",
                )
            )
            break

    for name in config.features:
        if name not in columns:
            failures.append(_failure(name, "feature column is missing from generated table"))
            continue
        expected_type = config.effective_columns[name].column_type
        for value in columns[name]:
            if value is None:
                continue
            if expected_type == "numeric" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
                failures.append(_failure(name, f"numeric column contains non-numeric value {value!r}"))
                break
            if expected_type == "categorical" and not isinstance(value, str):
                failures.append(_failure(name, f"categorical column contains non-string value {value!r}"))
                break
    return sorted(failures, key=lambda f: (f.column, f.stage, f.code, f.message))
