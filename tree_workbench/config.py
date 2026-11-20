"""Experiment configuration parsing and business validation."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable

from .errors import BusinessFailure
from .fingerprint import fingerprint


PRECISION_LEVELS = {"fast": 6, "balanced": 8, "high": 10, "strict": 12}


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    features: tuple[str, ...]
    tree_count: int
    tree_count_candidates: tuple[int, ...]
    max_depth: int
    learning_rate: float
    row_sample_ratio: float
    col_sample_ratio: float
    min_samples_leaf: int
    bin_count: int
    train_ratio: float
    thread_pool_size: int
    snapshot_interval: int
    precision_level: str
    precision_decimals: int
    ensemble_kind: str = "gradient_boosting"

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["features"] = list(self.features)
        value["tree_count_candidates"] = list(self.tree_count_candidates)
        return value

    def config_hash(self) -> str:
        return fingerprint({"schema": "experiment/v1", "config": self.as_dict()})

    def with_concurrency(self, concurrency: int) -> "ExperimentConfig":
        return replace(self, thread_pool_size=concurrency)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _get(mapping: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _failure(code: str, field: str, message: str) -> BusinessFailure:
    return BusinessFailure(code=code, field=field, message=message, stage="configuration")


def validate_experiment(
    config_data: dict[str, Any],
    spec_data: dict[str, Any],
    edit_batches: Iterable[dict[str, Any]] = (),
) -> tuple[ExperimentConfig | None, list[BusinessFailure]]:
    failures: list[BusinessFailure] = []
    batches = list(edit_batches)

    name = _get(config_data, ("experiment", "name"))
    if not isinstance(name, str) or not name:
        failures.append(_failure("CFG_MISSING_FIELD", "experiment.name", "experiment.name must be a non-empty string"))
        name = "unnamed"

    seed = _get(config_data, ("experiment", "seed"))
    if not _is_int(seed):
        failures.append(_failure("CFG_TYPE_MISMATCH", "experiment.seed", "experiment.seed must be an integer"))
        seed = 0

    features = _get(config_data, ("data", "features"))
    if not isinstance(features, list) or not features or not all(isinstance(item, str) and item for item in features):
        failures.append(_failure("CFG_MISSING_FIELD", "data.features", "data.features must be a non-empty string list"))
        features = []
    if len(set(features)) != len(features):
        failures.append(_failure("CFG_DUPLICATE_COLUMN", "data.features", "data.features contains duplicate names"))

    target_name = _get(config_data, ("data", "target"))
    if not isinstance(target_name, str) or not target_name:
        failures.append(_failure("CFG_TARGET_MISSING", "data.target", "data.target must name the binary target column"))
        target_name = ""

    columns = spec_data.get("columns")
    if not isinstance(columns, list):
        failures.append(_failure("CFG_TYPE_MISMATCH", "data_spec.columns", "data_spec.columns must be a list"))
        columns = []

    spec_columns: dict[str, dict[str, Any]] = {}
    for index, column in enumerate(columns):
        if not isinstance(column, dict):
            failures.append(_failure("CFG_TYPE_MISMATCH", f"data_spec.columns[{index}]", "column must be an object"))
            continue
        column_name = column.get("name")
        column_type = column.get("type")
        if not isinstance(column_name, str) or not column_name:
            failures.append(_failure("CFG_MISSING_FIELD", f"data_spec.columns[{index}].name", "column name is required"))
            continue
        if column_name in spec_columns:
            failures.append(_failure("CFG_DUPLICATE_COLUMN", f"data_spec.columns[{index}].name", "duplicate data-spec column"))
            continue
        copy_source = column.get("copy_of")
        if "copy_of" in column and (not isinstance(copy_source, str) or not copy_source):
            failures.append(_failure("CFG_UNKNOWN_COLUMN", f"{column_name}.copy_of", "copy_of must name a source column"))
            copy_source = None
        if column_type not in {"numeric", "categorical"}:
            failures.append(_failure("CFG_TYPE_MISMATCH", column_name, "column type must be numeric or categorical"))
        elif column_type == "numeric":
            if copy_source is None:
                if not _is_number(column.get("min")) or not _is_number(column.get("max")):
                    failures.append(_failure("CFG_TYPE_MISMATCH", column_name, "numeric columns require finite min and max"))
                elif float(column["min"]) > float(column["max"]):
                    failures.append(_failure("CFG_TYPE_MISMATCH", column_name, "numeric min must not exceed max"))
            missing_rate = column.get("missing_rate", 0.0)
            if not _is_number(missing_rate) or not (0.0 <= float(missing_rate) <= 1.0):
                failures.append(_failure("CFG_TYPE_MISMATCH", f"{column_name}.missing_rate", "missing_rate must be in [0, 1]"))
        else:
            categories = column.get("categories")
            if copy_source is None and (
                not isinstance(categories, list) or not categories or not all(isinstance(item, str) for item in categories)
            ):
                failures.append(
                    _failure("CFG_TYPE_MISMATCH", f"{column_name}.categories", "categorical categories must be a string list")
                )
            missing_rate = column.get("missing_rate", 0.0)
            if not _is_number(missing_rate) or not (0.0 <= float(missing_rate) <= 1.0):
                failures.append(_failure("CFG_TYPE_MISMATCH", f"{column_name}.missing_rate", "missing_rate must be in [0, 1]"))
            weights = column.get("weights")
            if copy_source is None and weights is not None:
                if (
                    not isinstance(categories, list)
                    or not isinstance(weights, list)
                    or len(weights) != len(categories)
                    or not all(_is_number(item) and float(item) >= 0.0 for item in weights)
                    or float(sum(float(item) for item in weights)) <= 0.0
                ):
                    failures.append(
                        _failure(
                            "CFG_TYPE_MISMATCH",
                            f"{column_name}.weights",
                            "categorical weights must be non-negative, positive in total, and match categories",
                        )
                    )
        spec_columns[column_name] = column

    for column_name, column in spec_columns.items():
        source = column.get("copy_of")
        if source is not None and isinstance(source, str):
            if source not in spec_columns:
                failures.append(
                    _failure("CFG_UNKNOWN_COLUMN", f"{column_name}.copy_of", f"copy_of source {source!r} is unknown")
                )
            elif column.get("type") != spec_columns[source].get("type"):
                failures.append(
                    _failure("CFG_TYPE_MISMATCH", column_name, "copy_of source and target types must match")
                )
            else:
                column_order = {item["name"]: index for index, item in enumerate(columns) if isinstance(item, dict) and "name" in item}
                if column_order.get(source, len(columns)) >= column_order.get(column_name, len(columns)):
                    failures.append(
                        _failure("CFG_TYPE_MISMATCH", column_name, "copy_of source must precede the copied column")
                    )

    spec_target = spec_data.get("target")
    if not isinstance(spec_target, dict):
        failures.append(_failure("CFG_TARGET_MISSING", "data_spec.target", "data_spec.target is required"))
        spec_target = {}
    else:
        spec_target_name = spec_target.get("name")
        if not isinstance(spec_target_name, str) or not spec_target_name:
            failures.append(_failure("CFG_TARGET_MISSING", "data_spec.target.name", "data-spec target name is required"))
        elif target_name and target_name != spec_target_name:
            failures.append(
                _failure(
                    "CFG_TARGET_MISSING",
                    "data.target",
                    f"configured target {target_name!r} is absent from data-spec target {spec_target_name!r}",
                )
            )
        if spec_target.get("type") != "binary":
            failures.append(_failure("CFG_TYPE_MISMATCH", "data_spec.target.type", "target type must be binary"))

    for feature in features:
        if feature not in spec_columns:
            failures.append(_failure("CFG_UNKNOWN_COLUMN", f"data.features.{feature}", "feature is not described by data_spec"))
    if target_name and target_name not in spec_columns and target_name != spec_target.get("name"):
        failures.append(_failure("CFG_TARGET_MISSING", "data.target", "target column is not described by data_spec"))

    tree_count = _get(config_data, ("model", "tree_count"))
    if not _is_int(tree_count) or tree_count <= 0:
        failures.append(_failure("CFG_TREE_COUNT_INVALID", "model.tree_count", "tree_count must be a positive integer"))

    candidates = _get(config_data, ("model", "tree_count_candidates"))
    if not isinstance(candidates, list) or not candidates or not all(_is_int(item) and item > 0 for item in candidates):
        failures.append(
            _failure(
                "CFG_TREE_COUNT_INVALID",
                "model.tree_count_candidates",
                "tree_count_candidates must be a non-empty list of positive integers",
            )
        )
        candidates = []
    else:
        candidates = list(candidates)

    max_depth = _get(config_data, ("model", "max_depth"))
    if not _is_int(max_depth) or max_depth <= 0:
        failures.append(_failure("CFG_MAX_DEPTH_INVALID", "model.max_depth", "max_depth must be a positive integer"))

    learning_rate = _get(config_data, ("model", "learning_rate"))
    if not _is_number(learning_rate) or float(learning_rate) < 0:
        failures.append(_failure("CFG_LEARNING_RATE_INVALID", "model.learning_rate", "learning_rate must be non-negative"))

    row_sample_ratio = _get(config_data, ("model", "row_sample_ratio"))
    if not _is_number(row_sample_ratio) or not (0 < float(row_sample_ratio) <= 1):
        failures.append(
            _failure("CFG_SAMPLE_RATIO_INVALID", "model.row_sample_ratio", "row_sample_ratio must be in (0, 1]")
        )

    col_sample_ratio = _get(config_data, ("model", "col_sample_ratio"))
    if not _is_number(col_sample_ratio) or not (0 < float(col_sample_ratio) <= 1):
        failures.append(
            _failure("CFG_SAMPLE_RATIO_INVALID", "model.col_sample_ratio", "col_sample_ratio must be in (0, 1]")
        )

    min_samples_leaf = _get(config_data, ("model", "min_samples_leaf"))
    if not _is_int(min_samples_leaf) or min_samples_leaf <= 0:
        failures.append(
            _failure("CFG_MIN_LEAF_INVALID", "model.min_samples_leaf", "min_samples_leaf must be a positive integer")
        )

    bin_count = _get(config_data, ("model", "bin_count"))
    if not _is_int(bin_count) or bin_count <= 0:
        failures.append(_failure("CFG_BIN_COUNT_INVALID", "model.bin_count", "bin_count must be a positive integer"))

    train_ratio = _get(config_data, ("model", "train_ratio"))
    if not _is_number(train_ratio) or not (0 < float(train_ratio) < 1):
        failures.append(_failure("CFG_SPLIT_RATIO_INVALID", "model.train_ratio", "train_ratio must be in (0, 1)"))

    concurrency = _get(config_data, ("runtime", "thread_pool_size"))
    if not _is_int(concurrency) or concurrency <= 0:
        failures.append(
            _failure("CFG_CONCURRENCY_INVALID", "runtime.thread_pool_size", "thread_pool_size must be a positive integer")
        )

    snapshot_interval = _get(config_data, ("runtime", "snapshot_interval"))
    if not _is_int(snapshot_interval) or snapshot_interval <= 0:
        failures.append(
            _failure(
                "CFG_SNAPSHOT_INTERVAL_INVALID",
                "runtime.snapshot_interval",
                "snapshot_interval must be a positive integer",
            )
        )

    precision_level = _get(config_data, ("precision", "level"))
    precision_decimals = _get(config_data, ("precision", "decimals"))
    if not isinstance(precision_level, str) or precision_level not in PRECISION_LEVELS:
        failures.append(
            _failure("CFG_PRECISION_INVALID", "precision.level", "precision.level must name a supported tier")
        )
    if precision_decimals is None and isinstance(precision_level, str):
        precision_decimals = PRECISION_LEVELS.get(precision_level, 8)
    if not _is_int(precision_decimals) or not (1 <= precision_decimals <= 15):
        failures.append(
            _failure("CFG_PRECISION_INVALID", "precision.decimals", "precision.decimals must be in [1, 15]")
        )

    ensemble_kind = _get(config_data, ("model", "kind"))
    if ensemble_kind not in {"gradient_boosting", "random_forest"}:
        failures.append(_failure("CFG_MODEL_KIND_INVALID", "model.kind", "model.kind must be gradient_boosting or random_forest"))

    if not failures and _is_int(tree_count) and candidates and tree_count not in candidates:
        failures.append(
            _failure(
                "CFG_TREE_COUNT_NOT_CANDIDATE",
                "model.tree_count",
                "tree_count must be selected from tree_count_candidates",
            )
        )

    failures.extend(_validate_edit_batches(batches, spec_columns, spec_target, features, target_name))

    if failures:
        return None, failures

    config = ExperimentConfig(
        name=name,
        seed=seed,
        features=tuple(features),
        tree_count=tree_count,
        tree_count_candidates=tuple(candidates),
        max_depth=max_depth,
        learning_rate=float(learning_rate),
        row_sample_ratio=float(row_sample_ratio),
        col_sample_ratio=float(col_sample_ratio),
        min_samples_leaf=min_samples_leaf,
        bin_count=bin_count,
        train_ratio=float(train_ratio),
        thread_pool_size=concurrency,
        snapshot_interval=snapshot_interval,
        precision_level=precision_level,
        precision_decimals=precision_decimals,
        ensemble_kind=ensemble_kind,
    )
    return config, []


def _validate_edit_batches(
    batches: list[dict[str, Any]],
    spec_columns: dict[str, dict[str, Any]],
    spec_target: dict[str, Any],
    features: list[str],
    target_name: str,
) -> list[BusinessFailure]:
    failures: list[BusinessFailure] = []
    current_tree_count: int | None = None
    for batch_index, batch in enumerate(batches):
        prefix = f"edits[{batch_index}]"
        if not isinstance(batch, dict):
            failures.append(_failure("CFG_TYPE_MISMATCH", prefix, "edit batch must be an object"))
            continue

        append = batch.get("append_rows")
        if append is not None:
            if not isinstance(append, dict):
                failures.append(_failure("CFG_TYPE_MISMATCH", f"{prefix}.append_rows", "append_rows must be an object"))
            else:
                count = append.get("count")
                append_seed = append.get("seed")
                if not _is_int(count) or count < 0:
                    failures.append(
                        _failure("CFG_TYPE_MISMATCH", f"{prefix}.append_rows.count", "append count must be non-negative integer")
                    )
                if not _is_int(append_seed):
                    failures.append(
                        _failure("CFG_TYPE_MISMATCH", f"{prefix}.append_rows.seed", "append seed must be an integer")
                    )

        corrections = batch.get("corrections", [])
        if corrections is None:
            corrections = []
        if not isinstance(corrections, list):
            failures.append(_failure("CFG_TYPE_MISMATCH", f"{prefix}.corrections", "corrections must be a list"))
            corrections = []
        for correction_index, correction in enumerate(corrections):
            location = f"{prefix}.corrections[{correction_index}]"
            if not isinstance(correction, dict):
                failures.append(_failure("CFG_TYPE_MISMATCH", location, "correction must be an object"))
                continue
            column = correction.get("column")
            if not isinstance(column, str) or not column:
                failures.append(_failure("CFG_UNKNOWN_COLUMN", f"{location}.column", "correction column is required"))
                continue
            if target_name and column == target_name:
                failures.append(_failure("CFG_TARGET_EDIT_FORBIDDEN", f"{location}.column", "target corrections are not allowed"))
                continue
            if column not in spec_columns:
                failures.append(
                    _failure("CFG_UNKNOWN_COLUMN", f"{location}.column", f"unknown correction column {column!r}")
                )
                continue
            if not _is_int(correction.get("row_id")) or correction.get("row_id") < 0:
                failures.append(
                    _failure("CFG_TYPE_MISMATCH", f"{location}.row_id", "row_id must be a non-negative integer")
                )
            value = correction.get("value")
            column_type = spec_columns[column].get("type")
            if column_type == "numeric" and not _is_number(value):
                failures.append(
                    _failure("CFG_TYPE_MISMATCH", f"{location}.value", "numeric correction value must be a finite number")
                )
            if column_type == "categorical" and not isinstance(value, str):
                failures.append(
                    _failure("CFG_TYPE_MISMATCH", f"{location}.value", "categorical correction value must be a string")
                )

        tree_edit_present = "tree_count" in batch or "tree_count_delta" in batch
        if tree_edit_present:
            if "tree_count" in batch and (not _is_int(batch["tree_count"]) or batch["tree_count"] <= 0):
                failures.append(_failure("CFG_TREE_COUNT_INVALID", f"{prefix}.tree_count", "tree_count edit must be positive"))
            if "tree_count_delta" in batch and (not _is_int(batch["tree_count_delta"]) or batch["tree_count_delta"] <= 0):
                failures.append(
                    _failure(
                        "CFG_TREE_COUNT_INVALID",
                        f"{prefix}.tree_count_delta",
                        "tree_count_delta must be a positive integer",
                    )
                )

        if "max_depth" in batch and (not _is_int(batch["max_depth"]) or batch["max_depth"] <= 0):
            failures.append(_failure("CFG_MAX_DEPTH_INVALID", f"{prefix}.max_depth", "max_depth edit must be positive"))
        if "learning_rate" in batch and (not _is_number(batch["learning_rate"]) or batch["learning_rate"] < 0):
            failures.append(
                _failure("CFG_LEARNING_RATE_INVALID", f"{prefix}.learning_rate", "learning_rate edit must be non-negative")
            )
        if "bin_count" in batch and (not _is_int(batch["bin_count"]) or batch["bin_count"] <= 0):
            failures.append(_failure("CFG_BIN_COUNT_INVALID", f"{prefix}.bin_count", "bin_count edit must be positive"))

    return failures


def apply_config_edits(config: ExperimentConfig, batches: Iterable[dict[str, Any]]) -> ExperimentConfig:
    updated = copy.deepcopy(config)
    tree_count = config.tree_count
    max_depth = config.max_depth
    learning_rate = config.learning_rate
    bin_count = config.bin_count
    for batch in batches:
        if "tree_count" in batch:
            next_tree_count = int(batch["tree_count"])
            if next_tree_count < tree_count:
                raise BusinessValidationError(
                    [
                        BusinessFailure(
                            code="CFG_TREE_COUNT_DECREASE",
                            field="edits.tree_count",
                            message="tree_count edits may only increase the ensemble",
                            stage="configuration",
                        )
                    ]
                )
            tree_count = next_tree_count
        if "tree_count_delta" in batch:
            next_tree_count = tree_count + int(batch["tree_count_delta"])
            if next_tree_count < tree_count:
                raise BusinessValidationError(
                    [
                        BusinessFailure(
                            code="CFG_TREE_COUNT_DECREASE",
                            field="edits.tree_count_delta",
                            message="tree_count_delta may only increase the ensemble",
                            stage="configuration",
                        )
                    ]
                )
            tree_count = next_tree_count
        if "max_depth" in batch:
            max_depth = int(batch["max_depth"])
        if "learning_rate" in batch:
            learning_rate = float(batch["learning_rate"])
        if "bin_count" in batch:
            bin_count = int(batch["bin_count"])
    updated = replace(
        updated,
        tree_count=tree_count,
        max_depth=max_depth,
        learning_rate=learning_rate,
        bin_count=bin_count,
    )
    if updated.tree_count not in updated.tree_count_candidates:
        raise BusinessValidationError(
            [
                BusinessFailure(
                    code="CFG_TREE_COUNT_NOT_CANDIDATE",
                    field="edits.tree_count",
                    message="effective tree_count must be selected from tree_count_candidates",
                    stage="configuration",
                )
            ]
        )
    return updated
