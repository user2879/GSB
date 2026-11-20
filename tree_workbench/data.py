"""Deterministic synthetic-data generation, editing, cleaning, and splitting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .config import ExperimentConfig
from .errors import BusinessFailure, BusinessValidationError
from .fingerprint import fingerprint, json_safe, rounded


@dataclass(frozen=True)
class PreparedData:
    row_ids: tuple[int, ...]
    raw_rows: tuple[dict[str, Any], ...]
    feature_names: tuple[str, ...]
    feature_types: dict[str, str]
    feature_values: tuple[tuple[float | int, ...], ...]
    target: tuple[int, ...]
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    numeric_fill: dict[str, float]
    categorical_fill: dict[str, str]
    category_codes: dict[str, dict[str, int]]
    base_probability: float
    base_score: float
    data_fingerprint: str
    decimals: int

    @property
    def row_count(self) -> int:
        return len(self.row_ids)

    def values_for_row(self, row_index: int) -> tuple[float | int, ...]:
        return self.feature_values[row_index]

    def summary(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "feature_names": list(self.feature_names),
            "feature_types": dict(sorted(self.feature_types.items())),
            "train_rows": [self.row_ids[index] for index in self.train_indices],
            "validation_rows": [self.row_ids[index] for index in self.validation_indices],
            "numeric_fill": {key: rounded(value, self.decimals) for key, value in sorted(self.numeric_fill.items())},
            "categorical_fill": dict(sorted(self.categorical_fill.items())),
            "category_codes": {key: dict(sorted(value.items())) for key, value in sorted(self.category_codes.items())},
            "base_probability": rounded(self.base_probability, self.decimals),
            "base_score": rounded(self.base_score, self.decimals),
            "data_fingerprint": self.data_fingerprint,
        }


def _stable_seed(seed: int, tag: int, count: int) -> np.random.Generator:
    sequence = np.random.SeedSequence([int(seed), int(tag), int(count)])
    return np.random.default_rng(sequence)


def _column_median(rows: list[dict[str, Any]], name: str) -> float:
    values = [float(row[name]) for row in rows if row.get(name) is not None]
    if not values:
        raise BusinessValidationError(
            [
                BusinessFailure(
                    code="DATA_NO_VALID_NUMERIC",
                    field=name,
                    message="numeric column has no non-missing values for median imputation",
                    stage="data",
                )
            ]
        )
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _category_mode(rows: list[dict[str, Any]], name: str) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(name)
        if value is not None:
            counts[str(value)] = counts.get(str(value), 0) + 1
    if not counts:
        return "__MISSING__"
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _generate_rows(spec: dict[str, Any], seed: int, count: int, start_row_id: int) -> list[dict[str, Any]]:
    if count == 0:
        return []
    rng = _stable_seed(seed, 0x51A7, count + start_row_id)
    columns = spec["columns"]
    generated: dict[str, list[Any]] = {column["name"]: [None] * count for column in columns}

    for column in columns:
        name = column["name"]
        if "copy_of" in column:
            source = column["copy_of"]
            if source not in generated:
                raise BusinessValidationError(
                    [
                        BusinessFailure(
                            code="CFG_UNKNOWN_COLUMN",
                            field=name,
                            message=f"copy_of source {source!r} is unknown",
                            stage="configuration",
                        )
                    ]
                )
            generated[name] = list(generated[source])
            continue

        column_type = column["type"]
        if column_type == "numeric":
            low = float(column["min"])
            high = float(column["max"])
            values = rng.uniform(low, high, size=count).astype(float)
            missing_rate = float(column.get("missing_rate", 0.0))
            missing_mask = rng.random(count) < missing_rate
            out: list[Any] = []
            for value, missing in zip(values, missing_mask):
                out.append(None if bool(missing) else float(value))
            generated[name] = out
        elif column_type == "categorical":
            categories = [str(value) for value in column["categories"]]
            weights = column.get("weights")
            if weights is None:
                probabilities = None
            else:
                total = float(sum(float(item) for item in weights))
                probabilities = [float(item) / total for item in weights]
            values = rng.choice(categories, size=count, p=probabilities)
            missing_rate = float(column.get("missing_rate", 0.0))
            missing_mask = rng.random(count) < missing_rate
            out = []
            for value, missing in zip(values, missing_mask):
                out.append(None if bool(missing) else str(value))
            generated[name] = out
        else:
            raise BusinessValidationError(
                [
                    BusinessFailure(
                        code="CFG_TYPE_MISMATCH",
                        field=name,
                        message="column type must be numeric or categorical",
                        stage="configuration",
                    )
                ]
            )

    # Target generation uses train-independent raw feature values. Missing numeric values
    # and categories are represented by neutral values for the latent score only; the
    # modeling imputation remains train-only.
    numeric_medians: dict[str, float] = {}
    for column in columns:
        if column["type"] == "numeric" and "copy_of" not in column:
            numeric_medians[column["name"]] = _column_median(
                [{"value": value} for value in generated[column["name"]]],
                "value",
            )

    target_spec = spec["target"]
    intercept = float(target_spec.get("intercept", 0.0))
    noise_sigma = float(target_spec.get("noise", 0.0))
    target_values: list[int] = []
    for row_offset in range(count):
        logit = intercept
        for column in columns:
            name = column["name"]
            if "copy_of" in column:
                continue
            value = generated[name][row_offset]
            if column["type"] == "numeric":
                coefficient = float(column.get("coefficient", 0.0))
                numeric_value = numeric_medians[name] if value is None else float(value)
                logit += coefficient * numeric_value
            else:
                effects = column.get("effects", {})
                if value is not None:
                    logit += float(effects.get(str(value), 0.0))
        logit += float(rng.normal(0.0, noise_sigma))
        target_values.append(1 if logit > 0 else 0)

    rows: list[dict[str, Any]] = []
    for row_offset in range(count):
        row = {column["name"]: generated[column["name"]][row_offset] for column in columns}
        row[target_spec["name"]] = target_values[row_offset]
        row["__row_id__"] = start_row_id + row_offset
        rows.append(row)
    return rows


def _apply_batches(
    spec: dict[str, Any],
    base_seed: int,
    batches: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = _generate_rows(spec, base_seed, int(spec["row_count"]), 0)
    column_types = {column["name"]: column["type"] for column in spec["columns"]}
    target_name = spec["target"]["name"]
    for batch_index, batch in enumerate(batches):
        append = batch.get("append_rows")
        if append is not None:
            rows.extend(
                _generate_rows(
                    spec,
                    int(append["seed"]),
                    int(append["count"]),
                    len(rows),
                )
            )
        for correction_index, correction in enumerate(batch.get("corrections", []) or []):
            row_id = int(correction["row_id"])
            column = str(correction["column"])
            if column == target_name:
                raise BusinessValidationError(
                    [
                        BusinessFailure(
                            code="CFG_TARGET_EDIT_FORBIDDEN",
                            field=f"edits[{batch_index}].corrections[{correction_index}].column",
                            message="target corrections are not allowed",
                            stage="configuration",
                        )
                    ]
                )
            if column not in column_types:
                raise BusinessValidationError(
                    [
                        BusinessFailure(
                            code="CFG_UNKNOWN_COLUMN",
                            field=f"edits[{batch_index}].corrections[{correction_index}].column",
                            message=f"unknown correction column {column!r}",
                            stage="configuration",
                        )
                    ]
                )
            if row_id < 0 or row_id >= len(rows):
                raise BusinessValidationError(
                    [
                        BusinessFailure(
                            code="EDIT_ROW_NOT_FOUND",
                            field=f"edits[{batch_index}].corrections[{correction_index}].row_id",
                            message=f"row_id {row_id} is outside the current data",
                            stage="data",
                        )
                    ]
                )
            value = correction.get("value")
            column_type = column_types[column]
            if column_type == "numeric":
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise BusinessValidationError(
                        [
                            BusinessFailure(
                                code="CFG_TYPE_MISMATCH",
                                field=f"edits[{batch_index}].corrections[{correction_index}].value",
                                message="numeric correction value must be a finite number",
                                stage="configuration",
                            )
                        ]
                    )
                rows[row_id][column] = float(value)
            else:
                if not isinstance(value, str):
                    raise BusinessValidationError(
                        [
                            BusinessFailure(
                                code="CFG_TYPE_MISMATCH",
                                field=f"edits[{batch_index}].corrections[{correction_index}].value",
                                message="categorical correction value must be a string",
                                stage="configuration",
                            )
                        ]
                    )
                rows[row_id][column] = value
    return rows


def prepare_data(
    config: ExperimentConfig,
    spec: dict[str, Any],
    batches: Iterable[dict[str, Any]],
) -> PreparedData:
    batches = list(batches)
    rows = _apply_batches(spec, int(spec.get("seed", config.seed)), batches)
    row_count = len(rows)
    if row_count < 2:
        raise BusinessValidationError(
            [
                BusinessFailure(
                    code="DATA_SPLIT_INVALID",
                    field="data",
                    message="at least two rows are required to create train and validation subsets",
                    stage="data",
                )
            ]
        )

    split_rng = _stable_seed(config.seed, 0x5EED, row_count)
    permutation = split_rng.permutation(row_count)
    train_count = int(round(row_count * config.train_ratio))
    train_count = max(1, min(row_count - 1, train_count))
    train_indices = tuple(sorted(int(index) for index in permutation[:train_count]))
    validation_indices = tuple(sorted(int(index) for index in permutation[train_count:]))
    if len(train_indices) < config.min_samples_leaf:
        raise BusinessValidationError(
            [
                BusinessFailure(
                    code="DATA_MIN_LEAF_UNSATISFIABLE",
                    field="model.min_samples_leaf",
                    message="training subset is smaller than min_samples_leaf",
                    stage="data",
                )
            ]
        )

    columns = {column["name"]: column for column in spec["columns"]}
    numeric_fill: dict[str, float] = {}
    categorical_fill: dict[str, str] = {}
    for feature in config.features:
        column = columns[feature]
        if column["type"] == "numeric":
            train_rows = [rows[index] for index in train_indices]
            numeric_fill[feature] = _column_median(train_rows, feature)
        else:
            train_rows = [rows[index] for index in train_indices]
            categorical_fill[feature] = _category_mode(train_rows, feature)

    encoded_values: list[tuple[float | int, ...]] = []
    category_codes: dict[str, dict[str, int]] = {}
    for feature in config.features:
        column = columns[feature]
        if column["type"] == "categorical":
            categories = {
                str(row[feature]) if row.get(feature) is not None else categorical_fill[feature]
                for row in rows
            }
            categories.add(categorical_fill[feature])
            category_codes[feature] = {value: index for index, value in enumerate(sorted(categories))}

    for row in rows:
        encoded: list[float | int] = []
        for feature in config.features:
            column = columns[feature]
            value = row.get(feature)
            if column["type"] == "numeric":
                encoded.append(numeric_fill[feature] if value is None else float(value))
            else:
                category = categorical_fill[feature] if value is None else str(value)
                encoded.append(category_codes[feature][category])
        encoded_values.append(tuple(encoded))

    target_name = spec["target"]["name"]
    target = tuple(int(row[target_name]) for row in rows)
    if len(set(target[index] for index in train_indices)) < 2:
        raise BusinessValidationError(
            [
                BusinessFailure(
                    code="DATA_TARGET_SINGLE_CLASS",
                    field=target_name,
                    message="training target must contain both binary classes",
                    stage="data",
                )
            ]
        )

    train_positive = sum(target[index] for index in train_indices)
    base_probability = train_positive / len(train_indices)
    clipped_probability = min(1.0 - 1e-12, max(1e-12, base_probability))
    base_score = math.log(clipped_probability / (1.0 - clipped_probability))

    raw_rows = tuple({key: value for key, value in row.items() if key != "__row_id__"} for row in rows)
    row_ids = tuple(int(row["__row_id__"]) for row in rows)
    fill_for_hash = {
        "numeric": {key: rounded(value, config.precision_decimals) for key, value in sorted(numeric_fill.items())},
        "categorical": dict(sorted(categorical_fill.items())),
    }
    data_fingerprint = fingerprint(
        {
            "schema": "prepared-data/v1",
            "seed": config.seed,
            "features": list(config.features),
            "row_ids": list(row_ids),
            "raw_rows": json_safe(list(raw_rows)),
            "feature_values": json_safe([list(row) for row in encoded_values]),
            "target": list(target),
            "train_indices": list(train_indices),
            "validation_indices": list(validation_indices),
            "fill": fill_for_hash,
            "category_codes": category_codes,
            "base_probability": rounded(base_probability, config.precision_decimals),
        }
    )

    return PreparedData(
        row_ids=row_ids,
        raw_rows=raw_rows,
        feature_names=tuple(config.features),
        feature_types={feature: columns[feature]["type"] for feature in config.features},
        feature_values=tuple(encoded_values),
        target=target,
        train_indices=train_indices,
        validation_indices=validation_indices,
        numeric_fill=numeric_fill,
        categorical_fill=categorical_fill,
        category_codes=category_codes,
        base_probability=base_probability,
        base_score=base_score,
        data_fingerprint=data_fingerprint,
        decimals=config.precision_decimals,
    )
