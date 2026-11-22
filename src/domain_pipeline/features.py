from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any

from .canonical import (
    IMPLEMENTATION_VERSION,
    canonical_json_text,
    fingerprint,
    stable_float,
)
from .config import PipelineConfig
from .loader import ValidSample, feature_input_fingerprint
from .scheduler import ScheduledTask, run_tasks


@dataclass
class FeatureRun:
    artifact: dict[str, Any] | None
    partial_columns: dict[str, Any]
    complete: bool
    pending: list[str]
    budget_report: dict[str, Any]
    rows_total: int
    rows_reused: int
    rows_computed: int
    row_cache: dict[str, dict[str, Any]]


def feature_cache_key(config: PipelineConfig, samples: list[ValidSample]) -> str:
    return fingerprint(
        "feature-stage:v1",
        {
            "implementation": IMPLEMENTATION_VERSION,
            "params": config.feature_params(),
            "input": feature_input_fingerprint(samples),
        },
    )


def _content_fingerprint(column: dict[str, Any]) -> str:
    return fingerprint("feature-column-content:v1", column)


def _numeric_quantile_boundaries(
    values: list[float | None], bins: int
) -> list[float]:
    present = sorted(value for value in values if value is not None)
    if not present or bins == 1:
        return []
    boundaries: list[float] = []
    for index in range(1, bins):
        position = math.floor(index * len(present) / bins)
        position = min(max(position, 0), len(present) - 1)
        boundaries.append(stable_float(present[position]))
    return boundaries


def _row_cache_key(
    sample: ValidSample,
    feature_params: dict[str, Any],
    schema_fingerprint: str,
) -> str:
    return fingerprint(
        "feature-row:v1",
        {
            "sample_id": sample.sample_id,
            "numeric": {
                column: sample.numeric[column]
                for column in sorted(sample.numeric)
            },
            "categorical": {
                column: sample.categorical[column]
                for column in sorted(sample.categorical)
            },
            "params": feature_params,
            "feature_schema_fingerprint": schema_fingerprint,
        },
    )


def _encode_row(
    sample: ValidSample,
    numeric_columns: list[dict[str, Any]],
    categorical_columns: list[dict[str, Any]],
    feature_names: list[str],
) -> dict[str, Any]:
    vector = [0.0] * len(feature_names)
    numeric_output: dict[str, dict[str, Any]] = {}
    categorical_output: dict[str, dict[str, Any]] = {}
    position = 0

    for column in numeric_columns:
        name = column["name"]
        bins = column["bins"]
        value = sample.numeric[name]
        missing = value is None
        bin_index = bins if missing else bisect.bisect_right(column["boundaries"], value)
        numeric_output[name] = {"bin": bin_index, "missing": missing}
        vector[position + bin_index] = 1.0
        position += bins + 1
        if missing:
            vector[position] = 1.0
        position += 1

    for column in categorical_columns:
        name = column["name"]
        value = sample.categorical[name]
        missing = value is None
        if missing:
            code = column["missing_code"]
        else:
            code = column["code_by_value"][canonical_json_text(value)]
        categorical_output[name] = {"code": code, "missing": missing}
        vector[position + code] = 1.0
        position += len(column["categories"]) + 1
        if missing:
            vector[position] = 1.0
        position += 1

    return {
        "sample_id": sample.sample_id,
        "numeric": numeric_output,
        "categorical": categorical_output,
        "vector": vector,
    }


def _column_artifacts(
    samples: list[ValidSample],
    config: PipelineConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], str]:
    bins = int(config.features["bins"])
    numeric_columns: list[dict[str, Any]] = []
    categorical_columns: list[dict[str, Any]] = []
    feature_names: list[str] = []

    for name in config.data["numeric_columns"]:
        values = [sample.numeric[name] for sample in samples]
        boundaries = _numeric_quantile_boundaries(values, bins)
        column: dict[str, Any] = {
            "name": name,
            "kind": "numeric",
            "bins": bins,
            "missing_bin": bins,
            "boundaries": boundaries,
            "input_values": values,
        }
        column["parameter_fingerprint"] = fingerprint(
            "numeric-column-params:v1",
            {"name": name, "bins": bins, "missing_bin": bins},
        )
        column["content_fingerprint"] = _content_fingerprint(column)
        numeric_columns.append(column)
        for bin_index in range(bins + 1):
            feature_names.append(f"{name}__bin_{bin_index}")
        feature_names.append(f"{name}__missing")

    for name in config.data["categorical_columns"]:
        categories = list(config.data["categorical_domains"][name])
        code_by_value = {
            canonical_json_text(value): index
            for index, value in enumerate(categories)
        }
        values = [sample.categorical[name] for sample in samples]
        column = {
            "name": name,
            "kind": "categorical",
            "categories": categories,
            "code_by_value": code_by_value,
            "missing_code": len(categories),
            "input_values": values,
        }
        column["parameter_fingerprint"] = fingerprint(
            "categorical-column-params:v1",
            {
                "name": name,
                "categories": categories,
                "missing_code": len(categories),
            },
        )
        column["content_fingerprint"] = _content_fingerprint(column)
        categorical_columns.append(column)
        for index in range(len(categories) + 1):
            feature_names.append(f"{name}__code_{index}")
        feature_names.append(f"{name}__missing")

    schema_fingerprint = fingerprint(
        "feature-encoding-schema:v1",
        {
            "feature_names": feature_names,
            "numeric_columns": [
                {
                    key: column[key]
                    for key in (
                        "name",
                        "bins",
                        "missing_bin",
                        "boundaries",
                    )
                }
                for column in numeric_columns
            ],
            "categorical_columns": [
                {
                    key: column[key]
                    for key in (
                        "name",
                        "categories",
                        "code_by_value",
                        "missing_code",
                    )
                }
                for column in categorical_columns
            ],
        },
    )
    return numeric_columns, categorical_columns, feature_names, schema_fingerprint


def compute_features(
    config: PipelineConfig,
    samples: list[ValidSample],
    *,
    row_cache: dict[str, dict[str, Any]],
) -> FeatureRun:
    feature_params = config.feature_params()
    numeric_columns, categorical_columns, feature_names, schema_fingerprint = (
        _column_artifacts(samples, config)
    )
    tasks: list[ScheduledTask] = []
    row_keys: dict[str, str] = {}
    newly_cached: dict[str, dict[str, Any]] = {}

    for sample in sorted(samples, key=lambda item: item.sample_id):
        row_key = _row_cache_key(sample, feature_params, schema_fingerprint)
        row_keys[sample.sample_id] = row_key
        cached = row_cache.get(row_key)
        if cached is not None:
            tasks.append(
                ScheduledTask(
                    key=f"feature:{sample.sample_id}",
                    cost=0,
                    function=lambda cached=cached: cached,
                )
            )
        else:
            tasks.append(
                ScheduledTask(
                    key=f"feature:{sample.sample_id}",
                    cost=1,
                    function=lambda sample=sample: _encode_row(
                        sample,
                        numeric_columns,
                        categorical_columns,
                        feature_names,
                    ),
                )
            )

    schedule = run_tasks(
        tasks,
        max_workers=int(config.concurrency["max_workers"]),
        budget=int(config.concurrency["batch_budget"]),
        budget_name="feature_rows",
    )
    rows: list[dict[str, Any]] = []
    rows_reused = 0
    rows_computed = 0
    for sample in sorted(samples, key=lambda item: item.sample_id):
        task_key = f"feature:{sample.sample_id}"
        if task_key not in schedule.results:
            continue
        row = schedule.results[task_key]
        rows.append(row)
        row_key = row_keys[sample.sample_id]
        if row_key in row_cache:
            rows_reused += 1
        else:
            rows_computed += 1
            newly_cached[row_key] = row

    updated_cache = dict(row_cache)
    updated_cache.update(newly_cached)
    columns = {
        "numeric": numeric_columns,
        "categorical": categorical_columns,
        "feature_names": feature_names,
        "schema_fingerprint": schema_fingerprint,
    }
    if schedule.pending:
        return FeatureRun(
            artifact=None,
            partial_columns=columns,
            complete=False,
            pending=schedule.pending,
            budget_report=schedule.budget_report,
            rows_total=len(samples),
            rows_reused=rows_reused,
            rows_computed=rows_computed,
            row_cache=updated_cache,
        )

    rows.sort(key=lambda row: row["sample_id"])
    artifact: dict[str, Any] = {
        "artifact_type": "features",
        "cache_key": feature_cache_key(config, samples),
        "implementation": IMPLEMENTATION_VERSION,
        "params_fingerprint": fingerprint("feature-params:v1", feature_params),
        "params": feature_params,
        "feature_input_fingerprint": feature_input_fingerprint(samples),
        "sample_ids": [sample.sample_id for sample in samples],
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "feature_names": feature_names,
        "feature_schema_fingerprint": schema_fingerprint,
        "rows": rows,
    }
    artifact["content_fingerprint"] = fingerprint("feature-artifact:v1", artifact)
    return FeatureRun(
        artifact=artifact,
        partial_columns=columns,
        complete=True,
        pending=[],
        budget_report=schedule.budget_report,
        rows_total=len(samples),
        rows_reused=rows_reused,
        rows_computed=rows_computed,
        row_cache=updated_cache,
    )
