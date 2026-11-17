"""Strict format v1/v2/v3 decoding and normalization."""

from __future__ import annotations

import math
from typing import Any, Iterable

from .errors import FormatError, RunError
from .metrics import verify_generated_series
from .models import (
    AGGREGATION_NAMES,
    AGGREGATION_VERSION,
    CURRENT_FORMAT_VERSION,
    SUPPORTED_FORMAT_VERSIONS,
    Aggregation,
    Library,
    MetricPoint,
    Parameters,
    Provenance,
    RunRecord,
)
from .util import type_name

_COMMON_FIELDS = {
    "format_version",
    "run_id",
    "run_version",
    "parameters",
    "final_metrics",
}
_ALLOWED_FIELDS = {
    1: _COMMON_FIELDS,
    2: _COMMON_FIELDS | {"step_metrics", "timestamp"},
    3: _COMMON_FIELDS | {"step_metrics", "timestamp", "aggregation", "parent_run_id", "provenance"},
}
_PARAMETER_FIELDS = {"seed", "model_dim", "learning_rate", "hyperparameters"}
_ENVELOPE_FIELDS = {"library_schema", "library_id", "runs", "history"}


def _fail(path: str, message: str, **details: Any) -> None:
    raise FormatError(path, message, **details)


def _expect_object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(path, f"expected object, got {type_name(value)}")
    return value


def _expect_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(path, f"expected array, got {type_name(value)}")
    return value


def _expect_string(value: Any, path: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        _fail(path, f"expected string, got {type_name(value)}")
    if nonempty and not value:
        _fail(path, "expected a non-empty string")
    return value


def _expect_int(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, f"expected integer, got {type_name(value)}")
    if minimum is not None and value < minimum:
        _fail(path, f"expected integer >= {minimum}, got {value}")
    return value


def _expect_number(value: Any, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, f"expected number, got {type_name(value)}")
    number = float(value)
    if not math.isfinite(number):
        _fail(path, "expected a finite number")
    if positive and number <= 0:
        _fail(path, f"expected a positive number, got {number}")
    return number


def _reject_unknown(
    value: dict[str, Any], allowed: set[str], path: str, *, context: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        first = unknown[0]
        _fail(f"{path}.{first}", f"unknown field in {context}", fields=unknown)


def _required(value: dict[str, Any], key: str, path: str) -> Any:
    if key not in value:
        _fail(f"{path}.{key}", "required field is missing")
    return value[key]


def _parse_hyperparameters(value: Any, path: str) -> dict[str, Any]:
    hyperparameters = _expect_object(value, path)
    result: dict[str, Any] = {}
    for key in sorted(hyperparameters):
        item_path = f"{path}.{key}"
        if not isinstance(key, str) or not key:
            _fail(item_path, "hyperparameter keys must be non-empty strings")
        item = hyperparameters[key]
        if isinstance(item, dict) or isinstance(item, list):
            _fail(item_path, "hyperparameter values must be JSON scalars")
        if item is not None and not isinstance(item, (str, int, float, bool)):
            _fail(item_path, f"unsupported hyperparameter type {type_name(item)}")
        if isinstance(item, float) and not math.isfinite(item):
            _fail(item_path, "expected a finite number")
        result[key] = item
    if "steps" not in result:
        _fail(f"{path}.steps", "required hyperparameter is missing")
    steps = result["steps"]
    if isinstance(steps, bool) or not isinstance(steps, int):
        _fail(f"{path}.steps", f"expected integer, got {type_name(steps)}")
    if steps < 1 or steps > 100_000:
        _fail(f"{path}.steps", "expected an integer between 1 and 100000")
    return result


def _parse_parameters(value: Any, path: str) -> Parameters:
    parameters = _expect_object(value, path)
    _reject_unknown(parameters, _PARAMETER_FIELDS, path, context="parameters")
    seed = _expect_int(_required(parameters, "seed", path), f"{path}.seed")
    model_dim = _expect_int(
        _required(parameters, "model_dim", path), f"{path}.model_dim", minimum=1
    )
    learning_rate = _expect_number(
        _required(parameters, "learning_rate", path),
        f"{path}.learning_rate",
        positive=True,
    )
    hyperparameters = _parse_hyperparameters(
        _required(parameters, "hyperparameters", path), f"{path}.hyperparameters"
    )
    return Parameters(
        seed=seed,
        model_dim=model_dim,
        learning_rate=learning_rate,
        hyperparameters=hyperparameters,
    )


def _parse_metric_value(value: Any, path: str, field: str) -> float:
    return _expect_number(value, f"{path}.{field}")


def _parse_final_metrics(value: Any, path: str) -> dict[str, float]:
    final = _expect_object(value, path)
    _reject_unknown(final, {"loss", "accuracy"}, path, context="final metrics")
    return {
        "loss": _parse_metric_value(_required(final, "loss", path), path, "loss"),
        "accuracy": _parse_metric_value(
            _required(final, "accuracy", path), path, "accuracy"
        ),
    }


def _parse_step_metrics(value: Any, path: str) -> tuple[MetricPoint, ...]:
    rows = _expect_list(value, path)
    if not rows:
        _fail(path, "step_metrics must contain at least one row")
    if len(rows) > 100_000:
        _fail(path, "step_metrics contains too many rows")
    points: list[MetricPoint] = []
    for index, raw in enumerate(rows):
        row_path = f"{path}[{index}]"
        row = _expect_object(raw, row_path)
        _reject_unknown(row, {"step", "loss", "accuracy"}, row_path, context="metric row")
        step = _expect_int(_required(row, "step", row_path), f"{row_path}.step", minimum=1)
        if step != index + 1:
            _fail(f"{row_path}.step", f"expected contiguous step {index + 1}, got {step}")
        points.append(
            MetricPoint(
                step=step,
                loss=_parse_metric_value(_required(row, "loss", row_path), row_path, "loss"),
                accuracy=_parse_metric_value(
                    _required(row, "accuracy", row_path), row_path, "accuracy"
                ),
            )
        )
    return tuple(points)


def _parse_aggregation(value: Any, path: str) -> Aggregation:
    aggregation = _expect_object(value, path)
    _reject_unknown(aggregation, {"name", "version"}, path, context="aggregation")
    name = _expect_string(_required(aggregation, "name", path), f"{path}.name")
    version = _expect_string(
        _required(aggregation, "version", path), f"{path}.version"
    )
    if name not in AGGREGATION_NAMES:
        _fail(
            f"{path}.name",
            f"unsupported aggregation name {name!r}",
            supported=list(AGGREGATION_NAMES),
        )
    if version != AGGREGATION_VERSION:
        _fail(
            f"{path}.version",
            f"unsupported aggregation version {version!r}",
            supported=[AGGREGATION_VERSION],
        )
    return Aggregation(name=name, version=version)


def _parse_provenance(value: Any, path: str) -> Provenance:
    provenance = _expect_object(value, path)
    _reject_unknown(
        provenance,
        {"source_format_version", "upgraded", "original_digest", "aggregation_defaulted"},
        path,
        context="provenance",
    )
    source_format = _expect_int(
        _required(provenance, "source_format_version", path),
        f"{path}.source_format_version",
        minimum=1,
    )
    if source_format not in SUPPORTED_FORMAT_VERSIONS:
        _fail(
            f"{path}.source_format_version",
            f"unsupported source format version {source_format}",
            supported=list(SUPPORTED_FORMAT_VERSIONS),
        )
    upgraded = _required(provenance, "upgraded", path)
    if not isinstance(upgraded, bool):
        _fail(f"{path}.upgraded", f"expected boolean, got {type_name(upgraded)}")
    aggregation_defaulted = _required(provenance, "aggregation_defaulted", path)
    if not isinstance(aggregation_defaulted, bool):
        _fail(
            f"{path}.aggregation_defaulted",
            f"expected boolean, got {type_name(aggregation_defaulted)}",
        )
    original_digest = _required(provenance, "original_digest", path)
    if original_digest is not None and not isinstance(original_digest, str):
        _fail(
            f"{path}.original_digest",
            f"expected string or null, got {type_name(original_digest)}",
        )
    return Provenance(
        source_format_version=source_format,
        upgraded=upgraded,
        original_digest=original_digest,
        aggregation_defaulted=aggregation_defaulted,
    )


def parse_record(raw: Any, path: str = "record") -> RunRecord:
    record = _expect_object(raw, path)
    format_value = _required(record, "format_version", path)
    format_version = _expect_int(format_value, f"{path}.format_version", minimum=1)
    if format_version not in SUPPORTED_FORMAT_VERSIONS:
        _fail(
            f"{path}.format_version",
            f"unsupported format version {format_version}",
            supported=list(SUPPORTED_FORMAT_VERSIONS),
            reason="unsupported_newer_version" if format_version > CURRENT_FORMAT_VERSION else "unsupported_old_version",
        )

    _reject_unknown(
        record,
        _ALLOWED_FIELDS[format_version],
        path,
        context=f"format v{format_version} record",
    )
    run_id = _expect_string(_required(record, "run_id", path), f"{path}.run_id")
    run_version = _expect_int(
        _required(record, "run_version", path), f"{path}.run_version", minimum=1
    )
    parameters = _parse_parameters(
        _required(record, "parameters", path), f"{path}.parameters"
    )
    final_metrics = _parse_final_metrics(
        _required(record, "final_metrics", path), f"{path}.final_metrics"
    )

    if format_version == 1:
        from .metrics import generate_step_metrics

        points = generate_step_metrics(parameters)
        aggregation = Aggregation("last")
        timestamp = 1_700_000_000 + (parameters.seed % 1_000_000)
        parent_run_id = None
        provenance = Provenance(
            source_format_version=1,
            upgraded=False,
            original_digest=None,
            aggregation_defaulted=True,
        )
    elif format_version == 2:
        points = _parse_step_metrics(
            _required(record, "step_metrics", path), f"{path}.step_metrics"
        )
        timestamp = _expect_int(
            _required(record, "timestamp", path), f"{path}.timestamp"
        )
        aggregation = Aggregation("last")
        parent_run_id = None
        provenance = Provenance(
            source_format_version=2,
            upgraded=False,
            original_digest=None,
            aggregation_defaulted=True,
        )
    else:
        points = _parse_step_metrics(
            _required(record, "step_metrics", path), f"{path}.step_metrics"
        )
        timestamp = _expect_int(
            _required(record, "timestamp", path), f"{path}.timestamp"
        )
        aggregation = _parse_aggregation(
            _required(record, "aggregation", path), f"{path}.aggregation"
        )
        parent_value = _required(record, "parent_run_id", path)
        parent_run_id = (
            None
            if parent_value is None
            else _expect_string(parent_value, f"{path}.parent_run_id")
        )
        provenance = (
            _parse_provenance(record["provenance"], f"{path}.provenance")
            if "provenance" in record
            else Provenance(source_format_version=3)
        )

    normalized = RunRecord(
        run_id=run_id,
        run_version=run_version,
        parameters=parameters,
        step_metrics=points,
        timestamp=timestamp,
        aggregation=aggregation,
        parent_run_id=parent_run_id,
        final_metrics=final_metrics,
        format_version=format_version,
        provenance=provenance,
    )
    verify_generated_series(normalized, path=f"{path}.step_metrics")
    return normalized


def validate_parent_graph(records: Iterable[RunRecord]) -> None:
    current = {record.run_id: record for record in records}
    parents = {run_id: record.parent_run_id for run_id, record in current.items()}
    for run_id in sorted(parents):
        parent = parents[run_id]
        if parent is not None and parent not in parents:
            raise RunError(
                run_id,
                [run_id, parent],
                f"parent run {parent!r} does not exist",
                constraint="parent_exists",
            )

    state: dict[str, int] = {run_id: 0 for run_id in parents}
    for start in sorted(parents):
        if state[start] == 2:
            continue
        path: list[str] = []
        positions: dict[str, int] = {}
        cursor: str | None = start
        while cursor is not None:
            if cursor in positions:
                cycle = path[positions[cursor] :] + [cursor]
                raise RunError(
                    cycle[0],
                    cycle,
                    "parent_run_id reference cycle detected",
                    constraint="parent_acyclic",
                )
            if state[cursor] == 2:
                break
            state[cursor] = 1
            positions[cursor] = len(path)
            path.append(cursor)
            cursor = parents[cursor]
        for member in path:
            state[member] = 2


def validate_record_versions(
    current: Iterable[RunRecord], history: Iterable[RunRecord]
) -> None:
    current_list = list(current)
    seen_current: dict[str, RunRecord] = {}
    for record in current_list:
        if record.run_id in seen_current:
            raise RunError(
                record.run_id,
                [record.run_id],
                "duplicate current run_id",
                constraint="run_id_unique",
            )
        seen_current[record.run_id] = record

    grouped: dict[str, list[RunRecord]] = {}
    for record in history:
        grouped.setdefault(record.run_id, []).append(record)
    for run_id, records in grouped.items():
        versions = [record.run_version for record in records]
        if len(versions) != len(set(versions)):
            raise RunError(
                run_id,
                [f"{run_id}@v{version}" for version in sorted(versions)],
                "duplicate historical run_version",
                constraint="run_version_monotonic",
                versions=versions,
            )
        current_record = seen_current.get(run_id)
        if current_record is None:
            raise RunError(
                run_id,
                [f"{run_id}@v{version}" for version in sorted(versions)],
                "history exists without a current record",
                constraint="run_version_monotonic",
                versions=versions,
            )
        if current_record.run_version <= max(versions):
            raise RunError(
                run_id,
                [f"{run_id}@v{version}" for version in sorted(versions + [current_record.run_version])],
                "current run_version must be greater than every historical version",
                constraint="run_version_monotonic",
                versions=sorted(versions + [current_record.run_version]),
            )


def parse_library(raw: Any, path: str = "$") -> Library:
    envelope = _expect_object(raw, path)
    _reject_unknown(envelope, _ENVELOPE_FIELDS, path, context="library")
    schema = envelope.get("library_schema", 1)
    schema_version = _expect_int(schema, f"{path}.library_schema", minimum=1)
    if schema_version != 1:
        _fail(
            f"{path}.library_schema",
            f"unsupported library schema {schema_version}",
            supported=[1],
        )
    library_id = _expect_string(
        _required(envelope, "library_id", path), f"{path}.library_id"
    )
    raw_runs = _expect_list(_required(envelope, "runs", path), f"{path}.runs")
    raw_history = _expect_list(envelope.get("history", []), f"{path}.history")

    current_records = [
        parse_record(item, f"{path}.runs[{index}]")
        for index, item in enumerate(raw_runs)
    ]
    history_records = [
        parse_record(item, f"{path}.history[{index}]")
        for index, item in enumerate(raw_history)
    ]

    validate_record_versions(current_records, history_records)
    validate_parent_graph(current_records)

    history: dict[str, list[RunRecord]] = {}
    for record in history_records:
        history.setdefault(record.run_id, []).append(record)
    for run_id in history:
        history[run_id].sort(key=lambda item: item.run_version)
    return Library(
        library_id=library_id,
        runs=sorted(current_records, key=lambda item: item.run_id),
        history=history,
    )


def parse_import_document(raw: Any, path: str = "$") -> Library:
    """Import declarations use the same envelope as the on-disk library."""

    return parse_library(raw, path=path)
