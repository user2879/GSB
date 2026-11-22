"""Strict JSONL sample loading and deterministic whole-state validation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from .errors import E_ROW_INVALID, make_failure
from .models import AppConfig, METADATA_FIELDS, Row


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON number {value}")


def _line_failure(
    message: str,
    *,
    line: int,
    row_id: str | None = None,
    entity_id: str | None = None,
    **details: Any,
) -> dict[str, Any]:
    payload = {"line": line}
    payload.update(details)
    return make_failure(
        E_ROW_INVALID,
        message,
        stage="load_validation",
        row_id=row_id,
        entity_id=entity_id,
        details=payload,
    )


def _normalise_feature_value(
    value: Any,
    *,
    field: str,
    missing_marker: str,
) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, str) and value == missing_marker:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"field {field!r} must be a finite number or missing marker")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"field {field!r} is not finite")
    return value


def parse_row_record(
    record: Any,
    *,
    line: int,
    config: AppConfig,
    raw_fields: Iterable[str],
) -> tuple[Row | None, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    if not isinstance(record, dict):
        return None, [_line_failure("sample row must be a JSON object", line=line)]
    row_id = record.get("row_id")
    entity_id = record.get("entity_id")
    timestamp = record.get("timestamp_ms")
    if not isinstance(row_id, str) or not row_id:
        failures.append(_line_failure("row_id must be a non-empty string", line=line))
    if not isinstance(entity_id, str) or not entity_id:
        failures.append(
            _line_failure(
                "entity_id must be a non-empty string",
                line=line,
                row_id=row_id if isinstance(row_id, str) else None,
            )
        )
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, int)
        or timestamp < 0
    ):
        failures.append(
            _line_failure(
                "timestamp_ms must be a non-negative integer",
                line=line,
                row_id=row_id if isinstance(row_id, str) else None,
                entity_id=entity_id if isinstance(entity_id, str) else None,
            )
        )
    values: dict[str, float | int | None] = {}
    raw_field_set = set(str(value) for value in raw_fields)
    missing_fields = record.get("missing_fields", [])
    if missing_fields is None:
        missing_fields = []
    if not isinstance(missing_fields, list) or not all(
        isinstance(item, str) for item in missing_fields
    ):
        failures.append(
            _line_failure(
                "missing_fields must be a list of strings",
                line=line,
                row_id=row_id if isinstance(row_id, str) else None,
                entity_id=entity_id if isinstance(entity_id, str) else None,
            )
        )
        missing_fields = []
    for field in sorted(raw_field_set):
        if field in record:
            try:
                values[field] = _normalise_feature_value(
                    record[field], field=field, missing_marker=config.missing_marker
                )
            except ValueError as exc:
                failures.append(
                    _line_failure(
                        str(exc),
                        line=line,
                        row_id=row_id if isinstance(row_id, str) else None,
                        entity_id=entity_id if isinstance(entity_id, str) else None,
                        feature=field,
                    )
                )
        elif field in missing_fields:
            values[field] = None
        else:
            # Sparse JSONL is treated as an explicit missing observation.
            values[field] = None
    # Unknown fields that are clearly numeric are checked too. This prevents a
    # typo in a raw field from smuggling NaN into the deterministic pipeline.
    for field, value in record.items():
        if field in METADATA_FIELDS or field in raw_field_set:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(float(value)):
                failures.append(
                    _line_failure(
                        f"unknown numeric field {field!r} is not finite",
                        line=line,
                        row_id=row_id if isinstance(row_id, str) else None,
                        entity_id=entity_id if isinstance(entity_id, str) else None,
                        feature=field,
                    )
                )
    if failures:
        return None, failures
    return (
        Row(
            row_id=row_id,
            entity_id=entity_id,
            timestamp_ms=timestamp,
            values=values,
        ),
        [],
    )


def validate_entity_sequences(rows: list[Row]) -> list[dict[str, Any]]:
    """Validate row-id uniqueness and per-entity strictly increasing order.

    The row list is intentionally kept in edit/insertion order. This catches a
    reversed append instead of hiding it by sorting before validation.
    """

    failures: list[dict[str, Any]] = []
    seen_rows: dict[str, int] = {}
    last_timestamp: dict[str, int] = {}
    for index, row in enumerate(rows):
        if row.row_id in seen_rows:
            failures.append(
                make_failure(
                    E_ROW_INVALID,
                    f"duplicate row_id {row.row_id}",
                    stage="load_validation",
                    entity_id=row.entity_id,
                    row_id=row.row_id,
                    details={"index": index, "first_index": seen_rows[row.row_id]},
                )
            )
        else:
            seen_rows[row.row_id] = index
        if row.timestamp_ms < 0:
            failures.append(
                make_failure(
                    E_ROW_INVALID,
                    "timestamp_ms must be non-negative",
                    stage="load_validation",
                    entity_id=row.entity_id,
                    row_id=row.row_id,
                    details={"timestamp_ms": row.timestamp_ms},
                )
            )
        previous = last_timestamp.get(row.entity_id)
        if previous is not None and row.timestamp_ms <= previous:
            failures.append(
                make_failure(
                    E_ROW_INVALID,
                    "timestamps must be strictly increasing within an entity",
                    stage="load_validation",
                    entity_id=row.entity_id,
                    row_id=row.row_id,
                    details={"previous_timestamp_ms": previous, "timestamp_ms": row.timestamp_ms},
                )
            )
        last_timestamp[row.entity_id] = row.timestamp_ms
    return failures


def load_samples(
    path: str | Path,
    *,
    config: AppConfig,
    raw_fields: Iterable[str],
) -> tuple[list[Row], list[dict[str, Any]]]:
    path = Path(path)
    rows: list[Row] = []
    failures: list[dict[str, Any]] = []
    seen_row_ids: set[str] = set()
    last_timestamp: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:  # noqa: BLE001
        return [], [
            make_failure(
                E_ROW_INVALID,
                f"cannot read samples JSONL: {exc}",
                stage="load_validation",
                details={"path": str(path)},
            )
        ]
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            failures.append(
                _line_failure("blank sample line is not allowed", line=line_number)
            )
            continue
        try:
            record = json.loads(line, parse_constant=_reject_json_constant)
        except Exception as exc:  # noqa: BLE001
            failures.append(
                _line_failure(
                    f"invalid JSON: {exc}",
                    line=line_number,
                )
            )
            continue
        row, row_failures = parse_row_record(
            record, line=line_number, config=config, raw_fields=raw_fields
        )
        if row_failures:
            failures.extend(row_failures)
            continue
        assert row is not None
        if row.row_id in seen_row_ids:
            failures.append(
                _line_failure(
                    f"duplicate row_id {row.row_id}",
                    line=line_number,
                    row_id=row.row_id,
                    entity_id=row.entity_id,
                )
            )
            continue
        previous = last_timestamp.get(row.entity_id)
        if previous is not None and row.timestamp_ms <= previous:
            failures.append(
                _line_failure(
                    "timestamps must be strictly increasing within an entity",
                    line=line_number,
                    row_id=row.row_id,
                    entity_id=row.entity_id,
                    previous_timestamp_ms=previous,
                    timestamp_ms=row.timestamp_ms,
                )
            )
            continue
        seen_row_ids.add(row.row_id)
        last_timestamp[row.entity_id] = row.timestamp_ms
        rows.append(row)
    return rows, failures


def rows_by_entity(rows: Iterable[Row]) -> dict[str, list[Row]]:
    grouped: dict[str, list[Row]] = {}
    for row in rows:
        grouped.setdefault(row.entity_id, []).append(row)
    for entity_rows in grouped.values():
        entity_rows.sort(key=lambda row: (row.timestamp_ms, row.row_id))
    return {entity: grouped[entity] for entity in sorted(grouped)}


def dataset_fingerprint(rows: Iterable[Row]) -> str:
    from .fingerprints import sha256_json

    return sha256_json(
        [row.content_fingerprint() for row in sorted(rows, key=lambda item: item.row_id)]
    )
