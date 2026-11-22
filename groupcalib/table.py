"""Feature-table parsing, validation, and deterministic fingerprints."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Iterable

from .config import CalibrationConfig
from .errors import E_CONFIG_INVALID, E_DATA_INVALID, E_GROUP_UNKNOWN, Failure, failure_sort_key
from .hashing import fingerprint_rows, sha256_value
from .io_utils import read_jsonl_strict


@dataclass(frozen=True)
class FeatureRow:
    order: int
    source_line: int
    row_id: str
    group_id: str
    values: dict[str, float | None]

    def normalised_values(self, columns: Iterable[str]) -> dict[str, float | None]:
        return {column: self.values.get(column) for column in columns}


@dataclass
class FeatureTable:
    rows: list[FeatureRow]
    columns: tuple[str, ...]

    def row_map(self) -> dict[str, FeatureRow]:
        return {row.row_id: row for row in self.rows}

    def rows_for_group(self, group_id: str) -> list[FeatureRow]:
        return [row for row in self.rows if row.group_id == group_id]

    def group_row_ids(self, group_id: str) -> list[str]:
        return sorted(row.row_id for row in self.rows if row.group_id == group_id)

    def table_hash(self) -> str:
        return sha256_value(
            {
                "columns": list(self.columns),
                "rows": [
                    {
                        "row_id": row.row_id,
                        "group_id": row.group_id,
                        "values": {column: row.values.get(column) for column in self.columns},
                    }
                    for row in sorted(self.rows, key=lambda item: item.row_id)
                ],
            }
        )

    def group_partition_hash(self, column_groups: dict[str, tuple[str, ...]]) -> str:
        payload = {
            "column_groups": {
                group_id: list(sorted(columns)) for group_id, columns in sorted(column_groups.items())
            },
            "row_assignments": [
                [row.row_id, row.group_id] for row in sorted(self.rows, key=lambda item: item.row_id)
            ],
        }
        return sha256_value(payload)

    def group_content_fingerprint(self, group_id: str, columns: Iterable[str]) -> str:
        return fingerprint_rows(
            (
                (row.row_id, row.group_id, {column: row.values.get(column) for column in columns})
                for row in self.rows
                if row.group_id == group_id
            ),
            columns,
            include_group=False,
        )


@dataclass
class TableLoadResult:
    table: FeatureTable
    failures: list[Failure]

    @property
    def ok(self) -> bool:
        return not self.failures


def _is_missing(value: Any, markers: tuple[Any, ...]) -> bool:
    for marker in markers:
        if value is marker:
            return True
        if type(value) is type(marker) and value == marker:
            return True
    return False


def normalise_feature_value(
    value: Any,
    markers: tuple[Any, ...],
) -> tuple[bool, float | None, str | None]:
    if _is_missing(value, markers):
        return True, None, None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False, None, "value must be a finite number or a configured missing marker"
    number = float(value)
    if not math.isfinite(number):
        return False, None, "value must be finite"
    if number == 0.0:
        number = 0.0
    return True, number, None


def _text_identifier(value: Any, field: str) -> tuple[str | None, str | None]:
    if isinstance(value, str) and value:
        return value, None
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value), None
    return None, f"{field} must be a non-empty string or integer"


def parse_feature_table(path: str | os.PathLike[str], config: CalibrationConfig) -> TableLoadResult:
    failures: list[Failure] = []
    rows: list[FeatureRow] = []
    seen_row_ids: set[str] = set()
    columns_seen: set[str] = set()
    meta_fields = {config.row_id_field, config.group_id_field}

    if not os.path.exists(path):
        message = f"feature table does not exist: {path}"
        failures.append(
            Failure(
                code=E_DATA_INVALID,
                message=message,
                sort_key=failure_sort_key(source="table", code=E_DATA_INVALID, field="path", message=message),
                details={"field": "path", "tag": "table", "path": os.fspath(path)},
            )
        )
        return TableLoadResult(FeatureTable([], ()), failures)

    try:
        raw_records = read_jsonl_strict(path)
    except OSError as exc:
        message = f"cannot read feature table: {exc}"
        failures.append(
            Failure(
                code=E_DATA_INVALID,
                message=message,
                sort_key=failure_sort_key(source="table", code=E_DATA_INVALID, field="io", message=message),
                details={"field": "io", "tag": "table"},
            )
        )
        return TableLoadResult(FeatureTable([], ()), failures)

    for line_number, record, parse_error in raw_records:
        if parse_error is not None:
            message = f"line {line_number}: {parse_error}"
            failures.append(
                Failure(
                    code=E_DATA_INVALID,
                    message=message,
                    sort_key=failure_sort_key(
                        source="table", line=line_number, code=E_DATA_INVALID, message=message
                    ),
                    details={"field": "$", "tag": "row", "line": line_number},
                )
            )
            continue
        if not isinstance(record, dict):
            message = f"line {line_number}: row must be a JSON object"
            failures.append(
                Failure(
                    code=E_DATA_INVALID,
                    message=message,
                    sort_key=failure_sort_key(
                        source="table", line=line_number, code=E_DATA_INVALID, message=message
                    ),
                    details={"field": "$", "tag": "row", "line": line_number},
                )
            )
            continue

        row_id, row_error = _text_identifier(record.get(config.row_id_field), config.row_id_field)
        group_id, group_error = _text_identifier(record.get(config.group_id_field), config.group_id_field)
        if row_error is not None:
            message = f"line {line_number}: {row_error}"
            failures.append(
                Failure(
                    code=E_DATA_INVALID,
                    message=message,
                    sort_key=failure_sort_key(
                        source="table", line=line_number, code=E_DATA_INVALID, field=config.row_id_field, message=message
                    ),
                    details={"field": config.row_id_field, "tag": "row", "line": line_number},
                )
            )
            continue
        if group_error is not None:
            message = f"line {line_number}: {group_error}"
            failures.append(
                Failure(
                    code=E_DATA_INVALID,
                    message=message,
                    sort_key=failure_sort_key(
                        source="table",
                        line=line_number,
                        code=E_DATA_INVALID,
                        field=config.group_id_field,
                        message=message,
                    ),
                    details={"field": config.group_id_field, "tag": "row", "line": line_number},
                )
            )
            continue
        assert row_id is not None and group_id is not None

        if row_id in seen_row_ids:
            message = f"line {line_number}: duplicate row_id {row_id!r}"
            failures.append(
                Failure(
                    code=E_DATA_INVALID,
                    message=message,
                    sort_key=failure_sort_key(
                        source="table",
                        line=line_number,
                        code=E_DATA_INVALID,
                        row_id=row_id,
                        field=config.row_id_field,
                        message=message,
                    ),
                    details={
                        "field": config.row_id_field,
                        "tag": "row",
                        "line": line_number,
                        "row_id": row_id,
                    },
                )
            )
            continue
        if group_id not in config.column_groups:
            message = f"line {line_number}: group_id {group_id!r} is not declared in column_groups"
            failures.append(
                Failure(
                    code=E_GROUP_UNKNOWN,
                    message=message,
                    sort_key=failure_sort_key(
                        source="table",
                        line=line_number,
                        code=E_GROUP_UNKNOWN,
                        row_id=row_id,
                        field=config.group_id_field,
                        message=message,
                    ),
                    details={
                        "field": config.group_id_field,
                        "tag": "row",
                        "line": line_number,
                        "row_id": row_id,
                        "group_id": group_id,
                    },
                )
            )
            continue

        values: dict[str, float | None] = {}
        row_invalid = False
        for column in sorted(key for key in record if key not in meta_fields):
            valid, number, value_error = normalise_feature_value(record[column], config.missing_value_markers)
            if not valid:
                message = f"line {line_number}: column {column!r}: {value_error}"
                failures.append(
                    Failure(
                        code=E_DATA_INVALID,
                        message=message,
                        sort_key=failure_sort_key(
                            source="table",
                            line=line_number,
                            code=E_DATA_INVALID,
                            row_id=row_id,
                            field=column,
                            message=message,
                        ),
                        details={
                            "field": column,
                            "tag": "row",
                            "line": line_number,
                            "row_id": row_id,
                        },
                    )
                )
                row_invalid = True
                break
            values[column] = number
        if row_invalid:
            continue

        seen_row_ids.add(row_id)
        columns_seen.update(values)
        rows.append(
            FeatureRow(
                order=len(rows),
                source_line=line_number,
                row_id=row_id,
                group_id=group_id,
                values=values,
            )
        )

    columns = tuple(sorted(columns_seen))
    return TableLoadResult(FeatureTable(rows, columns), failures)


def validate_table_schema(
    config: CalibrationConfig,
    table: FeatureTable,
    *,
    source: str = "schema",
) -> list[Failure]:
    """Validate group-column coverage, tensor axis width, and non-empty groups."""

    failures: list[Failure] = []
    table_columns = set(table.columns)
    configured_columns = set(config.all_columns)

    for group_id, columns in sorted(config.column_groups.items()):
        missing = sorted(set(columns) - table_columns)
        if missing:
            for column in missing:
                message = f"configured column {column!r} in group {group_id!r} is absent from the table schema"
                failures.append(
                    Failure(
                        code=E_CONFIG_INVALID,
                        message=message,
                        sort_key=failure_sort_key(
                            source=source, code=E_CONFIG_INVALID, field=column, message=message
                        ),
                        details={"field": column, "tag": "schema", "group_id": group_id},
                    )
                )

    extra = sorted(table_columns - configured_columns)
    for column in extra:
        message = f"table column {column!r} is not assigned to any column group"
        failures.append(
            Failure(
                code=E_CONFIG_INVALID,
                message=message,
                sort_key=failure_sort_key(source=source, code=E_CONFIG_INVALID, field=column, message=message),
                details={"field": column, "tag": "schema"},
            )
        )

    axis_dimension = config.tensor_dimensions[config.column_axis]
    if axis_dimension is not None and axis_dimension != len(table.columns):
        message = (
            f"column_axis dimension {axis_dimension} does not match the table schema width {len(table.columns)}"
        )
        failures.append(
            Failure(
                code=E_CONFIG_INVALID,
                message=message,
                sort_key=failure_sort_key(
                    source=source, code=E_CONFIG_INVALID, field="column_axis", message=message
                ),
                details={"field": "column_axis", "tag": "schema"},
            )
        )

    if not table.rows:
        message = "feature table contains no valid rows"
        failures.append(
            Failure(
                code=E_DATA_INVALID,
                message=message,
                sort_key=failure_sort_key(source=source, code=E_DATA_INVALID, field="rows", message=message),
                details={"field": "rows", "tag": "table"},
            )
        )
    return failures


def validate_nonmissing_columns(
    config: CalibrationConfig,
    table: FeatureTable,
    *,
    source: str = "data",
) -> list[Failure]:
    failures: list[Failure] = []
    for group_id, columns in sorted(config.column_groups.items()):
        group_rows = table.rows_for_group(group_id)
        if not group_rows:
            message = f"group {group_id!r} has no rows after applying edits"
            failures.append(
                Failure(
                    code=E_DATA_INVALID,
                    message=message,
                    sort_key=failure_sort_key(
                        source=source, code=E_DATA_INVALID, field="group_id", message=message
                    ),
                    details={"field": "group_id", "tag": "data", "group_id": group_id},
                )
            )
            continue
        for column in columns:
            if not any(row.values.get(column) is not None for row in group_rows):
                message = f"group {group_id!r} column {column!r} has no non-missing values"
                failures.append(
                    Failure(
                        code=E_DATA_INVALID,
                        message=message,
                        sort_key=failure_sort_key(
                            source=source,
                            code=E_DATA_INVALID,
                            field=column,
                            message=message,
                        ),
                        details={"field": column, "tag": "data", "group_id": group_id},
                    )
                )
    return failures

