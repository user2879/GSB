from __future__ import annotations

from decimal import Decimal
import hashlib
from itertools import groupby
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator, Sequence

from .analysis import row_error_for_comparison
from .database import (
    ColumnMeta,
    NUMERIC_TYPES,
    canonical_json,
    column_names,
    column_types,
    connect,
    normalize_sql_type,
    ordered_rows,
    primary_key_columns,
    read_column_meta,
    row_key_columns,
    row_key_for_row,
    table_exists,
    value_to_units,
)
from .errors import MigrationError, fail
from .fixedpoint import QuantizationSpec, comparison_bound_units, pow10, rescale_units
from .manifest import Manifest


def _calibration_map_from_argument(
    before: dict[str, QuantizationSpec] | Manifest | None,
    *,
    table: str,
    columns: Sequence[str],
) -> dict[str, QuantizationSpec]:
    result: dict[str, QuantizationSpec] = {}
    for column in columns:
        if isinstance(before, Manifest):
            spec = before.calibration_for(table, column)
        elif isinstance(before, dict):
            spec = before.get(column)
        else:
            spec = None
        if spec is not None:
            result[column] = spec
    return result


def _keyed_rows(
    connection: sqlite3.Connection,
    *,
    table: str,
    columns: Sequence[str],
    key_columns: Sequence[str],
    batch_size: int,
) -> Iterator[tuple[str, sqlite3.Row]]:
    for row in ordered_rows(
        connection, table, columns=columns, batch_size=max(1, batch_size)
    ):
        yield row_key_for_row(row, key_columns), row


def _merge_keys(
    left: Iterator[tuple[str, sqlite3.Row]],
    right: Iterator[tuple[str, sqlite3.Row]],
) -> Iterator[tuple[str, sqlite3.Row | None, sqlite3.Row | None]]:
    left_item = next(left, None)
    right_item = next(right, None)
    while left_item is not None or right_item is not None:
        if left_item is None:
            yield right_item[0], None, right_item[1]
            right_item = next(right, None)
            continue
        if right_item is None:
            yield left_item[0], left_item[1], None
            left_item = next(left, None)
            continue
        if left_item[0] == right_item[0]:
            yield left_item[0], left_item[1], right_item[1]
            left_item = next(left, None)
            right_item = next(right, None)
        elif left_item[0] < right_item[0]:
            yield left_item[0], left_item[1], None
            left_item = next(left, None)
        else:
            yield right_item[0], None, right_item[1]
            right_item = next(right, None)


def _spec_from_meta(meta: ColumnMeta) -> QuantizationSpec:
    if meta.spec is None:
        raise AssertionError("numeric meta has no spec")
    return meta.spec


def verify_drift(
    before_db_path: str | Path,
    after_db_path: str | Path,
    *,
    table: str = "orders",
    columns: Sequence[str] | None = None,
    before_calibrations: dict[str, QuantizationSpec] | Manifest | None = None,
    row_key: Sequence[str] | None = None,
    batch_size: int = 256,
    raise_on_drift: bool = True,
) -> dict[str, Any]:
    before_connection = connect(before_db_path, readonly=True)
    after_connection = connect(after_db_path, readonly=True)
    try:
        if not table_exists(before_connection, table) or not table_exists(after_connection, table):
            fail(
                "E_MIGRATE",
                "drift verification table is missing",
                step_id=None,
                field_path=f"tables.{table}",
            )
        before_types = column_types(before_connection, table)
        after_types = column_types(after_connection, table)
        if columns is None:
            selected_columns = sorted(
                column
                for column, sql_type in after_types.items()
                if sql_type in NUMERIC_TYPES and column in before_types
            )
        else:
            selected_columns = sorted(dict.fromkeys(columns))
        for column in selected_columns:
            if column not in before_types or column not in after_types:
                fail(
                    "E_MIGRATE",
                    "drift verification column is missing from one row set",
                    step_id=None,
                    field_path=f"tables.{table}.columns.{column}",
                )
            if before_types[column] not in NUMERIC_TYPES or after_types[column] not in NUMERIC_TYPES:
                fail(
                    "E_MIGRATE",
                    "drift verification only supports numeric columns",
                    step_id=None,
                    field_path=f"tables.{table}.columns.{column}",
                )
        key_columns = list(row_key) if row_key is not None else primary_key_columns(before_connection, table)
        if not key_columns:
            key_columns = row_key_columns(before_connection, table)
        for column in key_columns:
            if column != "_rowid_" and column not in before_types:
                fail(
                    "E_MIGRATE",
                    "drift row key references a missing column",
                    step_id=None,
                    field_path=f"tables.{table}.row_key.{column}",
                )
        before_specs = _calibration_map_from_argument(
            before_calibrations, table=table, columns=selected_columns
        )
        after_meta = read_column_meta(after_connection, table=table)
        before_meta: dict[str, ColumnMeta] = {}
        for column in selected_columns:
            spec = before_specs.get(column)
            if spec is None and (table, column) in after_meta:
                spec = _spec_from_meta(after_meta[(table, column)])
            if spec is None:
                fail(
                    "E_MIGRATE",
                    "a before-calibration is required for drift verification",
                    step_id=None,
                    field_path=f"initial_calibrations.{table}.{column}",
                )
            before_meta[column] = ColumnMeta(
                table=table,
                column=column,
                sql_type=before_types[column],
                numeric=True,
                spec=spec,
                version=spec.version(),
                updated_step=None,
            )
        for column in selected_columns:
            if (table, column) not in after_meta or not after_meta[(table, column)].numeric:
                fail(
                    "E_MIGRATE",
                    "after-calibration is missing for drift verification",
                    step_id=None,
                    field_path=f"tables.{table}.columns.{column}",
                )
        read_columns = sorted(set(selected_columns) | {c for c in key_columns if c != "_rowid_"})
        before_rows = _keyed_rows(
            before_connection,
            table=table,
            columns=read_columns,
            key_columns=key_columns,
            batch_size=batch_size,
        )
        after_rows = _keyed_rows(
            after_connection,
            table=table,
            columns=read_columns,
            key_columns=key_columns,
            batch_size=batch_size,
        )
        column_stats = {
            column: {
                "table": table,
                "column": column,
                "before_calibration": before_meta[column].to_dict(),
                "after_calibration": after_meta[(table, column)].to_dict(),
                "comparison_scale": max(
                    _spec_from_meta(before_meta[column]).scale,
                    _spec_from_meta(after_meta[(table, column)]).scale,
                ),
                "compared_rows": 0,
                "nonzero_deviation_rows": 0,
                "out_of_bound_rows": 0,
                "max_abs_diff_units": 0,
                "max_comparison_bound_units": 0,
                "max_margin_units": None,
                "first_out_of_bound": None,
                "first_nonzero": None,
                "conclusion": "precision_equivalent",
            }
            for column in selected_columns
        }
        row_set_mismatch: list[dict[str, Any]] = []
        compared_rows = 0
        for key, before_row, after_row in _merge_keys(before_rows, after_rows):
            if before_row is None or after_row is None:
                row_set_mismatch.append(
                    {
                        "row_key": key,
                        "missing_from": "before" if before_row is None else "after",
                    }
                )
                continue
            compared_rows += 1
            for column in selected_columns:
                before_spec = _spec_from_meta(before_meta[column])
                after_spec = _spec_from_meta(after_meta[(table, column)])
                common_scale = max(before_spec.scale, after_spec.scale)
                before_units, before_error = value_to_units(before_row[column], before_meta[column])
                after_units, base_after_error = value_to_units(
                    after_row[column], after_meta[(table, column)]
                )
                before_units = rescale_units(
                    before_units, before_spec.scale, common_scale, before_spec.truncation
                )
                after_units = rescale_units(
                    after_units, after_spec.scale, common_scale, after_spec.truncation
                )
                after_error, paths, error_step = row_error_for_comparison(
                    after_connection,
                    column=column,
                    row_key=key,
                    fallback_error=base_after_error,
                )
                before_error_common = before_error * pow10(common_scale - before_spec.scale)
                after_error_common = after_error * pow10(common_scale - after_spec.scale)
                bound = comparison_bound_units(before_error_common, after_error_common)
                difference = abs(after_units - before_units)
                stats = column_stats[column]
                stats["compared_rows"] += 1
                stats["max_comparison_bound_units"] = max(
                    int(stats["max_comparison_bound_units"]), bound
                )
                stats["max_abs_diff_units"] = max(
                    int(stats["max_abs_diff_units"]), difference
                )
                equivalent = difference == 0 or difference < bound
                margin = bound - difference
                if stats["max_margin_units"] is None or margin < int(stats["max_margin_units"]):
                    stats["max_margin_units"] = margin
                if difference > 0:
                    stats["nonzero_deviation_rows"] += 1
                    if stats["first_nonzero"] is None:
                        stats["first_nonzero"] = {
                            "row_key": key,
                            "difference_units": difference,
                            "bound_units": bound,
                            "paths": paths,
                            "error_step_id": error_step,
                        }
                if not equivalent:
                    stats["out_of_bound_rows"] += 1
                    stats["conclusion"] = "drift"
                    if stats["first_out_of_bound"] is None:
                        stats["first_out_of_bound"] = {
                            "row_key": key,
                            "difference_units": difference,
                            "bound_units": bound,
                            "excess_units": difference - bound,
                            "paths": paths,
                            "error_step_id": error_step,
                        }
        affected_columns = [
            column
            for column in selected_columns
            if column_stats[column]["conclusion"] == "drift"
        ]
        result: dict[str, Any] = {
            "table": table,
            "row_key": list(key_columns),
            "selected_columns": selected_columns,
            "compared_rows": compared_rows,
            "row_set_mismatch_count": len(row_set_mismatch),
            "row_set_mismatches": row_set_mismatch,
            "columns": [column_stats[column] for column in selected_columns],
            "drift_columns": affected_columns,
            "conclusion": (
                "drift"
                if affected_columns or row_set_mismatch
                else "precision_equivalent"
            ),
            "rule": "equivalent iff difference == 0 or difference < before_error + after_error",
        }
        result["report_digest"] = hashlib.sha256(
            canonical_json(result).encode("utf-8")
        ).hexdigest()
        if raise_on_drift and result["conclusion"] == "drift":
            raise MigrationError(
                "E_DRIFT",
                "migration data drift exceeded the composed fixed-point error bound",
                step_id=None,
                field_path="drift",
                details=result,
            )
        return result
    finally:
        before_connection.close()
        after_connection.close()
