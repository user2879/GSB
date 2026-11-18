from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Callable, Iterable, Sequence

from .database import (
    NUMERIC_TYPES,
    ColumnMeta,
    canonical_json,
    column_types,
    column_names,
    connect,
    normalize_sql_type,
    ordered_rows,
    quote_identifier,
    read_column_meta,
    row_key_columns,
    row_key_for_row,
    table_row_count,
    upsert_column_meta,
    value_to_units,
)
from .errors import MigrationError, fail
from .expression import EvalValue, ParsedExpression
from .fixedpoint import (
    QuantizationSpec,
    comparison_bound_units,
    normalize_error_terms,
    pow10,
)
from .ledger import insert_row_error, latest_row_errors_by_key, latest_row_error
from .manifest import Manifest, MigrationStep


def _base_source_path(table: str, column: str) -> tuple[str, ...]:
    return ("source", f"{table}.{column}")


def _derive_path(table: str, column: str) -> str:
    return f"derive:{table}.{column}"


def _parse_paths(paths_json: str) -> dict[tuple[str, ...], int]:
    result: dict[tuple[str, ...], int] = {}
    try:
        payload = json.loads(paths_json)
    except json.JSONDecodeError:
        return result
    for item in payload:
        if not isinstance(item, dict):
            continue
        path = tuple(str(part) for part in item.get("path", []))
        units = int(item.get("units", 0))
        if path and units >= 0:
            result[path] = result.get(path, 0) + units
    return result


def make_resolver(
    *,
    table: str,
    row: sqlite3.Row,
    row_key: str,
    meta: dict[tuple[str, str], ColumnMeta],
    preloaded_row_errors: dict[str, dict[str, sqlite3.Row]],
    step_id: str,
) -> Callable[[str], EvalValue]:
    def resolve(column: str) -> EvalValue:
        key = (table, column)
        column_meta = meta.get(key)
        if column_meta is None or not column_meta.numeric or column_meta.spec is None:
            fail(
                "E_MIGRATE",
                "backfill expression references a non-numeric column",
                step_id=step_id,
                field_path="expression",
                column=column,
            )
        units, base_error = value_to_units(row[column], column_meta, step_id=step_id)
        stored = preloaded_row_errors.get(column, {}).get(row_key)
        if stored is not None:
            error_bound = int(stored["error_bound"])
            terms = _parse_paths(str(stored["paths_json"]))
        else:
            error_bound = base_error
            terms = (
                {_base_source_path(table, column): error_bound}
                if error_bound > 0
                else {}
            )
        return EvalValue(
            nominal=units,
            scale=column_meta.spec.scale,
            lo=units - error_bound,
            hi=units + error_bound,
            terms=terms,
        )

    return resolve


def evaluate_expression_for_row(
    *,
    expression: ParsedExpression,
    spec: QuantizationSpec,
    resolver: Callable[[str], EvalValue],
    step_id: str,
    target_table: str,
    target_column: str,
) -> dict[str, Any]:
    try:
        result = expression.evaluate(
            resolver,
            target_scale=spec.scale,
            rounding=spec.truncation,
        )
    except ZeroDivisionError:
        fail(
            "E_MIGRATE",
            "backfill expression divided by zero",
            step_id=step_id,
            field_path="expression",
        )
    derive = _derive_path(target_table, target_column)
    units = int(result.units)
    bound = int(result.error_bound)
    saturation_contribution = 0
    if units < spec.min_units or units > spec.max_units:
        if spec.saturation == "reject":
            fail(
                "E_MIGRATE",
                "backfill result exceeds the declared saturation range",
                step_id=step_id,
                field_path="quantization",
                units=units,
                min_units=spec.min_units,
                max_units=spec.max_units,
            )
        clamped = min(spec.max_units, max(spec.min_units, units))
        saturation_contribution = abs(units - clamped)
        bound += saturation_contribution
        units = clamped
    transformed: dict[tuple[str, ...], int] = {
        tuple(path) + (derive,): units for path, units in result.term_units.items()
    }
    if saturation_contribution:
        saturation_path = ("saturation", derive)
        transformed[saturation_path] = transformed.get(saturation_path, 0) + saturation_contribution
    residual = ("rounding", derive)
    terms, bound = normalize_error_terms(
        transformed,
        bound=bound,
        residual_path=residual,
    )
    return {
        "units": units,
        "scale": result.scale,
        "error_bound": int(bound),
        "interval_bound_units": result.interval_bound_units,
        "paths": terms,
        "term_units": {tuple(term["path"]): int(term["units"]) for term in terms},
    }


def _row_predicate(
    table: str, row: sqlite3.Row, key_columns: Sequence[str]
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    values: list[Any] = []
    for column in key_columns:
        if column == "_rowid_":
            clauses.append("_rowid_ = ?")
            values.append(row["_rowid_"])
        else:
            clauses.append(f"{quote_identifier(column)} = ?")
            values.append(row[column])
    return " AND ".join(clauses), values


def execute_backfill(
    connection: sqlite3.Connection,
    *,
    step: MigrationStep,
    manifest: Manifest,
    revision: int,
    batch_size: int = 256,
    quantization: QuantizationSpec | None = None,
) -> tuple[int, dict[str, Any]]:
    effective_spec = quantization or step.quantization
    if step.expression is None or step.column is None or effective_spec is None:
        raise AssertionError("invalid backfill step")
    table = step.table
    target = step.column
    meta = read_column_meta(connection)
    target_meta = meta.get((table, target))
    if target_meta is None:
        fail(
            "E_MIGRATE",
            "backfill target metadata is missing",
            step_id=step.step_id,
            field_path="column",
        )
    previous_target_meta = target_meta
    updated_meta = upsert_column_meta(
        connection,
        table=table,
        column=target,
        sql_type=target_meta.sql_type,
        spec=effective_spec,
        step_id=step.step_id,
        revision=revision,
    )
    meta[(table, target)] = updated_meta
    preloaded = {
        reference: latest_row_errors_by_key(connection, reference)
        for reference in step.expression.references
    }
    connection.execute(
        "DELETE FROM _mig_row_errors WHERE step_id = ? AND column_name = ?",
        (step.step_id, target),
    )
    selected_columns = sorted(
        set(step.expression.references) | {target} | set(row_key_columns(connection, table))
    )
    count = 0
    max_bound = 0
    max_row_key: str | None = None
    for row in ordered_rows(
        connection, table, columns=selected_columns, batch_size=batch_size
    ):
        row_key = row_key_for_row(row, row_key_columns(connection, table))
        resolver = make_resolver(
            table=table,
            row=row,
            row_key=row_key,
            meta=meta,
            preloaded_row_errors=preloaded,
            step_id=step.step_id,
        )
        evaluated = evaluate_expression_for_row(
            expression=step.expression,
            spec=effective_spec,
            resolver=resolver,
            step_id=step.step_id,
            target_table=table,
            target_column=target,
        )
        db_value = (
            int(evaluated["units"])
            if updated_meta.sql_type != "REAL"
            else float(evaluated["units"]) / pow10(effective_spec.scale)
        )
        predicate, predicate_values = _row_predicate(
            table, row, row_key_columns(connection, table)
        )
        connection.execute(
            f"UPDATE {quote_identifier(table)} SET {quote_identifier(target)} = ? WHERE {predicate}",
            [db_value, *predicate_values],
        )
        insert_row_error(
            connection,
            step_id=step.step_id,
            column=target,
            row_key=row_key,
            revision=revision,
            units=int(evaluated["units"]),
            error_bound=int(evaluated["error_bound"]),
            paths=evaluated["paths"],
        )
        if int(evaluated["error_bound"]) > max_bound:
            max_bound = int(evaluated["error_bound"])
            max_row_key = row_key
        count += 1
    report = {
        "target": f"{table}.{target}",
        "expression": step.expression_text,
        "references": list(step.expression.references),
        "scale": effective_spec.scale,
        "truncation": effective_spec.truncation,
        "saturation": effective_spec.saturation,
        "previous_calibration": previous_target_meta.to_dict(),
        "updated_calibration": updated_meta.to_dict(),
        "rows_evaluated": count,
        "max_error_bound_units": max_bound,
        "max_error_row_key": max_row_key,
    }
    return count, report


def derivation_graph_payload(manifest: Manifest) -> dict[str, Any]:
    nodes = sorted(
        {node for edge in manifest.derivation_edges for node in edge}
        | set(manifest.derivation_order)
    )
    return {
        "nodes": nodes,
        "edges": [list(edge) for edge in sorted(manifest.derivation_edges)],
        "topological_order": list(manifest.derivation_order),
    }


def derivation_graph_version(manifest: Manifest) -> str:
    return hashlib.sha256(
        canonical_json(derivation_graph_payload(manifest)).encode("utf-8")
    ).hexdigest()


def _column_report_from_row_errors(
    connection: sqlite3.Connection,
    *,
    table: str,
    column: str,
    meta: ColumnMeta,
) -> dict[str, Any]:
    rows = connection.execute(
        """
        SELECT row_key, error_bound, units, revision, step_id, paths_json
        FROM _mig_row_errors
        WHERE column_name = ?
        ORDER BY row_key, revision DESC, step_id
        """,
        (column,),
    ).fetchall()
    chosen: dict[str, sqlite3.Row] = {}
    for row in rows:
        key = str(row["row_key"])
        if key not in chosen:
            chosen[key] = row
    representative: sqlite3.Row | None = None
    for key in sorted(chosen):
        row = chosen[key]
        if representative is None or int(row["error_bound"]) > int(representative["error_bound"]):
            representative = row
    max_bound = int(representative["error_bound"]) if representative else 0
    paths: list[dict[str, Any]] = []
    if representative is not None:
        paths = json.loads(str(representative["paths_json"]))
    return {
        "table": table,
        "column": column,
        "sql_type": meta.sql_type,
        "quantization": meta.spec.to_dict() if meta.spec else None,
        "calibration_version": meta.version,
        "error_source": "row_ledger",
        "rows_with_errors": len(chosen),
        "max_error_bound_units": max_bound,
        "representative_row_key": str(representative["row_key"]) if representative else None,
        "representative_step_id": str(representative["step_id"]) if representative else None,
        "source_paths": paths,
    }


def _column_report_from_values(
    connection: sqlite3.Connection,
    *,
    table: str,
    column: str,
    meta: ColumnMeta,
) -> dict[str, Any]:
    max_bound = 0
    max_row_key: str | None = None
    row_count = 0
    non_null_rows = 0
    key_columns = row_key_columns(connection, table)
    for row in ordered_rows(connection, table):
        row_count += 1
        if row[column] is None:
            continue
        non_null_rows += 1
        key = row_key_for_row(row, key_columns)
        units, error = value_to_units(row[column], meta)
        if error > max_bound:
            max_bound = error
            max_row_key = key
    path = _base_source_path(table, column)
    paths, bound = normalize_error_terms(
        {path: max_bound} if max_bound else {},
        bound=max_bound,
        residual_path=("rounding",),
    )
    return {
        "table": table,
        "column": column,
        "sql_type": meta.sql_type,
        "quantization": meta.spec.to_dict() if meta.spec else None,
        "calibration_version": meta.version,
        "error_source": "value_scan",
        "rows_with_errors": row_count if max_bound else 0,
        "non_null_rows": non_null_rows,
        "null_rows": row_count - non_null_rows,
        "max_error_bound_units": bound,
        "representative_row_key": max_row_key,
        "representative_step_id": meta.updated_step,
        "source_paths": paths,
    }


def column_error_report(
    connection: sqlite3.Connection,
    *,
    table: str,
    column: str,
) -> dict[str, Any]:
    meta = read_column_meta(connection, table=table).get((table, column))
    if meta is None or not meta.numeric or meta.spec is None:
        fail(
            "E_MIGRATE",
            "numeric column metadata is missing",
            step_id=None,
            field_path=f"tables.{table}.columns.{column}",
        )
    has_errors = connection.execute(
        "SELECT 1 FROM _mig_row_errors WHERE column_name = ? LIMIT 1", (column,)
    ).fetchone()
    if has_errors is not None:
        return _column_report_from_row_errors(
            connection, table=table, column=column, meta=meta
        )
    return _column_report_from_values(
        connection, table=table, column=column, meta=meta
    )


def table_numeric_column_report(
    connection: sqlite3.Connection, table: str
) -> list[dict[str, Any]]:
    meta = read_column_meta(connection, table=table)
    reports: list[dict[str, Any]] = []
    for (report_table, column), column_meta in sorted(meta.items()):
        if report_table != table or not column_meta.numeric:
            continue
        reports.append(column_error_report(connection, table=table, column=column))
    return reports


def build_step_result(
    connection: sqlite3.Connection,
    *,
    step: MigrationStep,
    manifest: Manifest,
    operation_report: dict[str, Any],
) -> dict[str, Any]:
    column_errors = table_numeric_column_report(connection, step.table)
    return {
        "step_id": step.step_id,
        "version": step.version,
        "operation": step.operation,
        "table": step.table,
        "operation_report": operation_report,
        "column_errors": column_errors,
        "derivation_graph_version": derivation_graph_version(manifest),
    }


def row_error_for_comparison(
    connection: sqlite3.Connection,
    *,
    column: str,
    row_key: str,
    fallback_error: int,
) -> tuple[int, list[dict[str, Any]], str | None]:
    stored = latest_row_error(connection, column=column, row_key=row_key)
    if stored is None:
        if fallback_error <= 0:
            return fallback_error, [], None
        terms, bound = normalize_error_terms(
            {("source", column): fallback_error},
            bound=fallback_error,
            residual_path=("rounding",),
        )
        return bound, terms, None
    return (
        int(stored["error_bound"]),
        json.loads(str(stored["paths_json"])),
        str(stored["step_id"]),
    )


def compute_error_report(
    db_path: str | os.PathLike[str],
    *,
    table: str,
    columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    connection = connect(db_path, readonly=True)
    try:
        meta = read_column_meta(connection, table=table)
        numeric_columns = [
            column
            for (meta_table, column), item in sorted(meta.items())
            if meta_table == table and item.numeric
        ]
        selected = sorted(dict.fromkeys(columns)) if columns is not None else numeric_columns
        reports = [
            column_error_report(connection, table=table, column=column)
            for column in selected
        ]
        digest = hashlib.sha256(canonical_json(reports).encode("utf-8")).hexdigest()
        return {
            "table": table,
            "columns": reports,
            "error_report_version": digest,
            "bound_rule": "integer grid units; paths and contributions are sorted lexicographically",
        }
    finally:
        connection.close()
