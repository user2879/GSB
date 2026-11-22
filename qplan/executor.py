from __future__ import annotations

import math
import re
import sqlite3
import statistics
import time
from typing import Any

from .canonical import canonical_json, content_digest, normalize
from .data import quote_identifier
from .errors import QPlanError
from .paths import AccessPath
from .spec import ExecutionConfig, QuerySpec, Spec


MEASUREMENT_METHOD = (
    "time.perf_counter_ns around cursor.execute() plus fetchall(); "
    "real_elapsed_ms is the median of measured repetitions after warmups; "
    "cumulative_ms is the sum of the measured repetitions. Result digest is "
    "computed from the complete sorted row set, so it is independent of row order."
)


def _typed_value(value: Any) -> Any:
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "integer", "value": int(value)}
    if isinstance(value, int):
        return {"type": "integer", "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise QPlanError("E_EXEC", "query returned a non-finite float", "execution.result")
        return {"type": "real", "value": round(value, 12) if value != 0 else 0.0}
    if isinstance(value, str):
        return {"type": "text", "value": value}
    if isinstance(value, bytes):
        return {"type": "blob", "value": value.hex()}
    return {"type": type(value).__name__, "value": str(value)}


def result_summary(columns: list[str], rows: list[tuple[Any, ...]]) -> dict[str, Any]:
    typed_rows = [[_typed_value(value) for value in row] for row in rows]
    encoded = [(canonical_json(row).encode("utf-8"), row) for row in typed_rows]
    encoded.sort(key=lambda item: item[0])
    sorted_rows = [item[1] for item in encoded]
    payload = {"columns": columns, "rows": sorted_rows}
    preview = sorted_rows[:3]
    return {
        "columns": columns,
        "row_count": len(sorted_rows),
        "digest": content_digest(payload),
        "digest_scope": "sha256(canonical_json(columns+typed_rows_sorted))",
        "rows_preview": preview,
        "row_order_sensitive": False,
    }


def _run_query(connection: sqlite3.Connection, path: AccessPath) -> tuple[list[str], list[tuple[Any, ...]], float]:
    start = time.perf_counter_ns()
    cursor = connection.execute(path.sql, path.params)
    rows = cursor.fetchall()
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    columns = [description[0] for description in cursor.description or []]
    return columns, rows, elapsed_ms


def explain_path(connection: sqlite3.Connection, path: AccessPath) -> list[dict[str, Any]]:
    rows = connection.execute("EXPLAIN QUERY PLAN " + path.sql, path.params).fetchall()
    result = []
    for row in rows:
        result.append(
            {
                "id": int(row[0]),
                "parent": int(row[1]),
                "notused": int(row[2]),
                "detail": str(row[3]),
            }
        )
    return result


def _observe_plan(plan_rows: list[dict[str, Any]], path: AccessPath) -> dict[str, Any]:
    details = [row["detail"] for row in plan_rows]
    index_uses = []
    covering_indexes = []
    table_order: list[str] = []
    known_tables = sorted(set(path.join_order), key=len, reverse=True)
    for detail in details:
        index_match = re.search(r"USING (COVERING )?INDEX ([A-Za-z_][A-Za-z0-9_]*)", detail)
        if index_match:
            if index_match.group(1):
                covering_indexes.append(index_match.group(2))
            index_uses.append(index_match.group(2))
        table_name = None
        for prefix in ("SCAN ", "SEARCH "):
            if detail.startswith(prefix):
                remainder = detail[len(prefix) :]
                for known in known_tables:
                    if remainder == known or remainder.startswith(known + " "):
                        table_name = known
                        break
                break
        if table_name is not None and table_name not in table_order:
            table_order.append(table_name)
        if path.index_name and f"USING COVERING INDEX {path.index_name}" in detail:
            if path.index_name not in covering_indexes:
                covering_indexes.append(path.index_name)
            if path.index_name not in index_uses:
                index_uses.append(path.index_name)
    return {
        "query_plan": plan_rows,
        "index_uses": index_uses,
        "covering_indexes": covering_indexes,
        "observed_table_order": table_order,
        "expected_index": path.index_name,
        "expected_covering": path.covering,
        "observed_expected_covering": bool(path.index_name and path.index_name in covering_indexes),
    }


def measure_path(
    connection: sqlite3.Connection,
    path: AccessPath,
    config: ExecutionConfig,
) -> dict[str, Any]:
    try:
        plan = _observe_plan(explain_path(connection, path), path)
        baseline_columns = None
        baseline_digest = None
        warmup_digests: list[str] = []
        for _ in range(config.warmup_runs):
            columns, rows, _ = _run_query(connection, path)
            summary = result_summary(columns, rows)
            warmup_digests.append(summary["digest"])
            baseline_columns = summary["columns"]
            baseline_digest = summary["digest"]
        measured_ms: list[float] = []
        measured_digests: list[str] = []
        summary = None
        for _ in range(config.repetitions):
            columns, rows, elapsed_ms = _run_query(connection, path)
            summary = result_summary(columns, rows)
            measured_ms.append(elapsed_ms)
            measured_digests.append(summary["digest"])
            baseline_columns = summary["columns"]
            if baseline_digest is None:
                baseline_digest = summary["digest"]
        if summary is None:
            raise QPlanError("E_EXEC", "no measured execution was performed", path.id)
        all_digests = warmup_digests + measured_digests
        if len(set(all_digests)) != 1 or baseline_digest != all_digests[0]:
            raise QPlanError(
                "E_EXEC",
                "result digest changed across repeated executions of the same deterministic path",
                path.id,
                {"digests": all_digests, "path_sql_digest": path.sql_digest},
            )
        real_elapsed_ms = round(float(statistics.median(measured_ms)), 6)
        cumulative_ms = round(float(sum(measured_ms)), 6)
        result = {
            "path_id": path.id,
            "sql_digest": path.sql_digest,
            "columns": baseline_columns,
            "row_count": summary["row_count"],
            "digest": summary["digest"],
            "result_digest": summary["digest"],
            "digest_scope": summary["digest_scope"],
            "rows_preview": summary["rows_preview"],
            "row_order_sensitive": False,
            "real_elapsed_ms": real_elapsed_ms,
            "elapsed_ms": real_elapsed_ms,
            "cumulative_ms": cumulative_ms,
            "cumulative_elapsed_ms": cumulative_ms,
            "measured_runs_ms": [round(item, 6) for item in measured_ms],
            "warmup_runs": config.warmup_runs,
            "measured_repetitions": config.repetitions,
            "measurement_method": MEASUREMENT_METHOD,
            "plan": plan,
        }
        result["execution_digest"] = content_digest(result)
        return result
    except QPlanError:
        raise
    except sqlite3.Error as exc:
        raise QPlanError("E_EXEC", f"SQLite execution failed: {exc}", path.id) from exc


def measure_filter_counts(connection: sqlite3.Connection, query: QuerySpec) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for table in query.tables:
        filters = query.filters_for(table)
        if not filters:
            count = int(connection.execute(f"SELECT COUNT(*) FROM {quote_identifier(table)}").fetchone()[0])
            result[table] = {"filtered_rows": count, "without_filters": True, "filters": []}
            continue
        params: list[Any] = []
        conditions = []
        for item in filters:
            column = f"{quote_identifier(table)}.{quote_identifier(item.column)}"
            if item.op == "between":
                params.extend(item.value)
                conditions.append(f"{column} BETWEEN ? AND ?")
            elif item.op == "in":
                params.extend(item.value)
                conditions.append(f"{column} IN ({','.join('?' for _ in item.value)})")
            else:
                sql_op = {"eq": "=", "ne": "<>", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}[item.op]
                params.append(item.value)
                conditions.append(f"{column} {sql_op} ?")
        sql = f"SELECT COUNT(*) FROM {quote_identifier(table)} WHERE {' AND '.join(conditions)}"
        count = int(connection.execute(sql, tuple(params)).fetchone()[0])
        result[table] = {
            "filtered_rows": count,
            "without_filters": False,
            "filters": [item.to_dict() for item in filters],
        }
    return result


def execute_query_paths(
    spec: Spec,
    connection: sqlite3.Connection,
    query: QuerySpec,
    paths: list[AccessPath],
) -> dict[str, Any]:
    executions = {}
    for path in sorted(paths, key=lambda item: item.id):
        executions[path.id] = measure_path(connection, path, spec.execution)
    filter_counts = measure_filter_counts(connection, query)
    return {
        "query_id": query.id,
        "paths": executions,
        "actual_table_filter_rows": normalize(filter_counts),
        "measurement_scope": "each path independently re-executed; no warm cache is shared deliberately",
    }
