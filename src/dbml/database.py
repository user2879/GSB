from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Callable, Iterable, Iterator, Sequence

from .errors import MigrationError, fail
from .fixedpoint import QuantizationSpec, parse_decimal, pow10, quantize_decimal


INTERNAL_PREFIX = "_mig_"
SUPPORTED_TYPES = ("INTEGER", "REAL", "DECIMAL", "TEXT")
NUMERIC_TYPES = frozenset({"INTEGER", "REAL", "DECIMAL"})


def connect(path: str | os.PathLike[str], *, readonly: bool = False) -> sqlite3.Connection:
    target = str(Path(path))
    if readonly:
        uri = f"file:{Path(target).as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, isolation_level=None)
    else:
        connection = sqlite3.connect(target, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    if not readonly:
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = FULL")
    return connection


def quote_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def normalize_sql_type(raw: str | None) -> str:
    if raw is None:
        return ""
    token = str(raw).strip().split("(", 1)[0].strip().upper()
    return token


def validate_supported_type(
    raw_type: Any,
    *,
    step_id: str | None,
    field_path: str,
) -> str:
    if not isinstance(raw_type, str) or not raw_type.strip():
        fail(
            "E_MIGRATE",
            "column type must be a non-empty string",
            step_id=step_id,
            field_path=field_path,
        )
    normalized = normalize_sql_type(raw_type)
    if normalized not in SUPPORTED_TYPES:
        fail(
            "E_MIGRATE",
            f"unsupported column type {raw_type!r}",
            step_id=step_id,
            field_path=field_path,
            supported=list(SUPPORTED_TYPES),
        )
    return normalized


def table_info(connection: sqlite3.Connection, table: str) -> list[sqlite3.Row]:
    return list(connection.execute(f"PRAGMA table_xinfo({quote_identifier(table)})"))


def table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    return row is not None


def user_tables(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute(
        """
        SELECT name FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '_mig_%'
        ORDER BY name
        """
    ).fetchall()
    return [str(row[0]) for row in rows]


def column_names(connection: sqlite3.Connection, table: str) -> list[str]:
    return [str(row["name"]) for row in table_info(connection, table) if not row["hidden"]]


def column_types(connection: sqlite3.Connection, table: str) -> dict[str, str]:
    return {
        str(row["name"]): normalize_sql_type(row["type"])
        for row in table_info(connection, table)
        if not row["hidden"]
    }


def primary_key_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    rows = [row for row in table_info(connection, table) if row["pk"] and not row["hidden"]]
    rows.sort(key=lambda row: int(row["pk"]))
    return [str(row["name"]) for row in rows]


def has_rowid(connection: sqlite3.Connection, table: str) -> bool:
    sql_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    if sql_row is None or sql_row[0] is None:
        return False
    return "WITHOUT ROWID" not in str(sql_row[0]).upper()


def canonical_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return {"bool": value}
    if isinstance(value, int):
        return {"int": str(value)}
    if isinstance(value, float):
        return {"float": value.hex()}
    if isinstance(value, Decimal):
        return {"decimal": str(value)}
    if isinstance(value, bytes):
        return {"blob": value.hex()}
    return {"text": str(value)}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _row_key_from_values(values: Sequence[Any]) -> str:
    if len(values) == 1:
        return str(values[0])
    return canonical_json([canonical_value(value) for value in values])


def row_key_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    pk = primary_key_columns(connection, table)
    if pk:
        return pk
    if has_rowid(connection, table):
        return ["_rowid_"]
    fail(
        "E_MIGRATE",
        "table has neither a primary key nor a rowid",
        step_id=None,
        field_path=f"tables.{table}.row_key",
    )


def row_key_for_row(
    row: sqlite3.Row | dict[str, Any], key_columns: Sequence[str]
) -> str:
    values: list[Any] = []
    for column in key_columns:
        if column == "_rowid_":
            values.append(row["_rowid_"])
        else:
            values.append(row[column])
    return _row_key_from_values(values)


def ordered_rows(
    connection: sqlite3.Connection,
    table: str,
    *,
    columns: Sequence[str] | None = None,
    batch_size: int = 256,
) -> Iterator[sqlite3.Row]:
    selected = list(columns) if columns is not None else column_names(connection, table)
    key_columns = row_key_columns(connection, table)
    expressions = [quote_identifier(column) for column in selected]
    order_terms: list[str] = []
    for column in key_columns:
        if column == "_rowid_":
            expressions.insert(0, "_rowid_ AS _rowid_")
            order_terms.append("_rowid_")
        else:
            order_terms.append(quote_identifier(column))
    sql = (
        f"SELECT {', '.join(expressions)} FROM {quote_identifier(table)} "
        f"ORDER BY {', '.join(order_terms)}"
    )
    cursor = connection.execute(sql)
    while True:
        rows = cursor.fetchmany(max(1, int(batch_size)))
        if not rows:
            break
        yield from rows


def table_row_count(connection: sqlite3.Connection, table: str) -> int:
    row = connection.execute(f"SELECT COUNT(*) FROM {quote_identifier(table)}").fetchone()
    return int(row[0])


def table_content_digest(connection: sqlite3.Connection, table: str) -> str:
    columns = sorted(column_names(connection, table))
    digest = hashlib.sha256()
    digest.update(canonical_json({"table": table, "columns": columns}).encode("utf-8"))
    key_columns = row_key_columns(connection, table)
    for row in ordered_rows(connection, table, columns=columns):
        payload: dict[str, Any] = {"__key__": row_key_for_row(row, key_columns)}
        for column in columns:
            payload[column] = canonical_value(row[column])
        digest.update(canonical_json(payload).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def schema_digest(connection: sqlite3.Connection) -> str:
    rows = connection.execute(
        """
        SELECT type, name, tbl_name, COALESCE(sql, '') AS sql
        FROM sqlite_master
        WHERE name NOT LIKE 'sqlite_%' AND name NOT LIKE '_mig_%'
        ORDER BY type, name
        """
    ).fetchall()
    payload = [
        {
            "type": str(row["type"]),
            "name": str(row["name"]),
            "table": str(row["tbl_name"]),
            "sql": str(row["sql"]),
        }
        for row in rows
    ]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def database_fingerprint(connection: sqlite3.Connection) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    for table in user_tables(connection):
        tables[table] = {
            "row_count": table_row_count(connection, table),
            "content_digest": table_content_digest(connection, table),
        }
    summary = {
        "schema_version": schema_digest(connection),
        "tables": tables,
        "total_rows": sum(item["row_count"] for item in tables.values()),
    }
    summary["data_version"] = hashlib.sha256(
        canonical_json(summary).encode("utf-8")
    ).hexdigest()
    return summary


def sqlite_integrity_digest(connection: sqlite3.Connection) -> str:
    return database_fingerprint(connection)["data_version"]


def write_json_atomic(path: str | os.PathLike[str], payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)


@dataclass(frozen=True)
class ColumnMeta:
    table: str
    column: str
    sql_type: str
    numeric: bool
    spec: QuantizationSpec | None
    version: str
    updated_step: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "column": self.column,
            "sql_type": self.sql_type,
            "numeric": self.numeric,
            "quantization": self.spec.to_dict() if self.spec else None,
            "calibration_version": self.version,
            "updated_step": self.updated_step,
        }


def ensure_internal_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS _mig_run (
            run_id INTEGER PRIMARY KEY CHECK (run_id = 1),
            manifest_id TEXT NOT NULL,
            manifest_sha256 TEXT NOT NULL,
            plan_json TEXT NOT NULL,
            base_schema_digest TEXT NOT NULL,
            status TEXT NOT NULL,
            current_step TEXT,
            revision INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS _mig_ledger (
            seq INTEGER PRIMARY KEY,
            step_id TEXT NOT NULL UNIQUE,
            version INTEGER NOT NULL,
            operation TEXT NOT NULL,
            table_name TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            before_columns_json TEXT NOT NULL,
            after_columns_json TEXT NOT NULL,
            result_json TEXT NOT NULL,
            content_digest TEXT NOT NULL,
            previous_chain_digest TEXT NOT NULL,
            chain_digest TEXT NOT NULL,
            revision INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS _mig_column_meta (
            table_name TEXT NOT NULL,
            column_name TEXT NOT NULL,
            sql_type TEXT NOT NULL,
            numeric INTEGER NOT NULL,
            scale INTEGER,
            truncation TEXT,
            saturation TEXT,
            min_units INTEGER,
            max_units INTEGER,
            calibration_version TEXT NOT NULL,
            updated_step TEXT,
            PRIMARY KEY (table_name, column_name)
        );
        CREATE TABLE IF NOT EXISTS _mig_row_errors (
            step_id TEXT NOT NULL,
            column_name TEXT NOT NULL,
            row_key TEXT NOT NULL,
            revision INTEGER NOT NULL,
            units INTEGER NOT NULL,
            error_bound INTEGER NOT NULL,
            paths_json TEXT NOT NULL,
            PRIMARY KEY (step_id, column_name, row_key)
        );
        CREATE TABLE IF NOT EXISTS _mig_constraints (
            constraint_name TEXT PRIMARY KEY,
            table_name TEXT NOT NULL,
            kind TEXT NOT NULL,
            definition TEXT NOT NULL,
            columns_json TEXT,
            created_step TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS _mig_recalibrations (
            seq INTEGER PRIMARY KEY,
            event_id TEXT NOT NULL UNIQUE,
            table_name TEXT NOT NULL,
            column_name TEXT NOT NULL,
            old_scale INTEGER NOT NULL,
            new_scale INTEGER NOT NULL,
            old_truncation TEXT NOT NULL,
            new_truncation TEXT NOT NULL,
            before_digest TEXT NOT NULL,
            after_digest TEXT NOT NULL,
            affected_steps_json TEXT NOT NULL,
            report_json TEXT NOT NULL,
            revision INTEGER NOT NULL
        );
        """
    )


def read_column_meta(
    connection: sqlite3.Connection, *, table: str | None = None
) -> dict[tuple[str, str], ColumnMeta]:
    clauses: list[str] = []
    params: list[Any] = []
    if table is not None:
        clauses.append("table_name = ?")
        params.append(table)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    rows = connection.execute(
        f"""
        SELECT table_name, column_name, sql_type, numeric, scale, truncation,
               saturation, min_units, max_units, calibration_version, updated_step
        FROM _mig_column_meta
        {where}
        ORDER BY table_name, column_name
        """,
        params,
    ).fetchall()
    result: dict[tuple[str, str], ColumnMeta] = {}
    for row in rows:
        spec = None
        if row["numeric"]:
            spec = QuantizationSpec(
                scale=int(row["scale"]),
                truncation=str(row["truncation"]),
                saturation=str(row["saturation"]),
                min_units=int(row["min_units"]),
                max_units=int(row["max_units"]),
            )
        key = (str(row["table_name"]), str(row["column_name"]))
        result[key] = ColumnMeta(
            table=key[0],
            column=key[1],
            sql_type=str(row["sql_type"]),
            numeric=bool(row["numeric"]),
            spec=spec,
            version=str(row["calibration_version"]),
            updated_step=row["updated_step"],
        )
    return result


def upsert_column_meta(
    connection: sqlite3.Connection,
    *,
    table: str,
    column: str,
    sql_type: str,
    spec: QuantizationSpec | None,
    step_id: str | None,
    revision: int,
) -> ColumnMeta:
    numeric = normalize_sql_type(sql_type) in NUMERIC_TYPES
    if numeric and spec is None:
        fail(
            "E_MIGRATE",
            "numeric column requires a quantization declaration",
            step_id=step_id,
            field_path=f"tables.{table}.columns.{column}.quantization",
        )
    version_payload = {
        "table": table,
        "column": column,
        "sql_type": normalize_sql_type(sql_type),
        "quantization": spec.to_dict() if spec else None,
    }
    version = hashlib.sha256(canonical_json(version_payload).encode("utf-8")).hexdigest()
    connection.execute(
        """
        INSERT INTO _mig_column_meta(
            table_name, column_name, sql_type, numeric, scale, truncation,
            saturation, min_units, max_units, calibration_version, updated_step
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(table_name, column_name) DO UPDATE SET
            sql_type = excluded.sql_type,
            numeric = excluded.numeric,
            scale = excluded.scale,
            truncation = excluded.truncation,
            saturation = excluded.saturation,
            min_units = excluded.min_units,
            max_units = excluded.max_units,
            calibration_version = excluded.calibration_version,
            updated_step = excluded.updated_step
        """,
        (
            table,
            column,
            normalize_sql_type(sql_type),
            int(numeric),
            spec.scale if spec else None,
            spec.truncation if spec else None,
            spec.saturation if spec else None,
            spec.min_units if spec else None,
            spec.max_units if spec else None,
            version,
            step_id,
        ),
    )
    return ColumnMeta(
        table=table,
        column=column,
        sql_type=normalize_sql_type(sql_type),
        numeric=numeric,
        spec=spec,
        version=version,
        updated_step=step_id,
    )


def calibration_snapshot(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    return [meta.to_dict() for meta in read_column_meta(connection).values()]


def value_to_units(value: Any, meta: ColumnMeta, *, step_id: str | None = None) -> tuple[int, int]:
    """Return (nominal units, quantization error bound) for a stored value."""

    if not meta.numeric or meta.spec is None:
        fail(
            "E_MIGRATE",
            "non-numeric column cannot participate in fixed-point evaluation",
            step_id=step_id,
            field_path=f"columns.{meta.column}",
        )
    if value is None:
        fail(
            "E_MIGRATE",
            "NULL cannot be evaluated in a numeric expression",
            step_id=step_id,
            field_path=f"columns.{meta.column}",
        )
    if meta.sql_type in {"DECIMAL", "INTEGER"}:
        try:
            return int(value), 0
        except (TypeError, ValueError):
            quantized = quantize_decimal(value, meta.spec, step_id=step_id)
            return quantized.units, quantized.error_units
    quantized = quantize_decimal(value, meta.spec, step_id=step_id)
    return quantized.units, quantized.error_units


def units_to_db_value(units: int, meta: ColumnMeta) -> Any:
    if meta.sql_type == "REAL":
        return float(Decimal(units) / Decimal(pow10(meta.spec.scale)))  # type: ignore[union-attr]
    return int(units)


def quantize_for_calibration(value: Any, spec: QuantizationSpec) -> int:
    return quantize_decimal(value, spec).units


def _split_top_level_constraints(sql: str) -> list[str]:
    start = sql.find("(")
    end = sql.rfind(")")
    if start < 0 or end <= start:
        return []
    body = sql[start + 1 : end]
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    current: list[str] = []
    index = 0
    while index < len(body):
        char = body[index]
        if quote:
            current.append(char)
            if char == quote:
                if index + 1 < len(body) and body[index + 1] == quote:
                    current.append(body[index + 1])
                    index += 1
                else:
                    quote = None
        elif char in {"'", '"', "`"}:
            quote = char
            current.append(char)
        elif char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        index += 1
    if current:
        parts.append("".join(current).strip())
    return [part for part in parts if part]


def _extract_check_expressions(sql: str) -> list[str]:
    expressions: list[str] = []
    upper = sql.upper()
    position = 0
    while True:
        match = re.search(r"\bCHECK\s*\(", upper[position:])
        if not match:
            break
        open_index = position + match.end() - 1
        depth = 0
        quote: str | None = None
        close_index = -1
        for index in range(open_index, len(sql)):
            char = sql[index]
            if quote:
                if char == quote:
                    if index + 1 < len(sql) and sql[index + 1] == quote:
                        continue
                    quote = None
                continue
            if char in {"'", '"', "`"}:
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    close_index = index
                    break
        if close_index < 0:
            break
        expressions.append(sql[open_index + 1 : close_index].strip())
        position = close_index + 1
    return expressions


def _existing_foreign_key_definitions(connection: sqlite3.Connection, table: str) -> list[str]:
    rows = connection.execute(
        f"PRAGMA foreign_key_list({quote_identifier(table)})"
    ).fetchall()
    grouped: dict[int, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(int(row["id"]), []).append(row)
    definitions: list[str] = []
    for identifier in sorted(grouped):
        group = sorted(grouped[identifier], key=lambda row: int(row["seq"]))
        first = group[0]
        local = ", ".join(quote_identifier(str(row["from"])) for row in group)
        remote = ", ".join(quote_identifier(str(row["to"])) for row in group)
        definition = (
            f"FOREIGN KEY ({local}) REFERENCES {quote_identifier(str(first['table']))} ({remote})"
        )
        if first["on_update"] and str(first["on_update"]).upper() != "NO ACTION":
            definition += f" ON UPDATE {str(first['on_update']).upper()}"
        if first["on_delete"] and str(first["on_delete"]).upper() != "NO ACTION":
            definition += f" ON DELETE {str(first['on_delete']).upper()}"
        definitions.append(definition)
    return definitions


def _create_table_clause(
    connection: sqlite3.Connection,
    table: str,
    type_overrides: dict[str, str],
    extra_checks: Sequence[str],
) -> tuple[str, list[str], list[int]]:
    info = [row for row in table_info(connection, table) if not row["hidden"]]
    pk = primary_key_columns(connection, table)
    single_integer_pk = (
        len(pk) == 1
        and normalize_sql_type(column_types(connection, table).get(pk[0], "")) == "INTEGER"
        and pk[0] not in type_overrides
    )
    definitions: list[str] = []
    for row in info:
        name = str(row["name"])
        raw_type = type_overrides.get(name, normalize_sql_type(row["type"]))
        column = f" {quote_identifier(name)} {raw_type}"
        if single_integer_pk and name == pk[0]:
            column += " PRIMARY KEY"
        else:
            if row["notnull"]:
                column += " NOT NULL"
            if row["dflt_value"] is not None:
                column += f" DEFAULT {row['dflt_value']}"
        definitions.append(column)
    if pk and not single_integer_pk:
        definitions.append(
            "PRIMARY KEY (" + ", ".join(quote_identifier(name) for name in pk) + ")"
        )
    original_sql_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    original_sql = str(original_sql_row[0]) if original_sql_row and original_sql_row[0] else ""
    checks = _extract_check_expressions(original_sql)
    try:
        managed_rows = connection.execute(
            "SELECT definition FROM _mig_constraints WHERE table_name = ? AND kind = 'check' ORDER BY constraint_name",
            (table,),
        ).fetchall()
        checks.extend(str(row[0]) for row in managed_rows)
    except sqlite3.OperationalError:
        pass
    seen_checks: set[str] = set()
    for expression in [*checks, *extra_checks]:
        normalized = expression.strip()
        if normalized and normalized not in seen_checks:
            seen_checks.add(normalized)
            definitions.append(f"CHECK ({normalized})")
    definitions.extend(_existing_foreign_key_definitions(connection, table))
    return "(\n    " + ",\n    ".join(definitions) + "\n)", [str(row["name"]) for row in info], [
        int(row["cid"]) for row in info
    ]


def rebuild_table(
    connection: sqlite3.Connection,
    table: str,
    *,
    type_overrides: dict[str, str] | None = None,
    value_transform: Callable[[str, Any, str], Any] | None = None,
    extra_checks: Sequence[str] = (),
) -> int:
    """Rebuild a table inside the caller's transaction.

    The package deliberately keeps the table shape explicit so the operation is
    deterministic and works on Windows with only the Python SQLite driver.
    """

    type_overrides = type_overrides or {}
    clause, columns, _ = _create_table_clause(
        connection, table, type_overrides, extra_checks
    )
    temp_table = f"{INTERNAL_PREFIX}tmp_{table}"
    connection.execute(f"DROP TABLE IF EXISTS {quote_identifier(temp_table)}")
    connection.execute(f"CREATE TABLE {quote_identifier(temp_table)} {clause}")
    indexes = connection.execute(
        """
        SELECT sql FROM sqlite_master
        WHERE type = 'index' AND tbl_name = ? AND sql IS NOT NULL
        ORDER BY name
        """,
        (table,),
    ).fetchall()
    rowid_available = has_rowid(connection, table)
    selected = (["_rowid_"] if rowid_available else []) + [quote_identifier(c) for c in columns]
    select_sql = f"SELECT {', '.join(selected)} FROM {quote_identifier(table)}"
    if rowid_available:
        select_sql += " ORDER BY _rowid_"
    else:
        key_columns = row_key_columns(connection, table)
        select_sql += " ORDER BY " + ", ".join(quote_identifier(c) for c in key_columns)
    cursor = connection.execute(select_sql)
    insert_columns = (["rowid"] if rowid_available else []) + columns
    placeholders = ", ".join("?" for _ in insert_columns)
    insert_sql = (
        f"INSERT INTO {quote_identifier(temp_table)} "
        f"({', '.join(quote_identifier(c) for c in insert_columns)}) VALUES ({placeholders})"
    )
    row_count = 0
    while True:
        rows = cursor.fetchmany(256)
        if not rows:
            break
        for row in rows:
            offset = 1 if rowid_available else 0
            rowid_value = row[0] if rowid_available else None
            values = [row[offset + index] for index in range(len(columns))]
            if value_transform is not None:
                for index, column in enumerate(columns):
                    transformed = value_transform(column, values[index], table)
                    values[index] = transformed
            if rowid_available:
                connection.execute(insert_sql, [rowid_value, *values])
            else:
                connection.execute(insert_sql, values)
            row_count += 1
    connection.execute(f"DROP TABLE {quote_identifier(table)}")
    connection.execute(
        f"ALTER TABLE {quote_identifier(temp_table)} RENAME TO {quote_identifier(table)}"
    )
    for index_row in indexes:
        index_sql = str(index_row[0])
        if index_sql:
            connection.execute(index_sql)
    return row_count


def find_latest_row_error(
    connection: sqlite3.Connection,
    *,
    column: str,
    row_key: str,
) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT step_id, revision, units, error_bound, paths_json
        FROM _mig_row_errors
        WHERE column_name = ? AND row_key = ?
        ORDER BY revision DESC, step_id
        LIMIT 1
        """,
        (column, row_key),
    ).fetchone()


def check_expression_valid(
    connection: sqlite3.Connection,
    table: str,
    expression: str,
) -> None:
    try:
        connection.execute(
            f"SELECT ({expression}) FROM {quote_identifier(table)} WHERE 0"
        ).fetchall()
    except sqlite3.Error as exc:
        raise ValueError(str(exc)) from exc


def check_constraint_violations(
    connection: sqlite3.Connection,
    table: str,
    expression: str,
) -> int:
    row = connection.execute(
        f"SELECT COUNT(*) FROM {quote_identifier(table)} WHERE NOT ({expression})"
    ).fetchone()
    return int(row[0])
