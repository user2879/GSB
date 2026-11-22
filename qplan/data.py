from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Iterator

from .canonical import canonical_json, content_digest
from .spec import ColumnSpec, Spec, TableSpec


MASK64 = (1 << 64) - 1


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def _text_word(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")


def stable_hash(*parts: Any) -> int:
    value = 0xCBF29CE484222325
    for part in parts:
        if isinstance(part, int):
            word = part & MASK64
        elif isinstance(part, float):
            word = int.from_bytes(repr(part).encode("ascii"), "little") & MASK64
        else:
            word = _text_word(str(part))
        value = splitmix64(value ^ word)
    return value


def deterministic_unit(*parts: Any) -> float:
    return stable_hash(*parts) / float(1 << 64)


def _target_unique_start(spec: Spec, table_name: str, column_name: str) -> int:
    column = spec.table(table_name).column(column_name)
    return int(column.generator.get("start", 1))


def generated_value(spec: Spec, table: TableSpec, column: ColumnSpec, row_index: int) -> Any:
    generator = column.generator or {}
    kind = generator.get("kind", "default")
    seed = spec.seed
    h = stable_hash(seed, table.name, column.name, row_index)
    if column.nullable and column.null_fraction > 0 and deterministic_unit(seed, "null", table.name, column.name, row_index) < column.null_fraction:
        if kind != "unique_int":
            return None
    if kind == "unique_int":
        return int(generator.get("start", 1)) + row_index
    if kind == "uniform_int":
        low = int(generator["min"])
        high = int(generator["max"])
        return low + int(h % (high - low + 1))
    if kind == "categorical":
        values = generator["values"]
        weights = generator.get("weights")
        if weights is None:
            return values[int(h % len(values))]
        unit = (h / float(1 << 64))
        total = float(sum(weights))
        threshold = unit * total
        accumulated = 0.0
        for value, weight in zip(values, weights):
            accumulated += float(weight)
            if threshold < accumulated:
                return value
        return values[-1]
    if kind == "uniform_float":
        low = float(generator["min"])
        high = float(generator["max"])
        decimals = int(generator.get("decimals", 6))
        return round(low + (h / float(1 << 64)) * (high - low), decimals)
    if kind == "foreign_key":
        target_table = spec.table(generator["table"])
        start = _target_unique_start(spec, target_table.name, generator["column"])
        return start + int(h % target_table.row_count)
    if kind == "constant":
        return generator["value"]
    if kind == "default":
        if column.type == "integer":
            return int(h % 1000)
        return round((h / float(1 << 64)) * 1000.0, 6)
    raise ValueError(f"unknown generator kind {kind!r}")


def generation_fingerprint(spec: Spec) -> str:
    return content_digest(
        {
            "model": "qplan-data-splitmix64-v1",
            "seed": spec.seed,
            "tables": [
                {
                    "name": table.name,
                    "row_count": table.row_count,
                    "columns": [
                        {
                            "name": column.name,
                            "type": column.type,
                            "nullable": column.nullable,
                            "null_fraction": column.null_fraction,
                            "generator": column.generator,
                        }
                        for column in table.columns
                    ],
                    "indexes": [
                        {"name": index.name, "columns": list(index.columns)} for index in table.candidate_indexes
                    ],
                }
                for table in spec.tables
            ],
        }
    )


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def sqlite_uri(path: str | Path, mode: str = "ro") -> str:
    """Build a cross-platform SQLite file URI, including Windows drive paths."""

    return Path(path).resolve().as_uri() + f"?mode={mode}"


def _iter_rows(spec: Spec, table: TableSpec) -> Iterator[tuple[Any, ...]]:
    columns = table.columns
    for row_index in range(table.row_count):
        yield tuple(generated_value(spec, table, column, row_index) for column in columns)


def _create_table(connection: sqlite3.Connection, table: TableSpec) -> None:
    column_sql = []
    for column in table.columns:
        sql_type = "INTEGER" if column.type == "integer" else "REAL"
        column_sql.append(f"{quote_identifier(column.name)} {sql_type}")
    connection.execute(f"CREATE TABLE {quote_identifier(table.name)} ({', '.join(column_sql)})")


def _insert_rows(connection: sqlite3.Connection, spec: Spec, table: TableSpec) -> None:
    placeholders = ",".join("?" for _ in table.columns)
    column_list = ",".join(quote_identifier(column.name) for column in table.columns)
    sql = f"INSERT INTO {quote_identifier(table.name)} ({column_list}) VALUES ({placeholders})"
    batch: list[tuple[Any, ...]] = []
    for row in _iter_rows(spec, table):
        batch.append(row)
        if len(batch) >= 2000:
            connection.executemany(sql, batch)
            batch.clear()
    if batch:
        connection.executemany(sql, batch)


def _create_indexes(connection: sqlite3.Connection, table: TableSpec) -> None:
    for index in table.candidate_indexes:
        columns = ",".join(quote_identifier(column) for column in index.columns)
        connection.execute(f"CREATE INDEX {quote_identifier(index.name)} ON {quote_identifier(table.name)} ({columns})")


def _database_matches(db_path: Path, fingerprint: str, spec: Spec) -> bool:
    if not db_path.is_file():
        return False
    try:
        connection = sqlite3.connect(sqlite_uri(db_path), uri=True)
        try:
            row = connection.execute("SELECT value FROM qplan_meta WHERE key='generation_fingerprint'").fetchone()
            if row is None or row[0] != fingerprint:
                return False
            for table in spec.tables:
                count = connection.execute(f"SELECT COUNT(*) FROM {quote_identifier(table.name)}").fetchone()[0]
                if int(count) != table.row_count:
                    return False
            return connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        finally:
            connection.close()
    except (sqlite3.Error, OSError):
        return False


def materialize_database(spec: Spec, db_path: str | Path, *, force: bool = False) -> dict[str, Any]:
    """Build the deterministic sample database atomically.

    A temporary SQLite file is populated and ``os.replace`` publishes it only
    after a clean commit, so a process kill cannot expose a half-built sample
    as the current data file.
    """

    target = Path(db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = generation_fingerprint(spec)
    if not force and _database_matches(target, fingerprint, spec):
        return {
            "database": str(target.resolve()),
            "generation_fingerprint": fingerprint,
            "reused": True,
            "tables": {table.name: table.row_count for table in spec.tables},
        }

    handle, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(handle)
    temp_path = Path(temp_name)
    try:
        connection = sqlite3.connect(temp_path)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("CREATE TABLE qplan_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute(
                "INSERT INTO qplan_meta(key,value) VALUES('generation_fingerprint',?)", (fingerprint,)
            )
            connection.execute(
                "INSERT INTO qplan_meta(key,value) VALUES('generator_model','qplan-data-splitmix64-v1')"
            )
            for table in spec.tables:
                _create_table(connection, table)
                _insert_rows(connection, spec, table)
            for table in spec.tables:
                _create_indexes(connection, table)
            connection.commit()
        finally:
            connection.close()
        os.replace(temp_path, target)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return {
        "database": str(target.resolve()),
        "generation_fingerprint": fingerprint,
        "reused": False,
        "tables": {table.name: table.row_count for table in spec.tables},
    }


def data_digest_summary(spec: Spec, db_path: str | Path) -> dict[str, Any]:
    """A lightweight, reproducible summary used in reports (not a full dump)."""

    connection = sqlite3.connect(sqlite_uri(db_path), uri=True)
    try:
        tables = []
        for table in spec.tables:
            count, minimum, maximum = connection.execute(
                f"SELECT COUNT(*), MIN(rowid), MAX(rowid) FROM {quote_identifier(table.name)}"
            ).fetchone()
            tables.append(
                {
                    "name": table.name,
                    "declared_rows": table.row_count,
                    "actual_rows": int(count),
                    "min_rowid": minimum,
                    "max_rowid": maximum,
                }
            )
        return {"tables": tables, "summary_digest": content_digest(tables)}
    finally:
        connection.close()
