from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sqlite3
from typing import Any, Iterable

from .database import (
    NUMERIC_TYPES,
    calibration_snapshot,
    canonical_json,
    column_types,
    normalize_sql_type,
    read_column_meta,
    table_exists,
    table_info,
    upsert_column_meta,
    user_tables,
    write_json_atomic,
)
from .errors import MigrationError, fail
from .fixedpoint import QuantizationSpec
from .manifest import Manifest, MigrationStep


GENESIS_CHAIN = "0" * 64


def current_revision(connection: sqlite3.Connection) -> int:
    ledger = connection.execute("SELECT COALESCE(MAX(revision), 0) FROM _mig_ledger").fetchone()[0]
    recalibrations = connection.execute(
        "SELECT COALESCE(MAX(revision), 0) FROM _mig_recalibrations"
    ).fetchone()[0]
    return max(int(ledger or 0), int(recalibrations or 0))


def seed_initial_calibrations(
    connection: sqlite3.Connection,
    manifest: Manifest,
    *,
    revision: int = 0,
) -> None:
    existing = read_column_meta(connection)
    for table in user_tables(connection):
        for row in table_info(connection, table):
            if row["hidden"]:
                continue
            column = str(row["name"])
            sql_type = normalize_sql_type(row["type"])
            key = (table, column)
            if key in existing:
                continue
            spec = manifest.initial_calibrations.get(key)
            if sql_type in NUMERIC_TYPES and spec is None:
                spec = manifest.default_quantization
            if sql_type in NUMERIC_TYPES and spec is None:
                fail(
                    "E_MIGRATE",
                    "numeric column has no initial calibration",
                    step_id=None,
                    field_path=f"$.initial_calibrations.{table}.{column}",
                )
            upsert_column_meta(
                connection,
                table=table,
                column=column,
                sql_type=sql_type,
                spec=spec,
                step_id=None,
                revision=revision,
            )


def _step_chain_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def record_step(
    connection: sqlite3.Connection,
    *,
    step: MigrationStep,
    before_columns: list[dict[str, Any]],
    after_columns: list[dict[str, Any]],
    row_count: int,
    result: dict[str, Any],
    content_digest: str,
    revision: int,
) -> dict[str, Any]:
    previous_row = connection.execute(
        "SELECT chain_digest FROM _mig_ledger ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    previous_chain = str(previous_row[0]) if previous_row else GENESIS_CHAIN
    chain_payload = {
        "step_id": step.step_id,
        "version": step.version,
        "operation": step.operation,
        "table": step.table,
        "row_count": row_count,
        "before_columns": before_columns,
        "after_columns": after_columns,
        "result": result,
        "content_digest": content_digest,
        "previous_chain_digest": previous_chain,
    }
    chain_digest = _step_chain_digest(chain_payload)
    connection.execute(
        """
        INSERT INTO _mig_ledger(
            step_id, version, operation, table_name, row_count,
            before_columns_json, after_columns_json, result_json,
            content_digest, previous_chain_digest, chain_digest, revision
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            step.step_id,
            step.version,
            step.operation,
            step.table,
            row_count,
            canonical_json(before_columns),
            canonical_json(after_columns),
            canonical_json(result),
            content_digest,
            previous_chain,
            chain_digest,
            revision,
        ),
    )
    return {
        "step_id": step.step_id,
        "version": step.version,
        "operation": step.operation,
        "table": step.table,
        "row_count": row_count,
        "before_columns": before_columns,
        "after_columns": after_columns,
        "result": result,
        "content_digest": content_digest,
        "previous_chain_digest": previous_chain,
        "chain_digest": chain_digest,
        "revision": revision,
    }


def ledger_rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute("SELECT * FROM _mig_ledger ORDER BY seq").fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                "seq": int(row["seq"]),
                "step_id": str(row["step_id"]),
                "version": int(row["version"]),
                "operation": str(row["operation"]),
                "table": str(row["table_name"]),
                "row_count": int(row["row_count"]),
                "before_columns": json.loads(str(row["before_columns_json"])),
                "after_columns": json.loads(str(row["after_columns_json"])),
                "result": json.loads(str(row["result_json"])),
                "content_digest": str(row["content_digest"]),
                "previous_chain_digest": str(row["previous_chain_digest"]),
                "chain_digest": str(row["chain_digest"]),
                "revision": int(row["revision"]),
            }
        )
    return result


def completed_step_ids(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute("SELECT step_id FROM _mig_ledger ORDER BY seq").fetchall()
    }


def ledger_version(connection: sqlite3.Connection) -> str:
    payload = {
        "ledger": ledger_rows(connection),
        "recalibrations": recalibration_rows(connection),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def insert_row_error(
    connection: sqlite3.Connection,
    *,
    step_id: str,
    column: str,
    row_key: str,
    revision: int,
    units: int,
    error_bound: int,
    paths: list[dict[str, Any]],
) -> None:
    connection.execute(
        """
        INSERT INTO _mig_row_errors(
            step_id, column_name, row_key, revision, units, error_bound, paths_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(step_id, column_name, row_key) DO UPDATE SET
            revision = excluded.revision,
            units = excluded.units,
            error_bound = excluded.error_bound,
            paths_json = excluded.paths_json
        """,
        (
            step_id,
            column,
            row_key,
            revision,
            int(units),
            int(error_bound),
            canonical_json(paths),
        ),
    )


def latest_row_error(
    connection: sqlite3.Connection, *, column: str, row_key: str
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


def row_errors_for_column(
    connection: sqlite3.Connection, column: str
) -> list[sqlite3.Row]:
    return list(
        connection.execute(
            """
            SELECT step_id, revision, row_key, units, error_bound, paths_json
            FROM _mig_row_errors
            WHERE column_name = ?
            ORDER BY row_key
            """,
            (column,),
        ).fetchall()
    )


def latest_row_errors_by_key(
    connection: sqlite3.Connection, column: str
) -> dict[str, sqlite3.Row]:
    result: dict[str, sqlite3.Row] = {}
    for row in connection.execute(
        """
        SELECT step_id, revision, row_key, units, error_bound, paths_json
        FROM _mig_row_errors
        WHERE column_name = ?
        ORDER BY row_key, revision DESC, step_id
        """,
        (column,),
    ).fetchall():
        key = str(row["row_key"])
        if key not in result:
            result[key] = row
    return result


def recalibration_rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute("SELECT * FROM _mig_recalibrations ORDER BY seq").fetchall()
    return [
        {
            "seq": int(row["seq"]),
            "event_id": str(row["event_id"]),
            "table": str(row["table_name"]),
            "column": str(row["column_name"]),
            "old_scale": int(row["old_scale"]),
            "new_scale": int(row["new_scale"]),
            "old_truncation": str(row["old_truncation"]),
            "new_truncation": str(row["new_truncation"]),
            "before_digest": str(row["before_digest"]),
            "after_digest": str(row["after_digest"]),
            "affected_steps": json.loads(str(row["affected_steps_json"])),
            "report": json.loads(str(row["report_json"])),
            "revision": int(row["revision"]),
        }
        for row in rows
    ]


def record_recalibration(
    connection: sqlite3.Connection,
    *,
    event_id: str,
    table: str,
    column: str,
    old_scale: int,
    new_scale: int,
    old_truncation: str,
    new_truncation: str,
    before_digest: str,
    after_digest: str,
    affected_steps: list[str],
    report: dict[str, Any],
    revision: int,
) -> dict[str, Any]:
    connection.execute(
        """
        INSERT INTO _mig_recalibrations(
            event_id, table_name, column_name, old_scale, new_scale,
            old_truncation, new_truncation, before_digest, after_digest,
            affected_steps_json, report_json, revision
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            table,
            column,
            old_scale,
            new_scale,
            old_truncation,
            new_truncation,
            before_digest,
            after_digest,
            canonical_json(sorted(affected_steps)),
            canonical_json(report),
            revision,
        ),
    )
    return {
        "event_id": event_id,
        "table": table,
        "column": column,
        "old_scale": old_scale,
        "new_scale": new_scale,
        "old_truncation": old_truncation,
        "new_truncation": new_truncation,
        "before_digest": before_digest,
        "after_digest": after_digest,
        "affected_steps": sorted(affected_steps),
        "report": report,
        "revision": revision,
    }


def ledger_view(connection: sqlite3.Connection) -> dict[str, Any]:
    run_row = connection.execute("SELECT * FROM _mig_run WHERE run_id = 1").fetchone()
    run = None
    if run_row is not None:
        run = {
            "manifest_id": str(run_row["manifest_id"]),
            "manifest_sha256": str(run_row["manifest_sha256"]),
            "plan": json.loads(str(run_row["plan_json"])),
            "base_schema_digest": str(run_row["base_schema_digest"]),
            "status": str(run_row["status"]),
            "current_step": run_row["current_step"],
            "revision": int(run_row["revision"]),
        }
    return {
        "run": run,
        "steps": ledger_rows(connection),
        "column_calibrations": calibration_snapshot(connection),
        "recalibrations": recalibration_rows(connection),
        "ledger_version": ledger_version(connection),
        "revision": current_revision(connection),
    }


def export_ledger(
    connection: sqlite3.Connection,
    *,
    ledger_dir: str | Path,
) -> dict[str, Any]:
    payload = ledger_view(connection)
    write_json_atomic(Path(ledger_dir) / "ledger.json", payload)
    return payload


def verify_ledger_chain(connection: sqlite3.Connection) -> tuple[bool, str | None]:
    previous = GENESIS_CHAIN
    for row in ledger_rows(connection):
        expected = _step_chain_digest(
            {
                "step_id": row["step_id"],
                "version": row["version"],
                "operation": row["operation"],
                "table": row["table"],
                "row_count": row["row_count"],
                "before_columns": row["before_columns"],
                "after_columns": row["after_columns"],
                "result": row["result"],
                "content_digest": row["content_digest"],
                "previous_chain_digest": previous,
            }
        )
        if row["previous_chain_digest"] != previous or row["chain_digest"] != expected:
            return False, str(row["step_id"])
        previous = str(row["chain_digest"])
    return True, None
