from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .errors import VersionMismatchError
from .loader import deserialize_parameter_snapshot
from .models import BorrowLoan, ConnectionState, ParameterSnapshot, WarningRecord


SCHEMA_VERSION = 1


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def snapshot_digest(snapshot: ParameterSnapshot) -> str:
    return hashlib.sha256(canonical_json(snapshot.to_dict()).encode("utf-8")).hexdigest()


def events_digest(events: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for event in events:
        digest.update(canonical_json(event).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class SQLiteLedger:
    """Transactional ledger for complete event batches only."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), isolation_level=None, timeout=15.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SQLiteLedger":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS parameter_versions (
                version INTEGER PRIMARY KEY,
                applied_after_seq INTEGER NOT NULL,
                active INTEGER NOT NULL,
                snapshot_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pending_batches (
                batch_id INTEGER PRIMARY KEY,
                start_seq INTEGER NOT NULL,
                end_seq INTEGER NOT NULL,
                parameter_version INTEGER NOT NULL,
                candidate_connections_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS batches (
                batch_id INTEGER PRIMARY KEY,
                start_seq INTEGER NOT NULL,
                end_seq INTEGER NOT NULL,
                event_count INTEGER NOT NULL,
                parameter_version INTEGER NOT NULL,
                event_digest TEXT NOT NULL,
                completed INTEGER NOT NULL DEFAULT 1,
                summary_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS batch_events (
                batch_id INTEGER NOT NULL,
                seq INTEGER NOT NULL,
                event_json TEXT NOT NULL,
                audit_json TEXT NOT NULL,
                PRIMARY KEY (batch_id, seq),
                FOREIGN KEY (batch_id) REFERENCES batches(batch_id)
            );
            CREATE TABLE IF NOT EXISTS pool_state (
                batch_id INTEGER NOT NULL,
                pool_id TEXT NOT NULL,
                parameter_version INTEGER NOT NULL,
                own_active INTEGER NOT NULL,
                borrowed_in INTEGER NOT NULL,
                lent_out INTEGER NOT NULL,
                active_connections INTEGER NOT NULL,
                capacity_in_use INTEGER NOT NULL,
                remaining_connections INTEGER NOT NULL,
                warning INTEGER NOT NULL,
                state_json TEXT NOT NULL,
                PRIMARY KEY (batch_id, pool_id),
                FOREIGN KEY (batch_id) REFERENCES batches(batch_id)
            );
            CREATE TABLE IF NOT EXISTS connection_state (
                batch_id INTEGER NOT NULL,
                connection_id TEXT NOT NULL,
                logical_pool TEXT NOT NULL,
                origin_pool TEXT NOT NULL,
                current_owner_pool TEXT NOT NULL,
                state TEXT NOT NULL,
                loan_id TEXT,
                parameter_version INTEGER NOT NULL,
                state_json TEXT NOT NULL,
                PRIMARY KEY (batch_id, connection_id),
                FOREIGN KEY (batch_id) REFERENCES batches(batch_id)
            );
            CREATE TABLE IF NOT EXISTS borrow_ledger (
                batch_id INTEGER NOT NULL,
                loan_id TEXT NOT NULL,
                lender_pool TEXT NOT NULL,
                borrower_pool TEXT NOT NULL,
                connection_id TEXT NOT NULL,
                amount INTEGER NOT NULL,
                deadline_ms INTEGER NOT NULL,
                status TEXT NOT NULL,
                stale INTEGER NOT NULL,
                borrowed_at_seq INTEGER NOT NULL,
                returned_at_seq INTEGER,
                parameter_version INTEGER NOT NULL,
                loan_json TEXT NOT NULL,
                PRIMARY KEY (batch_id, loan_id),
                FOREIGN KEY (batch_id) REFERENCES batches(batch_id)
            );
            CREATE TABLE IF NOT EXISTS warnings (
                batch_id INTEGER NOT NULL,
                warning_id TEXT NOT NULL,
                event_seq INTEGER NOT NULL,
                pool_id TEXT NOT NULL,
                parameter_version INTEGER NOT NULL,
                stale INTEGER NOT NULL,
                warning_json TEXT NOT NULL,
                PRIMARY KEY (batch_id, warning_id),
                FOREIGN KEY (batch_id) REFERENCES batches(batch_id)
            );
            CREATE TABLE IF NOT EXISTS parameter_transitions (
                batch_id INTEGER NOT NULL,
                version INTEGER NOT NULL,
                applied_after_seq INTEGER NOT NULL,
                transition_json TEXT NOT NULL,
                PRIMARY KEY (batch_id, version),
                FOREIGN KEY (batch_id) REFERENCES batches(batch_id)
            );
            CREATE INDEX IF NOT EXISTS idx_batch_events_seq ON batch_events(seq);
            CREATE INDEX IF NOT EXISTS idx_warnings_seq ON warnings(event_seq, pool_id);
            """
        )

    def _begin(self) -> None:
        self.connection.execute("BEGIN IMMEDIATE")

    def _commit(self) -> None:
        self.connection.execute("COMMIT")

    def _rollback(self) -> None:
        if self.connection.in_transaction:
            self.connection.execute("ROLLBACK")

    def initialize(self, initial_snapshot: ParameterSnapshot) -> None:
        digest = snapshot_digest(initial_snapshot)
        self._begin()
        try:
            existing_version = self._meta("initial_parameter_version")
            existing_digest = self._meta("initial_spec_sha256")
            if existing_version is not None:
                if int(existing_version) != initial_snapshot.version or existing_digest != digest:
                    raise VersionMismatchError(
                        "supplied initial parameters do not match the ledger",
                        path="pools.version",
                        details={
                            "ledger_initial_version": int(existing_version),
                            "supplied_initial_version": initial_snapshot.version,
                            "version_match": int(existing_version) == initial_snapshot.version,
                            "digest_match": existing_digest == digest,
                        },
                    )
                self._commit()
                return
            self.connection.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )
            self.connection.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?)",
                ("initial_parameter_version", str(initial_snapshot.version)),
            )
            self.connection.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?)",
                ("initial_spec_sha256", digest),
            )
            self.connection.execute(
                "INSERT INTO parameter_versions(version, applied_after_seq, active, snapshot_json) VALUES(?, ?, ?, ?)",
                (
                    initial_snapshot.version,
                    0,
                    1,
                    canonical_json(initial_snapshot.to_dict()),
                ),
            )
            self._commit()
        except Exception:
            self._rollback()
            raise

    def _meta(self, key: str) -> str | None:
        row = self.connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def next_batch_id(self) -> int:
        row = self.connection.execute("SELECT COALESCE(MAX(batch_id), 0) + 1 AS next_id FROM batches").fetchone()
        return int(row["next_id"])

    def begin_batch(
        self,
        *,
        batch_id: int,
        start_seq: int,
        end_seq: int,
        parameter_version: int,
        candidate_connections: Iterable[str],
    ) -> None:
        self._begin()
        try:
            self.connection.execute("DELETE FROM pending_batches")
            self.connection.execute(
                "INSERT INTO pending_batches(batch_id, start_seq, end_seq, parameter_version, candidate_connections_json) VALUES(?, ?, ?, ?, ?)",
                (
                    batch_id,
                    start_seq,
                    end_seq,
                    parameter_version,
                    canonical_json(sorted(set(candidate_connections))),
                ),
            )
            self._commit()
        except Exception:
            self._rollback()
            raise

    def commit_batch(
        self,
        *,
        batch_id: int,
        start_seq: int,
        end_seq: int,
        start_parameter_version: int,
        engine: Any,
        results: Iterable[Any],
        transitions: Iterable[tuple[ParameterSnapshot, dict[str, Any]]],
    ) -> None:
        result_list = list(results)
        transition_list = list(transitions)
        events_json = [result.event.to_dict() for result in result_list]
        digest = events_digest(events_json)
        pool_rows = engine.pool_rows()
        summary = {
            "batch_id": batch_id,
            "start_seq": start_seq,
            "end_seq": end_seq,
            "event_count": len(result_list),
            "start_parameter_version": start_parameter_version,
            "end_parameter_version": engine.parameter_version,
            "event_digest": digest,
            "pool_summary": pool_rows,
            "runtime": engine.runtime_state(),
            "transitions": [transition for _, transition in transition_list],
        }
        self._begin()
        try:
            self.connection.execute(
                """
                INSERT INTO batches(batch_id, start_seq, end_seq, event_count, parameter_version, event_digest, completed, summary_json)
                VALUES(?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    batch_id,
                    start_seq,
                    end_seq,
                    len(result_list),
                    engine.parameter_version,
                    digest,
                    canonical_json(summary),
                ),
            )
            for result in result_list:
                self.connection.execute(
                    "INSERT INTO batch_events(batch_id, seq, event_json, audit_json) VALUES(?, ?, ?, ?)",
                    (
                        batch_id,
                        result.event.seq,
                        canonical_json(result.event.to_dict()),
                        canonical_json(result.to_dict()),
                    ),
                )
            for row in pool_rows:
                self.connection.execute(
                    """
                    INSERT INTO pool_state(batch_id, pool_id, parameter_version, own_active, borrowed_in, lent_out,
                        active_connections, capacity_in_use, remaining_connections, warning, state_json)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        batch_id,
                        row["pool_id"],
                        row["parameter_version"],
                        row["own_active"],
                        row["borrowed_in"],
                        row["lent_out"],
                        row["active_connections"],
                        row["capacity_in_use"],
                        row["remaining_connections"],
                        1 if row["warning"] else 0,
                        canonical_json(row),
                    ),
                )
            for row in engine.connection_rows():
                self.connection.execute(
                    """
                    INSERT INTO connection_state(batch_id, connection_id, logical_pool, origin_pool, current_owner_pool,
                        state, loan_id, parameter_version, state_json)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        batch_id,
                        row["connection_id"],
                        row["logical_pool"],
                        row["origin_pool"],
                        row["current_owner_pool"],
                        row["state"],
                        row["loan_id"],
                        engine.parameter_version,
                        canonical_json(row),
                    ),
                )
            for row in engine.loan_rows():
                self.connection.execute(
                    """
                    INSERT INTO borrow_ledger(batch_id, loan_id, lender_pool, borrower_pool, connection_id, amount,
                        deadline_ms, status, stale, borrowed_at_seq, returned_at_seq, parameter_version, loan_json)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        batch_id,
                        row["loan_id"],
                        row["lender_pool"],
                        row["borrower_pool"],
                        row["connection_id"],
                        row["amount"],
                        row["deadline_ms"],
                        row["status"],
                        1 if row["stale"] else 0,
                        row["borrowed_at_seq"],
                        row["returned_at_seq"],
                        row["parameter_version"],
                        canonical_json(row),
                    ),
                )
            for row in engine.warning_rows():
                self.connection.execute(
                    """
                    INSERT INTO warnings(batch_id, warning_id, event_seq, pool_id, parameter_version, stale, warning_json)
                    VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        batch_id,
                        row["warning_id"],
                        row["event_seq"],
                        row["pool_id"],
                        row["parameter_version"],
                        1 if row["stale"] else 0,
                        canonical_json(row),
                    ),
                )
            for snapshot, transition in transition_list:
                self.connection.execute(
                    "UPDATE parameter_versions SET active=0 WHERE active=1",
                )
                self.connection.execute(
                    "INSERT INTO parameter_versions(version, applied_after_seq, active, snapshot_json) VALUES(?, ?, 1, ?)",
                    (
                        snapshot.version,
                        transition["after_event_seq"],
                        canonical_json(snapshot.to_dict()),
                    ),
                )
                self.connection.execute(
                    "INSERT INTO parameter_transitions(batch_id, version, applied_after_seq, transition_json) VALUES(?, ?, ?, ?)",
                    (
                        batch_id,
                        snapshot.version,
                        transition["after_event_seq"],
                        canonical_json(transition),
                    ),
                )
            self.connection.execute("DELETE FROM pending_batches WHERE batch_id=?", (batch_id,))
            self._commit()
        except Exception:
            self._rollback()
            raise

    def latest_batch(self) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM batches WHERE completed=1 ORDER BY batch_id DESC LIMIT 1"
        ).fetchone()
        return None if row is None else dict(row)

    def pending_batch(self) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM pending_batches ORDER BY batch_id DESC LIMIT 1").fetchone()
        if row is None:
            return None
        result = dict(row)
        result["candidate_connections"] = json.loads(result.pop("candidate_connections_json"))
        return result

    def current_snapshot(self) -> ParameterSnapshot:
        row = self.connection.execute(
            "SELECT snapshot_json FROM parameter_versions WHERE active=1 ORDER BY version DESC LIMIT 1"
        ).fetchone()
        if row is None:
            raise RuntimeError("ledger has no active parameter version")
        return deserialize_parameter_snapshot(json.loads(row["snapshot_json"]))

    def initial_snapshot(self) -> ParameterSnapshot:
        row = self.connection.execute(
            "SELECT snapshot_json FROM parameter_versions ORDER BY version ASC LIMIT 1"
        ).fetchone()
        if row is None:
            raise RuntimeError("ledger has no parameter versions")
        return deserialize_parameter_snapshot(json.loads(row["snapshot_json"]))

    def parameter_history(self) -> list[ParameterSnapshot]:
        rows = self.connection.execute(
            "SELECT snapshot_json FROM parameter_versions ORDER BY version ASC"
        ).fetchall()
        return [deserialize_parameter_snapshot(json.loads(row["snapshot_json"])) for row in rows]

    def parameter_versions(self) -> dict[int, ParameterSnapshot]:
        return {snapshot.version: snapshot for snapshot in self.parameter_history()}

    def recovery_state(self) -> dict[str, Any] | None:
        latest = self.latest_batch()
        if latest is None:
            return None
        batch_id = int(latest["batch_id"])
        summary = json.loads(latest["summary_json"])
        connection_rows = self.connection.execute(
            "SELECT state_json FROM connection_state WHERE batch_id=? ORDER BY connection_id",
            (batch_id,),
        ).fetchall()
        loan_rows = self.connection.execute(
            "SELECT loan_json FROM borrow_ledger WHERE batch_id=? ORDER BY loan_id",
            (batch_id,),
        ).fetchall()
        warning_rows = self.connection.execute(
            "SELECT warning_json FROM warnings WHERE batch_id=? ORDER BY event_seq, warning_id",
            (batch_id,),
        ).fetchall()
        pool_rows = self.connection.execute(
            "SELECT state_json FROM pool_state WHERE batch_id=? ORDER BY pool_id",
            (batch_id,),
        ).fetchall()
        connections = [ConnectionState(**json.loads(row["state_json"])) for row in connection_rows]
        loans = [BorrowLoan(**json.loads(row["loan_json"])) for row in loan_rows]
        warnings: list[WarningRecord] = []
        for row in warning_rows:
            data = json.loads(row["warning_json"])
            data.pop("type", None)
            warnings.append(WarningRecord(**data))
        return {
            "batch": latest,
            "summary": summary,
            "snapshot": self.current_snapshot(),
            "parameter_history": self.parameter_history(),
            "connections": connections,
            "loans": loans,
            "warnings": warnings,
            "pool_rows": [json.loads(row["state_json"]) for row in pool_rows],
            "runtime": summary["runtime"],
        }

    def all_event_audits(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT seq, event_json, audit_json FROM batch_events ORDER BY seq ASC"
        ).fetchall()
        return [
            {
                "seq": int(row["seq"]),
                "event": json.loads(row["event_json"]),
                "result": json.loads(row["audit_json"]),
            }
            for row in rows
        ]

    def transitions(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT version, applied_after_seq, transition_json FROM parameter_transitions ORDER BY applied_after_seq, version"
        ).fetchall()
        return [
            {
                "version": int(row["version"]),
                "applied_after_seq": int(row["applied_after_seq"]),
                "transition": json.loads(row["transition_json"]),
            }
            for row in rows
        ]

    def ledger_meta(self) -> dict[str, str]:
        rows = self.connection.execute("SELECT key, value FROM meta ORDER BY key").fetchall()
        return {str(row["key"]): str(row["value"]) for row in rows}

    def counts(self) -> dict[str, int]:
        tables = (
            "batches",
            "batch_events",
            "pool_state",
            "connection_state",
            "borrow_ledger",
            "warnings",
            "parameter_versions",
            "parameter_transitions",
            "pending_batches",
        )
        return {
            table: int(self.connection.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])
            for table in tables
        }
