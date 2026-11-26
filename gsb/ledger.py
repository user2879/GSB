"""SQLite ledger with atomic batch commits, snapshot replacement and replay."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from .canonical import canonical_json, sha256_hex
from .errors import DiagnosticError
from .model import SessionSpec
from .params import compute_batch_summary
from .scheduler import SchedulerJob

LEDGER_FORMAT_VERSION = 1


@dataclass(frozen=True)
class BatchLedgerRecord:
    sequence: int
    batch_id: str
    shard_id: str
    batch_order: int
    parameter_version: int
    parameter_digest: str
    loss_value: str
    loss_digest: str
    sample_ids: tuple[int, ...]
    lease_id: str
    lease_version: int
    lease_owner: str
    checkpoint_state: dict[str, Any]
    progress_cursor: int
    committed_at_ms: int
    format_version: int = LEDGER_FORMAT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "batch_id": self.batch_id,
            "shard_id": self.shard_id,
            "batch_order": self.batch_order,
            "parameter_version": self.parameter_version,
            "parameter_digest": self.parameter_digest,
            "loss_value": self.loss_value,
            "loss_digest": self.loss_digest,
            "sample_ids": list(self.sample_ids),
            "lease_id": self.lease_id,
            "lease_version": self.lease_version,
            "lease_owner": self.lease_owner,
            "checkpoint_state": self.checkpoint_state,
            "progress_cursor": self.progress_cursor,
            "committed_at_ms": self.committed_at_ms,
            "format_version": self.format_version,
        }

    @property
    def content_digest(self) -> str:
        return sha256_hex(self.to_dict())


class Ledger:
    def __init__(self, connection: sqlite3.Connection, path: str, session: SessionSpec) -> None:
        self.connection = connection
        self.path = path
        self.session = session

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "format_version": LEDGER_FORMAT_VERSION,
            "shard_set_version": self.session.shard_set_version,
            "parameter_versions": self.session.parameter_versions,
        }

    @classmethod
    def open(
        cls,
        path: str | Path,
        session: SessionSpec,
        *,
        replace_on_mismatch: bool = True,
    ) -> "Ledger":
        ledger_path = str(path)
        parent = Path(path).parent
        parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(ledger_path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        ledger = cls(connection, ledger_path, session)
        ledger._ensure_schema()
        needs_replace, reason = ledger._snapshot_needs_replace()
        if needs_replace:
            if not replace_on_mismatch:
                ledger.connection.close()
                raise DiagnosticError(
                    "E_REPLAY",
                    f"ledger snapshot identity/content mismatch: {reason}",
                    details={"ledger": ledger_path, "reason": reason},
                )
            ledger._replace_snapshot(reason=reason)
        return ledger

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield self.connection
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        else:
            self.connection.execute("COMMIT")

    def _ensure_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS snapshot (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                format_version INTEGER NOT NULL,
                shard_set_version TEXT NOT NULL,
                parameter_versions TEXT NOT NULL,
                content_digest TEXT NOT NULL,
                epoch INTEGER NOT NULL,
                replaced_reason TEXT
            );
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS batches (
                sequence INTEGER PRIMARY KEY,
                batch_id TEXT NOT NULL UNIQUE,
                shard_id TEXT NOT NULL,
                batch_order INTEGER NOT NULL,
                parameter_version INTEGER NOT NULL,
                parameter_digest TEXT NOT NULL,
                format_version INTEGER NOT NULL,
                loss_value TEXT NOT NULL,
                loss_digest TEXT NOT NULL,
                sample_ids TEXT NOT NULL,
                lease_id TEXT NOT NULL,
                lease_version INTEGER NOT NULL,
                lease_owner TEXT NOT NULL,
                checkpoint_state TEXT NOT NULL,
                checkpoint_digest TEXT NOT NULL,
                progress_cursor INTEGER NOT NULL,
                content_digest TEXT NOT NULL,
                committed_at_ms INTEGER NOT NULL,
                UNIQUE(shard_id, batch_order)
            );
            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL,
                shard_id TEXT NOT NULL,
                batch_order INTEGER NOT NULL,
                lease_id TEXT,
                lease_version INTEGER,
                status TEXT NOT NULL,
                error_code TEXT,
                error_message TEXT,
                created_at_ms INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                shard_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                state_digest TEXT NOT NULL,
                updated_at_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS leases (
                lease_id TEXT PRIMARY KEY,
                shard_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                state TEXT NOT NULL,
                updated_at_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS progress (
                shard_id TEXT PRIMARY KEY,
                cursor INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                state_digest TEXT NOT NULL,
                applied_at_ms INTEGER NOT NULL
            );
            """
        )

    def _snapshot_needs_replace(self) -> tuple[bool, str]:
        row = self.connection.execute("SELECT * FROM snapshot WHERE singleton=1").fetchone()
        if row is None:
            return True, "missing snapshot"
        if int(row["format_version"]) != LEDGER_FORMAT_VERSION:
            return True, "format_version mismatch"
        if row["shard_set_version"] != self.session.shard_set_version:
            return True, "shard_set_version mismatch"
        stored_versions = json.loads(row["parameter_versions"])
        if stored_versions != self.session.parameter_versions:
            return True, "parameter_versions mismatch"
        actual_digest = self._compute_content_digest()
        if row["content_digest"] != actual_digest:
            return True, "content_digest mismatch"
        meta_digest = self.connection.execute(
            "SELECT value FROM meta WHERE key='content_digest'"
        ).fetchone()
        if meta_digest is None or meta_digest["value"] != actual_digest:
            return True, "meta content_digest mismatch"
        return False, ""

    def _replace_snapshot(self, *, reason: str) -> None:
        with self._transaction() as conn:
            old = conn.execute("SELECT epoch FROM snapshot WHERE singleton=1").fetchone()
            epoch = (int(old["epoch"]) if old else 0) + 1
            for table in (
                "events",
                "runtime_state",
                "progress",
                "leases",
                "checkpoints",
                "attempts",
                "batches",
                "meta",
                "snapshot",
            ):
                conn.execute(f"DELETE FROM {table}")
            identity = self.identity
            digest = self._compute_content_digest(conn=conn)
            conn.execute(
                """
                INSERT INTO snapshot(
                    singleton, format_version, shard_set_version,
                    parameter_versions, content_digest, epoch, replaced_reason
                ) VALUES (1, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identity["format_version"],
                    identity["shard_set_version"],
                    canonical_json(identity["parameter_versions"]),
                    digest,
                    epoch,
                    reason,
                ),
            )
            conn.execute(
                "INSERT INTO meta(key,value) VALUES('content_digest',?)", (digest,)
            )
            conn.execute(
                "INSERT INTO meta(key,value) VALUES('format_version',?)",
                (str(LEDGER_FORMAT_VERSION),),
            )

    def _compute_content_digest(
        self, *, conn: sqlite3.Connection | None = None
    ) -> str:
        db = conn or self.connection
        batches = [
            dict(row)
            for row in db.execute(
                "SELECT sequence,batch_id,shard_id,batch_order,parameter_version,"
                "parameter_digest,format_version,loss_value,loss_digest,sample_ids,lease_id,"
                "lease_version,lease_owner,checkpoint_state,checkpoint_digest,"
                "progress_cursor,content_digest,committed_at_ms "
                "FROM batches ORDER BY sequence"
            ).fetchall()
        ]
        checkpoints = [
            dict(row)
            for row in db.execute(
                "SELECT checkpoint_id,shard_id,payload,state_digest FROM checkpoints "
                "ORDER BY checkpoint_id"
            ).fetchall()
        ]
        progress = [
            dict(row)
            for row in db.execute(
                "SELECT shard_id,cursor FROM progress ORDER BY shard_id"
            ).fetchall()
        ]
        runtime = {
            row["key"]: json.loads(row["value"])
            for row in db.execute("SELECT key,value FROM runtime_state ORDER BY key").fetchall()
        }
        events = [
            dict(row)
            for row in db.execute(
                "SELECT seq,event_type,payload,state_digest FROM events ORDER BY seq"
            ).fetchall()
        ]
        attempts = [
            dict(row)
            for row in db.execute(
                "SELECT attempt_id,batch_id,shard_id,batch_order,lease_id,lease_version,"
                "status,error_code,error_message,created_at_ms,updated_at_ms "
                "FROM attempts ORDER BY attempt_id"
            ).fetchall()
        ]
        leases = [
            dict(row)
            for row in db.execute(
                "SELECT lease_id,shard_id,payload,state,updated_at_ms "
                "FROM leases ORDER BY lease_id"
            ).fetchall()
        ]
        return sha256_hex(
            {
                "identity": self.identity,
                "batches": batches,
                "checkpoints": checkpoints,
                "progress": progress,
                "runtime_state": runtime,
                "events": events,
                "attempts": attempts,
                "leases": leases,
            }
        )

    def _refresh_digest(self, conn: sqlite3.Connection | None = None) -> str:
        db = conn or self.connection
        digest = self._compute_content_digest(conn=db)
        db.execute(
            "UPDATE snapshot SET content_digest=? WHERE singleton=1", (digest,)
        )
        db.execute(
            "INSERT INTO meta(key,value) VALUES('content_digest',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (digest,),
        )
        return digest

    @staticmethod
    def attempt_id(shard_id: str, batch_order: int, batch_id: str) -> str:
        return f"attempt:{shard_id}:{batch_order}:{batch_id}"

    def start_batch(
        self,
        job: SchedulerJob,
        lease: Mapping[str, Any],
        *,
        now_ms: int,
    ) -> str:
        attempt_id = self.attempt_id(job.shard.shard_id, job.batch.order, job.batch.batch_id)
        committed = self.connection.execute(
            "SELECT 1 FROM batches WHERE batch_id=?", (job.batch.batch_id,)
        ).fetchone()
        if committed is not None:
            raise DiagnosticError(
                "E_REPLAY",
                f"batch {job.batch.batch_id} is already committed",
                details={"batch_id": job.batch.batch_id, "shard_id": job.shard.shard_id},
            )
        lease_payload = canonical_json(dict(lease))
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO attempts(
                    attempt_id,batch_id,shard_id,batch_order,lease_id,lease_version,
                    status,error_code,error_message,created_at_ms,updated_at_ms
                ) VALUES(?,?,?,?,?,?,'pending',NULL,NULL,?,?)
                ON CONFLICT(attempt_id) DO UPDATE SET
                    lease_id=excluded.lease_id,
                    lease_version=excluded.lease_version,
                    status='pending',error_code=NULL,error_message=NULL,
                    updated_at_ms=excluded.updated_at_ms
                """,
                (
                    attempt_id,
                    job.batch.batch_id,
                    job.shard.shard_id,
                    job.batch.order,
                    str(lease["lease_id"]),
                    int(lease["lease_version"]),
                    now_ms,
                    now_ms,
                ),
            )
            conn.execute(
                """
                INSERT INTO leases(lease_id,shard_id,payload,state,updated_at_ms)
                VALUES(?,?,?,'active',?)
                ON CONFLICT(lease_id) DO UPDATE SET payload=excluded.payload,
                    state='active',updated_at_ms=excluded.updated_at_ms
                """,
                (str(lease["lease_id"]), job.shard.shard_id, lease_payload, now_ms),
            )
            self._refresh_digest(conn)
        return attempt_id

    def commit_batch(self, record: BatchLedgerRecord) -> str:
        if record.format_version != LEDGER_FORMAT_VERSION:
            raise DiagnosticError(
                "E_REPLAY",
                f"batch format_version {record.format_version} is not supported",
                details={
                    "field": "format_version",
                    "batch_id": record.batch_id,
                    "shard_id": record.shard_id,
                    "expected": LEDGER_FORMAT_VERSION,
                    "actual": record.format_version,
                },
            )
        attempt_id = self.attempt_id(record.shard_id, record.batch_order, record.batch_id)
        content_digest = record.content_digest
        checkpoint_digest = sha256_hex(record.checkpoint_state)
        with self._transaction() as conn:
            expected_digest = self._compute_content_digest(conn=conn)
            snapshot = conn.execute(
                "SELECT content_digest FROM snapshot WHERE singleton=1"
            ).fetchone()
            if snapshot is None or snapshot["content_digest"] != expected_digest:
                raise DiagnosticError(
                    "E_REPLAY", "ledger content digest changed outside the transaction"
                )
            conn.execute(
                """
                INSERT INTO batches(
                    sequence,batch_id,shard_id,batch_order,parameter_version,
                    parameter_digest,format_version,loss_value,loss_digest,sample_ids,lease_id,
                    lease_version,lease_owner,checkpoint_state,checkpoint_digest,
                    progress_cursor,content_digest,committed_at_ms
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.sequence,
                    record.batch_id,
                    record.shard_id,
                    record.batch_order,
                    record.parameter_version,
                    record.parameter_digest,
                    record.format_version,
                    record.loss_value,
                    record.loss_digest,
                    canonical_json(list(record.sample_ids)),
                    record.lease_id,
                    record.lease_version,
                    record.lease_owner,
                    canonical_json(record.checkpoint_state),
                    checkpoint_digest,
                    record.progress_cursor,
                    content_digest,
                    record.committed_at_ms,
                ),
            )
            attempt_update = conn.execute(
                """
                UPDATE attempts SET status='committed',updated_at_ms=?
                WHERE attempt_id=?
                """,
                (record.committed_at_ms, attempt_id),
            )
            if attempt_update.rowcount != 1:
                raise DiagnosticError(
                    "E_REPLAY",
                    "batch commit has no matching pending attempt",
                    details={
                        "field": "attempt_id",
                        "batch_id": record.batch_id,
                        "shard_id": record.shard_id,
                    },
                )
            conn.execute(
                """
                INSERT INTO progress(shard_id,cursor,updated_at_ms) VALUES(?,?,?)
                ON CONFLICT(shard_id) DO UPDATE SET cursor=MAX(progress.cursor,excluded.cursor),
                    updated_at_ms=excluded.updated_at_ms
                """,
                (record.shard_id, record.progress_cursor, record.committed_at_ms),
            )
            lease_update = conn.execute(
                """
                UPDATE leases SET state='released',updated_at_ms=? WHERE lease_id=?
                """,
                (record.committed_at_ms, record.lease_id),
            )
            if lease_update.rowcount != 1:
                raise DiagnosticError(
                    "E_REPLAY",
                    "batch commit has no matching lease ownership row",
                    details={
                        "field": "lease_id",
                        "batch_id": record.batch_id,
                        "shard_id": record.shard_id,
                    },
                )
            checkpoint_payloads = record.checkpoint_state.get("checkpoints", {})
            for checkpoint_id, payload in checkpoint_payloads.items():
                state_digest = sha256_hex(payload)
                conn.execute(
                    """
                    INSERT INTO checkpoints(
                        checkpoint_id,shard_id,payload,state_digest,updated_at_ms
                    ) VALUES(?,?,?,?,?)
                    ON CONFLICT(checkpoint_id) DO UPDATE SET
                        shard_id=excluded.shard_id,payload=excluded.payload,
                        state_digest=excluded.state_digest,updated_at_ms=excluded.updated_at_ms
                    """,
                    (
                        checkpoint_id,
                        str(payload.get("shard_id", record.shard_id)),
                        canonical_json(payload),
                        state_digest,
                        record.committed_at_ms,
                    ),
                )
            self._refresh_digest(conn)
        return content_digest

    def mark_unknown(
        self,
        *,
        batch_id: str,
        shard_id: str,
        batch_order: int,
        code: str,
        message: str,
        now_ms: int,
    ) -> None:
        attempt_id = self.attempt_id(shard_id, batch_order, batch_id)
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT lease_id FROM attempts WHERE attempt_id=?", (attempt_id,)
            ).fetchone()
            conn.execute(
                """
                UPDATE attempts SET status='unknown',error_code=?,error_message=?,
                    updated_at_ms=? WHERE attempt_id=? AND status!='committed'
                """,
                (code, message, now_ms, attempt_id),
            )
            if row and row["lease_id"]:
                conn.execute(
                    "UPDATE leases SET state='reclaimed',updated_at_ms=? WHERE lease_id=?",
                    (now_ms, row["lease_id"]),
                )
            self._refresh_digest(conn)

    def record_checkpoint_state(
        self, state: Mapping[str, Any], *, now_ms: int
    ) -> None:
        checkpoints = state.get("checkpoints", {})
        with self._transaction() as conn:
            for checkpoint_id, payload in checkpoints.items():
                conn.execute(
                    """
                    INSERT INTO checkpoints(
                        checkpoint_id,shard_id,payload,state_digest,updated_at_ms
                    ) VALUES(?,?,?,?,?)
                    ON CONFLICT(checkpoint_id) DO UPDATE SET
                        shard_id=excluded.shard_id,payload=excluded.payload,
                        state_digest=excluded.state_digest,updated_at_ms=excluded.updated_at_ms
                    """,
                    (
                        checkpoint_id,
                        str(payload.get("shard_id", "")),
                        canonical_json(payload),
                        sha256_hex(payload),
                        now_ms,
                    ),
                )
            self._refresh_digest(conn)

    def ensure_checkpoint_state(
        self, state: Mapping[str, Any], *, now_ms: int
    ) -> None:
        """Persist declared checkpoints without overwriting committed state."""

        checkpoints = state.get("checkpoints", {})
        with self._transaction() as conn:
            for checkpoint_id, payload in checkpoints.items():
                conn.execute(
                    """
                    INSERT OR IGNORE INTO checkpoints(
                        checkpoint_id,shard_id,payload,state_digest,updated_at_ms
                    ) VALUES(?,?,?,?,?)
                    """,
                    (
                        checkpoint_id,
                        str(payload.get("shard_id", "")),
                        canonical_json(payload),
                        sha256_hex(payload),
                        now_ms,
                    ),
                )
            self._refresh_digest(conn)

    def record_event(
        self,
        *,
        seq: int,
        event_type: str,
        payload: Mapping[str, Any],
        state_digest: str,
        now_ms: int,
    ) -> None:
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO events(seq,event_type,payload,state_digest,applied_at_ms)
                VALUES(?,?,?,?,?)
                ON CONFLICT(seq) DO UPDATE SET event_type=excluded.event_type,
                    payload=excluded.payload,state_digest=excluded.state_digest,
                    applied_at_ms=excluded.applied_at_ms
                """,
                (seq, event_type, canonical_json(dict(payload)), state_digest, now_ms),
            )
            self._refresh_digest(conn)

    def save_runtime_state(self, state: Mapping[str, Any], *, now_ms: int) -> None:
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO runtime_state(key,value) VALUES('checkpoint_manager',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (canonical_json(dict(state)),),
            )
            conn.execute(
                "INSERT INTO runtime_state(key,value) VALUES('saved_at_ms',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(now_ms),),
            )
            self._refresh_digest(conn)

    def recover(self, *, now_ms: int) -> dict[str, Any]:
        with self._transaction() as conn:
            pending = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM attempts WHERE status='pending' ORDER BY shard_id,batch_order,batch_id"
                ).fetchall()
            ]
            for attempt in pending:
                conn.execute(
                    """
                    UPDATE attempts SET status='unknown',
                        error_code='E_RECOVERY', error_message=?,updated_at_ms=?
                    WHERE attempt_id=?
                    """,
                    (
                        "worker or master terminated before the batch transaction committed",
                        now_ms,
                        attempt["attempt_id"],
                    ),
                )
                if attempt["lease_id"]:
                    conn.execute(
                        "UPDATE leases SET state='reclaimed',updated_at_ms=? WHERE lease_id=?",
                        (now_ms, attempt["lease_id"]),
                    )
            self._refresh_digest(conn)
        committed = [
            row["batch_id"]
            for row in self.connection.execute(
                "SELECT batch_id FROM batches ORDER BY sequence"
            ).fetchall()
        ]
        return {
            "unknown_attempts": pending,
            "committed_batches": committed,
            "requires_rerun": [row["batch_id"] for row in pending],
            "reclaimed_leases": [row["lease_id"] for row in pending if row["lease_id"]],
        }

    def committed_records(self) -> tuple[BatchLedgerRecord, ...]:
        records: list[BatchLedgerRecord] = []
        for row in self.connection.execute(
            "SELECT * FROM batches ORDER BY batch_order,shard_id,batch_id"
        ).fetchall():
            records.append(
                BatchLedgerRecord(
                    sequence=int(row["sequence"]),
                    batch_id=row["batch_id"],
                    shard_id=row["shard_id"],
                    batch_order=int(row["batch_order"]),
                    parameter_version=int(row["parameter_version"]),
                    parameter_digest=row["parameter_digest"],
                    loss_value=row["loss_value"],
                    loss_digest=row["loss_digest"],
                    sample_ids=tuple(int(value) for value in json.loads(row["sample_ids"])),
                    lease_id=row["lease_id"],
                    lease_version=int(row["lease_version"]),
                    lease_owner=row["lease_owner"],
                    checkpoint_state=json.loads(row["checkpoint_state"]),
                    progress_cursor=int(row["progress_cursor"]),
                    committed_at_ms=int(row["committed_at_ms"]),
                    format_version=int(row["format_version"]),
                )
            )
        return tuple(records)

    def replay(self) -> dict[str, Any]:
        """Recompute every committed summary and stop at the first mismatch."""

        records = self.committed_records()
        batch_by_id = {batch.batch_id: batch for batch in self.session.batches}
        shard_by_id = {shard.shard_id: shard for shard in self.session.shards}
        checked = 0
        for record in records:
            stored_digest = self.connection.execute(
                "SELECT content_digest,checkpoint_digest FROM batches WHERE batch_id=?",
                (record.batch_id,),
            ).fetchone()
            if stored_digest is None:
                raise DiagnosticError(
                    "E_REPLAY",
                    "committed batch disappeared during replay",
                    details={"batch_id": record.batch_id, "shard_id": record.shard_id, "field": "batch"},
                )
            if stored_digest["content_digest"] != record.content_digest:
                self._replay_error(record, "content_digest", record.content_digest, stored_digest["content_digest"])
            if stored_digest["checkpoint_digest"] != sha256_hex(record.checkpoint_state):
                self._replay_error(
                    record,
                    "checkpoint_state",
                    sha256_hex(record.checkpoint_state),
                    stored_digest["checkpoint_digest"],
                )
            batch = batch_by_id.get(record.batch_id)
            if batch is None:
                self._replay_error(record, "batch_id", record.batch_id, "missing from session")
            if record.shard_id not in shard_by_id:
                self._replay_error(record, "shard_id", record.shard_id, "missing from session")
            if batch.shard_id != record.shard_id:
                self._replay_error(record, "shard_id", batch.shard_id, record.shard_id)
            if tuple(batch.sample_ids) != record.sample_ids:
                self._replay_error(record, "sample_ids", list(batch.sample_ids), list(record.sample_ids))
            shard = shard_by_id[record.shard_id]
            recomputed = compute_batch_summary(
                shard, record.batch_id, record.sample_ids, record.parameter_version
            )
            comparisons = (
                ("parameter_version", record.parameter_version, record.parameter_version),
                ("parameter_digest", recomputed["parameter_digest"], record.parameter_digest),
                ("loss", str(recomputed["loss"]), record.loss_value),
                ("loss_digest", recomputed["loss_digest"], record.loss_digest),
            )
            for field, expected, actual in comparisons:
                if expected != actual:
                    self._replay_error(record, field, expected, actual)
            checked += 1
        actual_digest = self._compute_content_digest()
        expected_digest = self.connection.execute(
            "SELECT content_digest FROM snapshot WHERE singleton=1"
        ).fetchone()["content_digest"]
        if expected_digest != actual_digest:
            raise DiagnosticError(
                "E_REPLAY",
                "ledger content digest differs from its committed rows",
                details={"field": "content_digest", "expected": actual_digest, "actual": expected_digest},
            )
        return {
            "checked_batches": checked,
            "content_digest": actual_digest,
            "consistent": True,
        }

    @staticmethod
    def _replay_error(
        record: BatchLedgerRecord, field: str, expected: Any, actual: Any
    ) -> None:
        raise DiagnosticError(
            "E_REPLAY",
            f"replay mismatch for batch {record.batch_id}, shard {record.shard_id}, field {field}",
            details={
                "batch_id": record.batch_id,
                "shard_id": record.shard_id,
                "field": field,
                "expected": expected,
                "actual": actual,
            },
        )

    def snapshot_info(self) -> dict[str, Any]:
        row = self.connection.execute("SELECT * FROM snapshot WHERE singleton=1").fetchone()
        return dict(row) if row is not None else {}
