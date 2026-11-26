from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .canonical import canonical_json, digest_json
from .errors import GSBError

LEDGER_FORMAT_VERSION = 3
_HEX = set("0123456789abcdef")


class LedgerError(GSBError):
    def __init__(self, message: str, *, field: str | None = None, details: dict[str, Any] | None = None) -> None:
        super().__init__("E_LEDGER", message, field=field, details=details)


class LedgerConflict(LedgerError):
    def __init__(self, message: str, *, field: str | None = None, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, field=field, details=details)
        self.code = "E_CONFLICT"


@dataclass(frozen=True)
class BatchCommit:
    batch_id: str
    plan_version: int
    plan_id: str
    executor_id: str
    shard_id: str
    owner: str
    receipt_digest: str
    cost_digest: str
    invalidation_trace: list[dict[str, Any]]
    progress_cursor: int
    format_version: int = LEDGER_FORMAT_VERSION
    committed_mono_ns: int = field(default_factory=time.monotonic_ns)

    def content_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "plan_version": self.plan_version,
            "plan_id": self.plan_id,
            "executor_id": self.executor_id,
            "shard_id": self.shard_id,
            "owner": self.owner,
            "receipt_digest": self.receipt_digest,
            "cost_digest": self.cost_digest,
            "invalidation_trace": self.invalidation_trace,
            "progress_cursor": self.progress_cursor,
            "format_version": self.format_version,
            "committed_mono_ns": self.committed_mono_ns,
        }

    @property
    def content_digest(self) -> str:
        return digest_json(self.content_dict())


@dataclass(frozen=True)
class SnapshotResult:
    replaced: bool
    reason: str
    version: int
    content_digest: str
    previous_version: int | None
    previous_digest: str | None
    generation: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "replaced": self.replaced,
            "reason": self.reason,
            "version": self.version,
            "content_digest": self.content_digest,
            "previous_version": self.previous_version,
            "previous_digest": self.previous_digest,
            "generation": self.generation,
        }


@dataclass(frozen=True)
class RecoveryState:
    committed_batches: tuple[dict[str, Any], ...]
    unknown_batches: tuple[dict[str, Any], ...]
    progress_cursor: int
    plan_version: int
    snapshot: dict[str, Any] | None
    verified_content_digests: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "committed_batches": list(self.committed_batches),
            "unknown_batches": list(self.unknown_batches),
            "progress_cursor": self.progress_cursor,
            "plan_version": self.plan_version,
            "snapshot": self.snapshot,
            "verified_content_digests": list(self.verified_content_digests),
        }


def _is_digest(value: str) -> bool:
    return len(value) == 64 and all(character in _HEX for character in value)


def snapshot_digest(version: int, format_version: int, payload: Mapping[str, Any]) -> str:
    return digest_json({"version": version, "format_version": format_version, "payload": payload})


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA busy_timeout=10000")
        self._initialize()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
                CREATE TABLE IF NOT EXISTS ledger_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS session_snapshot (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    format_version INTEGER NOT NULL,
                    version INTEGER NOT NULL,
                    content_digest TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    generation INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS batch_intents (
                    batch_id TEXT PRIMARY KEY,
                    plan_version INTEGER NOT NULL,
                    plan_id TEXT NOT NULL,
                    executor_id TEXT NOT NULL,
                    shard_id TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    created_mono_ns INTEGER NOT NULL,
                    status TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS committed_batches (
                    batch_id TEXT PRIMARY KEY,
                    format_version INTEGER NOT NULL,
                    plan_version INTEGER NOT NULL,
                    plan_id TEXT NOT NULL,
                    executor_id TEXT NOT NULL,
                    shard_id TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    receipt_digest TEXT NOT NULL,
                    cost_digest TEXT NOT NULL,
                    invalidation_trace_json TEXT NOT NULL,
                    progress_cursor INTEGER NOT NULL,
                    content_digest TEXT NOT NULL,
                    committed_mono_ns INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS recovered_unknown_batches (
                    batch_id TEXT PRIMARY KEY,
                    format_version INTEGER NOT NULL,
                    intent_json TEXT NOT NULL,
                    marked_mono_ns INTEGER NOT NULL
                );
                """
        )
        self.connection.execute(
            "INSERT OR IGNORE INTO ledger_meta(key,value) VALUES('format_version',?)",
            (str(LEDGER_FORMAT_VERSION),),
        )
        self.connection.commit()

    def begin_batch_intent(
        self,
        *,
        batch_id: str,
        plan_version: int,
        plan_id: str,
        executor_id: str,
        shard_id: str,
        owner: str,
    ) -> dict[str, Any]:
        for field_name, value in (
            ("batch_id", batch_id),
            ("plan_id", plan_id),
            ("executor_id", executor_id),
            ("shard_id", shard_id),
            ("owner", owner),
        ):
            if not isinstance(value, str) or not value:
                raise LedgerError("intent field is empty", field=field_name)
        if plan_version < 0:
            raise LedgerError("plan_version must be non-negative", field="plan_version")
        intent = {
            "batch_id": batch_id,
            "plan_version": plan_version,
            "plan_id": plan_id,
            "executor_id": executor_id,
            "shard_id": shard_id,
            "owner": owner,
            "created_mono_ns": time.monotonic_ns(),
        }
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            existing = self.connection.execute(
                    "SELECT status FROM batch_intents WHERE batch_id=?", (batch_id,)
                ).fetchone()
            if existing is not None and existing["status"] == "committed":
                self.connection.execute("ROLLBACK")
                raise LedgerConflict("batch already has a committed intent", field="batch_id", details={"batch_id": batch_id})
            self.connection.execute(
                    """
                    INSERT INTO batch_intents(
                        batch_id,plan_version,plan_id,executor_id,shard_id,owner,created_mono_ns,status
                    ) VALUES(?,?,?,?,?,?,?,?)
                    ON CONFLICT(batch_id) DO UPDATE SET
                        plan_version=excluded.plan_version,
                        plan_id=excluded.plan_id,
                        executor_id=excluded.executor_id,
                        shard_id=excluded.shard_id,
                        owner=excluded.owner,
                        created_mono_ns=excluded.created_mono_ns,
                        status='pending'
                    """,
                    (
                        intent["batch_id"],
                        intent["plan_version"],
                        intent["plan_id"],
                        intent["executor_id"],
                        intent["shard_id"],
                        intent["owner"],
                        intent["created_mono_ns"],
                        "pending",
                    ),
                )
            self.connection.execute("COMMIT")
        except sqlite3.IntegrityError as exc:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise LedgerConflict(str(exc), field="batch_id") from exc
        return intent

    def commit_batch(self, commit: BatchCommit) -> tuple[dict[str, Any], bool]:
        if commit.format_version != LEDGER_FORMAT_VERSION:
            raise LedgerError(
                "unsupported ledger format version",
                field="format_version",
                details={"expected": LEDGER_FORMAT_VERSION, "actual": commit.format_version},
            )
        if not _is_digest(commit.receipt_digest):
            raise LedgerError("receipt_digest must be a 64-character hexadecimal digest", field="receipt_digest")
        if not _is_digest(commit.cost_digest):
            raise LedgerError("cost_digest must be a 64-character hexadecimal digest", field="cost_digest")
        if commit.progress_cursor < 0:
            raise LedgerError("progress_cursor must be non-negative", field="progress_cursor")
        trace_json = canonical_json(commit.invalidation_trace)
        content_digest = commit.content_digest
        record = {
            **commit.content_dict(),
            "invalidation_trace": commit.invalidation_trace,
            "content_digest": content_digest,
        }
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            existing = self.connection.execute(
                "SELECT content_digest FROM committed_batches WHERE batch_id=?",
                (commit.batch_id,),
            ).fetchone()
            if existing is not None:
                if existing["content_digest"] == content_digest:
                    self.connection.execute("ROLLBACK")
                    return record, False
                self.connection.execute("ROLLBACK")
                raise LedgerConflict(
                    "batch_id is already committed with different content",
                    field="batch_id",
                    details={"batch_id": commit.batch_id},
                )
            intent = self.connection.execute(
                "SELECT * FROM batch_intents WHERE batch_id=?",
                (commit.batch_id,),
            ).fetchone()
            if intent is None:
                self.connection.execute(
                    "INSERT INTO batch_intents(batch_id,plan_version,plan_id,executor_id,shard_id,owner,created_mono_ns,status) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        commit.batch_id,
                        commit.plan_version,
                        commit.plan_id,
                        commit.executor_id,
                        commit.shard_id,
                        commit.owner,
                        commit.committed_mono_ns,
                        "pending",
                    ),
                )
            self.connection.execute(
                """
                INSERT INTO committed_batches(
                    batch_id,format_version,plan_version,plan_id,executor_id,shard_id,owner,
                    receipt_digest,cost_digest,invalidation_trace_json,progress_cursor,
                    content_digest,committed_mono_ns
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    commit.batch_id,
                    commit.format_version,
                    commit.plan_version,
                    commit.plan_id,
                    commit.executor_id,
                    commit.shard_id,
                    commit.owner,
                    commit.receipt_digest,
                    commit.cost_digest,
                    trace_json,
                    commit.progress_cursor,
                    content_digest,
                    commit.committed_mono_ns,
                ),
            )
            self.connection.execute(
                "UPDATE batch_intents SET status='committed' WHERE batch_id=?",
                (commit.batch_id,),
            )
            self.connection.execute("COMMIT")
        except Exception:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise
        return record, True

    def ensure_snapshot(
        self,
        *,
        version: int,
        format_version: int,
        payload: Mapping[str, Any],
    ) -> SnapshotResult:
        if format_version != LEDGER_FORMAT_VERSION:
            raise LedgerError("unsupported snapshot format version", field="format_version")
        external_digest = snapshot_digest(version, format_version, payload)
        payload_json = canonical_json(payload)
        row = self.connection.execute("SELECT * FROM session_snapshot WHERE singleton=1").fetchone()
        reason = "insert_initial"
        replaced = True
        previous_version: int | None = None
        previous_digest: str | None = None
        generation = 0
        if row is not None:
            previous_version = int(row["version"])
            previous_digest = str(row["content_digest"])
            generation = int(row["generation"])
            internal_digest = None
            try:
                stored_payload = json.loads(row["payload_json"])
                internal_digest = snapshot_digest(
                    int(row["version"]),
                    int(row["format_version"]),
                    stored_payload,
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                internal_digest = None
            if internal_digest != previous_digest:
                reason = "stored_snapshot_digest_mismatch"
            elif previous_version != version:
                reason = "snapshot_version_mismatch"
            elif previous_digest != external_digest:
                reason = "snapshot_content_mismatch"
            else:
                reason = "snapshot_unchanged"
                replaced = False
        if replaced:
            generation += 1
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                self.connection.execute("DELETE FROM session_snapshot WHERE singleton=1")
                self.connection.execute(
                    """
                    INSERT INTO session_snapshot(
                        singleton,format_version,version,content_digest,payload_json,generation
                    ) VALUES(1,?,?,?,?,?)
                    """,
                    (format_version, version, external_digest, payload_json, generation),
                )
                self.connection.execute("COMMIT")
            except Exception:
                if self.connection.in_transaction:
                    self.connection.execute("ROLLBACK")
                raise
        return SnapshotResult(
            replaced=replaced,
            reason=reason,
            version=version,
            content_digest=external_digest,
            previous_version=previous_version,
            previous_digest=previous_digest,
            generation=generation,
        )

    def recover(self) -> RecoveryState:
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            pending = self.connection.execute(
                "SELECT * FROM batch_intents WHERE status='pending' ORDER BY batch_id"
            ).fetchall()
            marked_ns = time.monotonic_ns()
            for row in pending:
                intent_payload = {
                    "batch_id": row["batch_id"],
                    "plan_version": int(row["plan_version"]),
                    "plan_id": row["plan_id"],
                    "executor_id": row["executor_id"],
                    "shard_id": row["shard_id"],
                    "owner": row["owner"],
                    "created_mono_ns": int(row["created_mono_ns"]),
                }
                self.connection.execute(
                    """
                    INSERT INTO recovered_unknown_batches(batch_id,format_version,intent_json,marked_mono_ns)
                    VALUES(?,?,?,?)
                    ON CONFLICT(batch_id) DO UPDATE SET
                        intent_json=excluded.intent_json,
                        marked_mono_ns=excluded.marked_mono_ns
                    """,
                    (row["batch_id"], LEDGER_FORMAT_VERSION, canonical_json(intent_payload), marked_ns),
                )
                self.connection.execute(
                    "UPDATE batch_intents SET status='unknown' WHERE batch_id=?",
                    (row["batch_id"],),
                )
            self.connection.execute("COMMIT")

        committed_rows = self.connection.execute(
            "SELECT * FROM committed_batches ORDER BY progress_cursor,batch_id"
        ).fetchall()
        committed: list[dict[str, Any]] = []
        verified: list[str] = []
        for row in committed_rows:
            trace = json.loads(row["invalidation_trace_json"])
            record = {
                "batch_id": row["batch_id"],
                "plan_version": int(row["plan_version"]),
                "plan_id": row["plan_id"],
                "executor_id": row["executor_id"],
                "shard_id": row["shard_id"],
                "owner": row["owner"],
                "receipt_digest": row["receipt_digest"],
                "cost_digest": row["cost_digest"],
                "invalidation_trace": trace,
                "progress_cursor": int(row["progress_cursor"]),
                "format_version": int(row["format_version"]),
                "committed_mono_ns": int(row["committed_mono_ns"]),
                "content_digest": row["content_digest"],
            }
            content = {key: value for key, value in record.items() if key != "content_digest"}
            if digest_json(content) != record["content_digest"]:
                raise LedgerError(
                    "committed batch content digest mismatch",
                    field="content_digest",
                    details={"batch_id": record["batch_id"]},
                )
            verified.append(record["content_digest"])
            committed.append(record)

        unknown_rows = self.connection.execute(
            "SELECT * FROM recovered_unknown_batches ORDER BY batch_id"
        ).fetchall()
        unknown = [
            {
                "batch_id": row["batch_id"],
                "format_version": int(row["format_version"]),
                "intent": json.loads(row["intent_json"]),
                "marked_mono_ns": int(row["marked_mono_ns"]),
                "status": "unknown",
            }
            for row in unknown_rows
        ]
        progress_cursor = max((record["progress_cursor"] for record in committed), default=0)
        plan_version = max((record["plan_version"] for record in committed), default=0)
        snapshot = self.get_snapshot()
        return RecoveryState(
            committed_batches=tuple(committed),
            unknown_batches=tuple(unknown),
            progress_cursor=progress_cursor,
            plan_version=plan_version,
            snapshot=snapshot,
            verified_content_digests=tuple(verified),
        )

    def get_snapshot(self) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM session_snapshot WHERE singleton=1").fetchone()
        if row is None:
            return None
        return {
            "format_version": int(row["format_version"]),
            "version": int(row["version"]),
            "content_digest": row["content_digest"],
            "generation": int(row["generation"]),
            "payload": json.loads(row["payload_json"]),
        }

    def committed_batches(self) -> tuple[dict[str, Any], ...]:
        rows = self.connection.execute(
            "SELECT * FROM committed_batches ORDER BY progress_cursor,batch_id"
        ).fetchall()
        return tuple(
            {
                "batch_id": row["batch_id"],
                "plan_version": int(row["plan_version"]),
                "plan_id": row["plan_id"],
                "executor_id": row["executor_id"],
                "shard_id": row["shard_id"],
                "owner": row["owner"],
                "receipt_digest": row["receipt_digest"],
                "cost_digest": row["cost_digest"],
                "invalidation_trace": json.loads(row["invalidation_trace_json"]),
                "progress_cursor": int(row["progress_cursor"]),
                "format_version": int(row["format_version"]),
                "committed_mono_ns": int(row["committed_mono_ns"]),
                "content_digest": row["content_digest"],
            }
            for row in rows
        )
