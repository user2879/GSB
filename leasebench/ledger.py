"""SQLite ledger with batch transactions, digest chaining, and recovery."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .errors import SnapshotError
from .models import canonical_json, content_digest


LEDGER_FORMAT_VERSION = 1


@dataclass(frozen=True)
class RecoveryReport:
    committed_cursor: int
    last_batch_id: str | None
    pending_batch_ids: tuple[str, ...]
    unknown_lease_ids: tuple[str, ...]
    state: dict[str, Any] | None
    declaration_source: dict[str, Any] | None
    scale: int | None
    declaration_version: int | None
    status: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "committed_cursor": self.committed_cursor,
            "last_batch_id": self.last_batch_id,
            "pending_batch_ids": list(self.pending_batch_ids),
            "unknown_lease_ids": list(self.unknown_lease_ids),
            "declaration_version": self.declaration_version,
            "scale": self.scale,
            "status": self.status,
            "message": self.message,
        }


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = FULL")
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _create_schema(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS declarations (
                    version INTEGER PRIMARY KEY,
                    digest TEXT NOT NULL,
                    source_json TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 0,
                    applied_batch_id TEXT,
                    applied_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS batches (
                    sequence_no INTEGER PRIMARY KEY,
                    batch_id TEXT NOT NULL UNIQUE,
                    declaration_version INTEGER NOT NULL,
                    start_event_seq INTEGER NOT NULL,
                    end_event_seq INTEGER NOT NULL,
                    cursor_event_seq INTEGER NOT NULL,
                    arrival_ms INTEGER NOT NULL,
                    scale INTEGER NOT NULL,
                    result_json TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    content_digest TEXT NOT NULL,
                    prev_chain_digest TEXT NOT NULL,
                    chain_digest TEXT NOT NULL,
                    committed_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS pending_batches (
                    batch_id TEXT PRIMARY KEY,
                    declaration_version INTEGER NOT NULL,
                    start_event_seq INTEGER NOT NULL,
                    end_event_seq INTEGER NOT NULL,
                    candidate_lease_ids_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS unknown_leases (
                    batch_id TEXT NOT NULL,
                    lease_id TEXT NOT NULL,
                    detected_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    PRIMARY KEY (batch_id, lease_id)
                );

                CREATE TABLE IF NOT EXISTS batch_operations (
                    batch_id TEXT NOT NULL,
                    operation_seq INTEGER NOT NULL,
                    operation_type TEXT NOT NULL,
                    object_json TEXT NOT NULL,
                    PRIMARY KEY (batch_id, operation_seq),
                    FOREIGN KEY (batch_id) REFERENCES batches(batch_id)
                );

                CREATE TABLE IF NOT EXISTS watermarks (
                    batch_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    function_id TEXT,
                    used_concurrency INTEGER NOT NULL,
                    borrowed_concurrency INTEGER NOT NULL,
                    lent_concurrency INTEGER NOT NULL,
                    queue_length INTEGER NOT NULL,
                    capacity INTEGER NOT NULL,
                    ratio_fp INTEGER NOT NULL,
                    ratio_scale INTEGER NOT NULL,
                    single_value_error_fp INTEGER NOT NULL,
                    comparison_error_fp INTEGER NOT NULL,
                    borrow_return_error_fp INTEGER NOT NULL,
                    total_comparison_error_fp INTEGER NOT NULL,
                    dominant_error_segment TEXT NOT NULL,
                    comparison_affected_by_error INTEGER NOT NULL,
                    PRIMARY KEY (batch_id, scope, function_id),
                    FOREIGN KEY (batch_id) REFERENCES batches(batch_id)
                );

                CREATE TABLE IF NOT EXISTS invalidations (
                    new_version INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    lease_id TEXT,
                    event_seq INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    old_version INTEGER NOT NULL,
                    at_ms INTEGER NOT NULL,
                    batch_id TEXT NOT NULL,
                    PRIMARY KEY (new_version, kind, object_id),
                    FOREIGN KEY (batch_id) REFERENCES batches(batch_id)
                );
                """
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('ledger_format_version', ?)",
                (str(LEDGER_FORMAT_VERSION),),
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('cursor_event_seq', '0')"
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('chain_digest', '')"
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('last_batch_id', '')"
            )
        stored = self._meta("ledger_format_version")
        if stored != str(LEDGER_FORMAT_VERSION):
            raise SnapshotError(
                "ledger_format_version",
                "unsupported ledger format version",
                expected=LEDGER_FORMAT_VERSION,
                actual=stored,
            )

    def _meta(self, key: str) -> str:
        row = self.connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return "" if row is None else str(row["value"])

    def _set_meta(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    @property
    def current_cursor(self) -> int:
        return int(self._meta("cursor_event_seq") or 0)

    @property
    def last_batch_id(self) -> str | None:
        value = self._meta("last_batch_id")
        return value or None

    def begin_pending(
        self,
        *,
        batch_id: str,
        declaration_version: int,
        start_event_seq: int,
        end_event_seq: int,
        candidate_lease_ids: Iterable[str],
    ) -> None:
        now = _utc_now()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO pending_batches(
                    batch_id, declaration_version, start_event_seq,
                    end_event_seq, candidate_lease_ids_json, created_at
                ) VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(batch_id) DO UPDATE SET
                    declaration_version=excluded.declaration_version,
                    start_event_seq=excluded.start_event_seq,
                    end_event_seq=excluded.end_event_seq,
                    candidate_lease_ids_json=excluded.candidate_lease_ids_json,
                    created_at=excluded.created_at
                """,
                (
                    batch_id,
                    declaration_version,
                    start_event_seq,
                    end_event_seq,
                    canonical_json(sorted(set(candidate_lease_ids))),
                    now,
                ),
            )

    def append_batch(
        self,
        *,
        payload: dict[str, Any],
        state: dict[str, Any],
        declaration_source: dict[str, Any],
    ) -> str:
        self._verify_payload(payload, state)
        batch_id = str(payload["batch_id"])
        current_cursor = self.current_cursor
        expected_start = current_cursor + 1
        if int(payload["start_event_seq"]) != expected_start:
            raise SnapshotError(
                "start_event_seq",
                "batch does not continue from committed cursor",
                expected=expected_start,
                actual=payload["start_event_seq"],
                batch_id=batch_id,
            )
        if int(payload["cursor_event_seq"]) != int(payload["end_event_seq"]):
            raise SnapshotError(
                "cursor_event_seq",
                "batch cursor must equal its last event sequence",
                batch_id=batch_id,
            )
        current_declaration = self.connection.execute(
            "SELECT version, digest FROM declarations WHERE active = 1 ORDER BY version DESC LIMIT 1"
        ).fetchone()
        if current_declaration is not None:
            ledger_version = int(current_declaration["version"])
            requested_version = int(payload["declaration_version"])
            if requested_version < ledger_version:
                raise SnapshotError(
                    "declaration_version",
                    "batch declaration version regresses",
                    expected_min=ledger_version,
                    actual=requested_version,
                    batch_id=batch_id,
                )
            if (
                requested_version == ledger_version
                and str(current_declaration["digest"]) != str(payload["declaration_digest"])
            ):
                raise SnapshotError(
                    "declaration_digest",
                    "same declaration version has different content",
                    version=requested_version,
                    batch_id=batch_id,
                )
        result_json = canonical_json(payload)
        state_json = canonical_json(state)
        content = content_digest(payload)
        prev_chain = self._meta("chain_digest")
        chain = hashlib.sha256((prev_chain + content).encode("utf-8")).hexdigest()
        now = _utc_now()
        sequence_no = int(
            self.connection.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
        ) + 1

        with self.connection:
            existing = self.connection.execute(
                """
                INSERT INTO declarations(
                    version, digest, source_json, active, applied_batch_id, applied_at
                ) VALUES(?, ?, ?, 1, ?, ?)
                ON CONFLICT(version) DO UPDATE SET
                    digest=excluded.digest,
                    source_json=excluded.source_json,
                    active=1,
                    applied_batch_id=excluded.applied_batch_id,
                    applied_at=excluded.applied_at
                """,
                (
                    int(payload["declaration_version"]),
                    str(payload["declaration_digest"]),
                    canonical_json(declaration_source),
                    batch_id,
                    now,
                ),
            )
            self.connection.execute(
                "UPDATE declarations SET active = 0 WHERE version != ?",
                (int(payload["declaration_version"]),),
            )
            self.connection.execute(
                """
                INSERT INTO batches(
                    sequence_no, batch_id, declaration_version, start_event_seq,
                    end_event_seq, cursor_event_seq, arrival_ms, scale,
                    result_json, state_json, content_digest,
                    prev_chain_digest, chain_digest, committed_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sequence_no,
                    batch_id,
                    int(payload["declaration_version"]),
                    int(payload["start_event_seq"]),
                    int(payload["end_event_seq"]),
                    int(payload["cursor_event_seq"]),
                    int(payload["arrival_ms"]),
                    int(payload["scale"]),
                    result_json,
                    state_json,
                    content,
                    prev_chain,
                    chain,
                    now,
                ),
            )
            for operation_seq, operation in enumerate(payload["operations"], start=1):
                self.connection.execute(
                    """
                    INSERT INTO batch_operations(
                        batch_id, operation_seq, operation_type, object_json
                    ) VALUES(?, ?, ?, ?)
                    """,
                    (
                        batch_id,
                        operation_seq,
                        str(operation["type"]),
                        canonical_json(operation),
                    ),
                )
            for item in payload["watermarks"]:
                self.connection.execute(
                    """
                    INSERT INTO watermarks(
                        batch_id, scope, function_id, used_concurrency,
                        borrowed_concurrency, lent_concurrency, queue_length,
                        capacity, ratio_fp, ratio_scale, single_value_error_fp,
                        comparison_error_fp, borrow_return_error_fp,
                        total_comparison_error_fp, dominant_error_segment,
                        comparison_affected_by_error
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        batch_id,
                        item["scope"],
                        item.get("function_id"),
                        item["used_concurrency"],
                        item["borrowed_concurrency"],
                        item["lent_concurrency"],
                        item["queue_length"],
                        item["capacity"],
                        item["ratio_fp"],
                        item["ratio_scale"],
                        item["single_value_error_fp"],
                        item["comparison_error_fp"],
                        item["borrow_return_error_fp"],
                        item["total_comparison_error_fp"],
                        item["dominant_error_segment"],
                        1 if item["comparison_affected_by_error"] else 0,
                    ),
                )
            for item in state.get("invalidations", []):
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO invalidations(
                        new_version, kind, object_id, lease_id, event_seq,
                        reason, old_version, at_ms, batch_id
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item["new_version"],
                        item["kind"],
                        item["object_id"],
                        item.get("lease_id"),
                        item["event_seq"],
                        item["reason"],
                        item["old_version"],
                        item["at_ms"],
                        batch_id,
                    ),
                )
            self.connection.execute(
                "DELETE FROM pending_batches WHERE batch_id = ?", (batch_id,)
            )
            self.connection.execute(
                "DELETE FROM unknown_leases WHERE batch_id = ?", (batch_id,)
            )
            self._set_meta("cursor_event_seq", str(payload["cursor_event_seq"]))
            self._set_meta("last_batch_id", batch_id)
            self._set_meta("chain_digest", chain)
            self._set_meta(
                "current_declaration_version", str(payload["declaration_version"])
            )
        return batch_id

    def _verify_payload(self, payload: dict[str, Any], state: dict[str, Any]) -> None:
        if int(payload.get("format_version", -1)) != 1:
            raise SnapshotError(
                "format_version", "unsupported batch format", batch_id=payload.get("batch_id")
            )
        state_digest = content_digest(state)
        if state_digest != payload.get("state_digest"):
            raise SnapshotError(
                "state_digest",
                "batch state digest does not match payload",
                batch_id=payload.get("batch_id"),
                expected=payload.get("state_digest"),
                actual=state_digest,
            )

    def recover(self, *, record_unknowns: bool = True) -> RecoveryReport:
        self.verify_all()
        pending = list(
            self.connection.execute(
                "SELECT * FROM pending_batches ORDER BY created_at, batch_id"
            )
        )
        unknown: list[str] = []
        for row in pending:
            candidates = json.loads(row["candidate_lease_ids_json"])
            unknown.extend(str(item) for item in candidates)
        unknown = sorted(set(unknown))
        if record_unknowns and pending:
            now = _utc_now()
            with self.connection:
                for row in pending:
                    for lease_id in json.loads(row["candidate_lease_ids_json"]):
                        self.connection.execute(
                            """
                            INSERT OR IGNORE INTO unknown_leases(
                                batch_id, lease_id, detected_at, status
                            ) VALUES(?, ?, ?, 'UNKNOWN_RECOMPUTE_REQUIRED')
                            """,
                            (row["batch_id"], lease_id, now),
                        )
        latest = self.connection.execute(
            "SELECT * FROM batches ORDER BY sequence_no DESC LIMIT 1"
        ).fetchone()
        if latest is None:
            status = "PENDING_RECOMPUTE" if pending else "EMPTY"
            message = (
                "No complete event batch is committed; pending leases are unknown and must be recomputed."
                if pending
                else "Ledger is empty."
            )
            return RecoveryReport(
                committed_cursor=0,
                last_batch_id=None,
                pending_batch_ids=tuple(sorted(row["batch_id"] for row in pending)),
                unknown_lease_ids=tuple(unknown),
                state=None,
                declaration_source=None,
                scale=None,
                declaration_version=None,
                status=status,
                message=message,
            )
        state = json.loads(latest["state_json"])
        payload = json.loads(latest["result_json"])
        declaration_row = self.connection.execute(
            "SELECT * FROM declarations WHERE version = ?",
            (latest["declaration_version"],),
        ).fetchone()
        if declaration_row is None:
            raise SnapshotError(
                "declarations",
                "latest batch references a missing declaration",
                batch_id=latest["batch_id"],
            )
        status = "PENDING_RECOMPUTE" if pending else "CLEAN"
        message = (
            "Committed batches recovered; uncommitted pending lease conclusions are unknown and require recomputation."
            if pending
            else "Only complete committed batches were loaded."
        )
        return RecoveryReport(
            committed_cursor=int(latest["cursor_event_seq"]),
            last_batch_id=str(latest["batch_id"]),
            pending_batch_ids=tuple(sorted(row["batch_id"] for row in pending)),
            unknown_lease_ids=tuple(unknown),
            state=state,
            declaration_source=json.loads(declaration_row["source_json"]),
            scale=int(latest["scale"]),
            declaration_version=int(latest["declaration_version"]),
            status=status,
            message=message,
        )

    def verify_all(self) -> None:
        rows = list(self.connection.execute("SELECT * FROM batches ORDER BY sequence_no"))
        previous_chain = ""
        previous_cursor = 0
        previous_version = 0
        for row in rows:
            try:
                payload = json.loads(row["result_json"])
                state = json.loads(row["state_json"])
            except json.JSONDecodeError as exc:
                raise SnapshotError(
                    "result_json",
                    f"invalid JSON in ledger batch: {exc.msg}",
                    batch_id=row["batch_id"],
                ) from exc
            actual_content = content_digest(payload)
            if actual_content != row["content_digest"]:
                raise SnapshotError(
                    "content_digest",
                    "ledger batch content digest mismatch",
                    batch_id=row["batch_id"],
                    expected=row["content_digest"],
                    actual=actual_content,
                )
            actual_state = content_digest(state)
            if actual_state != payload.get("state_digest"):
                raise SnapshotError(
                    "state_digest",
                    "ledger state digest mismatch",
                    batch_id=row["batch_id"],
                    expected=payload.get("state_digest"),
                    actual=actual_state,
                )
            expected_chain = hashlib.sha256(
                (previous_chain + actual_content).encode("utf-8")
            ).hexdigest()
            if expected_chain != row["chain_digest"] or row["prev_chain_digest"] != previous_chain:
                raise SnapshotError(
                    "chain_digest",
                    "ledger hash chain mismatch",
                    batch_id=row["batch_id"],
                    expected=expected_chain,
                    actual=row["chain_digest"],
                )
            if int(row["start_event_seq"]) != previous_cursor + 1:
                raise SnapshotError(
                    "start_event_seq",
                    "ledger batch sequence is not contiguous",
                    batch_id=row["batch_id"],
                    expected=previous_cursor + 1,
                    actual=row["start_event_seq"],
                )
            if int(row["declaration_version"]) < previous_version:
                raise SnapshotError(
                    "declaration_version",
                    "declaration version regressed in ledger",
                    batch_id=row["batch_id"],
                )
            previous_chain = row["chain_digest"]
            previous_cursor = int(row["cursor_event_seq"])
            previous_version = int(row["declaration_version"])
        if rows:
            if int(self._meta("cursor_event_seq") or -1) != previous_cursor:
                raise SnapshotError(
                    "meta.cursor_event_seq",
                    "ledger cursor metadata does not match last batch",
                )
            if self._meta("chain_digest") != previous_chain:
                raise SnapshotError(
                    "meta.chain_digest",
                    "ledger chain metadata does not match last batch",
                )
        from .validation import parse_declaration

        for row in self.connection.execute(
            "SELECT version, digest, source_json FROM declarations ORDER BY version"
        ):
            try:
                source = json.loads(row["source_json"])
                declaration = parse_declaration(source)
            except Exception as exc:
                raise SnapshotError(
                    "declarations.source_json",
                    f"stored declaration cannot be validated: {type(exc).__name__}",
                    version=row["version"],
                ) from exc
            if declaration.digest != row["digest"]:
                raise SnapshotError(
                    "declarations.digest",
                    "stored declaration digest mismatch",
                    version=row["version"],
                    expected=row["digest"],
                    actual=declaration.digest,
                )

    def load_batches(self) -> list[dict[str, Any]]:
        self.verify_all()
        result = []
        for row in self.connection.execute("SELECT * FROM batches ORDER BY sequence_no"):
            result.append(
                {
                    "sequence_no": row["sequence_no"],
                    "batch_id": row["batch_id"],
                    "declaration_version": row["declaration_version"],
                    "scale": row["scale"],
                    "cursor_event_seq": row["cursor_event_seq"],
                    "payload": json.loads(row["result_json"]),
                    "state": json.loads(row["state_json"]),
                }
            )
        return result

    def declaration_sources(self) -> dict[int, dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT version, source_json FROM declarations ORDER BY version"
        )
        return {int(row["version"]): json.loads(row["source_json"]) for row in rows}

    def load_view_rows(self) -> dict[str, list[dict[str, Any]]]:
        self.verify_all()
        latest = self.last_batch_id
        if latest is None:
            return {"watermarks": [], "invalidations": [], "unknown_leases": []}
        watermarks = [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM watermarks WHERE batch_id = ? ORDER BY scope, function_id",
                (latest,),
            )
        ]
        invalidations = [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM invalidations ORDER BY new_version, kind, object_id"
            )
        ]
        unknown = [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM unknown_leases ORDER BY batch_id, lease_id"
            )
        ]
        return {
            "watermarks": watermarks,
            "invalidations": invalidations,
            "unknown_leases": unknown,
        }

    def tamper_content_digest(self) -> str:
        row = self.connection.execute(
            "SELECT batch_id FROM batches ORDER BY sequence_no LIMIT 1"
        ).fetchone()
        if row is None:
            raise SnapshotError("batches", "no batch available for snapshot injection")
        batch_id = str(row["batch_id"])
        with self.connection:
            self.connection.execute(
                "UPDATE batches SET content_digest = 'injected-corruption' WHERE batch_id = ?",
                (batch_id,),
            )
        return batch_id


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
