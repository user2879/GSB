from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .canonical import canonical_json, content_digest, json_loads_canonical
from .data import sqlite_uri
from .errors import QPlanError


LEDGER_SCHEMA_VERSION = "qplan-ledger-v1"


def compute_snapshot_content_digest(
    spec: dict[str, Any], stats: dict[str, Any], weights: dict[str, int]
) -> str:
    return content_digest({"spec": spec, "stats": stats, "weights": weights})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class Ledger:
    def __init__(self, path: str | Path, *, readonly: bool = False):
        self.path = Path(path)
        self.readonly = readonly
        if not readonly:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
            self.connection.execute("PRAGMA busy_timeout=30000")
            self._initialize()
        else:
            if not self.path.is_file():
                raise QPlanError("E_LEDGER", f"ledger does not exist: {self.path}", str(self.path))
            self.connection = sqlite3.connect(sqlite_uri(self.path), uri=True, timeout=30.0)
            self.connection.execute("PRAGMA busy_timeout=30000")

    @classmethod
    def read_only(cls, path: str | Path) -> "Ledger":
        return cls(path, readonly=True)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        if self.readonly:
            raise QPlanError("E_LEDGER", "ledger is read-only", str(self.path))
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield self.connection
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def _initialize(self) -> None:
        with self.transaction() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS qplan_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            existing = connection.execute(
                "SELECT value FROM qplan_meta WHERE key='ledger_schema_version'"
            ).fetchone()
            if existing is not None and existing[0] != LEDGER_SCHEMA_VERSION:
                raise QPlanError(
                    "E_LEDGER",
                    f"unsupported ledger schema {existing[0]!r}; expected {LEDGER_SCHEMA_VERSION}",
                    str(self.path),
                )
            connection.execute(
                "INSERT OR IGNORE INTO qplan_meta(key,value) VALUES('ledger_schema_version',?)",
                (LEDGER_SCHEMA_VERSION,),
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_version INTEGER PRIMARY KEY,
                    plan_version INTEGER NOT NULL,
                    content_digest TEXT NOT NULL,
                    spec_digest TEXT NOT NULL,
                    stats_digest TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    stats_json TEXT NOT NULL,
                    weights_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    invalidation_json TEXT NOT NULL,
                    replacement_reason TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL UNIQUE,
                    snapshot_version INTEGER NOT NULL,
                    plan_version INTEGER NOT NULL,
                    spec_digest TEXT NOT NULL,
                    snapshot_digest TEXT NOT NULL,
                    stats_digest TEXT NOT NULL,
                    expected_batches INTEGER NOT NULL,
                    batch_plan_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    unknown_batches_json TEXT,
                    recovery_json TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS batches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    snapshot_version INTEGER NOT NULL,
                    plan_version INTEGER NOT NULL,
                    batch_no INTEGER NOT NULL,
                    content_digest TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    committed_at TEXT NOT NULL,
                    UNIQUE(session_id, batch_no)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_batches_session ON batches(session_id, batch_no)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_snapshot ON sessions(snapshot_version, id)"
            )

    def _snapshot_row(self, row: tuple[Any, ...] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "snapshot_version": int(row[0]),
            "plan_version": int(row[1]),
            "content_digest": row[2],
            "spec_digest": row[3],
            "stats_digest": row[4],
            "spec": json_loads_canonical(row[5]),
            "stats": json_loads_canonical(row[6]),
            "weights": json_loads_canonical(row[7]),
            "status": row[8],
            "invalidation": json_loads_canonical(row[9]),
            "replacement_reason": row[10],
            "created_at": row[11],
        }

    def latest_snapshot(self) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT snapshot_version, plan_version, content_digest, spec_digest, stats_digest,
                   spec_json, stats_json, weights_json, status, invalidation_json,
                   replacement_reason, created_at
            FROM snapshots ORDER BY snapshot_version DESC LIMIT 1
            """
        ).fetchone()
        return self._snapshot_row(row)

    def current_snapshot(self) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT snapshot_version, plan_version, content_digest, spec_digest, stats_digest,
                   spec_json, stats_json, weights_json, status, invalidation_json,
                   replacement_reason, created_at
            FROM snapshots WHERE status='current' ORDER BY snapshot_version DESC LIMIT 1
            """
        ).fetchone()
        return self._snapshot_row(row)

    def snapshot(self, version: int) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT snapshot_version, plan_version, content_digest, spec_digest, stats_digest,
                   spec_json, stats_json, weights_json, status, invalidation_json,
                   replacement_reason, created_at
            FROM snapshots WHERE snapshot_version=?
            """,
            (int(version),),
        ).fetchone()
        return self._snapshot_row(row)

    def _latest_session(self, snapshot_version: int) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT id, session_id, snapshot_version, plan_version, spec_digest, snapshot_digest,
                   stats_digest, expected_batches, batch_plan_json, status, started_at, ended_at,
                   unknown_batches_json, recovery_json
            FROM sessions WHERE snapshot_version=? ORDER BY id DESC LIMIT 1
            """,
            (int(snapshot_version),),
        ).fetchone()
        if row is None:
            return None
        return {
            "row_id": int(row[0]),
            "session_id": row[1],
            "snapshot_version": int(row[2]),
            "plan_version": int(row[3]),
            "spec_digest": row[4],
            "snapshot_digest": row[5],
            "stats_digest": row[6],
            "expected_batches": int(row[7]),
            "batch_plan": json_loads_canonical(row[8]),
            "status": row[9],
            "started_at": row[10],
            "ended_at": row[11],
            "unknown_batches": json_loads_canonical(row[12]) if row[12] else None,
            "recovery": json_loads_canonical(row[13]) if row[13] else None,
        }

    def latest_session(self, snapshot_version: int) -> dict[str, Any] | None:
        return self._latest_session(snapshot_version)

    def prepare_snapshot(
        self,
        *,
        spec: dict[str, Any],
        spec_digest: str,
        stats: dict[str, Any],
        stats_digest: str,
        weights: dict[str, int],
        content_digest_value: str,
        invalidation: dict[str, Any],
        expected_snapshot_version: int | None = None,
        expected_snapshot_digest: str | None = None,
        force_new_version: bool = False,
        force_new_reason: str | None = None,
    ) -> dict[str, Any]:
        latest = self.latest_snapshot()
        expected_mismatch = False
        if expected_snapshot_version is not None:
            expected_mismatch = latest is None or int(expected_snapshot_version) != latest["snapshot_version"]
        if expected_snapshot_digest is not None:
            expected_mismatch = expected_mismatch or latest is None or expected_snapshot_digest != latest["content_digest"]
        if latest is not None and latest["status"] == "current" and latest["content_digest"] == content_digest_value:
            if not force_new_version and not expected_mismatch:
                return {"snapshot": latest, "reused": True, "replaced": False, "replacement_reason": None}

        spec_json = canonical_json(spec)
        stats_json = canonical_json(stats)
        weights_json = canonical_json(weights)
        invalidation_json = canonical_json(invalidation)
        replacement_reason = None
        if expected_mismatch:
            replacement_reason = "expected_snapshot_version_or_digest_mismatch"
        elif force_new_reason is not None:
            replacement_reason = force_new_reason
        elif force_new_version:
            replacement_reason = "forced_new_version"
        elif latest is not None and latest["content_digest"] != content_digest_value:
            replacement_reason = "content_digest_changed"
        elif latest is not None:
            replacement_reason = "new_conclusion_view"

        with self.transaction() as connection:
            max_snapshot = int(connection.execute("SELECT COALESCE(MAX(snapshot_version),0) FROM snapshots").fetchone()[0])
            max_plan = int(connection.execute("SELECT COALESCE(MAX(plan_version),0) FROM snapshots").fetchone()[0])
            new_version = max(max_snapshot, max_plan) + 1
            connection.execute("UPDATE snapshots SET status='stale' WHERE status='current'")
            connection.execute(
                "UPDATE batches SET status='stale' WHERE status='committed' AND snapshot_version IN (SELECT snapshot_version FROM snapshots WHERE status='stale')"
            )
            connection.execute("UPDATE sessions SET status='stale' WHERE status IN ('running','completed','recovered_complete','recovered_partial')")
            connection.execute(
                """
                INSERT INTO snapshots(
                    snapshot_version, plan_version, content_digest, spec_digest, stats_digest,
                    spec_json, stats_json, weights_json, status, invalidation_json,
                    replacement_reason, created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    new_version,
                    new_version,
                    content_digest_value,
                    spec_digest,
                    stats_digest,
                    spec_json,
                    stats_json,
                    weights_json,
                    "current",
                    invalidation_json,
                    replacement_reason,
                    _utc_now(),
                ),
            )
        snapshot = self.snapshot(new_version)
        assert snapshot is not None
        return {
            "snapshot": snapshot,
            "reused": False,
            "replaced": latest is not None,
            "replacement_reason": replacement_reason,
            "previous_snapshot_version": latest["snapshot_version"] if latest else None,
        }

    def begin_session(
        self,
        *,
        snapshot: dict[str, Any],
        batch_plan: list[list[str]],
    ) -> dict[str, Any]:
        if not batch_plan or any(not batch for batch in batch_plan):
            raise QPlanError("E_LEDGER", "batch plan must contain non-empty query batches", "batch_plan")
        session_id = uuid.uuid4().hex
        started_at = _utc_now()
        with self.transaction() as connection:
            current = connection.execute(
                "SELECT status FROM snapshots WHERE snapshot_version=?", (snapshot["snapshot_version"],)
            ).fetchone()
            if current is None or current[0] != "current":
                raise QPlanError("E_LEDGER", "cannot start a session on a non-current snapshot", str(snapshot["snapshot_version"]))
            connection.execute(
                "UPDATE sessions SET status='superseded' WHERE snapshot_version=? AND status IN ('completed','recovered_complete','recovered_partial','running')",
                (snapshot["snapshot_version"],),
            )
            connection.execute(
                "UPDATE batches SET status='superseded' WHERE snapshot_version=? AND status='committed'",
                (snapshot["snapshot_version"],),
            )
            connection.execute(
                """
                INSERT INTO sessions(
                    session_id, snapshot_version, plan_version, spec_digest, snapshot_digest,
                    stats_digest, expected_batches, batch_plan_json, status, started_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    session_id,
                    snapshot["snapshot_version"],
                    snapshot["plan_version"],
                    snapshot["spec_digest"],
                    snapshot["content_digest"],
                    snapshot["stats_digest"],
                    len(batch_plan),
                    canonical_json(batch_plan),
                    "running",
                    started_at,
                ),
            )
        return {
            "session_id": session_id,
            "snapshot_version": snapshot["snapshot_version"],
            "plan_version": snapshot["plan_version"],
            "expected_batches": len(batch_plan),
            "batch_plan": batch_plan,
            "status": "running",
            "started_at": started_at,
        }

    def commit_batch(
        self,
        *,
        session_id: str,
        batch_no: int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        session_row = self.connection.execute(
            "SELECT snapshot_version, plan_version, expected_batches, batch_plan_json, status FROM sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
        if session_row is None:
            raise QPlanError("E_LEDGER", f"unknown session {session_id}", session_id)
        snapshot_version, plan_version, expected_batches, batch_plan_json, status = session_row
        if status != "running":
            raise QPlanError("E_LEDGER", f"session is not running (status={status})", session_id)
        if batch_no < 1 or batch_no > int(expected_batches):
            raise QPlanError("E_LEDGER", "batch number is outside the committed session manifest", session_id)
        batch_plan = json_loads_canonical(batch_plan_json)
        expected_queries = list(batch_plan[batch_no - 1])
        actual_queries = [item.get("query_id") for item in payload.get("queries", [])]
        if actual_queries != expected_queries:
            raise QPlanError(
                "E_LEDGER",
                "batch payload query order does not match the session manifest",
                session_id,
                {"expected": expected_queries, "actual": actual_queries},
            )
        bound = {
            "session_id": session_id,
            "snapshot_version": int(snapshot_version),
            "plan_version": int(plan_version),
            "batch_no": int(batch_no),
            "payload": payload,
        }
        digest_value = content_digest(bound)
        payload_json = canonical_json(payload)
        committed_at = _utc_now()
        try:
            with self.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO batches(
                        session_id, snapshot_version, plan_version, batch_no,
                        content_digest, payload_json, status, committed_at
                    ) VALUES(?,?,?,?,?,?,?,?)
                    """,
                    (
                        session_id,
                        int(snapshot_version),
                        int(plan_version),
                        int(batch_no),
                        digest_value,
                        payload_json,
                        "committed",
                        committed_at,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise QPlanError("E_LEDGER", f"batch {batch_no} was already committed", session_id) from exc
        return {
            "session_id": session_id,
            "batch_no": int(batch_no),
            "content_digest": digest_value,
            "committed_at": committed_at,
        }

    def complete_session(self, session_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT expected_batches, status FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if row is None:
            raise QPlanError("E_LEDGER", f"unknown session {session_id}", session_id)
        expected_batches, status = row
        if status != "running":
            raise QPlanError("E_LEDGER", f"session is not running (status={status})", session_id)
        committed = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM batches WHERE session_id=? AND status='committed'", (session_id,)
            ).fetchone()[0]
        )
        if committed != int(expected_batches):
            raise QPlanError(
                "E_LEDGER",
                "cannot complete a session with missing committed batches",
                session_id,
                {"expected": int(expected_batches), "committed": committed},
            )
        with self.transaction() as connection:
            connection.execute(
                "UPDATE sessions SET status='completed', ended_at=? WHERE session_id=?",
                (_utc_now(), session_id),
            )
        session = self._latest_session(int(self.connection.execute(
            "SELECT snapshot_version FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()[0]))
        return session or {"session_id": session_id, "status": "completed"}

    def recover(self) -> dict[str, Any]:
        snapshot = self.current_snapshot()
        if snapshot is None:
            return {"code": "E_RECOVERY", "status": "no_snapshot", "committed_batches": [], "unknown_batches": []}
        session = self._latest_session(snapshot["snapshot_version"])
        if session is None:
            return {
                "code": "OK",
                "status": "no_session",
                "snapshot_version": snapshot["snapshot_version"],
                "committed_batches": [],
                "unknown_batches": [],
            }
        expected = [int(index + 1) for index in range(session["expected_batches"])]
        rows = self.connection.execute(
            "SELECT batch_no, content_digest FROM batches WHERE session_id=? AND status='committed' ORDER BY batch_no",
            (session["session_id"],),
        ).fetchall()
        committed = [{"batch_no": int(row[0]), "content_digest": row[1]} for row in rows]
        committed_numbers = {item["batch_no"] for item in committed}
        unknown = []
        for batch_no in expected:
            if batch_no not in committed_numbers:
                unknown.append(
                    {
                        "batch_no": batch_no,
                        "query_ids": session["batch_plan"][batch_no - 1],
                        "status": "unknown",
                        "reason": "not committed before process termination",
                    }
                )
        if session["status"] == "running":
            recovered_status = "recovered_complete" if not unknown else "recovered_partial"
            recovery = {
                "code": "OK" if not unknown else "E_RECOVERY",
                "status": recovered_status,
                "recovered_at": _utc_now(),
                "snapshot_version": snapshot["snapshot_version"],
                "session_id": session["session_id"],
                "committed_batches": committed,
                "unknown_batches": unknown,
                "message": "only committed batches were recovered; missing batches are explicitly unknown",
            }
            with self.transaction() as connection:
                connection.execute(
                    """
                    UPDATE sessions SET status=?, ended_at=?, unknown_batches_json=?, recovery_json=?
                    WHERE session_id=?
                    """,
                    (
                        recovered_status,
                        _utc_now(),
                        canonical_json(unknown),
                        canonical_json(recovery),
                        session["session_id"],
                    ),
                )
            return recovery
        return {
            "code": "OK",
            "status": session["status"],
            "snapshot_version": snapshot["snapshot_version"],
            "session_id": session["session_id"],
            "committed_batches": committed,
            "unknown_batches": session["unknown_batches"] or [],
            "message": "session was already finished; no recovery mutation was needed",
        }

    def load_current_view(self) -> dict[str, Any]:
        snapshot = self.current_snapshot()
        if snapshot is None:
            return {"code": "E_LEDGER", "message": "no current snapshot", "snapshot": None, "batches": []}
        session = self._latest_session(snapshot["snapshot_version"])
        if session is None:
            return {
                "code": "OK",
                "snapshot": snapshot,
                "session": None,
                "batches": [],
                "unknown_batches": [],
            }
        rows = self.connection.execute(
            """
            SELECT batch_no, content_digest, payload_json, status, committed_at
            FROM batches WHERE session_id=? ORDER BY batch_no
            """,
            (session["session_id"],),
        ).fetchall()
        batches = []
        for row in rows:
            batches.append(
                {
                    "batch_no": int(row[0]),
                    "content_digest": row[1],
                    "payload": json_loads_canonical(row[2]),
                    "status": row[3],
                    "committed_at": row[4],
                }
            )
        return {
            "code": "OK" if session["status"] in {"completed", "recovered_complete"} else "E_RECOVERY",
            "snapshot": snapshot,
            "session": session,
            "batches": batches,
            "unknown_batches": session["unknown_batches"] or [],
            "view_semantics": "latest current snapshot and latest session only; no prior-version merge",
        }

    def history(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT snapshot_version, plan_version, content_digest, status, replacement_reason, created_at
            FROM snapshots ORDER BY snapshot_version DESC
            """
        ).fetchall()
        return [
            {
                "snapshot_version": int(row[0]),
                "plan_version": int(row[1]),
                "content_digest": row[2],
                "status": row[3],
                "replacement_reason": row[4],
                "created_at": row[5],
            }
            for row in rows
        ]

    def verify_storage_digests(self, view: dict[str, Any]) -> list[dict[str, Any]]:
        errors: list[dict[str, Any]] = []
        snapshot = view.get("snapshot")
        if snapshot is None:
            errors.append({"code": "E_SNAPSHOT", "message": "missing current snapshot"})
            return errors
        recomputed_snapshot = content_digest(
            {
                "spec": snapshot["spec"],
                "stats": snapshot["stats"],
                "weights": snapshot["weights"],
            }
        )
        if recomputed_snapshot != snapshot["content_digest"]:
            errors.append(
                {
                    "code": "E_SNAPSHOT",
                    "message": "snapshot content digest mismatch; view must be replaced",
                    "stored": snapshot["content_digest"],
                    "recomputed": recomputed_snapshot,
                }
            )
        for batch in view.get("batches", []):
            bound = {
                "session_id": view["session"]["session_id"],
                "snapshot_version": int(snapshot["snapshot_version"]),
                "plan_version": int(snapshot["plan_version"]),
                "batch_no": int(batch["batch_no"]),
                "payload": batch["payload"],
            }
            recomputed = content_digest(bound)
            if recomputed != batch["content_digest"]:
                errors.append(
                    {
                        "code": "E_LEDGER",
                        "message": "batch content digest mismatch",
                        "batch_no": int(batch["batch_no"]),
                        "stored": batch["content_digest"],
                        "recomputed": recomputed,
                    }
                )
        return errors
