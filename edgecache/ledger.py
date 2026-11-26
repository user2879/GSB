"""SQLite write-ahead ledger, conservation audit and immutable snapshots."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import FORMAT_VERSION
from .canonical import canonical_bytes, content_digest, digest_bytes
from .errors import ConserveError, EdgeError


def _now_ms() -> int:
    return int(time.time() * 1000)


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    spec_digest TEXT NOT NULL,
    status TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    created_at_ms INTEGER NOT NULL,
    finished_at_ms INTEGER,
    run_digest TEXT
);
CREATE TABLE IF NOT EXISTS quota_state (
    run_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    available INTEGER NOT NULL,
    quota_version INTEGER NOT NULL,
    PRIMARY KEY (run_id, node_id)
);
CREATE TABLE IF NOT EXISTS global_quota_state (
    run_id TEXT PRIMARY KEY,
    available INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    run_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    fragment_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    revision_id INTEGER NOT NULL,
    sequence INTEGER NOT NULL,
    state TEXT NOT NULL,
    evict_blocks INTEGER NOT NULL,
    block_weight INTEGER NOT NULL,
    quota_version INTEGER NOT NULL,
    node_delta INTEGER NOT NULL DEFAULT 0,
    global_delta INTEGER NOT NULL DEFAULT 0,
    node_before INTEGER,
    node_after INTEGER,
    global_before INTEGER,
    global_after INTEGER,
    format_version INTEGER NOT NULL,
    content_digest TEXT,
    sample_bytes BLOB,
    sample_json TEXT,
    elapsed_monotonic_ms INTEGER,
    error_code TEXT,
    error_json TEXT,
    requires_rerun INTEGER NOT NULL DEFAULT 0,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    PRIMARY KEY (run_id, batch_id)
);
CREATE TABLE IF NOT EXISTS conservation_checks (
    check_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    quota_version INTEGER NOT NULL,
    evicted_blocks INTEGER NOT NULL,
    block_weight INTEGER NOT NULL,
    expected_delta INTEGER NOT NULL,
    node_delta INTEGER NOT NULL,
    global_delta INTEGER NOT NULL,
    node_diff INTEGER NOT NULL,
    global_diff INTEGER NOT NULL,
    ok INTEGER NOT NULL,
    format_version INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    canonical_bytes BLOB NOT NULL,
    created_at_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    run_id TEXT NOT NULL,
    snapshot_id INTEGER NOT NULL,
    format_version INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    canonical_bytes BLOB NOT NULL,
    created_at_ms INTEGER NOT NULL,
    PRIMARY KEY (run_id, snapshot_id)
);
CREATE TABLE IF NOT EXISTS recovery_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    previous_state TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    created_at_ms INTEGER NOT NULL,
    level TEXT NOT NULL,
    code TEXT NOT NULL,
    node_id TEXT,
    batch_id TEXT,
    details_json TEXT NOT NULL,
    content_digest TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_batches_state ON batches(run_id, state);
CREATE INDEX IF NOT EXISTS idx_checks_phase ON conservation_checks(run_id, phase, batch_id);
"""


class Ledger:
    def __init__(self, path: str | Path, *, reset: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset and self.path.exists():
            self.path.unlink()
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=10)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=FULL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)
            self._conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('format_version', ?)",
                (str(FORMAT_VERSION),),
            )
            self._conn.commit()
        stored_version = self._conn.execute("SELECT value FROM meta WHERE key='format_version'").fetchone()[0]
        if int(stored_version) != FORMAT_VERSION:
            raise EdgeError(
                "E_SNAPSHOT",
                "ledger format version mismatch; replacement is required",
                details={"stored": int(stored_version), "required": FORMAT_VERSION},
                exit_code=11,
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def create_run(
        self,
        run_id: str,
        spec_digest: str,
        node_quotas: dict[str, int],
        global_quota: int,
    ) -> None:
        with self._lock, self._conn:
            existing = self._conn.execute(
                "SELECT spec_digest, format_version FROM runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if existing is not None:
                if existing["spec_digest"] != spec_digest:
                    raise EdgeError(
                        "E_SNAPSHOT",
                        "run identifier was created from a different spec digest",
                        details={"run_id": run_id, "stored": existing["spec_digest"], "incoming": spec_digest},
                        exit_code=11,
                    )
                if int(existing["format_version"]) != FORMAT_VERSION:
                    raise EdgeError("E_SNAPSHOT", "run format version mismatch", exit_code=11)
                return
            self._conn.execute(
                "INSERT INTO runs(run_id, spec_digest, status, format_version, created_at_ms) VALUES(?,?,?,?,?)",
                (run_id, spec_digest, "open", FORMAT_VERSION, _now_ms()),
            )
            for node_id, quota in sorted(node_quotas.items()):
                version = 0
                self._conn.execute(
                    "INSERT INTO quota_state(run_id, node_id, available, quota_version) VALUES(?,?,?,?)",
                    (run_id, node_id, int(quota), version),
                )
            self._conn.execute(
                "INSERT INTO global_quota_state(run_id, available) VALUES(?,?)",
                (run_id, int(global_quota)),
            )

    def find_open_runs(self, spec_digest: str) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id FROM runs WHERE spec_digest=? AND status='open' ORDER BY created_at_ms DESC",
                (spec_digest,),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def set_run_status(self, run_id: str, status: str, run_digest: str | None = None) -> None:
        with self._lock, self._conn:
            finished = _now_ms() if status in {"complete", "interrupted", "failed"} else None
            self._conn.execute(
                "UPDATE runs SET status=?, finished_at_ms=?, run_digest=? WHERE run_id=?",
                (status, finished, run_digest, run_id),
            )

    def register_batch_intent(
        self,
        run_id: str,
        *,
        batch_id: str,
        fragment_id: str,
        node_id: str,
        revision_id: int,
        sequence: int,
        evict_blocks: int,
        block_weight: int,
        quota_version: int,
    ) -> str:
        now = _now_ms()
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT state FROM batches WHERE run_id=? AND batch_id=?",
                (run_id, batch_id),
            ).fetchone()
            if row is None:
                state = "pending"
                self._conn.execute(
                    """
                    INSERT INTO batches(
                        run_id,batch_id,fragment_id,node_id,revision_id,sequence,state,
                        evict_blocks,block_weight,quota_version,format_version,created_at_ms,updated_at_ms
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        run_id,
                        batch_id,
                        fragment_id,
                        node_id,
                        revision_id,
                        sequence,
                        state,
                        evict_blocks,
                        block_weight,
                        quota_version,
                        FORMAT_VERSION,
                        now,
                        now,
                    ),
                )
                return state
            state = str(row["state"])
            if state == "committed":
                return state
            if state in {"unknown", "failed"}:
                self._conn.execute(
                    "INSERT INTO recovery_events(run_id,batch_id,reason,previous_state,created_at_ms) VALUES(?,?,?,?,?)",
                    (run_id, batch_id, "explicit_rerun", state, now),
                )
            self._conn.execute(
                """
                UPDATE batches SET state='pending', error_code=NULL, error_json=NULL,
                    requires_rerun=0, updated_at_ms=?
                WHERE run_id=? AND batch_id=?
                """,
                (now, run_id, batch_id),
            )
            return "pending"

    def mark_unknown(
        self,
        run_id: str,
        batch_id: str,
        reason: str,
        error_code: str | None = None,
    ) -> bool:
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT state FROM batches WHERE run_id=? AND batch_id=?",
                (run_id, batch_id),
            ).fetchone()
            if row is None or row["state"] == "committed":
                return False
            self._conn.execute(
                """
                UPDATE batches SET state='unknown', error_code=?, requires_rerun=1,
                    error_json=?, updated_at_ms=?
                WHERE run_id=? AND batch_id=?
                """,
                (error_code, canonical_bytes({"reason": reason}).decode("utf-8"), _now_ms(), run_id, batch_id),
            )
            return True

    def fail_batch(self, run_id: str, batch_id: str, error: EdgeError) -> bool:
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT state FROM batches WHERE run_id=? AND batch_id=?",
                (run_id, batch_id),
            ).fetchone()
            if row is None or row["state"] == "committed":
                return False
            self._conn.execute(
                """
                UPDATE batches SET state='failed', error_code=?, error_json=?,
                    requires_rerun=0, updated_at_ms=?
                WHERE run_id=? AND batch_id=?
                """,
                (
                    error.code,
                    canonical_bytes(error.to_dict()).decode("utf-8"),
                    _now_ms(),
                    run_id,
                    batch_id,
                ),
            )
            return True

    def mark_interrupted_pending_unknown(self, run_id: str, reason: str = "uncommitted_after_restart") -> list[str]:
        with self._lock, self._conn:
            rows = self._conn.execute(
                "SELECT batch_id FROM batches WHERE run_id=? AND state='pending' ORDER BY node_id, batch_id",
                (run_id,),
            ).fetchall()
            batch_ids = [str(row[0]) for row in rows]
            now = _now_ms()
            for batch_id in batch_ids:
                self._conn.execute(
                    """
                    UPDATE batches SET state='unknown', requires_rerun=1, error_code='E_UNKNOWN',
                        error_json=?, updated_at_ms=?
                    WHERE run_id=? AND batch_id=?
                    """,
                    (
                        canonical_bytes({"reason": reason, "requires_rerun": True}).decode("utf-8"),
                        now,
                        run_id,
                        batch_id,
                    ),
                )
                self._conn.execute(
                    "INSERT INTO recovery_events(run_id,batch_id,reason,previous_state,created_at_ms) VALUES(?,?,?,?,?)",
                    (run_id, batch_id, reason, "pending", now),
                )
            return batch_ids

    def commit_batch(
        self,
        run_id: str,
        batch_id: str,
        *,
        sampling_payload: dict[str, Any],
        elapsed_monotonic_ms: int,
        expected_delta: int,
    ) -> dict[str, int]:
        sample_bytes = canonical_bytes(sampling_payload)
        sample_digest = digest_bytes(sample_bytes)
        now = _now_ms()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                batch = self._conn.execute(
                    "SELECT * FROM batches WHERE run_id=? AND batch_id=?",
                    (run_id, batch_id),
                ).fetchone()
                if batch is None:
                    raise EdgeError("E_LEDGER", "batch intent does not exist", exit_code=12)
                if batch["state"] == "committed":
                    self._conn.commit()
                    return {
                        "node_before": int(batch["node_before"]),
                        "node_after": int(batch["node_after"]),
                        "global_before": int(batch["global_before"]),
                        "global_after": int(batch["global_after"]),
                        "node_delta": int(batch["node_delta"]),
                        "global_delta": int(batch["global_delta"]),
                    }
                node = self._conn.execute(
                    "SELECT available, quota_version FROM quota_state WHERE run_id=? AND node_id=?",
                    (run_id, batch["node_id"]),
                ).fetchone()
                global_row = self._conn.execute(
                    "SELECT available FROM global_quota_state WHERE run_id=?",
                    (run_id,),
                ).fetchone()
                if node is None or global_row is None:
                    raise EdgeError("E_LEDGER", "quota state is missing", exit_code=12)
                node_before = int(node["available"])
                global_before = int(global_row["available"])
                delta = int(expected_delta)
                if delta < 0 or node_before < delta or global_before < delta:
                    raise EdgeError(
                        "E_QUOTA",
                        "quota is insufficient at commit",
                        details={
                            "batch_id": batch_id,
                            "expected_delta": delta,
                            "node_before": node_before,
                            "global_before": global_before,
                        },
                        exit_code=10,
                    )
                node_after = node_before - delta
                global_after = global_before - delta
                node_diff = delta - delta
                global_diff = delta - delta
                if node_diff != 0 or global_diff != 0:
                    raise ConserveError(
                        batch_id,
                        node_diff,
                        global_diff,
                        int(batch["quota_version"]),
                    )
                self._conn.execute(
                    "UPDATE quota_state SET available=?, quota_version=? WHERE run_id=? AND node_id=?",
                    (node_after, int(batch["quota_version"]), run_id, batch["node_id"]),
                )
                self._conn.execute(
                    "UPDATE global_quota_state SET available=? WHERE run_id=?",
                    (global_after, run_id),
                )
                self._conn.execute(
                    """
                    UPDATE batches SET state='committed', node_delta=?, global_delta=?,
                        node_before=?, node_after=?, global_before=?, global_after=?,
                        sample_bytes=?, sample_json=?, content_digest=?,
                        elapsed_monotonic_ms=?, requires_rerun=0, updated_at_ms=?
                    WHERE run_id=? AND batch_id=?
                    """,
                    (
                        delta,
                        delta,
                        node_before,
                        node_after,
                        global_before,
                        global_after,
                        sample_bytes,
                        sample_bytes.decode("utf-8"),
                        sample_digest,
                        int(elapsed_monotonic_ms),
                        now,
                        run_id,
                        batch_id,
                    ),
                )
                self._conn.commit()
                return {
                    "node_before": node_before,
                    "node_after": node_after,
                    "global_before": global_before,
                    "global_after": global_after,
                    "node_delta": delta,
                    "global_delta": delta,
                }
            except Exception:
                self._conn.rollback()
                raise

    def record_conservation_check(
        self,
        run_id: str,
        phase: str,
        *,
        batch_id: str,
        node_id: str,
        quota_version: int,
        evicted_blocks: int,
        block_weight: int,
        node_delta: int,
        global_delta: int,
        node_before: int,
        global_before: int,
    ) -> dict[str, Any]:
        expected = evicted_blocks * block_weight
        node_diff = node_delta - expected
        global_diff = global_delta - expected
        ok = node_diff == 0 and global_diff == 0
        payload = {
            "run_id": run_id,
            "phase": phase,
            "batch_id": batch_id,
            "node_id": node_id,
            "quota_version": quota_version,
            "evicted_blocks": evicted_blocks,
            "block_weight": block_weight,
            "expected_delta": expected,
            "node_delta": node_delta,
            "global_delta": global_delta,
            "node_diff": node_diff,
            "global_diff": global_diff,
            "node_before": node_before,
            "global_before": global_before,
            "ok": ok,
        }
        data = canonical_bytes(payload)
        digest = digest_bytes(data)
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO conservation_checks(
                    run_id,phase,batch_id,node_id,quota_version,evicted_blocks,block_weight,
                    expected_delta,node_delta,global_delta,node_diff,global_diff,ok,
                    format_version,content_digest,canonical_bytes,created_at_ms
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    phase,
                    batch_id,
                    node_id,
                    quota_version,
                    evicted_blocks,
                    block_weight,
                    expected,
                    node_delta,
                    global_delta,
                    node_diff,
                    global_diff,
                    int(ok),
                    FORMAT_VERSION,
                    digest,
                    data,
                    _now_ms(),
                ),
            )
        return payload

    def conservation_conclusion(self, run_id: str, phase: str) -> dict[str, Any]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT batch_id, ok, node_diff, global_diff, quota_version, content_digest
                FROM conservation_checks WHERE run_id=? AND phase=? ORDER BY batch_id
                """,
                (run_id, phase),
            ).fetchall()
        checks = [
            {
                "batch_id": str(row["batch_id"]),
                "ok": bool(row["ok"]),
                "node_diff": int(row["node_diff"]),
                "global_diff": int(row["global_diff"]),
                "quota_version": int(row["quota_version"]),
                "content_digest": str(row["content_digest"]),
            }
            for row in rows
        ]
        conclusion = {
            "phase": phase,
            "all_ok": all(check["ok"] for check in checks),
            "checked_batches": len(checks),
            "checks": checks,
        }
        conclusion["content_digest"] = content_digest(conclusion)
        return conclusion

    def clear_conservation_checks(self, run_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM conservation_checks WHERE run_id=?", (run_id,))

    def save_snapshot(self, run_id: str, payload: dict[str, Any], snapshot_id: int | None = None) -> dict[str, Any]:
        data = canonical_bytes(payload)
        digest = digest_bytes(data)
        with self._lock, self._conn:
            if snapshot_id is None:
                row = self._conn.execute(
                    "SELECT MAX(snapshot_id) FROM snapshots WHERE run_id=?",
                    (run_id,),
                ).fetchone()
                snapshot_id = int(row[0] or 0) + 1
            existing = self._conn.execute(
                "SELECT format_version, content_digest, canonical_bytes FROM snapshots WHERE run_id=? AND snapshot_id=?",
                (run_id, snapshot_id),
            ).fetchone()
            replaced = False
            if existing is not None:
                existing_bytes = bytes(existing["canonical_bytes"])
                existing_digest = digest_bytes(existing_bytes)
                valid = (
                    int(existing["format_version"]) == FORMAT_VERSION
                    and existing_digest == existing["content_digest"]
                )
                if not valid:
                    self._conn.execute("DELETE FROM snapshots WHERE run_id=?", (run_id,))
                    replaced = True
                elif existing_bytes == data:
                    return {
                        "snapshot_id": snapshot_id,
                        "format_version": FORMAT_VERSION,
                        "content_digest": digest,
                        "action": "unchanged",
                    }
            self._conn.execute(
                """
                INSERT OR REPLACE INTO snapshots(run_id,snapshot_id,format_version,content_digest,canonical_bytes,created_at_ms)
                VALUES(?,?,?,?,?,?)
                """,
                (run_id, snapshot_id, FORMAT_VERSION, digest, data, _now_ms()),
            )
        return {
            "snapshot_id": snapshot_id,
            "format_version": FORMAT_VERSION,
            "content_digest": digest,
            "action": "replaced" if replaced else "saved",
        }

    def load_snapshot(self, run_id: str, snapshot_id: int | None = None) -> dict[str, Any]:
        with self._lock:
            if snapshot_id is None:
                row = self._conn.execute(
                    "SELECT * FROM snapshots WHERE run_id=? ORDER BY snapshot_id DESC LIMIT 1",
                    (run_id,),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT * FROM snapshots WHERE run_id=? AND snapshot_id=?",
                    (run_id, snapshot_id),
                ).fetchone()
        if row is None:
            return {"action": "missing", "payload": None}
        data = bytes(row["canonical_bytes"])
        actual_digest = digest_bytes(data)
        version_ok = int(row["format_version"]) == FORMAT_VERSION
        digest_ok = actual_digest == row["content_digest"]
        if not version_ok or not digest_ok:
            return {
                "action": "replace",
                "payload": None,
                "reason": "version_mismatch" if not version_ok else "digest_mismatch",
                "stored_digest": row["content_digest"],
                "actual_digest": actual_digest,
            }
        import json

        return {
            "action": "valid",
            "payload": json.loads(data.decode("utf-8")),
            "format_version": int(row["format_version"]),
            "content_digest": actual_digest,
        }

    def list_snapshots(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT snapshot_id, format_version, content_digest, created_at_ms FROM snapshots WHERE run_id=? ORDER BY snapshot_id",
                (run_id,),
            ).fetchall()
        return [
            {
                "snapshot_id": int(row["snapshot_id"]),
                "format_version": int(row["format_version"]),
                "content_digest": str(row["content_digest"]),
                "created_at_ms": int(row["created_at_ms"]),
            }
            for row in rows
        ]

    def iter_batches(self, run_id: str, states: Iterable[str] | None = None) -> Iterator[dict[str, Any]]:
        state_values = tuple(states) if states is not None else None
        query = "SELECT * FROM batches WHERE run_id=?"
        params: list[Any] = [run_id]
        if state_values:
            placeholders = ",".join("?" for _ in state_values)
            query += f" AND state IN ({placeholders})"
            params.extend(state_values)
        query += " ORDER BY node_id, sequence, batch_id"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        for row in rows:
            yield {key: row[key] for key in row.keys()}

    def get_batch(self, run_id: str, batch_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM batches WHERE run_id=? AND batch_id=?",
                (run_id, batch_id),
            ).fetchone()
        return {key: row[key] for key in row.keys()} if row is not None else None

    def unknown_batches(self, run_id: str) -> list[dict[str, Any]]:
        return list(self.iter_batches(run_id, states=("unknown",)))

    def committed_batches(self, run_id: str) -> list[dict[str, Any]]:
        return list(self.iter_batches(run_id, states=("committed",)))

    def quota_snapshot(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            nodes = self._conn.execute(
                "SELECT node_id, available, quota_version FROM quota_state WHERE run_id=? ORDER BY node_id",
                (run_id,),
            ).fetchall()
            global_row = self._conn.execute(
                "SELECT available FROM global_quota_state WHERE run_id=?",
                (run_id,),
            ).fetchone()
        return {
            "global_available": int(global_row["available"]) if global_row is not None else None,
            "nodes": [
                {
                    "node_id": str(row["node_id"]),
                    "available": int(row["available"]),
                    "quota_version": int(row["quota_version"]),
                }
                for row in nodes
            ],
        }

    def add_log(
        self,
        *,
        run_id: str | None,
        level: str,
        code: str,
        node_id: str | None = None,
        batch_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "timestamp_ms": _now_ms(),
            "level": level,
            "code": code,
            "node_id": node_id,
            "batch_id": batch_id,
            "details": details or {},
        }
        data = canonical_bytes(payload)
        digest = digest_bytes(data)
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO logs(run_id,created_at_ms,level,code,node_id,batch_id,details_json,content_digest)
                VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    payload["timestamp_ms"],
                    level,
                    code,
                    node_id,
                    batch_id,
                    canonical_bytes(payload["details"]).decode("utf-8"),
                    digest,
                ),
            )
        return payload

    def list_logs(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM logs WHERE run_id=? ORDER BY log_id",
                (run_id,),
            ).fetchall()
        return [
            {
                "log_id": int(row["log_id"]),
                "created_at_ms": int(row["created_at_ms"]),
                "level": str(row["level"]),
                "code": str(row["code"]),
                "node_id": row["node_id"],
                "batch_id": row["batch_id"],
                "details": row["details_json"],
                "content_digest": str(row["content_digest"]),
            }
            for row in rows
        ]

    def ledger_digest(self, run_id: str) -> str:
        payload = {
            "run_id": run_id,
            "quota": self.quota_snapshot(run_id),
            "batches": [
                {
                    "batch_id": batch["batch_id"],
                    "fragment_id": batch["fragment_id"],
                    "node_id": batch["node_id"],
                    "revision_id": batch["revision_id"],
                    "state": batch["state"],
                    "evict_blocks": batch["evict_blocks"],
                    "node_delta": batch["node_delta"],
                    "global_delta": batch["global_delta"],
                    "content_digest": batch["content_digest"],
                    "error_code": batch["error_code"],
                    "requires_rerun": bool(batch["requires_rerun"]),
                }
                for batch in self.iter_batches(run_id)
            ],
        }
        return content_digest(payload)

    def all_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id,spec_digest,status,format_version,created_at_ms,finished_at_ms,run_digest FROM runs ORDER BY created_at_ms",
            ).fetchall()
        return [
            {
                "run_id": str(row["run_id"]),
                "spec_digest": str(row["spec_digest"]),
                "status": str(row["status"]),
                "format_version": int(row["format_version"]),
                "created_at_ms": int(row["created_at_ms"]),
                "finished_at_ms": row["finished_at_ms"],
                "run_digest": row["run_digest"],
            }
            for row in rows
        ]
