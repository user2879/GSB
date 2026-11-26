from __future__ import annotations

import json
import os
import shutil
import sqlite3
import stat
import threading
from pathlib import Path
from typing import Any, Iterable

from . import FORMAT_VERSION
from .canonical import canonical_bytes, canonical_json, sha256_hex
from .clockutil import monotonic_ms
from .errors import LedgerError


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS volumes (
    volume_id TEXT PRIMARY KEY,
    format_version INTEGER NOT NULL,
    block_size INTEGER NOT NULL,
    block_count INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    baseline_version INTEGER NOT NULL,
    spec_json TEXT NOT NULL,
    content_digest TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    volume_id TEXT NOT NULL,
    snapshot_version INTEGER NOT NULL,
    format_version INTEGER NOT NULL,
    captured_seq INTEGER NOT NULL,
    status TEXT NOT NULL,
    expected_json TEXT NOT NULL,
    content_digest TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_effects (
    seq INTEGER PRIMARY KEY,
    event_type TEXT NOT NULL,
    entry_id TEXT,
    volume_id TEXT,
    payload_json TEXT NOT NULL,
    content_digest TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT UNIQUE NOT NULL,
    format_version INTEGER NOT NULL,
    rollback_id TEXT NOT NULL,
    event_seq INTEGER NOT NULL,
    volume_id TEXT NOT NULL,
    snapshot_id TEXT NOT NULL,
    batch_index INTEGER NOT NULL,
    block_set_json TEXT NOT NULL,
    status TEXT NOT NULL,
    wait_started_ms INTEGER NOT NULL,
    wait_ended_ms INTEGER,
    started_ms INTEGER,
    ended_ms INTEGER,
    quota_global_limit INTEGER NOT NULL,
    quota_per_volume_limit INTEGER NOT NULL,
    max_blocks_limit INTEGER NOT NULL,
    quota_occupancy_json TEXT NOT NULL,
    content_digest TEXT
);

CREATE TABLE IF NOT EXISTS block_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    rollback_id TEXT NOT NULL,
    volume_id TEXT NOT NULL,
    block_index INTEGER NOT NULL,
    block_version INTEGER NOT NULL,
    baseline_version INTEGER NOT NULL,
    expected_digest TEXT NOT NULL,
    restored_digest TEXT,
    sample_selected INTEGER NOT NULL,
    verification_status TEXT NOT NULL,
    reason TEXT,
    worker_pid INTEGER,
    worker_returncode INTEGER,
    force_killed INTEGER NOT NULL DEFAULT 0,
    format_version INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    created_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS block_state_latest (
    rollback_id TEXT NOT NULL,
    volume_id TEXT NOT NULL,
    block_index INTEGER NOT NULL,
    batch_id TEXT NOT NULL,
    block_version INTEGER NOT NULL,
    baseline_version INTEGER NOT NULL,
    expected_digest TEXT NOT NULL,
    restored_digest TEXT,
    sample_selected INTEGER NOT NULL,
    verification_status TEXT NOT NULL,
    reason TEXT,
    worker_pid INTEGER,
    worker_returncode INTEGER,
    force_killed INTEGER NOT NULL DEFAULT 0,
    format_version INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    updated_ms INTEGER NOT NULL,
    PRIMARY KEY (rollback_id, volume_id, block_index)
);

CREATE TABLE IF NOT EXISTS progress_cursors (
    rollback_id TEXT PRIMARY KEY,
    next_batch_index INTEGER NOT NULL,
    total_blocks INTEGER NOT NULL,
    completed_blocks INTEGER NOT NULL,
    passed_blocks INTEGER NOT NULL,
    corrupt_blocks INTEGER NOT NULL,
    unknown_blocks INTEGER NOT NULL,
    unchecked_blocks INTEGER NOT NULL,
    invalidated_blocks INTEGER NOT NULL DEFAULT 0,
    format_version INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    updated_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS quota_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    rollback_id TEXT NOT NULL,
    volume_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    global_active INTEGER NOT NULL,
    volume_active INTEGER NOT NULL,
    wait_started_ms INTEGER,
    wait_ended_ms INTEGER,
    created_ms INTEGER NOT NULL,
    detail_json TEXT
);

CREATE TABLE IF NOT EXISTS commit_records (
    commit_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    scope_id TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    committed_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS invalidations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    change_id TEXT NOT NULL,
    volume_id TEXT NOT NULL,
    block_index INTEGER NOT NULL,
    kind TEXT NOT NULL,
    source_id TEXT,
    reason TEXT NOT NULL,
    step_order INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    created_ms INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_outcomes_lookup
    ON block_outcomes(rollback_id, volume_id, block_index, id);
CREATE INDEX IF NOT EXISTS idx_batches_status ON batches(status);
CREATE INDEX IF NOT EXISTS idx_invalidations_change ON invalidations(change_id, step_order, volume_id, block_index);
"""


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.init_schema()

    def init_schema(self) -> None:
        with self.lock, self.conn:
            self.conn.executescript(SCHEMA_SQL)
            self.conn.execute("PRAGMA user_version=1")

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def set_meta(self, key: str, value: Any) -> None:
        encoded = value if isinstance(value, str) else canonical_json(value)
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, encoded),
            )

    def get_meta(self, key: str) -> Any | None:
        with self.lock:
            row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return row["value"]

    def initialize_identity(self, identity: dict[str, Any]) -> None:
        payload = dict(identity)
        payload["format_version"] = FORMAT_VERSION
        self.set_meta("session_identity", payload)
        self.set_meta("format_version", str(FORMAT_VERSION))

    def save_replay(self, replay: Any) -> str:
        from .engine import replay_canonical

        canonical = replay_canonical(replay)
        digest = sha256_hex(canonical_bytes(canonical))
        now = monotonic_ms()
        with self.lock, self.conn:
            for volume_id, state in replay.volumes.items():
                volume = state["config"]
                volume_payload = {
                    **volume,
                    "baseline_version_after_replay": state["baseline_version"],
                }
                self.conn.execute(
                    "INSERT INTO volumes(volume_id,format_version,block_size,block_count,seed,"
                    "baseline_version,spec_json,content_digest) VALUES(?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(volume_id) DO UPDATE SET "
                    "format_version=excluded.format_version,block_size=excluded.block_size,"
                    "block_count=excluded.block_count,seed=excluded.seed,"
                    "baseline_version=excluded.baseline_version,spec_json=excluded.spec_json,"
                    "content_digest=excluded.content_digest",
                    (
                        volume_id,
                        FORMAT_VERSION,
                        volume["block_size"],
                        volume["block_count"],
                        volume["seed"],
                        state["baseline_version"],
                        canonical_json(volume_payload),
                        sha256_hex(canonical_bytes(volume_payload)),
                    ),
                )
            for snapshot_id, snapshot in replay.snapshots.items():
                payload = snapshot.to_dict()
                self.conn.execute(
                    "INSERT INTO snapshots(snapshot_id,volume_id,snapshot_version,format_version,"
                    "captured_seq,status,expected_json,content_digest) VALUES(?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(snapshot_id) DO UPDATE SET status=excluded.status,"
                    "expected_json=excluded.expected_json,content_digest=excluded.content_digest",
                    (
                        snapshot_id,
                        snapshot.volume_id,
                        snapshot.version,
                        FORMAT_VERSION,
                        snapshot.captured_seq,
                        snapshot.status,
                        canonical_json(payload),
                        sha256_hex(canonical_bytes(payload)),
                    ),
                )
            for effect in replay.effects:
                effect_digest = sha256_hex(canonical_bytes(effect))
                self.conn.execute(
                    "INSERT OR REPLACE INTO event_effects(seq,event_type,entry_id,volume_id,"
                    "payload_json,content_digest) VALUES(?,?,?,?,?,?)",
                    (
                        effect["seq"],
                        effect["type"],
                        effect.get("entry_id"),
                        effect.get("volume_id"),
                        canonical_json(effect),
                        effect_digest,
                    ),
                )
            self._write_commit_record_locked("replay-session", digest, digest, now)
            self._set_ledger_digest_locked(now)
        return digest

    def start_batch_lease(self, task: dict[str, Any], global_active: int, volume_active: int) -> None:
        now = monotonic_ms()
        occupancy = {
            "global_slots": 1,
            "volume_slots": 1,
            "block_count": len(task["blocks"]),
            "global_active_after_acquire": global_active,
            "volume_active_after_acquire": volume_active,
            "max_blocks_per_rollback": task["max_blocks"],
        }
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT INTO batches(batch_id,format_version,rollback_id,event_seq,volume_id,"
                "snapshot_id,batch_index,block_set_json,status,wait_started_ms,wait_ended_ms,"
                "started_ms,quota_global_limit,quota_per_volume_limit,max_blocks_limit,"
                "quota_occupancy_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    task["batch_id"],
                    FORMAT_VERSION,
                    task["rollback_id"],
                    task["event_seq"],
                    task["volume_id"],
                    task["snapshot_id"],
                    task["batch_index"],
                    canonical_json(task["blocks"]),
                    "running",
                    task["wait_started_ms"],
                    task["wait_ended_ms"],
                    now,
                    task["global_concurrency"],
                    task["per_volume_concurrency"],
                    task["max_blocks"],
                    canonical_json(occupancy),
                ),
            )
            self.conn.execute(
                "INSERT INTO quota_events(batch_id,rollback_id,volume_id,event_type,"
                "global_active,volume_active,wait_started_ms,wait_ended_ms,created_ms,detail_json) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    task["batch_id"],
                    task["rollback_id"],
                    task["volume_id"],
                    "acquired",
                    global_active,
                    volume_active,
                    task["wait_started_ms"],
                    task["wait_ended_ms"],
                    now,
                    canonical_json(occupancy),
                ),
            )

    def complete_batch(
        self,
        batch_id: str,
        outcomes: list[dict[str, Any]],
        *,
        status: str = "completed",
    ) -> dict[str, Any]:
        now = monotonic_ms()
        with self.lock, self.conn:
            batch = self.conn.execute("SELECT * FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
            if batch is None:
                raise LedgerError(f"unknown batch {batch_id}", field="batch_id")
            outcome_rows: list[dict[str, Any]] = []
            for outcome in outcomes:
                payload = {
                    "batch_id": batch_id,
                    "rollback_id": batch["rollback_id"],
                    "volume_id": outcome["volume_id"],
                    "block_index": outcome["block_index"],
                    "block_version": outcome["block_version"],
                    "baseline_version": outcome["baseline_version"],
                    "expected_digest": outcome["expected_digest"],
                    "restored_digest": outcome.get("restored_digest"),
                    "sample_selected": bool(outcome.get("sample_selected", True)),
                    "verification_status": outcome["verification_status"],
                    "reason": outcome.get("reason"),
                    "worker_pid": outcome.get("worker_pid"),
                    "worker_returncode": outcome.get("worker_returncode"),
                    "force_killed": bool(outcome.get("force_killed", False)),
                }
                digest = sha256_hex(canonical_bytes({"format_version": FORMAT_VERSION, "outcome": payload}))
                payload["content_digest"] = digest
                outcome_rows.append(payload)
                self.conn.execute(
                    "INSERT INTO block_outcomes(batch_id,rollback_id,volume_id,block_index,"
                    "block_version,baseline_version,expected_digest,restored_digest,"
                    "sample_selected,verification_status,reason,worker_pid,worker_returncode,"
                    "force_killed,format_version,content_digest,created_ms) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        batch_id,
                        batch["rollback_id"],
                        outcome["volume_id"],
                        outcome["block_index"],
                        outcome["block_version"],
                        outcome["baseline_version"],
                        outcome["expected_digest"],
                        outcome.get("restored_digest"),
                        1 if outcome.get("sample_selected", True) else 0,
                        outcome["verification_status"],
                        outcome.get("reason"),
                        outcome.get("worker_pid"),
                        outcome.get("worker_returncode"),
                        1 if outcome.get("force_killed") else 0,
                        FORMAT_VERSION,
                        digest,
                        now,
                    ),
                )
                self.conn.execute(
                    "INSERT INTO block_state_latest(rollback_id,volume_id,block_index,batch_id,"
                    "block_version,baseline_version,expected_digest,restored_digest,"
                    "sample_selected,verification_status,reason,worker_pid,worker_returncode,"
                    "force_killed,format_version,content_digest,updated_ms) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(rollback_id,volume_id,block_index) DO UPDATE SET "
                    "batch_id=excluded.batch_id,block_version=excluded.block_version,"
                    "baseline_version=excluded.baseline_version,expected_digest=excluded.expected_digest,"
                    "restored_digest=excluded.restored_digest,sample_selected=excluded.sample_selected,"
                    "verification_status=excluded.verification_status,reason=excluded.reason,"
                    "worker_pid=excluded.worker_pid,worker_returncode=excluded.worker_returncode,"
                    "force_killed=excluded.force_killed,format_version=excluded.format_version,"
                    "content_digest=excluded.content_digest,updated_ms=excluded.updated_ms",
                    (
                        batch["rollback_id"],
                        outcome["volume_id"],
                        outcome["block_index"],
                        batch_id,
                        outcome["block_version"],
                        outcome["baseline_version"],
                        outcome["expected_digest"],
                        outcome.get("restored_digest"),
                        1 if outcome.get("sample_selected", True) else 0,
                        outcome["verification_status"],
                        outcome.get("reason"),
                        outcome.get("worker_pid"),
                        outcome.get("worker_returncode"),
                        1 if outcome.get("force_killed") else 0,
                        FORMAT_VERSION,
                        digest,
                        now,
                    ),
                )

            cursor = self._recompute_cursor_locked(batch["rollback_id"], batch["event_seq"])
            batch_payload = {
                "batch_id": batch_id,
                "format_version": FORMAT_VERSION,
                "outcomes": outcome_rows,
                "cursor": cursor,
                "quota_occupancy": json.loads(batch["quota_occupancy_json"]),
            }
            batch_digest = sha256_hex(canonical_bytes(batch_payload))
            terminal_status = status
            if status == "completed" and cursor["unknown_blocks"]:
                terminal_status = "completed_unknown"
            self.conn.execute(
                "UPDATE batches SET status=?,ended_ms=?,content_digest=? WHERE batch_id=?",
                (terminal_status, now, batch_digest, batch_id),
            )
            self.conn.execute(
                "INSERT INTO quota_events(batch_id,rollback_id,volume_id,event_type,"
                "global_active,volume_active,created_ms,detail_json) VALUES(?,?,?,?,0,0,?,?)",
                (
                    batch_id,
                    batch["rollback_id"],
                    batch["volume_id"],
                    "released",
                    now,
                    canonical_json({"reason": terminal_status, "block_count": len(outcomes)}),
                ),
            )
            self._write_commit_record_locked(batch_id, batch_digest, batch_digest, now)
            ledger_digest = self._set_ledger_digest_locked(now)
            cursor["ledger_content_digest"] = ledger_digest
            return {"batch_id": batch_id, "status": terminal_status, "digest": batch_digest, "cursor": cursor}

    def _recompute_cursor_locked(self, rollback_id: str, event_seq: int) -> dict[str, Any]:
        rows = self.conn.execute(
            "SELECT verification_status, COUNT(*) AS n FROM block_state_latest "
            "WHERE rollback_id=? GROUP BY verification_status",
            (rollback_id,),
        ).fetchall()
        counts = {row["verification_status"]: row["n"] for row in rows}
        total = sum(counts.values())
        passed = int(counts.get("pass", 0))
        corrupt = int(counts.get("corrupt", 0))
        unknown = int(counts.get("unknown", 0))
        unchecked = int(counts.get("unchecked", 0))
        invalidated = int(counts.get("invalidated", 0))
        completed = passed + corrupt + unchecked + invalidated
        max_batch = self.conn.execute(
            "SELECT COALESCE(MAX(batch_index), -1) AS m FROM batches WHERE rollback_id=?",
            (rollback_id,),
        ).fetchone()["m"]
        cursor = {
            "rollback_id": rollback_id,
            "event_seq": event_seq,
            "next_batch_index": int(max_batch) + 1,
            "total_blocks": total,
            "completed_blocks": completed,
            "passed_blocks": passed,
            "corrupt_blocks": corrupt,
            "unknown_blocks": unknown,
            "unchecked_blocks": unchecked,
            "invalidated_blocks": invalidated,
            "format_version": FORMAT_VERSION,
            "updated_ms": monotonic_ms(),
        }
        digest = sha256_hex(canonical_bytes(cursor))
        cursor["content_digest"] = digest
        self.conn.execute(
            "INSERT INTO progress_cursors(rollback_id,next_batch_index,total_blocks,"
            "completed_blocks,passed_blocks,corrupt_blocks,unknown_blocks,unchecked_blocks,"
            "invalidated_blocks,format_version,content_digest,updated_ms) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(rollback_id) DO UPDATE SET "
            "next_batch_index=excluded.next_batch_index,total_blocks=excluded.total_blocks,"
            "completed_blocks=excluded.completed_blocks,passed_blocks=excluded.passed_blocks,"
            "corrupt_blocks=excluded.corrupt_blocks,unknown_blocks=excluded.unknown_blocks,"
            "unchecked_blocks=excluded.unchecked_blocks,invalidated_blocks=excluded.invalidated_blocks,"
            "format_version=excluded.format_version,content_digest=excluded.content_digest,"
            "updated_ms=excluded.updated_ms",
            (
                rollback_id,
                cursor["next_batch_index"],
                total,
                completed,
                passed,
                corrupt,
                unknown,
                unchecked,
                invalidated,
                FORMAT_VERSION,
                digest,
                cursor["updated_ms"],
            ),
        )
        return cursor

    def recover_interrupted(self) -> list[dict[str, Any]]:
        now = monotonic_ms()
        recovered: list[dict[str, Any]] = []
        with self.lock, self.conn:
            rows = self.conn.execute("SELECT * FROM batches WHERE status='running' ORDER BY id").fetchall()
            for batch in rows:
                block_set = json.loads(batch["block_set_json"])
                outcomes: list[dict[str, Any]] = []
                for block in block_set:
                    outcome = {
                        "volume_id": block["volume_id"] if "volume_id" in block else batch["volume_id"],
                        "block_index": block["block_index"],
                        "block_version": block.get("replay_block_version", block.get("block_version", 0)),
                        "baseline_version": block["baseline_version"],
                        "expected_digest": block["expected_digest"],
                        "restored_digest": None,
                        "sample_selected": True,
                        "verification_status": "unknown",
                        "reason": "restart_recovered_uncommitted_batch",
                        "worker_pid": None,
                        "worker_returncode": None,
                        "force_killed": False,
                    }
                    outcomes.append(outcome)
                    payload_digest = sha256_hex(
                        canonical_bytes({"format_version": FORMAT_VERSION, "outcome": outcome})
                    )
                    self.conn.execute(
                        "INSERT INTO block_outcomes(batch_id,rollback_id,volume_id,block_index,"
                        "block_version,baseline_version,expected_digest,restored_digest,"
                        "sample_selected,verification_status,reason,format_version,content_digest,created_ms) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            batch["batch_id"],
                            batch["rollback_id"],
                            outcome["volume_id"],
                            outcome["block_index"],
                            outcome["block_version"],
                            outcome["baseline_version"],
                            outcome["expected_digest"],
                            None,
                            1,
                            "unknown",
                            outcome["reason"],
                            FORMAT_VERSION,
                            payload_digest,
                            now,
                        ),
                    )
                    self.conn.execute(
                        "INSERT INTO block_state_latest(rollback_id,volume_id,block_index,batch_id,"
                        "block_version,baseline_version,expected_digest,restored_digest,"
                        "sample_selected,verification_status,reason,format_version,content_digest,updated_ms) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                        "ON CONFLICT(rollback_id,volume_id,block_index) DO UPDATE SET "
                        "batch_id=excluded.batch_id,verification_status='unknown',"
                        "reason=excluded.reason,restored_digest=NULL,updated_ms=excluded.updated_ms",
                        (
                            batch["rollback_id"],
                            outcome["volume_id"],
                            outcome["block_index"],
                            batch["batch_id"],
                            outcome["block_version"],
                            outcome["baseline_version"],
                            outcome["expected_digest"],
                            None,
                            1,
                            "unknown",
                            outcome["reason"],
                            FORMAT_VERSION,
                            payload_digest,
                            now,
                        ),
                    )
                self.conn.execute(
                    "UPDATE batches SET status='interrupted_unknown',ended_ms=? WHERE batch_id=?",
                    (now, batch["batch_id"]),
                )
                self.conn.execute(
                    "INSERT INTO quota_events(batch_id,rollback_id,volume_id,event_type,"
                    "global_active,volume_active,created_ms,detail_json) VALUES(?,?,?,?,0,0,?,?)",
                    (
                        batch["batch_id"],
                        batch["rollback_id"],
                        batch["volume_id"],
                        "recovery_release",
                        now,
                        canonical_json({"recovered_blocks": len(outcomes)}),
                    ),
                )
                cursor = self._recompute_cursor_locked(batch["rollback_id"], batch["event_seq"])
                digest = sha256_hex(canonical_bytes({"recovered": batch["batch_id"], "outcomes": outcomes}))
                self._write_commit_record_locked(batch["batch_id"] + ":recover", digest, digest, now)
                recovered.append(
                    {"batch_id": batch["batch_id"], "rollback_id": batch["rollback_id"], "blocks": len(outcomes), "cursor": cursor}
                )
            if recovered:
                self._set_ledger_digest_locked(now)
        return recovered

    def record_invalidation(
        self,
        change_id: str,
        affected: list[dict[str, Any]],
        steps: list[dict[str, Any]],
    ) -> dict[str, Any]:
        now = monotonic_ms()
        with self.lock, self.conn:
            for step_order, step in enumerate(steps):
                self.conn.execute(
                    "INSERT INTO invalidations(change_id,volume_id,block_index,kind,source_id,"
                    "reason,step_order,payload_json,created_ms) VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        change_id,
                        step.get("volume_id", "*"),
                        int(step.get("block_index", -1)),
                        step.get("kind", "propagation"),
                        step.get("source_id"),
                        step.get("reason", ""),
                        step_order,
                        canonical_json(step),
                        now,
                    ),
                )
            changed_keys = 0
            for item in affected:
                if item.get("ledger") is False:
                    continue
                row = self.conn.execute(
                    "SELECT * FROM block_state_latest WHERE rollback_id=? AND volume_id=? AND block_index=?",
                    (item["rollback_id"], item["volume_id"], item["block_index"]),
                ).fetchone()
                if row is None:
                    continue
                outcome = {
                    "volume_id": item["volume_id"],
                    "block_index": item["block_index"],
                    "block_version": row["block_version"],
                    "baseline_version": row["baseline_version"],
                    "expected_digest": row["expected_digest"],
                    "restored_digest": row["restored_digest"],
                    "sample_selected": bool(row["sample_selected"]),
                    "verification_status": "invalidated",
                    "reason": item.get("reason", "snapshot_dependency_invalidated"),
                    "worker_pid": None,
                    "worker_returncode": None,
                    "force_killed": False,
                }
                digest = sha256_hex(canonical_bytes({"format_version": FORMAT_VERSION, "outcome": outcome, "invalidated": True}))
                self.conn.execute(
                    "INSERT INTO block_outcomes(batch_id,rollback_id,volume_id,block_index,"
                    "block_version,baseline_version,expected_digest,restored_digest,"
                    "sample_selected,verification_status,reason,format_version,content_digest,created_ms) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        "invalidation:" + change_id,
                        item["rollback_id"],
                        item["volume_id"],
                        item["block_index"],
                        row["block_version"],
                        row["baseline_version"],
                        row["expected_digest"],
                        row["restored_digest"],
                        row["sample_selected"],
                        "invalidated",
                        outcome["reason"],
                        FORMAT_VERSION,
                        digest,
                        now,
                    ),
                )
                self.conn.execute(
                    "UPDATE block_state_latest SET verification_status='invalidated',reason=?,"
                    "content_digest=?,updated_ms=? WHERE rollback_id=? AND volume_id=? AND block_index=?",
                    (outcome["reason"], digest, now, item["rollback_id"], item["volume_id"], item["block_index"]),
                )
                self._recompute_cursor_locked(item["rollback_id"], 0)
                changed_keys += 1
            digest = sha256_hex(canonical_bytes({"change_id": change_id, "affected": affected, "steps": steps}))
            self._write_commit_record_locked("invalidation:" + change_id, digest, digest, now)
            ledger_digest = self._set_ledger_digest_locked(now)
        return {"change_id": change_id, "ledger_rows_changed": changed_keys, "content_digest": ledger_digest}

    def unknown_tasks(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM block_state_latest WHERE verification_status='unknown' "
                "ORDER BY rollback_id,volume_id,block_index"
            ).fetchall()
        return [dict(row) for row in rows]

    def latest_rows(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM block_state_latest ORDER BY rollback_id,volume_id,block_index"
            ).fetchall()
        return [dict(row) for row in rows]

    def authoritative_state(self) -> dict[str, Any]:
        """Return only durable state usable by restart recovery."""
        with self.lock:
            snapshots = {
                row["snapshot_id"]: json.loads(row["expected_json"])
                for row in self.conn.execute(
                    "SELECT snapshot_id,expected_json FROM snapshots ORDER BY snapshot_id"
                )
            }
            blocks = self.latest_rows()
            cursors = [
                dict(row)
                for row in self.conn.execute("SELECT * FROM progress_cursors ORDER BY rollback_id")
            ]
            identity = self.get_meta("session_identity")
            digest = self.get_meta("ledger_content_digest")
        return {
            "identity": identity,
            "ledger_content_digest": digest,
            "snapshot_count": len(snapshots),
            "snapshots": snapshots,
            "block_count": len(blocks),
            "blocks": blocks,
            "cursors": cursors,
        }

    def batch_wait_metrics(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT batch_id,rollback_id,volume_id,status,wait_started_ms,wait_ended_ms,"
                "COALESCE(wait_ended_ms-wait_started_ms,0) AS queue_wait_ms,"
                "quota_occupancy_json FROM batches ORDER BY id"
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["quota_occupancy"] = json.loads(item.pop("quota_occupancy_json"))
            result.append(item)
        return result

    def _write_commit_record_locked(
        self, scope_id: str, content_digest: str, payload_digest: str, now: int
    ) -> None:
        self.conn.execute(
            "INSERT INTO commit_records(scope_id,format_version,content_digest,payload_digest,committed_ms) "
            "VALUES(?,?,?,?,?)",
            (scope_id, FORMAT_VERSION, content_digest, payload_digest, now),
        )

    def ledger_content_digest(self) -> str:
        with self.lock:
            return self._ledger_digest_locked()

    def _ledger_digest_locked(self) -> str:
        tables = {
            "meta_identity": self.get_meta("session_identity"),
            "volumes": [dict(r) for r in self.conn.execute("SELECT * FROM volumes ORDER BY volume_id")],
            "snapshots": [dict(r) for r in self.conn.execute("SELECT snapshot_id,volume_id,snapshot_version,format_version,captured_seq,status,expected_json,content_digest FROM snapshots ORDER BY snapshot_id")],
            "latest": [dict(r) for r in self.conn.execute("SELECT rollback_id,volume_id,block_index,batch_id,block_version,baseline_version,expected_digest,restored_digest,sample_selected,verification_status,reason,format_version,content_digest FROM block_state_latest ORDER BY rollback_id,volume_id,block_index")],
            "cursors": [dict(r) for r in self.conn.execute("SELECT rollback_id,next_batch_index,total_blocks,completed_blocks,passed_blocks,corrupt_blocks,unknown_blocks,unchecked_blocks,invalidated_blocks,format_version,content_digest FROM progress_cursors ORDER BY rollback_id")],
        }
        return sha256_hex(canonical_bytes(tables))

    def _set_ledger_digest_locked(self, now: int) -> str:
        digest = self._ledger_digest_locked()
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES('ledger_content_digest',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (digest,),
        )
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES('ledger_updated_ms',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (json.dumps(now),),
        )
        return digest


def make_readonly(path: Path) -> None:
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    except FileNotFoundError:
        pass


def _read_sqlite_identity(ledger_path: Path) -> dict[str, Any] | None:
    try:
        conn = sqlite3.connect(f"file:{ledger_path.resolve()}?mode=ro", uri=True)
        try:
            row = conn.execute("SELECT value FROM meta WHERE key='session_identity'").fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        value = json.loads(row[0])
        return value if isinstance(value, dict) else None
    except (sqlite3.Error, json.JSONDecodeError, OSError):
        return None


def open_session_ledger(
    spec: dict[str, Any],
    ledger_path: str | Path,
    session_path: str | Path,
) -> tuple[Ledger, dict[str, Any] | None]:
    """Open ledger, replacing the whole database when session identity differs."""
    from .canonical import spec_identity

    identity = spec_identity(spec)
    ledger_path = Path(ledger_path)
    session_path = Path(session_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    session_path.parent.mkdir(parents=True, exist_ok=True)

    old_identity: dict[str, Any] | None = None
    diff: dict[str, Any] | None = None
    replacement_needed = False

    if session_path.exists():
        try:
            old_identity = json.loads(session_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            old_identity = {"content_digest": "invalid-session-json"}
        replacement_needed = any(old_identity.get(key) != identity[key] for key in identity)
    elif ledger_path.exists():
        old_identity = _read_sqlite_identity(ledger_path)
        if old_identity is None:
            replacement_needed = True
        else:
            replacement_needed = any(old_identity.get(key) != identity[key] for key in identity)

    if replacement_needed and (ledger_path.exists() or session_path.exists()):
        backup_dir = ledger_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stem = ledger_path.stem
        index = 1
        while (backup_dir / f"{stem}.replaced-{index:03d}.sqlite").exists():
            index += 1
        backup_db = backup_dir / f"{stem}.replaced-{index:03d}.sqlite"
        diff_path = backup_dir / f"{stem}.replaced-{index:03d}.diff.json"

        if ledger_path.exists():
            source = sqlite3.connect(str(ledger_path))
            try:
                target = sqlite3.connect(str(backup_db))
                source.backup(target)
                target.close()
            finally:
                source.close()
            for suffix in ("-wal", "-shm"):
                side = Path(str(ledger_path) + suffix)
                if side.exists():
                    try:
                        side.unlink()
                    except OSError:
                        pass
            ledger_path.unlink(missing_ok=True)
            make_readonly(backup_db)

        old_session_backup = backup_dir / f"{stem}.replaced-{index:03d}.session.json"
        if session_path.exists():
            shutil.copy2(session_path, old_session_backup)
            make_readonly(old_session_backup)

        diff = {
            "replacement_index": index,
            "reason": "session_identity_mismatch",
            "old_identity": old_identity,
            "new_identity": identity,
            "differences": [
                {
                    "field": key,
                    "old": None if old_identity is None else old_identity.get(key),
                    "new": identity[key],
                }
                for key in identity
                if old_identity is None or old_identity.get(key) != identity[key]
            ],
            "backup_database": str(backup_db),
            "backup_session": str(old_session_backup) if old_session_backup.exists() else None,
        }
        diff_path.write_text(canonical_json(diff) + "\n", encoding="utf-8")
        make_readonly(diff_path)
        session_path.unlink(missing_ok=True)

    ledger = Ledger(ledger_path)
    ledger.initialize_identity(identity)
    session_payload = {**identity, "ledger_file": str(ledger_path), "schema_name": "blockledger-session"}
    temporary = session_path.with_suffix(".tmp")
    temporary.write_text(canonical_json(session_payload) + "\n", encoding="utf-8")
    os.replace(temporary, session_path)
    return ledger, diff


def open_readonly_ledger(path: str | Path) -> sqlite3.Connection:
    ledger_path = Path(path)
    if not ledger_path.exists():
        raise LedgerError(f"ledger does not exist: {ledger_path}", field="ledger")
    uri = f"file:{ledger_path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn
