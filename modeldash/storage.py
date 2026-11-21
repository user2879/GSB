"""Transactional SQLite ledger, session replacement and snapshot restoration."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import LedgerError, SpecError
from .receipts import Receipt
from .spec import Spec
from .util import canonical_json, digest_json, ensure_directory, utc_now


LEDGER_FORMAT_VERSION = 1


SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    spec_version INTEGER NOT NULL,
    spec_digest TEXT NOT NULL,
    budget_digest TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    is_stale INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    state_json TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    replacement_reason TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS rounds (
    session_id TEXT NOT NULL,
    round_seq INTEGER NOT NULL,
    cursor INTEGER NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    committed_at TEXT NOT NULL,
    is_stale INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, round_seq)
);
CREATE TABLE IF NOT EXISTS tasks (
    session_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    priority INTEGER NOT NULL,
    config_version INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    state TEXT NOT NULL,
    completed_json TEXT NOT NULL,
    output_summary TEXT NOT NULL,
    wait_ms INTEGER NOT NULL,
    used_quota_ms INTEGER NOT NULL,
    error_code TEXT NOT NULL,
    is_stale INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, task_id)
);
CREATE TABLE IF NOT EXISTS attempts (
    session_id TEXT NOT NULL,
    attempt_key TEXT NOT NULL,
    task_id TEXT NOT NULL,
    shard_index INTEGER NOT NULL,
    attempt_no INTEGER NOT NULL,
    request_id TEXT NOT NULL,
    error_code TEXT NOT NULL,
    compute_ms INTEGER NOT NULL,
    started_at_ms INTEGER NOT NULL,
    finished_at_ms INTEGER NOT NULL,
    is_stale INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, attempt_key)
);
CREATE TABLE IF NOT EXISTS preemptions (
    session_id TEXT NOT NULL,
    preemption_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    by_task_id TEXT NOT NULL,
    requested_shard INTEGER NOT NULL,
    suspended_shard INTEGER NOT NULL,
    resources_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    is_stale INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, preemption_id)
);
CREATE TABLE IF NOT EXISTS preemption_decisions (
    session_id TEXT NOT NULL,
    decision_id TEXT NOT NULL,
    high_task_id TEXT NOT NULL,
    victim_task_id TEXT NOT NULL,
    verdict TEXT NOT NULL,
    reason TEXT NOT NULL,
    preemptable INTEGER NOT NULL,
    details_json TEXT NOT NULL,
    is_stale INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, decision_id)
);
CREATE TABLE IF NOT EXISTS receipts (
    session_id TEXT NOT NULL,
    receipt_key TEXT NOT NULL,
    node_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    config_version INTEGER NOT NULL,
    shard_index INTEGER NOT NULL,
    shard_count INTEGER NOT NULL,
    item_keys_json TEXT NOT NULL,
    input_digest TEXT NOT NULL,
    output_digest TEXT NOT NULL,
    generated_version INTEGER NOT NULL,
    decay_count INTEGER NOT NULL,
    untrust_score INTEGER NOT NULL,
    status TEXT NOT NULL,
    reason TEXT NOT NULL,
    dependency_refs_json TEXT NOT NULL,
    is_stale INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, receipt_key)
);
CREATE TABLE IF NOT EXISTS node_untrust (
    session_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    score INTEGER NOT NULL,
    rank INTEGER NOT NULL,
    receipt_count INTEGER NOT NULL,
    untrusted_receipts INTEGER NOT NULL,
    max_decay INTEGER NOT NULL,
    status_counts_json TEXT NOT NULL,
    current_version INTEGER NOT NULL,
    digest TEXT NOT NULL,
    is_stale INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, node_id)
);
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    label TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    is_stale INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS evolutions (
    evolution_id INTEGER PRIMARY KEY AUTOINCREMENT,
    old_session_id TEXT NOT NULL,
    new_session_id TEXT NOT NULL,
    old_spec_version INTEGER NOT NULL,
    new_spec_version INTEGER NOT NULL,
    reason TEXT NOT NULL,
    affected_nodes_json TEXT NOT NULL,
    affected_tasks_json TEXT NOT NULL,
    stale_summary_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_current ON sessions(is_stale, spec_version);
CREATE INDEX IF NOT EXISTS idx_rounds_session ON rounds(session_id, round_seq);
CREATE INDEX IF NOT EXISTS idx_receipts_session ON receipts(session_id, node_id, receipt_key);
CREATE INDEX IF NOT EXISTS idx_snapshots_session ON snapshots(session_id, snapshot_id);
"""


@dataclass
class ActivationResult:
    session_id: str
    action: str
    state: dict[str, Any] | None
    reason: str = ""
    stale_summary: dict[str, Any] | None = None


@dataclass
class SnapshotResult:
    snapshot_id: str
    action: str
    state: dict[str, Any] | None
    reason: str = ""


class Ledger:
    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
        ensure_directory(self.path.parent)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(
            str(self.path), timeout=30.0, isolation_level=None, check_same_thread=False
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=30000")
        with self._lock:
            self.conn.executescript(SCHEMA)
            current = self._meta("ledger_format_version")
            if current is None:
                self._set_meta("ledger_format_version", str(LEDGER_FORMAT_VERSION))
            elif int(current) != LEDGER_FORMAT_VERSION:
                raise LedgerError(
                    "E_LEDGER_FORMAT",
                    "ledger.format_version",
                    f"expected {LEDGER_FORMAT_VERSION}, found {current}",
                )

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def _meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM ledger_meta WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def _set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO ledger_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def current_session(self) -> sqlite3.Row | None:
        with self._lock:
            return self.conn.execute(
                "SELECT * FROM sessions WHERE is_stale=0 ORDER BY spec_version DESC, rowid DESC LIMIT 1"
            ).fetchone()

    @staticmethod
    def _session_id(spec: Spec, suffix: str = "") -> str:
        base = f"s{spec.spec_version}-{spec.digest[:12]}"
        return base + suffix

    def _insert_session(self, session_id: str, spec: Spec, reason: str = "") -> None:
        timestamp = utc_now()
        empty_state = {
            "format_version": LEDGER_FORMAT_VERSION,
            "session_id": session_id,
            "spec_version": spec.spec_version,
            "spec_digest": spec.digest,
            "budget_digest": spec.budget_digest,
            "current_version": spec.current_version,
            "cursor": 0,
            "round_seq": 0,
            "tasks": {},
            "preemptions": [],
            "decisions": [],
            "attempts": [],
            "receipts": [],
            "node_untrust": [],
            "recovery_notes": [],
        }
        state_json = canonical_json(empty_state)
        content_digest = digest_json(empty_state)
        self.conn.execute(
            "INSERT INTO sessions(session_id,spec_version,spec_digest,budget_digest,format_version,is_stale,"
            "created_at,updated_at,content_digest,state_json,spec_json,replacement_reason) "
            "VALUES(?,?,?,?,?,0,?,?,?,?,?,?)",
            (
                session_id,
                spec.spec_version,
                spec.digest,
                spec.budget_digest,
                LEDGER_FORMAT_VERSION,
                timestamp,
                timestamp,
                content_digest,
                state_json,
                canonical_json(spec.raw),
                reason,
            ),
        )

    def _session_stale_summary(self, row: sqlite3.Row) -> dict[str, Any]:
        session_id = str(row["session_id"])
        node_rows = self.conn.execute(
            "SELECT DISTINCT node_id FROM receipts WHERE session_id=? "
            "UNION SELECT DISTINCT node_id FROM node_untrust WHERE session_id=? ORDER BY node_id",
            (session_id, session_id),
        ).fetchall()
        task_rows = self.conn.execute(
            "SELECT task_id FROM tasks WHERE session_id=? ORDER BY task_id", (session_id,)
        ).fetchall()
        counts = {
            "rounds": int(
                self.conn.execute("SELECT COUNT(*) FROM rounds WHERE session_id=?", (session_id,)).fetchone()[0]
            ),
            "preemptions": int(
                self.conn.execute("SELECT COUNT(*) FROM preemptions WHERE session_id=?", (session_id,)).fetchone()[0]
            ),
            "receipts": int(
                self.conn.execute("SELECT COUNT(*) FROM receipts WHERE session_id=?", (session_id,)).fetchone()[0]
            ),
            "snapshots": int(
                self.conn.execute("SELECT COUNT(*) FROM snapshots WHERE session_id=?", (session_id,)).fetchone()[0]
            ),
        }
        current_state = self._decode_state(row)
        untrust = current_state.get("node_untrust", []) if current_state else []
        return {
            "session_id": session_id,
            "spec_version": int(row["spec_version"]),
            "content_digest": str(row["content_digest"]),
            "affected_nodes": [str(item["node_id"]) for item in node_rows],
            "affected_tasks": [str(item["task_id"]) for item in task_rows],
            "conclusion_counts": counts,
            "untrust_ranking": untrust,
            "summary_digest": digest_json(
                {
                    "session_id": session_id,
                    "content_digest": str(row["content_digest"]),
                    "counts": counts,
                    "nodes": [str(item["node_id"]) for item in node_rows],
                    "tasks": [str(item["task_id"]) for item in task_rows],
                }
            ),
        }

    def _archive_session(self, row: sqlite3.Row, reason: str) -> dict[str, Any]:
        session_id = str(row["session_id"])
        summary = self._session_stale_summary(row)
        self.conn.execute("UPDATE sessions SET is_stale=1, replacement_reason=?, updated_at=? WHERE session_id=?", (reason, utc_now(), session_id))
        for table in ("rounds", "tasks", "attempts", "preemptions", "preemption_decisions", "receipts", "node_untrust", "snapshots"):
            self.conn.execute(f"UPDATE {table} SET is_stale=1 WHERE session_id=?", (session_id,))
        summary["reason"] = reason
        return summary

    def activate_spec(self, spec: Spec) -> ActivationResult:
        with self._lock:
            latest = self.conn.execute(
                "SELECT * FROM sessions WHERE is_stale=0 ORDER BY spec_version DESC, rowid DESC LIMIT 1"
            ).fetchone()
            if latest is None:
                session_id = self._session_id(spec)
                existing = self.conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
                if existing is None:
                    self.conn.execute("BEGIN IMMEDIATE")
                    try:
                        self._insert_session(session_id, spec)
                        self.conn.execute("COMMIT")
                    except Exception:
                        self.conn.execute("ROLLBACK")
                        raise
                    return ActivationResult(session_id, "created", None)
                if int(existing["is_stale"]) == 0:
                    return ActivationResult(session_id, "resumed", self._decode_state(existing))
                raise LedgerError("E_SESSION_STALE", "session", "matching stale session has no current replacement")

            if str(latest["spec_digest"]) == spec.digest:
                return ActivationResult(str(latest["session_id"]), "resumed", self._decode_state(latest))

            if spec.spec_version <= int(latest["spec_version"]):
                raise SpecError(
                    "spec_version",
                    f"changed specification must increase spec_version beyond {int(latest['spec_version'])}",
                )

            previous_spec = json.loads(str(latest["spec_json"]))
            previous_configs = {
                int(item["version"]): canonical_json(item) for item in previous_spec.get("configs", [])
            }
            for item in spec.raw.get("configs", []):
                version = int(item["version"])
                if version in previous_configs and canonical_json(item) != previous_configs[version]:
                    raise SpecError(
                        "configs",
                        f"config version {version} was already committed with different content; increment the version",
                    )
            previous_current = max(previous_configs) if previous_configs else 0
            if spec.current_version < previous_current:
                raise SpecError(
                    "configs",
                    f"current config version {spec.current_version} is older than committed version {previous_current}",
                )

            self.conn.execute("BEGIN IMMEDIATE")
            try:
                stale_summary = self._archive_session(latest, "spec_revision_replaced_current_view")
                session_id = self._session_id(spec)
                duplicate = self.conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
                if duplicate is not None:
                    self.conn.execute("UPDATE sessions SET is_stale=1 WHERE session_id=?", (session_id,))
                    suffix_index = 1
                    while True:
                        candidate = f"{session_id}-r{suffix_index}"
                        if self.conn.execute("SELECT 1 FROM sessions WHERE session_id=?", (candidate,)).fetchone() is None:
                            session_id = candidate
                            break
                        suffix_index += 1
                self._insert_session(session_id, spec, "whole_view_replacement")
                self.conn.execute(
                    "INSERT INTO evolutions(old_session_id,new_session_id,old_spec_version,new_spec_version,reason,"
                    "affected_nodes_json,affected_tasks_json,stale_summary_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        str(latest["session_id"]),
                        session_id,
                        int(latest["spec_version"]),
                        spec.spec_version,
                        "spec_revision_replaced_current_view",
                        canonical_json(stale_summary["affected_nodes"]),
                        canonical_json(stale_summary["affected_tasks"]),
                        canonical_json(stale_summary),
                        utc_now(),
                    ),
                )
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
            return ActivationResult(session_id, "replaced", None, "spec_revision_replaced_current_view", stale_summary)

    def _decode_state(self, row: sqlite3.Row) -> dict[str, Any]:
        try:
            state = json.loads(str(row["state_json"]))
        except json.JSONDecodeError as exc:
            raise LedgerError("E_LEDGER_DIGEST", "sessions.state_json", f"invalid state JSON: {exc}") from exc
        actual = digest_json(state)
        expected = str(row["content_digest"])
        if actual != expected:
            raise LedgerError(
                "E_LEDGER_DIGEST",
                "sessions.content_digest",
                f"state content digest mismatch: expected {expected}, got {actual}",
            )
        return state

    def latest_state(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if row is None:
                raise LedgerError("E_SESSION", "session_id", f"unknown session {session_id}")
            state = self._decode_state(row)
            return state if state.get("round_seq", 0) > 0 else None

    def commit_state(self, session_id: str, state: dict[str, Any], kind: str, event: dict[str, Any]) -> int:
        """Commit one scheduler round and all derived rows in one transaction."""

        with self._lock:
            state_copy = json.loads(canonical_json(state))
            round_seq = int(state_copy["round_seq"])
            cursor = int(state_copy["cursor"])
            payload = {"kind": kind, "event": event, "state": state_copy}
            payload_json = canonical_json(payload)
            content_digest = digest_json(payload)
            state_json = canonical_json(state_copy)
            session_digest = digest_json(state_copy)
            timestamp = utc_now()
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                self.conn.execute(
                    "INSERT INTO rounds(session_id,round_seq,cursor,kind,payload_json,content_digest,committed_at,is_stale) "
                    "VALUES(?,?,?,?,?,?,?,0)",
                    (session_id, round_seq, cursor, kind, payload_json, content_digest, timestamp),
                )
                self.conn.execute(
                    "UPDATE sessions SET updated_at=?,content_digest=?,state_json=? WHERE session_id=? AND is_stale=0",
                    (timestamp, session_digest, state_json, session_id),
                )
                self._write_task_rows(session_id, state_copy)
                self._write_attempt_rows(session_id, state_copy)
                self._write_preemptions(session_id, state_copy)
                self._write_decisions(session_id, state_copy)
                self._write_receipts(session_id, state_copy)
                self._write_node_untrust(session_id, state_copy)
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
            return round_seq

    def _write_task_rows(self, session_id: str, state: dict[str, Any]) -> None:
        for task_id in sorted(state.get("tasks", {})):
            task = state["tasks"][task_id]
            self.conn.execute(
                "INSERT INTO tasks(session_id,task_id,priority,config_version,node_id,state,completed_json,"
                "output_summary,wait_ms,used_quota_ms,error_code,is_stale) VALUES(?,?,?,?,?,?,?,?,?,?,?,0) "
                "ON CONFLICT(session_id,task_id) DO UPDATE SET priority=excluded.priority,"
                "config_version=excluded.config_version,node_id=excluded.node_id,state=excluded.state,"
                "completed_json=excluded.completed_json,output_summary=excluded.output_summary,"
                "wait_ms=excluded.wait_ms,used_quota_ms=excluded.used_quota_ms,error_code=excluded.error_code",
                (
                    session_id,
                    task_id,
                    int(task["priority"]),
                    int(task["config_version"]),
                    str(task["node_id"]),
                    str(task["state"]),
                    canonical_json(task.get("completed_shards", [])),
                    str(task.get("output_summary", "")),
                    int(task.get("wait_ms", 0)),
                    int(task.get("used_quota_ms", 0)),
                    str(task.get("error_code", "")),
                ),
            )

    def _write_attempt_rows(self, session_id: str, state: dict[str, Any]) -> None:
        for attempt in state.get("attempts", []):
            self.conn.execute(
                "INSERT INTO attempts(session_id,attempt_key,task_id,shard_index,attempt_no,request_id,error_code,"
                "compute_ms,started_at_ms,finished_at_ms,is_stale) VALUES(?,?,?,?,?,?,?,?,?,?,0) "
                "ON CONFLICT(session_id,attempt_key) DO UPDATE SET error_code=excluded.error_code,"
                "compute_ms=excluded.compute_ms,finished_at_ms=excluded.finished_at_ms",
                (
                    session_id,
                    str(attempt["attempt_key"]),
                    str(attempt["task_id"]),
                    int(attempt["shard_index"]),
                    int(attempt["attempt_no"]),
                    str(attempt["request_id"]),
                    str(attempt.get("error_code", "")),
                    int(attempt.get("compute_ms", 0)),
                    int(attempt["started_at_ms"]),
                    int(attempt["finished_at_ms"]),
                ),
            )

    def _write_preemptions(self, session_id: str, state: dict[str, Any]) -> None:
        for item in state.get("preemptions", []):
            self.conn.execute(
                "INSERT INTO preemptions(session_id,preemption_id,task_id,by_task_id,requested_shard,"
                "suspended_shard,resources_json,status,created_at,is_stale) VALUES(?,?,?,?,?,?,?,?,?,0) "
                "ON CONFLICT(session_id,preemption_id) DO UPDATE SET suspended_shard=excluded.suspended_shard,"
                "resources_json=excluded.resources_json,status=excluded.status",
                (
                    session_id,
                    str(item["preemption_id"]),
                    str(item["task_id"]),
                    str(item["by_task_id"]),
                    int(item["requested_shard"]),
                    int(item["suspended_shard"]),
                    canonical_json(item.get("resources", [])),
                    str(item["status"]),
                    str(item.get("created_at", utc_now())),
                ),
            )

    def _write_decisions(self, session_id: str, state: dict[str, Any]) -> None:
        for item in state.get("decisions", []):
            self.conn.execute(
                "INSERT INTO preemption_decisions(session_id,decision_id,high_task_id,victim_task_id,verdict,"
                "reason,preemptable,details_json,is_stale) VALUES(?,?,?,?,?,?,?,?,0) "
                "ON CONFLICT(session_id,decision_id) DO UPDATE SET verdict=excluded.verdict,reason=excluded.reason,"
                "preemptable=excluded.preemptable,details_json=excluded.details_json",
                (
                    session_id,
                    str(item["decision_id"]),
                    str(item["high_task_id"]),
                    str(item["victim_task_id"]),
                    str(item["verdict"]),
                    str(item["reason"]),
                    1 if item.get("preemptable") else 0,
                    canonical_json(item.get("details", {})),
                ),
            )

    def _write_receipts(self, session_id: str, state: dict[str, Any]) -> None:
        for item in state.get("receipts", []):
            self.conn.execute(
                "INSERT INTO receipts(session_id,receipt_key,node_id,task_id,config_version,shard_index,"
                "shard_count,item_keys_json,input_digest,output_digest,generated_version,decay_count,"
                "untrust_score,status,reason,dependency_refs_json,is_stale) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0) ON CONFLICT(session_id,receipt_key) DO UPDATE SET "
                "decay_count=excluded.decay_count,untrust_score=excluded.untrust_score,status=excluded.status,"
                "reason=excluded.reason,dependency_refs_json=excluded.dependency_refs_json",
                (
                    session_id,
                    str(item["receipt_key"]),
                    str(item["node_id"]),
                    str(item["task_id"]),
                    int(item["config_version"]),
                    int(item["shard_index"]),
                    int(item["shard_count"]),
                    canonical_json(item.get("item_keys", [])),
                    str(item["input_digest"]),
                    str(item["output_summary"]),
                    int(item["generated_version"]),
                    int(item.get("decay_count", 0)),
                    int(item.get("untrust_score", 0)),
                    str(item.get("status", "CURRENT")),
                    str(item.get("reason", "")),
                    canonical_json(item.get("dependency_refs", [])),
                ),
            )

    def _write_node_untrust(self, session_id: str, state: dict[str, Any]) -> None:
        for item in state.get("node_untrust", []):
            self.conn.execute(
                "INSERT INTO node_untrust(session_id,node_id,score,rank,receipt_count,untrusted_receipts,"
                "max_decay,status_counts_json,current_version,digest,is_stale) VALUES(?,?,?,?,?,?,?,?,?,?,0) "
                "ON CONFLICT(session_id,node_id) DO UPDATE SET score=excluded.score,rank=excluded.rank,"
                "receipt_count=excluded.receipt_count,untrusted_receipts=excluded.untrusted_receipts,"
                "max_decay=excluded.max_decay,status_counts_json=excluded.status_counts_json,"
                "current_version=excluded.current_version,digest=excluded.digest",
                (
                    session_id,
                    str(item["node_id"]),
                    int(item["score"]),
                    int(item["rank"]),
                    int(item.get("receipt_count", 0)),
                    int(item.get("untrusted_receipts", 0)),
                    int(item.get("max_decay", 0)),
                    canonical_json(item.get("status_counts", {})),
                    int(item.get("current_version", 0)),
                    str(item.get("digest", "")),
                ),
            )

    def load_rounds(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM rounds WHERE session_id=? AND is_stale=0 ORDER BY round_seq", (session_id,)
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            payload = json.loads(str(row["payload_json"]))
            digest = digest_json(payload)
            if digest != str(row["content_digest"]):
                raise LedgerError(
                    "E_LEDGER_DIGEST",
                    f"rounds[{row['round_seq']}].content_digest",
                    f"expected {row['content_digest']}, got {digest}",
                )
            result.append(
                {
                    "round_seq": int(row["round_seq"]),
                    "cursor": int(row["cursor"]),
                    "kind": str(row["kind"]),
                    "content_digest": digest,
                    "payload": payload,
                }
            )
        return result

    def create_snapshot(self, session_id: str, state: dict[str, Any], label: str) -> dict[str, Any]:
        with self._lock:
            row = self.conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if row is None:
                raise LedgerError("E_SESSION", "session_id", f"unknown session {session_id}")
            spec = json.loads(str(row["spec_json"]))
            payload = {
                "format_version": LEDGER_FORMAT_VERSION,
                "session_id": session_id,
                "spec_digest": str(row["spec_digest"]),
                "spec_version": int(row["spec_version"]),
                "label": label,
                "state": state,
                "spec": spec,
            }
            digest = digest_json(payload)
            count = int(
                self.conn.execute("SELECT COUNT(*) FROM snapshots WHERE session_id=?", (session_id,)).fetchone()[0]
            )
            snapshot_id = f"snap-{count + 1:04d}-{digest[:12]}"
            self.conn.execute(
                "INSERT INTO snapshots(snapshot_id,session_id,label,format_version,content_digest,payload_json,"
                "created_at,is_stale) VALUES(?,?,?,?,?,?,?,0)",
                (snapshot_id, session_id, label, LEDGER_FORMAT_VERSION, digest, canonical_json(payload), utc_now()),
            )
            snapshot_dir = ensure_directory(self.path.parent / "snapshots")
            target = snapshot_dir / f"{snapshot_id}.json"
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
            os.replace(temporary, target)
            return {"snapshot_id": snapshot_id, "content_digest": digest, "session_id": session_id, "label": label}

    def latest_snapshot(self, session_id: str) -> sqlite3.Row | None:
        with self._lock:
            return self.conn.execute(
                "SELECT * FROM snapshots WHERE session_id=? AND is_stale=0 ORDER BY rowid DESC LIMIT 1", (session_id,)
            ).fetchone()

    def _snapshot_payload(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = json.loads(str(row["payload_json"]))
        actual = digest_json(payload)
        expected = str(row["content_digest"])
        if int(row["format_version"]) != LEDGER_FORMAT_VERSION:
            raise LedgerError(
                "E_SNAPSHOT_FORMAT",
                "snapshot.format_version",
                f"expected {LEDGER_FORMAT_VERSION}, found {row['format_version']}",
            )
        if actual != expected:
            raise LedgerError(
                "E_SNAPSHOT_DIGEST",
                "snapshot.content_digest",
                f"expected {expected}, got {actual}",
            )
        return payload

    def load_snapshot(self, snapshot_id: str, spec: Spec | None = None) -> SnapshotResult:
        with self._lock:
            row = self.conn.execute("SELECT * FROM snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone()
            if row is None:
                raise LedgerError("E_SNAPSHOT", "snapshot_id", f"unknown snapshot {snapshot_id}")
            if int(row["is_stale"]):
                return SnapshotResult(snapshot_id, "STALE", None, "snapshot_marked_stale")
            try:
                payload = self._snapshot_payload(row)
                if spec is not None:
                    if str(payload.get("spec_digest")) != spec.digest:
                        raise LedgerError(
                            "E_SNAPSHOT_SPEC",
                            "snapshot.spec_digest",
                            "snapshot belongs to a different specification",
                        )
                state = payload.get("state")
                if not isinstance(state, dict):
                    raise LedgerError("E_SNAPSHOT", "snapshot.state", "snapshot has no state object")
                return SnapshotResult(snapshot_id, "RESTORED", state)
            except LedgerError as exc:
                self.conn.execute("UPDATE snapshots SET is_stale=1 WHERE snapshot_id=?", (snapshot_id,))
                return SnapshotResult(snapshot_id, "REPLACED", None, str(exc))

    def restore_snapshot_or_replace(self, session_id: str, snapshot_id: str, spec: Spec) -> SnapshotResult:
        result = self.load_snapshot(snapshot_id, spec)
        if result.action == "RESTORED":
            return result
        with self._lock:
            row = self.conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if row is None or int(row["is_stale"]):
                return SnapshotResult(snapshot_id, "REPLACED", None, result.reason)
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                summary = self._archive_session(row, "snapshot_version_or_digest_mismatch")
                new_session = self._session_id(spec, f"-snapreset{int(row['spec_version'])}")
                index = 1
                base = new_session
                while self.conn.execute("SELECT 1 FROM sessions WHERE session_id=?", (new_session,)).fetchone() is not None:
                    new_session = f"{base}-{index}"
                    index += 1
                self._insert_session(new_session, spec, "snapshot_mismatch_whole_replacement")
                self.conn.execute(
                    "INSERT INTO evolutions(old_session_id,new_session_id,old_spec_version,new_spec_version,reason,"
                    "affected_nodes_json,affected_tasks_json,stale_summary_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        session_id,
                        new_session,
                        int(row["spec_version"]),
                        spec.spec_version,
                        "snapshot_version_or_digest_mismatch",
                        canonical_json(summary["affected_nodes"]),
                        canonical_json(summary["affected_tasks"]),
                        canonical_json(summary),
                        utc_now(),
                    ),
                )
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
            return SnapshotResult(snapshot_id, "REPLACED", None, result.reason)

    def list_snapshots(self, session_id: str | None = None, include_stale: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            if session_id is None:
                session = self.current_session()
                if session is None:
                    return []
                session_id = str(session["session_id"])
            stale_clause = "" if include_stale else " AND is_stale=0"
            rows = self.conn.execute(
                "SELECT snapshot_id,session_id,label,format_version,content_digest,created_at,is_stale "
                "FROM snapshots WHERE session_id=?" + stale_clause + " ORDER BY snapshot_id",
                (session_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def diagnostic_view(self, session_id: str | None = None) -> dict[str, Any]:
        """Build a read-only, stably ordered view for Tkinter/headless diagnostics."""

        with self._lock:
            if session_id is None:
                session = self.current_session()
            else:
                session = self.conn.execute(
                    "SELECT * FROM sessions WHERE session_id=? AND is_stale=0", (session_id,)
                ).fetchone()
            if session is None:
                return {
                    "current_version": None,
                    "session": None,
                    "priority_queue": [],
                    "preemptions": [],
                    "decisions": [],
                    "receipts": [],
                    "node_untrust": [],
                    "snapshots": [],
                    "flash": {},
                }
            sid = str(session["session_id"])
            tasks = [dict(row) for row in self.conn.execute(
                "SELECT * FROM tasks WHERE session_id=? AND is_stale=0", (sid,)
            ).fetchall()]
            for task in tasks:
                task["completed_shards"] = json.loads(task.pop("completed_json"))
            tasks.sort(key=lambda item: (-int(item["priority"]), str(item["task_id"])))
            preemptions = [dict(row) for row in self.conn.execute(
                "SELECT * FROM preemptions WHERE session_id=? AND is_stale=0 ORDER BY preemption_id", (sid,)
            ).fetchall()]
            for row in preemptions:
                row["resources"] = json.loads(row.pop("resources_json"))
            decisions = [dict(row) for row in self.conn.execute(
                "SELECT * FROM preemption_decisions WHERE session_id=? AND is_stale=0 ORDER BY decision_id", (sid,)
            ).fetchall()]
            for row in decisions:
                row["details"] = json.loads(row.pop("details_json"))
                row["preemptable"] = bool(row["preemptable"])
            receipts = [dict(row) for row in self.conn.execute(
                "SELECT * FROM receipts WHERE session_id=? AND is_stale=0 ORDER BY node_id,task_id,shard_index,receipt_key",
                (sid,),
            ).fetchall()]
            for row in receipts:
                row["item_keys"] = json.loads(row.pop("item_keys_json"))
                row["dependency_refs"] = json.loads(row.pop("dependency_refs_json"))
            untrust = [dict(row) for row in self.conn.execute(
                "SELECT * FROM node_untrust WHERE session_id=? AND is_stale=0 ORDER BY rank,node_id", (sid,)
            ).fetchall()]
            for row in untrust:
                row["status_counts"] = json.loads(row.pop("status_counts_json"))
            snapshots = self.list_snapshots(sid)
            evolutions = [dict(row) for row in self.conn.execute(
                "SELECT * FROM evolutions WHERE old_session_id=? OR new_session_id=? ORDER BY evolution_id", (sid, sid)
            ).fetchall()]
            for row in evolutions:
                row["affected_nodes"] = json.loads(row.pop("affected_nodes_json"))
                row["affected_tasks"] = json.loads(row.pop("affected_tasks_json"))
                row["stale_summary"] = json.loads(row.pop("stale_summary_json"))
            return {
                "current_version": int(json.loads(str(session["state_json"]))["current_version"]),
                "session": {
                    "session_id": sid,
                    "spec_version": int(session["spec_version"]),
                    "spec_digest": str(session["spec_digest"]),
                    "budget_digest": str(session["budget_digest"]),
                    "content_digest": str(session["content_digest"]),
                    "state_json": str(session["state_json"]),
                },
                "priority_queue": tasks,
                "preemptions": preemptions,
                "decisions": decisions,
                "receipts": receipts,
                "node_untrust": untrust,
                "snapshots": snapshots,
                "evolutions": evolutions,
            }

    def current_session_id(self) -> str | None:
        row = self.current_session()
        return None if row is None else str(row["session_id"])

    def stale_evolutions(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute("SELECT * FROM evolutions ORDER BY evolution_id").fetchall()
            result = []
            for row in rows:
                value = dict(row)
                value["stale_summary"] = json.loads(value.pop("stale_summary_json"))
                value["affected_nodes"] = json.loads(value.pop("affected_nodes_json"))
                value["affected_tasks"] = json.loads(value.pop("affected_tasks_json"))
                result.append(value)
            return result
