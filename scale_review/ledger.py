from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from . import LEDGER_FORMAT_VERSION, SNAPSHOT_FORMAT_VERSION
from .canonical import canonical_bytes, sha256_hex
from .errors import ReplayError


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ledger_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS plan_versions (
    version INTEGER PRIMARY KEY,
    change_type TEXT NOT NULL,
    status TEXT NOT NULL,
    spec_digest TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    selection_json TEXT NOT NULL,
    propagation_json TEXT,
    digest TEXT NOT NULL,
    created_wall_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS rounds (
    round_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id TEXT UNIQUE NOT NULL,
    plan_version INTEGER NOT NULL,
    minute INTEGER NOT NULL,
    batch_id TEXT NOT NULL,
    status TEXT NOT NULL,
    input_digest TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    ledger_digest_after TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    started_monotonic_ms INTEGER,
    completed_monotonic_ms INTEGER,
    created_wall_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS minute_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    round_seq INTEGER NOT NULL REFERENCES rounds(round_seq),
    service_id TEXT NOT NULL,
    minute INTEGER NOT NULL,
    record_type TEXT NOT NULL,
    field_digest TEXT NOT NULL,
    canonical_json TEXT NOT NULL,
    UNIQUE(round_seq, service_id, minute, record_type)
);
CREATE TABLE IF NOT EXISTS leases (
    lease_id TEXT PRIMARY KEY,
    round_seq INTEGER NOT NULL REFERENCES rounds(round_seq),
    service_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    status TEXT NOT NULL,
    granted_monotonic_ms INTEGER NOT NULL,
    expires_monotonic_ms INTEGER NOT NULL,
    replica_count INTEGER NOT NULL,
    slots_json TEXT NOT NULL,
    lease_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS quota_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    round_seq INTEGER REFERENCES rounds(round_seq),
    event_seq INTEGER,
    event_type TEXT NOT NULL,
    lease_id TEXT,
    service_id TEXT,
    effective INTEGER NOT NULL,
    event_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS quota_consumption (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    round_seq INTEGER NOT NULL REFERENCES rounds(round_seq),
    scope TEXT NOT NULL,
    target TEXT NOT NULL,
    consumed INTEGER NOT NULL,
    remaining INTEGER NOT NULL,
    canonical_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS wait_spans (
    span_id TEXT PRIMARY KEY,
    round_seq INTEGER NOT NULL REFERENCES rounds(round_seq),
    task_id TEXT NOT NULL,
    service_id TEXT NOT NULL,
    minute INTEGER NOT NULL,
    queued_at_monotonic_ms INTEGER NOT NULL,
    dispatched_at_monotonic_ms INTEGER NOT NULL,
    waited_ms INTEGER NOT NULL,
    wait_reason TEXT NOT NULL,
    lease_id TEXT NOT NULL,
    span_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS progress_cursor (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_committed_round_seq INTEGER NOT NULL,
    last_round_id TEXT NOT NULL,
    last_minute INTEGER NOT NULL,
    minute_count INTEGER NOT NULL,
    plan_version INTEGER NOT NULL,
    updated_wall_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS round_attempts (
    attempt_id TEXT PRIMARY KEY,
    round_id TEXT NOT NULL,
    minute INTEGER NOT NULL,
    plan_version INTEGER NOT NULL,
    status TEXT NOT NULL,
    reserved_json TEXT NOT NULL,
    started_wall_ms INTEGER NOT NULL,
    recovered_wall_ms INTEGER,
    resolution TEXT
);
"""


def wall_ms() -> int:
    return time.time_ns() // 1_000_000


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path != Path(":memory:"):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()
        self._initialize_meta()

    def _initialize_meta(self) -> None:
        with self._lock:
            row = self.conn.execute(
                "SELECT value FROM ledger_meta WHERE key='ledger_format_version'"
            ).fetchone()
            if row is None:
                self.conn.execute(
                    "INSERT INTO ledger_meta(key,value) VALUES('ledger_format_version',?)",
                    (str(LEDGER_FORMAT_VERSION),),
                )
                self.conn.execute(
                    "INSERT INTO ledger_meta(key,value) VALUES('content_digest',?)",
                    (sha256_hex(b"scale-review-ledger-genesis"),),
                )
                self.conn.commit()
            elif int(row["value"]) != LEDGER_FORMAT_VERSION:
                raise ReplayError(
                    "unsupported ledger format version",
                    field="ledger_format_version",
                    details={"found": row["value"], "expected": LEDGER_FORMAT_VERSION},
                )

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def meta(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM ledger_meta WHERE key=?", (key,)
        ).fetchone()
        return None if row is None else row["value"]

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO ledger_meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            self.conn.commit()

    def save_plan_version(self, plan: Any) -> None:
        spec_dict = plan.spec.canonical_dict()
        spec_json = json.dumps(spec_dict, ensure_ascii=False, sort_keys=True)
        record = plan.versions[-1]
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO plan_versions(version,change_type,status,"
                "spec_digest,spec_json,selection_json,propagation_json,digest,created_wall_ms) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    plan.version,
                    record["change_type"],
                    "current",
                    sha256_hex(plan.spec.canonical_bytes()),
                    spec_json,
                    json.dumps(record.get("selection", plan.selection.ledger_basis()), sort_keys=True),
                    json.dumps(record.get("propagation"), sort_keys=True)
                    if record.get("propagation") is not None
                    else None,
                    plan.digest(),
                    wall_ms(),
                ),
            )
            self.conn.execute(
                "UPDATE plan_versions SET status='stale' WHERE version <> ?",
                (plan.version,),
            )
            self.conn.execute(
                "INSERT INTO ledger_meta(key,value) VALUES('latest_plan_version',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(plan.version),),
            )
            self.conn.commit()

    def commit_round(self, payload: dict[str, Any]) -> dict[str, Any]:
        required = [
            "round_id",
            "plan_version",
            "minute",
            "batch_id",
            "spec",
            "minute_records",
            "leases",
            "events",
            "consumption",
            "wait_spans",
            "cursor",
        ]
        for field in required:
            if field not in payload:
                raise ReplayError(f"round payload missing {field}", field=field)

        canonical_payload = canonical_bytes(payload).decode("utf-8")
        payload_digest = sha256_hex(canonical_payload.encode("utf-8"))
        with self._lock:
            conn = self.conn
            try:
                conn.execute("BEGIN IMMEDIATE")
                previous_digest = self.meta("content_digest") or sha256_hex(
                    b"scale-review-ledger-genesis"
                )
                ledger_digest = sha256_hex(
                    (previous_digest + payload_digest).encode("utf-8")
                )
                spec_json = json.dumps(
                    payload["spec"], ensure_ascii=False, sort_keys=True
                )
                cursor_result = conn.execute(
                    "INSERT INTO rounds(round_id,plan_version,minute,batch_id,status,"
                    "input_digest,payload_digest,ledger_digest_after,payload_json,"
                    "spec_json,started_monotonic_ms,completed_monotonic_ms,created_wall_ms) "
                    "VALUES(?,?,?,?, 'committed',?,?,?,?,?,?,?,?)",
                    (
                        payload["round_id"],
                        payload["plan_version"],
                        payload["minute"],
                        payload["batch_id"],
                        payload.get("input_digest", sha256_hex(spec_json)),
                        payload_digest,
                        ledger_digest,
                        canonical_payload,
                        spec_json,
                        payload.get("started_monotonic_ms"),
                        payload.get("completed_monotonic_ms"),
                        wall_ms(),
                    ),
                )
                round_seq = cursor_result.lastrowid

                for record in payload["minute_records"]:
                    record_json = canonical_bytes(record["value"]).decode("utf-8")
                    conn.execute(
                        "INSERT INTO minute_results(round_seq,service_id,minute,"
                        "record_type,field_digest,canonical_json) VALUES(?,?,?,?,?,?)",
                        (
                            round_seq,
                            record["service_id"],
                            record["minute"],
                            record["record_type"],
                            sha256_hex(record_json.encode("utf-8")),
                            record_json,
                        ),
                    )

                for lease in payload["leases"]:
                    lease_dict = lease.to_dict() if hasattr(lease, "to_dict") else dict(lease)
                    conn.execute(
                        "INSERT INTO leases(lease_id,round_seq,service_id,batch_id,"
                        "task_id,version,status,granted_monotonic_ms,"
                        "expires_monotonic_ms,replica_count,slots_json,lease_json) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            lease_dict["lease_id"],
                            round_seq,
                            lease_dict["service_id"],
                            lease_dict["batch_id"],
                            lease_dict["task_id"],
                            lease_dict["version"],
                            lease_dict["status"],
                            lease_dict["granted_at_monotonic_ms"],
                            lease_dict["expires_at_monotonic_ms"],
                            lease_dict["replica_count"],
                            json.dumps(
                                {
                                    "global_slots": lease_dict["global_slots"],
                                    "service_slots": lease_dict["service_slots"],
                                },
                                sort_keys=True,
                            ),
                            json.dumps(lease_dict, sort_keys=True),
                        ),
                    )

                for event in payload["events"]:
                    conn.execute(
                        "INSERT INTO quota_events(round_seq,event_seq,event_type,"
                        "lease_id,service_id,effective,event_json) VALUES(?,?,?,?,?,?,?)",
                        (
                            round_seq,
                            event.get("event_seq"),
                            event.get("event_type", "event"),
                            event.get("lease_id"),
                            event.get("service_id"),
                            1 if event.get("effective") else 0,
                            json.dumps(event, ensure_ascii=False, sort_keys=True),
                        ),
                    )

                for consumption in payload["consumption"]:
                    conn.execute(
                        "INSERT INTO quota_consumption(round_seq,scope,target,"
                        "consumed,remaining,canonical_json) VALUES(?,?,?,?,?,?)",
                        (
                            round_seq,
                            consumption["scope"],
                            consumption["target"],
                            consumption["consumed"],
                            consumption["remaining"],
                            canonical_bytes(consumption).decode("utf-8"),
                        ),
                    )

                for span in payload["wait_spans"]:
                    conn.execute(
                        "INSERT INTO wait_spans(span_id,round_seq,task_id,service_id,"
                        "minute,queued_at_monotonic_ms,dispatched_at_monotonic_ms,"
                        "waited_ms,wait_reason,lease_id,span_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            span["task_id"],
                            round_seq,
                            span["task_id"],
                            span["service_id"],
                            span["minute"],
                            span["queued_at_monotonic_ms"],
                            span["dispatched_at_monotonic_ms"],
                            span["waited_ms"],
                            span["wait_reason"],
                            span["lease_id"],
                            json.dumps(span, sort_keys=True),
                        ),
                    )

                cursor = payload["cursor"]
                conn.execute(
                    "INSERT INTO progress_cursor(id,last_committed_round_seq,"
                    "last_round_id,last_minute,minute_count,plan_version,updated_wall_ms) "
                    "VALUES(1,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                    "last_committed_round_seq=excluded.last_committed_round_seq,"
                    "last_round_id=excluded.last_round_id,"
                    "last_minute=excluded.last_minute,"
                    "minute_count=excluded.minute_count,"
                    "plan_version=excluded.plan_version,"
                    "updated_wall_ms=excluded.updated_wall_ms",
                    (
                        round_seq,
                        payload["round_id"],
                        cursor["last_minute"],
                        cursor["minute_count"],
                        payload["plan_version"],
                        wall_ms(),
                    ),
                )
                conn.execute(
                    "UPDATE ledger_meta SET value=? WHERE key='content_digest'",
                    (ledger_digest,),
                )
                conn.execute(
                    "INSERT INTO ledger_meta(key,value) VALUES('latest_round_seq',?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(round_seq),),
                )
                conn.commit()
                return {
                    "round_seq": round_seq,
                    "round_id": payload["round_id"],
                    "payload_digest": payload_digest,
                    "ledger_digest": ledger_digest,
                }
            except Exception:
                conn.rollback()
                raise

    def begin_attempt(
        self,
        attempt_id: str,
        round_id: str,
        minute: int,
        plan_version: int,
        reserved: dict[str, Any],
    ) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO round_attempts(attempt_id,round_id,minute,"
                "plan_version,status,reserved_json,started_wall_ms) "
                "VALUES(?,?,?,?,'running',?,?)",
                (
                    attempt_id,
                    round_id,
                    minute,
                    plan_version,
                    json.dumps(reserved, sort_keys=True),
                    wall_ms(),
                ),
            )
            self.conn.commit()

    def recover_unknown_attempts(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM round_attempts WHERE status='running' ORDER BY started_wall_ms,attempt_id"
            ).fetchall()
            recovered: list[dict[str, Any]] = []
            for row in rows:
                reserved = json.loads(row["reserved_json"])
                event = {
                    "event_type": "uncommitted_attempt_released",
                    "effective": True,
                    "attempt_id": row["attempt_id"],
                    "round_id": row["round_id"],
                    "minute": row["minute"],
                    "service_id": reserved.get("service_id"),
                    "lease_id": reserved.get("lease_id"),
                    "global_slots": reserved.get("global_slots", []),
                    "service_slots": reserved.get("service_slots", []),
                    "basis": "forced_termination_uncommitted_round_marked_unknown",
                    "lease_version": reserved.get("version"),
                }
                cursor = self.conn.execute("BEGIN IMMEDIATE")
                try:
                    self.conn.execute(
                        "UPDATE round_attempts SET status='unknown',"
                        "recovered_wall_ms=?, resolution=? WHERE attempt_id=?",
                        (
                            wall_ms(),
                            "released_uncommitted_reservation_and_replay_from_last_cursor",
                            row["attempt_id"],
                        ),
                    )
                    self.conn.execute(
                        "INSERT INTO quota_events(round_seq,event_seq,event_type,"
                        "lease_id,service_id,effective,event_json) VALUES(NULL,NULL,?,?,?,?,?)",
                        (
                            "uncommitted_attempt_released",
                            reserved.get("lease_id"),
                            reserved.get("service_id"),
                            1,
                            json.dumps(event, sort_keys=True),
                        ),
                    )
                    self.conn.commit()
                except Exception:
                    self.conn.rollback()
                    raise
                recovered.append(
                    {
                        "attempt_id": row["attempt_id"],
                        "round_id": row["round_id"],
                        "minute": row["minute"],
                        "status": "unknown",
                        "reserved": reserved,
                        "resolution": "released_uncommitted_reservation",
                        "recovery_event": event,
                    }
                )
            return recovered

    def get_cursor(self) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM progress_cursor WHERE id=1").fetchone()
        return None if row is None else dict(row)

    def committed_rounds(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM rounds WHERE status='committed' ORDER BY round_seq"
        ).fetchall()
        return [dict(row) for row in rows]

    def round_rows(self, round_seq: int, table: str) -> list[dict[str, Any]]:
        allowed = {
            "minute_results",
            "leases",
            "quota_events",
            "quota_consumption",
            "wait_spans",
        }
        if table not in allowed:
            raise ValueError("unsafe table")
        rows = self.conn.execute(
            f"SELECT * FROM {table} WHERE round_seq=? ORDER BY rowid", (round_seq,)
        ).fetchall()
        return [dict(row) for row in rows]

    def snapshot_state(self) -> dict[str, Any]:
        cursor = self.get_cursor()
        rounds = self.committed_rounds()
        plans = [
            dict(row)
            for row in self.conn.execute(
                "SELECT version,change_type,status,spec_digest,selection_json,"
                "propagation_json,digest,created_wall_ms FROM plan_versions ORDER BY version"
            ).fetchall()
        ]
        attempts = [
            dict(row)
            for row in self.conn.execute(
                "SELECT attempt_id,round_id,minute,plan_version,status,reserved_json,"
                "started_wall_ms,recovered_wall_ms,resolution FROM round_attempts ORDER BY attempt_id"
            ).fetchall()
        ]
        return {
            "ledger_format_version": LEDGER_FORMAT_VERSION,
            "ledger_content_digest": self.meta("content_digest"),
            "latest_plan_version": int(self.meta("latest_plan_version") or "1"),
            "latest_round_seq": int(self.meta("latest_round_seq") or "0"),
            "cursor": cursor,
            "round_digests": [
                {
                    "round_seq": row["round_seq"],
                    "round_id": row["round_id"],
                    "minute": row["minute"],
                    "payload_digest": row["payload_digest"],
                    "ledger_digest_after": row["ledger_digest_after"],
                }
                for row in rounds
            ],
            "plan_versions": plans,
            "attempts": attempts,
        }

    def write_snapshot(self, path: str | Path) -> dict[str, Any]:
        state = self.snapshot_state()
        envelope = {
            "format": "scale-review-snapshot",
            "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
            "version": state["latest_plan_version"],
            "content_digest": sha256_hex(canonical_bytes(state)),
            "state": state,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return envelope

    def open_or_replace_snapshot(self, path: str | Path) -> tuple[dict[str, Any], bool, str | None]:
        path = Path(path)
        reason: str | None = None
        envelope: dict[str, Any] | None = None
        try:
            candidate = json.loads(path.read_text(encoding="utf-8"))
            state = candidate["state"]
            expected_digest = sha256_hex(canonical_bytes(state))
            if candidate.get("format") != "scale-review-snapshot":
                reason = "snapshot_format_mismatch"
            elif candidate.get("snapshot_format_version") != SNAPSHOT_FORMAT_VERSION:
                reason = "snapshot_version_mismatch"
            elif not isinstance(candidate.get("version"), int):
                reason = "snapshot_plan_version_not_integer"
            elif candidate.get("version") != state.get("latest_plan_version"):
                reason = "snapshot_plan_version_mismatch"
            elif candidate.get("content_digest") != expected_digest:
                reason = "snapshot_digest_mismatch"
            else:
                envelope = candidate
        except FileNotFoundError:
            reason = "snapshot_missing"
        except (KeyError, TypeError, json.JSONDecodeError):
            reason = "snapshot_unreadable"

        if envelope is not None and reason is None:
            return envelope, False, None

        replacement = self.write_snapshot(path)
        replacement["replacement_reason"] = reason or "snapshot_invalid"
        return replacement, True, reason
