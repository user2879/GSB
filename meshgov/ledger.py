from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .budget import BudgetSnapshot
from .circuit import CircuitRuntime
from .rules import RequestSpec, RuleSet
from .selector import Selection
from .util import canonical_json, mono_ms, utc_now_iso


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS rounds (
    round_id TEXT PRIMARY KEY,
    rule_version TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at_utc TEXT NOT NULL,
    started_mono_ms REAL NOT NULL,
    ended_at_utc TEXT,
    ended_mono_ms REAL,
    global_limit INTEGER NOT NULL,
    input_sha256 TEXT NOT NULL,
    observed_rule_version TEXT,
    rules_changed_during_round INTEGER NOT NULL DEFAULT 0,
    snapshot_id TEXT,
    summary_json TEXT,
    meta_json TEXT
);

CREATE TABLE IF NOT EXISTS requests (
    round_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    request_index INTEGER NOT NULL,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    service TEXT NOT NULL,
    route_prefix TEXT NOT NULL,
    timeout_ms INTEGER NOT NULL,
    max_retries INTEGER NOT NULL,
    status TEXT NOT NULL,
    queued_mono_ms REAL NOT NULL,
    started_mono_ms REAL,
    finished_mono_ms REAL,
    final_status TEXT,
    final_code TEXT,
    attempts_used INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (round_id, request_id),
    FOREIGN KEY(round_id) REFERENCES rounds(round_id)
);

CREATE TABLE IF NOT EXISTS request_checkpoints (
    round_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    request_index INTEGER NOT NULL,
    service TEXT NOT NULL,
    path TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts_completed INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL,
    retry_budget INTEGER NOT NULL,
    queued_mono_ms REAL NOT NULL,
    started_mono_ms REAL,
    updated_mono_ms REAL NOT NULL,
    budget_json TEXT NOT NULL,
    quota_json TEXT NOT NULL,
    final_status TEXT,
    final_code TEXT,
    PRIMARY KEY (round_id, request_id),
    FOREIGN KEY(round_id) REFERENCES rounds(round_id)
);

CREATE TABLE IF NOT EXISTS attempts (
    round_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    attempt_no INTEGER NOT NULL,
    request_index INTEGER NOT NULL,
    service TEXT NOT NULL,
    route_prefix TEXT NOT NULL,
    target_instance TEXT NOT NULL,
    target_port INTEGER NOT NULL,
    selection_ordinal INTEGER NOT NULL,
    selection_algorithm TEXT NOT NULL,
    targeted INTEGER NOT NULL DEFAULT 1,
    selected INTEGER NOT NULL DEFAULT 0,
    dispatched INTEGER NOT NULL DEFAULT 0,
    timeout_ms INTEGER NOT NULL,
    retry_budget INTEGER NOT NULL,
    queue_ms REAL NOT NULL DEFAULT 0,
    start_mono_ms REAL NOT NULL,
    end_mono_ms REAL,
    duration_ms REAL,
    result_status TEXT,
    code TEXT,
    detail TEXT,
    http_status INTEGER,
    global_active_at_start INTEGER,
    service_active_at_start INTEGER,
    permit_start_mono_ms REAL,
    permit_end_mono_ms REAL,
    circuit_admitted_mono_ms REAL,
    complete INTEGER NOT NULL DEFAULT 0,
    created_at_utc TEXT NOT NULL,
    PRIMARY KEY (round_id, request_id, attempt_no),
    FOREIGN KEY(round_id) REFERENCES rounds(round_id)
);

CREATE INDEX IF NOT EXISTS attempts_round_start
    ON attempts(round_id, start_mono_ms);
CREATE INDEX IF NOT EXISTS attempts_request
    ON attempts(round_id, request_id, attempt_no);

CREATE TABLE IF NOT EXISTS circuit_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id TEXT,
    instance TEXT NOT NULL,
    service TEXT NOT NULL,
    host TEXT NOT NULL,
    port INTEGER NOT NULL,
    event TEXT NOT NULL,
    reason TEXT NOT NULL,
    state TEXT NOT NULL,
    opened_at_mono_ms REAL,
    cooldown_until_mono_ms REAL,
    created_at_utc TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS circuit_events_round
    ON circuit_events(round_id, event_id);

CREATE TABLE IF NOT EXISTS circuit_state (
    instance_key TEXT PRIMARY KEY,
    instance TEXT NOT NULL,
    service TEXT NOT NULL,
    host TEXT NOT NULL,
    port INTEGER NOT NULL,
    state TEXT NOT NULL,
    consecutive_failures INTEGER NOT NULL,
    failure_threshold INTEGER NOT NULL,
    cooldown_ms INTEGER NOT NULL,
    opened_at_mono_ms REAL,
    cooldown_until_mono_ms REAL,
    probe_in_flight INTEGER NOT NULL DEFAULT 0,
    updated_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    round_id TEXT NOT NULL,
    rule_version TEXT NOT NULL,
    path TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    readonly INTEGER NOT NULL DEFAULT 1,
    created_at_utc TEXT NOT NULL,
    UNIQUE(round_id)
);

CREATE TABLE IF NOT EXISTS runtime_kv (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL
);
"""


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.path),
            timeout=30,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=FULL")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _write(self, sql: str, params: Iterable[Any] = ()) -> None:
        with self._lock, self._conn:
            self._conn.execute(sql, tuple(params))

    def _rows(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            cursor = self._conn.execute(sql, tuple(params))
            return [dict(row) for row in cursor.fetchall()]

    def begin_round(
        self,
        round_id: str,
        rules: RuleSet,
        input_sha256: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self._write(
            """
            INSERT INTO rounds(
                round_id, rule_version, status, started_at_utc,
                started_mono_ms, global_limit, input_sha256, meta_json
            ) VALUES (?, ?, 'running', ?, ?, ?, ?, ?)
            """,
            (
                round_id,
                rules.rules_version,
                utc_now_iso(),
                mono_ms(),
                rules.global_concurrency,
                input_sha256,
                json.dumps(meta or {}, ensure_ascii=False, sort_keys=True),
            ),
        )

    def queue_request(
        self, round_id: str, request: RequestSpec, queued_mono_ms: float
    ) -> None:
        budget = {
            "max_attempts": request.route.max_retries + 1,
            "retry_budget": request.route.max_retries,
            "timeout_ms": request.route.timeout_ms,
            "per_attempt_timeout_ms": request.route.timeout_ms,
        }
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO requests(
                    round_id, request_id, request_index, method, path, service,
                    route_prefix, timeout_ms, max_retries, status, queued_mono_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?)
                """,
                (
                    round_id,
                    request.request_id,
                    request.index,
                    request.method,
                    request.path,
                    request.route.service,
                    request.route.prefix,
                    request.route.timeout_ms,
                    request.route.max_retries,
                    queued_mono_ms,
                ),
            )
            self._conn.execute(
                """
                INSERT INTO request_checkpoints(
                    round_id, request_id, request_index, service, path, status,
                    attempts_completed, max_attempts, retry_budget,
                    queued_mono_ms, updated_mono_ms, budget_json, quota_json
                ) VALUES (?, ?, ?, ?, ?, 'queued', 0, ?, ?, ?, ?, ?, '{}')
                """,
                (
                    round_id,
                    request.request_id,
                    request.index,
                    request.route.service,
                    request.path,
                    request.route.max_retries + 1,
                    request.route.max_retries,
                    queued_mono_ms,
                    mono_ms(),
                    canonical_json(budget),
                ),
            )

    def request_started(self, round_id: str, request_id: str) -> None:
        now = mono_ms()
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE requests SET status='running', started_mono_ms=?
                WHERE round_id=? AND request_id=?
                """,
                (now, round_id, request_id),
            )
            self._conn.execute(
                """
                UPDATE request_checkpoints SET status='running', started_mono_ms=?,
                    updated_mono_ms=?
                WHERE round_id=? AND request_id=?
                """,
                (now, now, round_id, request_id),
            )

    def persist_budget(self, round_id: str, budget: BudgetSnapshot) -> None:
        payload = {
            "global": {
                "limit": budget.global_limit,
                "active": budget.global_active,
                "peak": budget.global_peak,
            },
            "services": budget.services,
            "saved_mono_ms": mono_ms(),
        }
        self._write(
            """
            INSERT INTO runtime_kv(key, value_json, updated_at_utc)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value_json=excluded.value_json,
                updated_at_utc=excluded.updated_at_utc
            """,
            (f"{round_id}:budget", canonical_json(payload), utc_now_iso()),
        )

    def attempt_started(
        self,
        round_id: str,
        request: RequestSpec,
        attempt_no: int,
        selection: Selection,
        *,
        selected: bool,
        queue_ms: float,
        global_active_at_start: int | None,
        service_active_at_start: int | None,
        permit_start_mono_ms: float | None,
        circuit_admitted_mono_ms: float | None,
        start_mono_ms: float,
    ) -> None:
        self._write(
            """
            INSERT INTO attempts(
                round_id, request_id, attempt_no, request_index, service,
                route_prefix, target_instance, target_port, selection_ordinal,
                selection_algorithm, targeted, selected, dispatched, timeout_ms,
                retry_budget, queue_ms, start_mono_ms, global_active_at_start,
                service_active_at_start, permit_start_mono_ms,
                circuit_admitted_mono_ms, complete, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (
                round_id,
                request.request_id,
                attempt_no,
                request.index,
                request.route.service,
                request.route.prefix,
                selection.target.name,
                selection.target.port,
                selection.ordinal,
                selection.algorithm,
                1 if selected else 0,
                1 if selected else 0,
                request.route.timeout_ms,
                request.route.max_retries,
                queue_ms,
                start_mono_ms,
                global_active_at_start,
                service_active_at_start,
                permit_start_mono_ms,
                circuit_admitted_mono_ms,
                utc_now_iso(),
            ),
        )

    def attempt_permit_released(
        self,
        round_id: str,
        request_id: str,
        attempt_no: int,
        permit_end_mono_ms: float,
    ) -> None:
        self._write(
            """
            UPDATE attempts SET permit_end_mono_ms=?
            WHERE round_id=? AND request_id=? AND attempt_no=?
            """,
            (permit_end_mono_ms, round_id, request_id, attempt_no),
        )

    def attempt_finished(
        self,
        round_id: str,
        request_id: str,
        attempt_no: int,
        *,
        end_mono_ms: float,
        result_status: str,
        code: str | None,
        detail: str,
        http_status: int | None,
        selected: bool,
    ) -> None:
        with self._lock, self._conn:
            row = self._conn.execute(
                """
                SELECT start_mono_ms FROM attempts
                WHERE round_id=? AND request_id=? AND attempt_no=?
                """,
                (round_id, request_id, attempt_no),
            ).fetchone()
            start = float(row["start_mono_ms"]) if row else end_mono_ms
            self._conn.execute(
                """
                UPDATE attempts SET end_mono_ms=?, duration_ms=?, result_status=?,
                    code=?, detail=?, http_status=?, selected=?, dispatched=?,
                    complete=1
                WHERE round_id=? AND request_id=? AND attempt_no=?
                """,
                (
                    end_mono_ms,
                    end_mono_ms - start,
                    result_status,
                    code,
                    detail,
                    http_status,
                    1 if selected else 0,
                    1 if selected else 0,
                    round_id,
                    request_id,
                    attempt_no,
                ),
            )
            attempt_count = self._conn.execute(
                """
                SELECT COUNT(*) AS total FROM attempts
                WHERE round_id=? AND request_id=? AND complete=1
                """,
                (round_id, request_id),
            ).fetchone()["total"]
            self._conn.execute(
                """
                UPDATE request_checkpoints SET attempts_completed=?,
                    updated_mono_ms=?
                WHERE round_id=? AND request_id=?
                """,
                (attempt_count, mono_ms(), round_id, request_id),
            )

    def request_finished(
        self,
        round_id: str,
        request_id: str,
        *,
        final_status: str,
        final_code: str | None,
        attempts_used: int,
        finished_mono_ms: float,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE requests SET status='completed', final_status=?, final_code=?,
                    attempts_used=?, finished_mono_ms=?
                WHERE round_id=? AND request_id=?
                """,
                (
                    final_status,
                    final_code,
                    attempts_used,
                    finished_mono_ms,
                    round_id,
                    request_id,
                ),
            )
            self._conn.execute(
                """
                UPDATE request_checkpoints SET status='completed', final_status=?,
                    final_code=?, attempts_completed=?, updated_mono_ms=?
                WHERE round_id=? AND request_id=?
                """,
                (
                    final_status,
                    final_code,
                    attempts_used,
                    finished_mono_ms,
                    round_id,
                    request_id,
                ),
            )

    def save_circuit_state(self, state: CircuitRuntime) -> None:
        self._write(
            """
            INSERT INTO circuit_state(
                instance_key, instance, service, host, port, state,
                consecutive_failures, failure_threshold, cooldown_ms,
                opened_at_mono_ms, cooldown_until_mono_ms, probe_in_flight,
                updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(instance_key) DO UPDATE SET
                instance=excluded.instance,
                service=excluded.service,
                host=excluded.host,
                port=excluded.port,
                state=excluded.state,
                consecutive_failures=excluded.consecutive_failures,
                failure_threshold=excluded.failure_threshold,
                cooldown_ms=excluded.cooldown_ms,
                opened_at_mono_ms=excluded.opened_at_mono_ms,
                cooldown_until_mono_ms=excluded.cooldown_until_mono_ms,
                probe_in_flight=excluded.probe_in_flight,
                updated_at_utc=excluded.updated_at_utc
            """,
            (
                state.key,
                state.instance,
                state.service,
                state.host,
                state.port,
                state.state,
                state.consecutive_failures,
                state.failure_threshold,
                state.cooldown_ms,
                state.opened_at_mono_ms,
                state.cooldown_until_mono_ms,
                1 if state.probe_in_flight else 0,
                state.updated_at_utc,
            ),
        )

    def add_circuit_event(self, round_id: str, event: dict[str, Any]) -> None:
        self._write(
            """
            INSERT INTO circuit_events(
                round_id, instance, service, host, port, event, reason, state,
                opened_at_mono_ms, cooldown_until_mono_ms, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                round_id,
                event["instance"],
                event["service"],
                event["host"],
                event["port"],
                event["event"],
                event["reason"],
                event["state"],
                event.get("opened_at_mono_ms"),
                event.get("cooldown_until_mono_ms"),
                event["created_at_utc"],
            ),
        )

    def finish_round(
        self,
        round_id: str,
        *,
        status: str,
        observed_rule_version: str,
        rules_changed_during_round: bool,
        summary: dict[str, Any],
    ) -> None:
        self._write(
            """
            UPDATE rounds SET status=?, ended_at_utc=?, ended_mono_ms=?,
                observed_rule_version=?, rules_changed_during_round=?,
                summary_json=?
            WHERE round_id=?
            """,
            (
                status,
                utc_now_iso(),
                mono_ms(),
                observed_rule_version,
                1 if rules_changed_during_round else 0,
                canonical_json(summary),
                round_id,
            ),
        )

    def register_snapshot(
        self,
        snapshot_id: str,
        round_id: str,
        rule_version: str,
        path: str,
        payload_sha256: str,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO snapshots(
                    snapshot_id, round_id, rule_version, path, payload_sha256,
                    readonly, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    snapshot_id,
                    round_id,
                    rule_version,
                    path,
                    payload_sha256,
                    utc_now_iso(),
                ),
            )
            self._conn.execute(
                "UPDATE rounds SET snapshot_id=? WHERE round_id=?",
                (snapshot_id, round_id),
            )

    def snapshot_for_round(self, round_id: str) -> dict[str, Any] | None:
        rows = self._rows(
            "SELECT * FROM snapshots WHERE round_id=?",
            (round_id,),
        )
        return rows[0] if rows else None

    def get_round(self, round_id: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM rounds WHERE round_id=?", (round_id,))
        return rows[0] if rows else None

    def latest_round(self) -> dict[str, Any] | None:
        rows = self._rows(
            "SELECT * FROM rounds ORDER BY started_mono_ms DESC LIMIT 1"
        )
        return rows[0] if rows else None

    def round_data(self, round_id: str) -> dict[str, Any]:
        return {
            "round": self.get_round(round_id),
            "requests": self._rows(
                "SELECT * FROM requests WHERE round_id=? ORDER BY request_index",
                (round_id,),
            ),
            "attempts": self._rows(
                """
                SELECT * FROM attempts WHERE round_id=? AND complete=1
                ORDER BY request_index, attempt_no
                """,
                (round_id,),
            ),
            "incomplete_attempts": self._rows(
                """
                SELECT * FROM attempts WHERE round_id=? AND complete=0
                ORDER BY request_index, attempt_no
                """,
                (round_id,),
            ),
            "circuit_events": self._rows(
                "SELECT * FROM circuit_events WHERE round_id=? ORDER BY event_id",
                (round_id,),
            ),
            "checkpoints": self._rows(
                """
                SELECT * FROM request_checkpoints WHERE round_id=?
                ORDER BY request_index
                """,
                (round_id,),
            ),
        }

    def budget_state(self, round_id: str) -> dict[str, Any] | None:
        rows = self._rows(
            "SELECT value_json FROM runtime_kv WHERE key=?",
            (f"{round_id}:budget",),
        )
        return json.loads(rows[0]["value_json"]) if rows else None

    def current_circuit_states(self) -> dict[str, dict[str, Any]]:
        return {
            row["instance_key"]: row
            for row in self._rows("SELECT * FROM circuit_state ORDER BY service, port")
        }

    def recovery_report(self) -> dict[str, Any]:
        rounds = self._rows(
            """
            SELECT * FROM rounds
            WHERE status NOT IN ('completed','failed','aborted')
               OR (status='completed' AND snapshot_id IS NULL)
            ORDER BY started_mono_ms
            """
        )
        completed_rounds: list[dict[str, Any]] = []
        unknown_items: list[dict[str, Any]] = []
        recovered_items: list[dict[str, Any]] = []
        unknown_attempts: list[dict[str, Any]] = []
        circuit_states = self._rows(
            "SELECT * FROM circuit_state ORDER BY service, port"
        )
        budget_states: list[dict[str, Any]] = []
        for round_row in rounds:
            round_id = round_row["round_id"]
            requests = self._rows(
                """
                SELECT * FROM requests WHERE round_id=? ORDER BY request_index
                """,
                (round_id,),
            )
            for request in requests:
                if request["status"] == "completed":
                    recovered_items.append(
                        {
                            "round_id": round_id,
                            "request_id": request["request_id"],
                            "status": "completed",
                            "final_status": request["final_status"],
                            "final_code": request["final_code"],
                            "attempts_used": request["attempts_used"],
                        }
                    )
                else:
                    checkpoint = self._rows(
                        """
                        SELECT * FROM request_checkpoints
                        WHERE round_id=? AND request_id=?
                        """,
                        (round_id, request["request_id"]),
                    )
                    unknown_items.append(
                        {
                            "round_id": round_id,
                            "request_id": request["request_id"],
                            "request_index": request["request_index"],
                            "status": "unknown_requires_rerun",
                            "persisted_status": request["status"],
                            "attempts_completed": (
                                checkpoint[0]["attempts_completed"] if checkpoint else 0
                            ),
                            "budget": (
                                json.loads(checkpoint[0]["budget_json"])
                                if checkpoint
                                else None
                            ),
                        }
                    )
                    self._write(
                        """
                        UPDATE request_checkpoints SET status='unknown', updated_mono_ms=?
                        WHERE round_id=? AND request_id=? AND status NOT IN ('completed')
                        """,
                        (mono_ms(), round_id, request["request_id"]),
                    )
            for attempt in self._rows(
                """
                SELECT * FROM attempts WHERE round_id=? AND complete=0
                ORDER BY request_index, attempt_no
                """,
                (round_id,),
            ):
                unknown_attempts.append(
                    {
                        "round_id": round_id,
                        "request_id": attempt["request_id"],
                        "attempt_no": attempt["attempt_no"],
                        "target_instance": attempt["target_instance"],
                        "target_port": attempt["target_port"],
                        "status": "unknown_incomplete_attempt",
                    }
                )
            budget = self.budget_state(round_id)
            if budget is not None:
                budget_states.append({"round_id": round_id, **budget})
            completed_rounds.append(
                {
                    "round_id": round_id,
                    "status": round_row["status"],
                    "rule_version": round_row["rule_version"],
                    "started_at_utc": round_row["started_at_utc"],
                    "snapshot_id": round_row.get("snapshot_id"),
                    "missing_snapshot": round_row.get("snapshot_id") is None,
                }
            )
        return {
            "schema_version": 1,
            "generated_at_utc": utc_now_iso(),
            "exit_report_supported": False,
            "reason": "the interrupted round has unknown in-flight outcomes; it must be rerun",
            "interrupted_rounds": completed_rounds,
            "recovered_items": recovered_items,
            "unknown_items": unknown_items,
            "unknown_attempts": unknown_attempts,
            "persisted_circuit_states": circuit_states,
            "persisted_quota_states": budget_states,
        }
