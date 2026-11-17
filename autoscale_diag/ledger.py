"""SQLite decision ledger with one transaction per completed minute."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .engine import MinuteResult, WorkloadState
from .errors import BusinessError, E_LEDGER, ledger_error
from .spec import ClusterSpec


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    declaration_version TEXT NOT NULL,
    spec_sha256 TEXT NOT NULL,
    policy_json TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    declaration_file TEXT,
    budget INTEGER NOT NULL,
    minute_count INTEGER NOT NULL,
    path TEXT NOT NULL,
    window_size INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS minute_commits (
    run_id TEXT NOT NULL,
    minute_index INTEGER NOT NULL,
    minute INTEGER NOT NULL,
    path TEXT NOT NULL,
    window_size INTEGER NOT NULL,
    peak_memory_estimate_bytes INTEGER NOT NULL,
    budget_used INTEGER NOT NULL,
    budget_json TEXT NOT NULL,
    decision_digest TEXT NOT NULL,
    committed_at TEXT NOT NULL,
    PRIMARY KEY (run_id, minute_index),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS workload_decisions (
    run_id TEXT NOT NULL,
    minute_index INTEGER NOT NULL,
    minute INTEGER NOT NULL,
    workload_id TEXT NOT NULL,
    selected_rule_id TEXT NOT NULL,
    expected_replicas INTEGER NOT NULL,
    previous_replicas INTEGER NOT NULL,
    attempted_replicas INTEGER NOT NULL,
    budget_target_replicas INTEGER NOT NULL,
    final_replicas INTEGER NOT NULL,
    decision_type TEXT NOT NULL,
    budget_clipped INTEGER NOT NULL,
    cooldown_active INTEGER NOT NULL,
    cooldown_suppressed INTEGER NOT NULL,
    cooldown_code TEXT,
    cooldown_trigger_minute INTEGER,
    cooldown_until_minute INTEGER,
    cooldown_message TEXT,
    change_took_effect INTEGER NOT NULL,
    observation_json TEXT NOT NULL,
    stats_json TEXT NOT NULL,
    rule_evaluations_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    decision_json TEXT NOT NULL,
    PRIMARY KEY (run_id, minute_index, workload_id),
    FOREIGN KEY (run_id, minute_index)
        REFERENCES minute_commits(run_id, minute_index) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS cooling_suppressions (
    run_id TEXT NOT NULL,
    minute_index INTEGER NOT NULL,
    workload_id TEXT NOT NULL,
    code TEXT NOT NULL,
    trigger_minute INTEGER,
    attempted_replicas INTEGER NOT NULL,
    final_replicas INTEGER NOT NULL,
    message TEXT NOT NULL,
    PRIMARY KEY (run_id, minute_index, workload_id),
    FOREIGN KEY (run_id, minute_index, workload_id)
        REFERENCES workload_decisions(run_id, minute_index, workload_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS recovery_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    minute_index INTEGER,
    minute INTEGER,
    detail_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (run_id, event_type, minute_index)
);

CREATE INDEX IF NOT EXISTS idx_workload_decisions_run_minute
    ON workload_decisions(run_id, minute_index, workload_id);
CREATE INDEX IF NOT EXISTS idx_cooling_suppressions_run_minute
    ON cooling_suppressions(run_id, minute_index, workload_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=10)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = FULL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.conn.commit()
        finally:
            self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _backup_for_replace(self) -> Path | None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return None
        backup_path = self.path.with_name(self.path.name + ".bak")
        destination = sqlite3.connect(backup_path)
        try:
            self.conn.backup(destination)
            destination.commit()
        finally:
            destination.close()
        return backup_path

    def create_run(
        self,
        spec: ClusterSpec,
        *,
        run_id: str,
        path: str,
        declaration_file: str | None = None,
        replace: bool = False,
    ) -> dict[str, Any]:
        if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for char in run_id):
            raise ledger_error("run_id may contain only letters, digits, dot, underscore and dash")
        existing = self.get_run(run_id)
        backup_path: str | None = None
        if existing is not None:
            if not replace:
                raise ledger_error(
                    "run already exists; use resume or explicit --replace",
                    "run_id",
                    run_id=run_id,
                    status=existing["status"],
                )
            backup = self._backup_for_replace()
            backup_path = str(backup) if backup else None
            with self.conn:
                self.conn.execute("DELETE FROM recovery_events WHERE run_id = ?", (run_id,))
                self.conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        now = _now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO runs (
                    run_id, declaration_version, spec_sha256, policy_json,
                    spec_json, declaration_file, budget, minute_count, path,
                    window_size, status, created_at, updated_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, NULL)
                """,
                (
                    run_id,
                    spec.declaration_version,
                    spec.sha256(),
                    _json(spec.policy_payload()),
                    spec.canonical_json(),
                    declaration_file,
                    spec.global_replica_budget,
                    spec.minute_count,
                    path,
                    spec.window_size,
                    now,
                    now,
                ),
            )
        result = self.get_run(run_id)
        assert result is not None
        if backup_path:
            result["replacement_backup"] = backup_path
        return result

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["policy"] = json.loads(result.pop("policy_json"))
        result["spec"] = json.loads(result.pop("spec_json"))
        return result

    def list_runs(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT run_id, declaration_version, status, budget, minute_count, path, created_at "
            "FROM runs ORDER BY run_id"
        ).fetchall()
        return [dict(row) for row in rows]

    def committed_count(self, run_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM minute_commits WHERE run_id = ?", (run_id,)
        ).fetchone()
        return int(row["n"])

    def committed_indices(self, run_id: str) -> list[int]:
        rows = self.conn.execute(
            "SELECT minute_index FROM minute_commits WHERE run_id = ? ORDER BY minute_index",
            (run_id,),
        ).fetchall()
        return [int(row["minute_index"]) for row in rows]

    def commit_minute(self, run_id: str, result: MinuteResult, total_minutes: int) -> str:
        run = self.get_run(run_id)
        if run is None:
            raise ledger_error("run does not exist", "run_id", run_id=run_id)
        if run["status"] == "complete":
            raise ledger_error("cannot append to a complete run", "run_id", run_id=run_id)
        if result.path != run["path"]:
            raise ledger_error(
                "aggregation path changed inside a run",
                "path",
                expected=run["path"],
                actual=result.path,
            )
        if result.budget_used > run["budget"]:
            raise ledger_error(
                "minute exceeds global replica budget",
                "budget_used",
                budget=run["budget"],
                budget_used=result.budget_used,
            )

        expected_index = self.committed_count(run_id)
        if result.minute_index != expected_index:
            raise ledger_error(
                "minute commit would create a gap or duplicate",
                "minute_index",
                expected=expected_index,
                actual=result.minute_index,
            )
        if expected_index >= total_minutes:
            raise ledger_error("run already contains all expected minutes", "minute_index")
        digest = result.canonical_payload()
        decision_payload = json.dumps(
            digest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        minute_digest = __import__("hashlib").sha256(decision_payload.encode("utf-8")).hexdigest()
        now = _now()
        try:
            with self.conn:
                self.conn.execute(
                    """
                    INSERT INTO minute_commits (
                        run_id, minute_index, minute, path, window_size,
                        peak_memory_estimate_bytes, budget_used, budget_json,
                        decision_digest, committed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        result.minute_index,
                        result.minute,
                        result.path,
                        result.window_size,
                        result.peak_memory_estimate_bytes,
                        result.budget_used,
                        _json(result.budget.to_dict()),
                        minute_digest,
                        now,
                    ),
                )
                for decision in result.decisions:
                    self.conn.execute(
                        """
                        INSERT INTO workload_decisions (
                            run_id, minute_index, minute, workload_id, selected_rule_id,
                            expected_replicas, previous_replicas, attempted_replicas,
                            budget_target_replicas, final_replicas, decision_type,
                            budget_clipped, cooldown_active, cooldown_suppressed,
                            cooldown_code, cooldown_trigger_minute,
                            cooldown_until_minute, cooldown_message,
                            change_took_effect,
                            observation_json, stats_json, rule_evaluations_json,
                            evidence_json, decision_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            decision.minute_index,
                            decision.minute,
                            decision.workload_id,
                            decision.selected_rule_id,
                            decision.expected_replicas,
                            decision.previous_replicas,
                            decision.attempted_replicas,
                            decision.budget_target_replicas,
                            decision.final_replicas,
                            decision.decision_type,
                            int(decision.budget_clipped),
                            int(decision.cooldown_active),
                            int(decision.cooldown_suppressed),
                            decision.cooldown_code,
                            decision.cooldown_trigger_minute,
                            decision.cooldown_until_minute,
                            decision.cooldown_message,
                            int(decision.change_took_effect),
                            _json(decision.observation.to_dict()),
                            _json(decision.window_stats.to_dict()),
                            _json([item.to_dict() for item in decision.rule_evaluations]),
                            _json(list(decision.evidence)),
                            _json(decision.to_dict()),
                        ),
                    )
                    if decision.cooldown_suppressed:
                        self.conn.execute(
                            """
                            INSERT INTO cooling_suppressions (
                                run_id, minute_index, workload_id, code,
                                trigger_minute, attempted_replicas, final_replicas,
                                message
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                run_id,
                                decision.minute_index,
                                decision.workload_id,
                                decision.cooldown_code or "E_COOLDOWN",
                                decision.cooldown_trigger_minute,
                                decision.attempted_replicas,
                                decision.final_replicas,
                                decision.cooldown_message or "E_COOLDOWN",
                            ),
                        )
                self.conn.execute(
                    "UPDATE runs SET updated_at = ? WHERE run_id = ?", (now, run_id)
                )
        except sqlite3.Error as exc:
            raise ledger_error("minute transaction failed and was rolled back", details={"error": str(exc)})
        return minute_digest

    def prepare_resume(self, run_id: str, total_minutes: int) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run is None:
            raise ledger_error("run does not exist", "run_id", run_id=run_id)
        if run["status"] == "complete":
            return run
        indices = self.committed_indices(run_id)
        first_missing: int | None = None
        for expected in range(total_minutes):
            if expected >= len(indices) or indices[expected] != expected:
                first_missing = expected
                break
        now = _now()
        with self.conn:
            if first_missing is not None:
                next_minute = run["spec"]["workloads"][0]["observations"][first_missing]["minute"]
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO recovery_events (
                        run_id, event_type, minute_index, minute, detail_json, created_at
                    ) VALUES (?, 'UNKNOWN_MINUTE_REQUIRES_RERUN', ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        first_missing,
                        next_minute,
                        _json(
                            {
                                "reason": "no committed minute transaction exists",
                                "committed_minutes": len(indices),
                                "action": "rerun from first missing minute",
                            }
                        ),
                        now,
                    ),
                )
            elif len(indices) == total_minutes:
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO recovery_events (
                        run_id, event_type, minute_index, minute, detail_json, created_at
                    ) VALUES (?, 'COMPLETION_UNCONFIRMED', ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        total_minutes,
                        None,
                        _json(
                            {
                                "reason": "all minute rows exist but final run commit is absent",
                                "action": "verify committed prefix before finalizing",
                            }
                        ),
                        now,
                    ),
                )
            self.conn.execute(
                "UPDATE runs SET status = 'interrupted', updated_at = ? WHERE run_id = ?",
                (now, run_id),
            )
        return self.get_run(run_id) or run

    def mark_running(self, run_id: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET status = 'running', updated_at = ? WHERE run_id = ?",
                (_now(), run_id),
            )

    def finalize_run(self, run_id: str, total_minutes: int) -> None:
        run = self.get_run(run_id)
        if run is None:
            raise ledger_error("run does not exist", "run_id", run_id=run_id)
        indices = self.committed_indices(run_id)
        if indices != list(range(total_minutes)):
            raise ledger_error(
                "cannot finalize: committed minutes are incomplete or contain a gap",
                "minute_index",
                committed_count=len(indices),
                expected_count=total_minutes,
            )
        now = _now()
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET status = 'complete', completed_at = ?, updated_at = ? WHERE run_id = ?",
                (now, now, run_id),
            )

    def latest_states(self, run_id: str, workload_ids: Iterable[str]) -> dict[str, WorkloadState]:
        states: dict[str, WorkloadState] = {}
        for workload_id in sorted(workload_ids):
            row = self.conn.execute(
                """
                SELECT final_replicas FROM workload_decisions
                WHERE run_id = ? AND workload_id = ?
                ORDER BY minute_index DESC LIMIT 1
                """,
                (run_id, workload_id),
            ).fetchone()
            change = self.conn.execute(
                """
                SELECT MAX(minute) AS minute FROM workload_decisions
                WHERE run_id = ? AND workload_id = ? AND change_took_effect = 1
                """,
                (run_id, workload_id),
            ).fetchone()
            if row is not None:
                states[workload_id] = WorkloadState(
                    replicas=int(row["final_replicas"]),
                    last_change_minute=(
                        int(change["minute"]) if change and change["minute"] is not None else None
                    ),
                )
        return states

    def load_committed(
        self, run_id: str, *, limit: int | None = None
    ) -> list[dict[str, Any]]:
        sql = (
            "SELECT * FROM minute_commits WHERE run_id = ? "
            "ORDER BY minute_index ASC"
        )
        params: list[Any] = [run_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        headers = [dict(row) for row in self.conn.execute(sql, tuple(params)).fetchall()]
        result: list[dict[str, Any]] = []
        for header in headers:
            rows = self.conn.execute(
                """
                SELECT * FROM workload_decisions
                WHERE run_id = ? AND minute_index = ?
                ORDER BY workload_id ASC
                """,
                (run_id, header["minute_index"]),
            ).fetchall()
            header["budget"] = json.loads(header.pop("budget_json"))
            header["decisions"] = []
            for row in rows:
                record = dict(row)
                record["budget_clipped"] = bool(record["budget_clipped"])
                record["cooldown_active"] = bool(record["cooldown_active"])
                record["cooldown_suppressed"] = bool(record["cooldown_suppressed"])
                record["change_took_effect"] = bool(record["change_took_effect"])
                record["observation"] = json.loads(record.pop("observation_json"))
                record["window_stats"] = json.loads(record.pop("stats_json"))
                record["rule_evaluations"] = json.loads(record.pop("rule_evaluations_json"))
                record["evidence"] = json.loads(record.pop("evidence_json"))
                record["decision"] = json.loads(record.pop("decision_json"))
                header["decisions"].append(record)
            result.append(header)
        return result

    def cooling_suppressions(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT minute_index, workload_id, code, trigger_minute,
                   attempted_replicas, final_replicas, message
            FROM cooling_suppressions WHERE run_id = ?
            ORDER BY minute_index ASC, workload_id ASC
            """,
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def recovery_events(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT event_type, minute_index, minute, detail_json, created_at
            FROM recovery_events WHERE run_id = ? ORDER BY id ASC
            """,
            (run_id,),
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["detail"] = json.loads(item.pop("detail_json"))
            result.append(item)
        return result

    def ledger_version(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run is None:
            raise ledger_error("run does not exist", "run_id", run_id=run_id)
        return {
            "run_id": run_id,
            "declaration_version": run["declaration_version"],
            "spec_sha256": run["spec_sha256"],
            "status": run["status"],
            "path": run["path"],
            "created_at": run["created_at"],
        }
