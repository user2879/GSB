"""本地 SQLite 路由账本。

一个路由轮次（round）在单个数据库事务内完整提交：轮次元数据、单元结果、
任务轨迹、版本信息要么全部可见，要么全部不可见。强制终止时未 COMMIT 的
事务被 SQLite 自动回滚，重启只能看到已完整提交的轮次。
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

from .determinism import content_digest

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS rounds (
    round_id INTEGER PRIMARY KEY,
    committed_at TEXT NOT NULL,
    status TEXT NOT NULL,
    config_digest TEXT NOT NULL,
    ring_digest TEXT NOT NULL,
    dependency_digest TEXT NOT NULL,
    budget_digest TEXT NOT NULL,
    config_version INTEGER NOT NULL,
    ring_version INTEGER NOT NULL,
    dependency_version INTEGER NOT NULL,
    budget_version INTEGER NOT NULL,
    cache_digest TEXT NOT NULL,
    history_digest TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    config_json TEXT NOT NULL,
    units_json TEXT NOT NULL,
    invalidated_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS unit_results (
    round_id INTEGER NOT NULL,
    unit_id TEXT NOT NULL,
    table_name TEXT NOT NULL,
    route_key TEXT NOT NULL,
    normalized_predicate TEXT NOT NULL,
    is_broadcast INTEGER NOT NULL,
    broadcast_reason TEXT,
    targets_json TEXT NOT NULL,
    summary TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    recomputed INTEGER NOT NULL,
    PRIMARY KEY (round_id, unit_id)
);
CREATE TABLE IF NOT EXISTS task_traces (
    round_id INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    dbs_json TEXT NOT NULL,
    enqueue_ms INTEGER NOT NULL,
    wait_start_ms INTEGER NOT NULL,
    wait_end_ms INTEGER,
    run_start_ms INTEGER,
    run_end_ms INTEGER,
    outcome_code TEXT NOT NULL,
    detail TEXT NOT NULL,
    max_global_running INTEGER NOT NULL,
    max_per_db_running_json TEXT NOT NULL,
    PRIMARY KEY (round_id, task_id)
);
CREATE TABLE IF NOT EXISTS round_events (
    round_id INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    event TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (round_id, seq)
);
"""


class Ledger:
    def __init__(self, path: str) -> None:
        self.path = path
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def last_committed_round_id(self) -> int | None:
        row = self.conn.execute(
            "SELECT MAX(round_id) AS rid FROM rounds WHERE status = 'committed'"
        ).fetchone()
        return row["rid"] if row and row["rid"] is not None else None

    def list_committed_rounds(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM rounds WHERE status = 'committed' ORDER BY round_id"
        ).fetchall()
        return [dict(row) for row in rows]

    def begin_round(self) -> int:
        """登记一个 unknown 轮次占位；未提交即崩溃时重启可见其为未完成。"""
        next_id = self._next_round_id()
        self.conn.execute(
            "INSERT OR REPLACE INTO rounds (round_id, committed_at, status, "
            "config_digest, ring_digest, dependency_digest, budget_digest, "
            "config_version, ring_version, dependency_version, budget_version, "
            "cache_digest, history_digest, payload_digest, config_json, "
            "units_json, invalidated_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                next_id,
                "",
                "unknown",
                "", "", "", "",
                0, 0, 0, 0,
                "", "", "", "{}", "{}", "[]",
            ),
        )
        self.conn.commit()
        return next_id

    def _next_round_id(self) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(round_id), 0) + 1 AS rid FROM rounds"
        ).fetchone()
        return int(row["rid"])

    def commit_round(
        self,
        round_id: int,
        *,
        committed_at: str,
        versions: dict[str, Any],
        cache_digest: str,
        history_digest: str,
        config_json: str,
        units_json: str,
        invalidated: list[str],
        unit_records: list[dict[str, Any]],
        trace_records: list[dict[str, Any]],
        events: list[dict[str, Any]],
    ) -> str:
        payload_digest = content_digest(
            {
                "versions": versions,
                "units": unit_records,
                "traces": trace_records,
                "invalidated": invalidated,
            }
        )
        with self.conn:
            self.conn.execute(
                "UPDATE rounds SET committed_at=?, status='committed', "
                "config_digest=?, ring_digest=?, dependency_digest=?, "
                "budget_digest=?, config_version=?, ring_version=?, "
                "dependency_version=?, budget_version=?, cache_digest=?, "
                "history_digest=?, payload_digest=?, config_json=?, "
                "units_json=?, invalidated_json=? WHERE round_id=?",
                (
                    committed_at,
                    versions["config_digest"],
                    versions["ring_digest"],
                    versions["dependency_digest"],
                    versions["budget_digest"],
                    versions["config_version"],
                    versions["ring_version"],
                    versions["dependency_version"],
                    versions["budget_version"],
                    cache_digest,
                    history_digest,
                    payload_digest,
                    config_json,
                    units_json,
                    json.dumps(invalidated, ensure_ascii=False),
                    round_id,
                ),
            )
            self.conn.executemany(
                "INSERT INTO unit_results (round_id, unit_id, table_name, "
                "route_key, normalized_predicate, is_broadcast, broadcast_reason, "
                "targets_json, summary, plan_json, recomputed) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        round_id,
                        record["unit_id"],
                        record["table"],
                        record["route_key"],
                        record["normalized_predicate"],
                        1 if record["is_broadcast"] else 0,
                        record.get("broadcast_reason"),
                        json.dumps(record["targets"], ensure_ascii=False),
                        record["summary"],
                        json.dumps(record["plan"], ensure_ascii=False),
                        1 if record.get("recomputed") else 0,
                    )
                    for record in unit_records
                ],
            )
            self.conn.executemany(
                "INSERT INTO task_traces (round_id, task_id, unit_id, dbs_json, "
                "enqueue_ms, wait_start_ms, wait_end_ms, run_start_ms, run_end_ms, "
                "outcome_code, detail, max_global_running, max_per_db_running_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        round_id,
                        record["task_id"],
                        record["unit_id"],
                        json.dumps(record["dbs"], ensure_ascii=False),
                        record["enqueue_ms"],
                        record["wait_start_ms"],
                        record["wait_end_ms"],
                        record["run_start_ms"],
                        record["run_end_ms"],
                        record["outcome_code"],
                        record["detail"],
                        record["max_global_running"],
                        json.dumps(
                            record["max_per_db_running"], ensure_ascii=False
                        ),
                    )
                    for record in trace_records
                ],
            )
            self.conn.executemany(
                "INSERT INTO round_events (round_id, seq, event, payload_json) "
                "VALUES (?,?,?,?)",
                [
                    (
                        round_id,
                        index,
                        event["event"],
                        json.dumps(event.get("payload", {}), ensure_ascii=False),
                    )
                    for index, event in enumerate(events)
                ],
            )
        return payload_digest

    def load_round(self, round_id: int) -> dict[str, Any]:
        round_row = self.conn.execute(
            "SELECT * FROM rounds WHERE round_id=?", (round_id,)
        ).fetchone()
        if round_row is None:
            raise KeyError(f"账本中不存在轮次 {round_id}")
        units = [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM unit_results WHERE round_id=? ORDER BY unit_id",
                (round_id,),
            ).fetchall()
        ]
        traces = [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM task_traces WHERE round_id=? ORDER BY task_id",
                (round_id,),
            ).fetchall()
        ]
        events = [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM round_events WHERE round_id=? ORDER BY seq",
                (round_id,),
            ).fetchall()
        ]
        return {
            "round": dict(round_row),
            "units": units,
            "traces": traces,
            "events": events,
        }

    def unknown_rounds(self) -> list[int]:
        rows = self.conn.execute(
            "SELECT round_id FROM rounds WHERE status != 'committed' ORDER BY round_id"
        ).fetchall()
        return [row["round_id"] for row in rows]

    def mark_unknown_for_rerun(self, round_ids: list[int]) -> None:
        with self.conn:
            for round_id in round_ids:
                self.conn.execute(
                    "UPDATE rounds SET status='requires_rerun' WHERE round_id=?",
                    (round_id,),
                )
