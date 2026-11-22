from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from .canonical import canonical_json, content_digest
from .errors import AppError


LEDGER_FORMAT_VERSION = 1


class Ledger:
    """SQLite ledger with per-shard atomic commits and content integrity."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self._create_schema()
        self._initialize_meta()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value
            );
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                plan_digest TEXT NOT NULL,
                plan_json TEXT NOT NULL,
                replacement_json TEXT NOT NULL,
                state TEXT NOT NULL,
                current INTEGER NOT NULL DEFAULT 0,
                stale INTEGER NOT NULL DEFAULT 0,
                created_ms INTEGER NOT NULL,
                finished_ms INTEGER,
                completed_cursor INTEGER NOT NULL DEFAULT 0,
                peak_json TEXT
            );
            CREATE TABLE IF NOT EXISTS run_shards (
                run_id TEXT NOT NULL,
                shard_id TEXT NOT NULL,
                cluster_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                shard_index INTEGER NOT NULL,
                revision_version INTEGER NOT NULL,
                resource_units INTEGER NOT NULL,
                state TEXT NOT NULL,
                PRIMARY KEY (run_id, shard_id)
            );
            CREATE TABLE IF NOT EXISTS shards (
                run_id TEXT NOT NULL,
                shard_id TEXT NOT NULL,
                cluster_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                shard_index INTEGER NOT NULL,
                revision_version INTEGER NOT NULL,
                resource_units INTEGER NOT NULL,
                outcome TEXT NOT NULL,
                aggregate_json TEXT,
                scan_rows INTEGER,
                byte_digest TEXT,
                summary_digest TEXT,
                progress_cursor INTEGER NOT NULL,
                committed_ms INTEGER NOT NULL,
                wait_started_ms INTEGER,
                wait_ended_ms INTEGER,
                started_ms INTEGER,
                finished_ms INTEGER,
                elapsed_ms INTEGER,
                node_pid INTEGER,
                error_code TEXT,
                error_message TEXT,
                stale INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (run_id, shard_id)
            );
            CREATE INDEX IF NOT EXISTS shards_order_idx
                ON shards (run_id, cluster_id, task_id, shard_index);
            """
        )
        self.connection.commit()

    def _meta_value(self, key: str) -> str | None:
        row = self.connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def _initialize_meta(self) -> None:
        with self.connection:
            version = self._meta_value("format_version")
            fresh = version is None
            if version is None:
                self.connection.execute(
                    "INSERT INTO meta(key, value) VALUES('format_version', ?)",
                    (LEDGER_FORMAT_VERSION,),
                )
            elif int(version) != LEDGER_FORMAT_VERSION:
                raise AppError(
                    "E_LEDGER",
                    "账本格式版本不受支持",
                    "format_version",
                    str(self.path),
                    {"found": int(version), "supported": LEDGER_FORMAT_VERSION},
                )
            if self._meta_value("generation") is None:
                self.connection.execute("INSERT INTO meta(key, value) VALUES('generation', 0)")
            if fresh:
                self._write_content_digest()
            elif self._meta_value("content_digest") is None:
                raise AppError("E_LEDGER", "账本缺少内容摘要", "content_digest", str(self.path))

    def _ledger_content(self) -> dict[str, Any]:
        meta = [
            {"key": row["key"], "value": row["value"]}
            for row in self.connection.execute(
                "SELECT key, value FROM meta WHERE key != 'content_digest' ORDER BY key"
            )
        ]
        runs = [dict(row) for row in self.connection.execute("SELECT * FROM runs ORDER BY run_id")]
        run_shards = [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM run_shards ORDER BY run_id, cluster_id, task_id, shard_index, shard_id"
            )
        ]
        shards = [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM shards ORDER BY run_id, cluster_id, task_id, shard_index, shard_id"
            )
        ]
        return {
            "format_version": LEDGER_FORMAT_VERSION,
            "meta": meta,
            "runs": runs,
            "run_shards": run_shards,
            "shards": shards,
        }

    def _write_content_digest(self) -> None:
        digest = content_digest(self._ledger_content())
        self.connection.execute(
            "INSERT INTO meta(key, value) VALUES('content_digest', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (digest,),
        )

    def content_digest(self) -> str:
        value = self._meta_value("content_digest")
        if value is None:
            raise AppError("E_LEDGER", "账本缺少内容摘要", "content_digest", str(self.path))
        return value

    def verify_integrity(self) -> None:
        expected = content_digest(self._ledger_content())
        actual = self.content_digest()
        if expected != actual:
            raise AppError(
                "E_LEDGER",
                "账本内容摘要不一致",
                "content_digest",
                str(self.path),
                {"expected": expected, "actual": actual},
            )

    def begin_replacement(
        self,
        plan: dict[str, Any],
        replacement: dict[str, Any],
    ) -> str:
        now = int(time.time() * 1000)
        with self.connection:
            generation = int(self._meta_value("generation") or "0") + 1
            self.connection.execute(
                "INSERT INTO meta(key, value) VALUES('generation', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (generation,),
            )
            run_id = f"g{generation:04d}-{plan['plan_digest'][:16]}"
            self.connection.execute("UPDATE runs SET current = 0, stale = 1 WHERE current = 1")
            self.connection.execute("UPDATE shards SET stale = 1 WHERE stale = 0")
            self.connection.execute(
                "INSERT INTO runs("
                "run_id, plan_digest, plan_json, replacement_json, state, current, stale, created_ms, completed_cursor"
                ") VALUES(?, ?, ?, ?, 'running', 1, 0, ?, 0)",
                (
                    run_id,
                    plan["plan_digest"],
                    canonical_json(plan),
                    canonical_json(replacement),
                    now,
                ),
            )
            self.connection.executemany(
                "INSERT INTO run_shards("
                "run_id, shard_id, cluster_id, task_id, shard_index, revision_version, resource_units, state"
                ") VALUES(?, ?, ?, ?, ?, ?, ?, 'pending')",
                [
                    (
                        run_id,
                        shard["shard_id"],
                        shard["cluster_id"],
                        shard["task_id"],
                        shard["shard_index"],
                        shard["revision_version"],
                        shard["resource_units"],
                    )
                    for shard in plan.get("shards", [])
                ],
            )
            self._write_content_digest()
        return run_id

    def commit_shard(self, run_id: str, trace: dict[str, Any], shard: dict[str, Any]) -> None:
        aggregate = trace.get("aggregate")
        aggregate_json = canonical_json(aggregate) if aggregate is not None else None
        scan_rows = int(aggregate["scan_rows"]) if aggregate is not None else None
        byte_digest = str(aggregate["byte_digest"]) if aggregate is not None else None
        summary_digest = str(aggregate["summary_digest"]) if aggregate is not None else None
        now = int(time.time() * 1000)
        with self.connection:
            run = self.connection.execute(
                "SELECT completed_cursor FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                raise AppError("E_LEDGER", "账本运行不存在", "run_id", run_id)
            cursor = int(run["completed_cursor"]) + 1
            self.connection.execute(
                "UPDATE runs SET completed_cursor = ? WHERE run_id = ?",
                (cursor, run_id),
            )
            self.connection.execute(
                "INSERT INTO shards("
                "run_id, shard_id, cluster_id, task_id, shard_index, revision_version, resource_units, "
                "outcome, aggregate_json, scan_rows, byte_digest, summary_digest, progress_cursor, committed_ms, "
                "wait_started_ms, wait_ended_ms, started_ms, finished_ms, elapsed_ms, node_pid, error_code, "
                "error_message, stale"
                ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0) "
                "ON CONFLICT(run_id, shard_id) DO UPDATE SET "
                "outcome=excluded.outcome, aggregate_json=excluded.aggregate_json, "
                "scan_rows=excluded.scan_rows, byte_digest=excluded.byte_digest, "
                "summary_digest=excluded.summary_digest, progress_cursor=excluded.progress_cursor, "
                "committed_ms=excluded.committed_ms, wait_started_ms=excluded.wait_started_ms, "
                "wait_ended_ms=excluded.wait_ended_ms, started_ms=excluded.started_ms, "
                "finished_ms=excluded.finished_ms, elapsed_ms=excluded.elapsed_ms, node_pid=excluded.node_pid, "
                "error_code=excluded.error_code, error_message=excluded.error_message, stale=0",
                (
                    run_id,
                    shard["shard_id"],
                    shard["cluster_id"],
                    shard["task_id"],
                    shard["shard_index"],
                    shard["revision_version"],
                    shard["resource_units"],
                    trace["outcome"],
                    aggregate_json,
                    scan_rows,
                    byte_digest,
                    summary_digest,
                    cursor,
                    now,
                    trace.get("wait_started_ms"),
                    trace.get("wait_ended_ms"),
                    trace.get("started_ms"),
                    trace.get("finished_ms"),
                    trace.get("elapsed_ms"),
                    trace.get("node_pid"),
                    trace.get("error_code"),
                    trace.get("error_message"),
                ),
            )
            self.connection.execute(
                "UPDATE run_shards SET state = 'committed' WHERE run_id = ? AND shard_id = ?",
                (run_id, shard["shard_id"]),
            )
            self._write_content_digest()

    def finish_run(self, run_id: str, result: dict[str, Any]) -> None:
        counts = result.get("counts", {})
        state = "completed"
        if counts.get("failed", 0) or counts.get("blocked", 0):
            state = "completed_with_failures"
        with self.connection:
            self.connection.execute(
                "UPDATE runs SET state = ?, finished_ms = ?, peak_json = ? WHERE run_id = ?",
                (state, int(time.time() * 1000), canonical_json(result.get("peak", {})), run_id),
            )
            self._write_content_digest()

    def current_run(self) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM runs WHERE current = 1 AND stale = 0 ORDER BY created_ms DESC LIMIT 1"
        ).fetchone()

    def current_shards(self, include_failed: bool = True) -> list[dict[str, Any]]:
        run = self.current_run()
        if run is None:
            return []
        query = "SELECT * FROM shards WHERE run_id = ? AND stale = 0"
        if not include_failed:
            query += " AND outcome = 'completed'"
        query += " ORDER BY cluster_id, task_id, shard_index, shard_id"
        return [dict(row) for row in self.connection.execute(query, (run["run_id"],))]

    def preview(self) -> dict[str, Any]:
        run = self.current_run()
        if run is None:
            return {
                "run_id": None,
                "state": "empty",
                "groups": [],
                "stale_shards": self.stale_count(),
            }
        rows = {row["shard_id"]: row for row in self.current_shards(include_failed=True)}
        expected = [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM run_shards WHERE run_id = ? "
                "ORDER BY cluster_id, task_id, shard_index, shard_id",
                (run["run_id"],),
            )
        ]
        nested: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for planned in expected:
            row = rows.get(planned["shard_id"])
            if row is None:
                shard_view = {
                    "shard_id": planned["shard_id"],
                    "shard_index": planned["shard_index"],
                    "outcome": "unknown",
                    "scan_rows": None,
                    "byte_digest": None,
                    "summary_digest": None,
                    "aggregate": None,
                    "progress_cursor": None,
                }
            else:
                aggregate = json.loads(row["aggregate_json"]) if row["aggregate_json"] else None
                shard_view = {
                    "shard_id": row["shard_id"],
                    "shard_index": row["shard_index"],
                    "outcome": row["outcome"],
                    "scan_rows": row["scan_rows"],
                    "byte_digest": row["byte_digest"],
                    "summary_digest": row["summary_digest"],
                    "aggregate": aggregate,
                    "progress_cursor": row["progress_cursor"],
                }
            nested.setdefault((planned["cluster_id"], planned["task_id"]), []).append(shard_view)
        groups = []
        for (cluster_id, task_id), shards in sorted(nested.items()):
            completed = sum(1 for shard in shards if shard["outcome"] == "completed")
            unknown = sum(1 for shard in shards if shard["outcome"] == "unknown")
            groups.append(
                {
                    "cluster_id": cluster_id,
                    "task_id": task_id,
                    "progress": {"completed": completed, "unknown": unknown, "total": len(shards)},
                    "shards": sorted(shards, key=lambda item: (item["shard_index"], item["shard_id"])),
                }
            )
        return {
            "run_id": run["run_id"],
            "state": run["state"],
            "plan_digest": run["plan_digest"],
            "completed_cursor": run["completed_cursor"],
            "groups": groups,
            "stale_shards": self.stale_count(),
        }

    def stale_count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS count FROM shards WHERE stale = 1").fetchone()
        return int(row["count"])

    def stale_shards(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM shards WHERE stale = 1 ORDER BY run_id, cluster_id, task_id, shard_index"
            )
        ]

    def recover(self, run_id: str | None = None) -> dict[str, Any]:
        with self.connection:
            if run_id is None:
                row = self.connection.execute(
                    "SELECT * FROM runs WHERE state = 'running' ORDER BY created_ms DESC LIMIT 1"
                ).fetchone()
            else:
                row = self.connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                raise AppError("E_RECOVERY", "没有可恢复的运行", "run_id", run_id or "current")
            actual_run_id = row["run_id"]
            committed = [dict(item) for item in self.connection.execute(
                "SELECT s.* FROM shards s WHERE s.run_id = ? ORDER BY s.cluster_id, s.task_id, s.shard_index, s.shard_id",
                (actual_run_id,),
            )]
            committed_ids = {item["shard_id"] for item in committed}
            expected = [dict(item) for item in self.connection.execute(
                "SELECT * FROM run_shards WHERE run_id = ? ORDER BY cluster_id, task_id, shard_index, shard_id",
                (actual_run_id,),
            )]
            unknown = []
            for item in expected:
                if item["shard_id"] not in committed_ids:
                    unknown.append(item["shard_id"])
                    self.connection.execute(
                        "UPDATE run_shards SET state = 'unknown' WHERE run_id = ? AND shard_id = ?",
                        (actual_run_id, item["shard_id"]),
                    )
            self.connection.execute(
                "UPDATE runs SET state = 'interrupted_recovered' WHERE run_id = ?",
                (actual_run_id,),
            )
            self._write_content_digest()
        return {
            "run_id": actual_run_id,
            "recovered_from": "committed_shard_records_only",
            "committed_shards": [item["shard_id"] for item in committed],
            "unknown_shards": unknown,
            "requires_rerun": sorted(unknown),
            "exit_report_generated": False,
        }

    def run_plan(self, run_id: str | None = None) -> dict[str, Any]:
        if run_id is None:
            row = self.current_run()
        else:
            row = self.connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise AppError("E_LEDGER", "当前没有运行计划", "run_id", run_id or "current")
        return json.loads(row["plan_json"])

    def current_trace(self) -> dict[str, Any]:
        run = self.current_run()
        if run is None:
            return {"run_id": None, "plan_digest": None, "replacement": {}, "traces": [], "unknown_shards": []}
        traces = []
        for row in self.current_shards(include_failed=True):
            record = dict(row)
            record["aggregate"] = json.loads(record["aggregate_json"]) if record["aggregate_json"] else None
            record.pop("aggregate_json", None)
            traces.append(record)
        unknown = [
            row["shard_id"]
            for row in self.connection.execute(
                "SELECT shard_id FROM run_shards WHERE run_id = ? AND state = 'unknown' "
                "ORDER BY cluster_id, task_id, shard_index, shard_id",
                (run["run_id"],),
            )
        ]
        return {
            "run_id": run["run_id"],
            "plan_digest": run["plan_digest"],
            "state": run["state"],
            "replacement": json.loads(run["replacement_json"]),
            "traces": traces,
            "unknown_shards": unknown,
        }

    def current_replacement(self) -> dict[str, Any]:
        run = self.current_run()
        if run is None:
            return {}
        return json.loads(run["replacement_json"])

    def ledger_versions(self) -> dict[str, Any]:
        run = self.current_run()
        return {
            "format_version": LEDGER_FORMAT_VERSION,
            "content_digest": self.content_digest(),
            "run_id": run["run_id"] if run is not None else None,
        }
