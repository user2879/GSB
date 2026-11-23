from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterator

from .errors import LabError


LEDGER_FORMAT_VERSION = 1


class Ledger:
    """SQLite ledger with atomic plan batches and whole-session snapshots."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._ensure_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _ensure_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_meta (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                schema_version INTEGER NOT NULL,
                format_name TEXT NOT NULL
            )
            """
        )
        row = self.connection.execute("SELECT schema_version FROM schema_meta WHERE id=1").fetchone()
        if row is None:
            with self._transaction():
                self.connection.execute(
                    "INSERT INTO schema_meta(id, schema_version, format_name) VALUES(1, ?, ?)",
                    (LEDGER_FORMAT_VERSION, "joinlab-ledger"),
                )
        elif row["schema_version"] != LEDGER_FORMAT_VERSION:
            raise LabError(
                "E_LEDGER_VERSION",
                "账本整数格式版本不兼容",
                field="schema_meta.schema_version",
                expected=LEDGER_FORMAT_VERSION,
                actual=row["schema_version"],
            )

        statements = [
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                query_digest TEXT NOT NULL,
                workload_digest TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                created_ns INTEGER NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                batch_index INTEGER NOT NULL,
                candidate_id TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('RUNNING','COMMITTED','UNKNOWN','FAILED')),
                started_ns INTEGER NOT NULL,
                ended_ns INTEGER,
                error TEXT,
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS batches (
                run_id TEXT NOT NULL,
                batch_index INTEGER NOT NULL,
                batch_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                format_version INTEGER NOT NULL,
                content_digest TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                snapshot_version INTEGER NOT NULL,
                committed_ns INTEGER NOT NULL,
                PRIMARY KEY(run_id, batch_index),
                UNIQUE(batch_id),
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS estimates (
                run_id TEXT NOT NULL,
                batch_index INTEGER NOT NULL,
                candidate_id TEXT NOT NULL,
                total_cost_fx INTEGER NOT NULL,
                estimate_json TEXT NOT NULL,
                PRIMARY KEY(run_id, batch_index, candidate_id),
                FOREIGN KEY(run_id, batch_index) REFERENCES batches(run_id, batch_index) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS selections (
                run_id TEXT NOT NULL,
                batch_index INTEGER NOT NULL,
                best_candidate_id TEXT NOT NULL,
                selection_json TEXT NOT NULL,
                PRIMARY KEY(run_id, batch_index),
                FOREIGN KEY(run_id, batch_index) REFERENCES batches(run_id, batch_index) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS executions (
                run_id TEXT NOT NULL,
                batch_index INTEGER NOT NULL,
                candidate_id TEXT NOT NULL,
                role TEXT NOT NULL,
                raw_digest TEXT NOT NULL,
                grouped_digest TEXT NOT NULL,
                elapsed_ms REAL NOT NULL,
                execution_json TEXT NOT NULL,
                PRIMARY KEY(run_id, batch_index),
                FOREIGN KEY(run_id, batch_index) REFERENCES batches(run_id, batch_index) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS scheduler_tasks (
                run_id TEXT NOT NULL,
                batch_index INTEGER NOT NULL,
                ordinal INTEGER NOT NULL,
                task_id TEXT NOT NULL,
                start_ns INTEGER NOT NULL,
                end_ns INTEGER NOT NULL,
                completed_rows INTEGER NOT NULL,
                peak_memory_bytes INTEGER NOT NULL,
                task_json TEXT NOT NULL,
                PRIMARY KEY(run_id, batch_index, ordinal),
                FOREIGN KEY(run_id, batch_index) REFERENCES batches(run_id, batch_index) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS progress (
                run_id TEXT NOT NULL,
                batch_index INTEGER NOT NULL,
                cursor INTEGER NOT NULL,
                total INTEGER NOT NULL,
                progress_json TEXT NOT NULL,
                PRIMARY KEY(run_id, batch_index),
                FOREIGN KEY(run_id, batch_index) REFERENCES batches(run_id, batch_index) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS snapshots (
                run_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                format_version INTEGER NOT NULL,
                content_digest TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_ns INTEGER NOT NULL,
                PRIMARY KEY(run_id, version),
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS invalidation_events (
                event_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                affected_json TEXT NOT NULL,
                diff_json TEXT NOT NULL,
                created_ns INTEGER NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            )
            """,
        ]
        with self._transaction():
            for statement in statements:
                self.connection.execute(statement)

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        else:
            self.connection.execute("COMMIT")

    def ensure_run(
        self,
        run_id: str,
        *,
        query_digest: str,
        workload_digest: str,
        fingerprint: str,
    ) -> None:
        row = self.connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            with self._transaction():
                self.connection.execute(
                    """
                    INSERT INTO runs(run_id, query_digest, workload_digest, fingerprint, created_ns)
                    VALUES(?,?,?,?,?)
                    """,
                    (run_id, query_digest, workload_digest, fingerprint, time.time_ns()),
                )
            return
        if row["fingerprint"] != fingerprint:
            raise LabError(
                "E_LEDGER",
                "同一 run_id 的运行指纹不一致，拒绝混写账本",
                field="runs.fingerprint",
                expected=row["fingerprint"],
                actual=fingerprint,
            )

    def begin_attempt(self, run_id: str, batch_index: int, candidate_id: str) -> str:
        attempt_id = "A-" + hashlib.sha256(
            f"{run_id}:{batch_index}:{candidate_id}:{time.time_ns()}".encode("utf-8")
        ).hexdigest()[:24]
        with self._transaction():
            self.connection.execute(
                """
                INSERT INTO attempts(attempt_id, run_id, batch_index, candidate_id, status, started_ns)
                VALUES(?,?,?,?,?,?)
                """,
                (attempt_id, run_id, batch_index, candidate_id, "RUNNING", time.time_ns()),
            )
        return attempt_id

    def fail_attempt(self, attempt_id: str, error: str) -> None:
        with self._transaction():
            self.connection.execute(
                "UPDATE attempts SET status='FAILED', ended_ns=?, error=? WHERE attempt_id=? AND status='RUNNING'",
                (time.time_ns(), error, attempt_id),
            )

    def recover_incomplete_attempts(self, run_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM attempts WHERE status='RUNNING'"
        params: tuple[Any, ...] = ()
        if run_id is not None:
            query += " AND run_id=?"
            params = (run_id,)
        rows = self.connection.execute(query + " ORDER BY started_ns, attempt_id", params).fetchall()
        recovered: list[dict[str, Any]] = []
        if rows:
            with self._transaction():
                for row in rows:
                    self.connection.execute(
                        "UPDATE attempts SET status='UNKNOWN', ended_ns=? WHERE attempt_id=? AND status='RUNNING'",
                        (time.time_ns(), row["attempt_id"]),
                    )
                    recovered.append(
                        {
                            "attempt_id": row["attempt_id"],
                            "run_id": row["run_id"],
                            "batch_index": row["batch_index"],
                            "candidate_id": row["candidate_id"],
                            "status": "UNKNOWN",
                            "requires_rerun": True,
                        }
                    )
        return recovered

    def commit_batch(
        self,
        *,
        run_id: str,
        attempt_id: str,
        payload: dict[str, Any],
        snapshot_payload: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = self._normalize_batch_payload(payload)
        payload_json = _canonical_json(normalized)
        content_digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        batch_index = normalized["batch_index"]
        batch_id = normalized["batch_id"]
        candidate_id = normalized["candidate_id"]
        snapshot_version = self._next_snapshot_version(run_id)
        snapshot_json = _canonical_json(snapshot_payload)
        snapshot_digest = hashlib.sha256(snapshot_json.encode("utf-8")).hexdigest()
        now = time.time_ns()
        attempt = self.connection.execute(
            "SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)
        ).fetchone()
        if attempt is None or attempt["status"] != "RUNNING":
            raise LabError(
                "E_LEDGER",
                "批次提交缺少有效的 RUNNING 尝试",
                field="attempts.status",
                expected="RUNNING",
                actual=attempt["status"] if attempt else None,
            )
        if attempt["run_id"] != run_id or attempt["batch_index"] != batch_index:
            raise LabError("E_LEDGER", "尝试与批次身份不匹配", field="attempts.batch_index")

        with self._transaction():
            self.connection.execute(
                """
                INSERT INTO snapshots(run_id, version, format_version, content_digest, payload_json, created_ns)
                VALUES(?,?,?,?,?,?)
                """,
                (run_id, snapshot_version, LEDGER_FORMAT_VERSION, snapshot_digest, snapshot_json, now),
            )
            self.connection.execute(
                """
                INSERT INTO batches(run_id, batch_index, batch_id, candidate_id, format_version,
                                    content_digest, payload_json, snapshot_version, committed_ns)
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    batch_index,
                    batch_id,
                    candidate_id,
                    LEDGER_FORMAT_VERSION,
                    content_digest,
                    payload_json,
                    snapshot_version,
                    now,
                ),
            )
            for estimate in normalized["estimates"]:
                self.connection.execute(
                    """
                    INSERT INTO estimates(run_id, batch_index, candidate_id, total_cost_fx, estimate_json)
                    VALUES(?,?,?,?,?)
                    """,
                    (
                        run_id,
                        batch_index,
                        estimate["candidate_id"],
                        estimate["total_cost_fx"],
                        _canonical_json(estimate),
                    ),
                )
            self.connection.execute(
                """
                INSERT INTO selections(run_id, batch_index, best_candidate_id, selection_json)
                VALUES(?,?,?,?)
                """,
                (
                    run_id,
                    batch_index,
                    normalized["selection"]["best_candidate_id"],
                    _canonical_json(normalized["selection"]),
                ),
            )
            execution = normalized["execution"]
            self.connection.execute(
                """
                INSERT INTO executions(run_id, batch_index, candidate_id, role, raw_digest,
                                       grouped_digest, elapsed_ms, execution_json)
                VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    batch_index,
                    execution["candidate_id"],
                    execution["role"],
                    execution["raw_digest"],
                    execution["grouped_digest"],
                    execution["elapsed_ms"],
                    _canonical_json(execution),
                ),
            )
            for ordinal, task in enumerate(normalized["scheduler_tasks"]):
                self.connection.execute(
                    """
                    INSERT INTO scheduler_tasks(run_id, batch_index, ordinal, task_id, start_ns,
                                                end_ns, completed_rows, peak_memory_bytes, task_json)
                    VALUES(?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        run_id,
                        batch_index,
                        ordinal,
                        task["task_id"],
                        task["start_ns"],
                        task["end_ns"],
                        task["completed_rows"],
                        task["peak_memory_bytes"],
                        _canonical_json(task),
                    ),
                )
            self.connection.execute(
                """
                INSERT INTO progress(run_id, batch_index, cursor, total, progress_json)
                VALUES(?,?,?,?,?)
                """,
                (
                    run_id,
                    batch_index,
                    normalized["progress"]["cursor"],
                    normalized["progress"]["total"],
                    _canonical_json(normalized["progress"]),
                ),
            )
            updated = self.connection.execute(
                "UPDATE attempts SET status='COMMITTED', ended_ns=? WHERE attempt_id=? AND status='RUNNING'",
                (now, attempt_id),
            )
            if updated.rowcount != 1:
                raise LabError("E_LEDGER", "尝试状态提交冲突", field="attempts.status")
        return {
            "batch_id": batch_id,
            "batch_index": batch_index,
            "candidate_id": candidate_id,
            "content_digest": content_digest,
            "snapshot_version": snapshot_version,
            "snapshot_digest": snapshot_digest,
            "format_version": LEDGER_FORMAT_VERSION,
        }

    def committed_batches(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM batches WHERE run_id=? ORDER BY batch_index", (run_id,)
        ).fetchall()
        return [
            {
                "batch_id": row["batch_id"],
                "batch_index": row["batch_index"],
                "candidate_id": row["candidate_id"],
                "content_digest": row["content_digest"],
                "snapshot_version": row["snapshot_version"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def latest_snapshot(self, run_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM snapshots WHERE run_id=? ORDER BY version DESC LIMIT 1", (run_id,)
        ).fetchone()
        return self._snapshot_row(row) if row else None

    def get_snapshot(self, run_id: str, version: int) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM snapshots WHERE run_id=? AND version=?", (run_id, version)
        ).fetchone()
        if row is None:
            raise LabError(
                "E_RECOVERY",
                "指定版本快照不存在",
                field="snapshots.version",
                expected=version,
                actual=None,
            )
        return self._snapshot_row(row)

    def _snapshot_row(self, row: sqlite3.Row) -> dict[str, Any]:
        digest = hashlib.sha256(row["payload_json"].encode("utf-8")).hexdigest()
        if digest != row["content_digest"]:
            raise LabError(
                "E_LEDGER",
                "快照内容摘要不符",
                field="snapshots.content_digest",
                expected=row["content_digest"],
                actual=digest,
                context={"version": row["version"]},
            )
        return {
            "version": row["version"],
            "format_version": row["format_version"],
            "content_digest": row["content_digest"],
            "payload": json.loads(row["payload_json"]),
            "created_ns": row["created_ns"],
        }

    def record_invalidation(
        self,
        *,
        event_id: str,
        run_id: str,
        generation: int,
        affected: dict[str, Any],
        diff: dict[str, Any],
    ) -> None:
        with self._transaction():
            self.connection.execute(
                """
                INSERT INTO invalidation_events(event_id, run_id, generation, affected_json, diff_json, created_ns)
                VALUES(?,?,?,?,?,?)
                """,
                (event_id, run_id, generation, _canonical_json(affected), _canonical_json(diff), time.time_ns()),
            )

    def verify_integrity(self, run_id: str | None = None) -> dict[str, Any]:
        sql = "SELECT * FROM batches"
        params: tuple[Any, ...] = ()
        if run_id is not None:
            sql += " WHERE run_id=?"
            params = (run_id,)
        rows = self.connection.execute(sql + " ORDER BY run_id, batch_index", params).fetchall()
        checked = 0
        for row in rows:
            payload_json = row["payload_json"]
            actual_digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
            if actual_digest != row["content_digest"]:
                raise LabError(
                    "E_LEDGER",
                    "账本批次内容摘要不符",
                    field="batches.content_digest",
                    expected=row["content_digest"],
                    actual=actual_digest,
                    context={"run_id": row["run_id"], "batch_index": row["batch_index"]},
                )
            payload = json.loads(payload_json)
            reconstructed = self._reconstruct_batch(row["run_id"], row["batch_index"])
            if _canonical_json(reconstructed) != payload_json:
                mismatch = _first_json_mismatch(payload, reconstructed, "batch")
                raise LabError(
                    "E_LEDGER",
                    "账本规范化明细与批次摘要内容不一致",
                    field=mismatch["path"],
                    expected=mismatch["expected"],
                    actual=mismatch["actual"],
                    context={"run_id": row["run_id"], "batch_index": row["batch_index"]},
                )
            snapshot_ref = self.connection.execute(
                "SELECT 1 FROM snapshots WHERE run_id=? AND version=?",
                (row["run_id"], row["snapshot_version"]),
            ).fetchone()
            if snapshot_ref is None:
                raise LabError(
                    "E_LEDGER",
                    "批次引用的快照不存在",
                    field="batches.snapshot_version",
                    expected=row["snapshot_version"],
                    actual=None,
                    context={"run_id": row["run_id"], "batch_index": row["batch_index"]},
                )
            checked += 1
        snap_rows = self.connection.execute("SELECT * FROM snapshots ORDER BY run_id, version").fetchall()
        for row in snap_rows:
            actual = hashlib.sha256(row["payload_json"].encode("utf-8")).hexdigest()
            if actual != row["content_digest"]:
                raise LabError(
                    "E_LEDGER",
                    "账本快照内容摘要不符",
                    field="snapshots.content_digest",
                    expected=row["content_digest"],
                    actual=actual,
                    context={"run_id": row["run_id"], "version": row["version"]},
                )
        return {"ok": True, "batches_checked": checked, "snapshots_checked": len(snap_rows)}

    def _reconstruct_batch(self, run_id: str, batch_index: int) -> dict[str, Any]:
        batch = self.connection.execute(
            "SELECT * FROM batches WHERE run_id=? AND batch_index=?", (run_id, batch_index)
        ).fetchone()
        if batch is None:
            raise LabError("E_LEDGER", "批次不存在", field="batches.batch_index")
        estimates = [
            json.loads(row["estimate_json"])
            for row in self.connection.execute(
                "SELECT estimate_json FROM estimates WHERE run_id=? AND batch_index=? ORDER BY candidate_id",
                (run_id, batch_index),
            ).fetchall()
        ]
        selection_row = self.connection.execute(
            "SELECT selection_json FROM selections WHERE run_id=? AND batch_index=?",
            (run_id, batch_index),
        ).fetchone()
        execution_row = self.connection.execute(
            "SELECT execution_json FROM executions WHERE run_id=? AND batch_index=?",
            (run_id, batch_index),
        ).fetchone()
        tasks = [
            json.loads(row["task_json"])
            for row in self.connection.execute(
                "SELECT task_json FROM scheduler_tasks WHERE run_id=? AND batch_index=? ORDER BY ordinal",
                (run_id, batch_index),
            ).fetchall()
        ]
        progress_row = self.connection.execute(
            "SELECT progress_json FROM progress WHERE run_id=? AND batch_index=?",
            (run_id, batch_index),
        ).fetchone()
        return {
            "run_id": run_id,
            "batch_index": batch_index,
            "batch_id": batch["batch_id"],
            "candidate_id": batch["candidate_id"],
            "estimates": estimates,
            "selection": json.loads(selection_row["selection_json"]),
            "execution": json.loads(execution_row["execution_json"]),
            "scheduler_tasks": tasks,
            "progress": json.loads(progress_row["progress_json"]),
        }

    def _next_snapshot_version(self, run_id: str) -> int:
        row = self.connection.execute(
            "SELECT COALESCE(MAX(version), 0) AS version FROM snapshots WHERE run_id=?", (run_id,)
        ).fetchone()
        return int(row["version"]) + 1

    def _normalize_batch_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        required = {
            "run_id",
            "batch_index",
            "batch_id",
            "candidate_id",
            "estimates",
            "selection",
            "execution",
            "scheduler_tasks",
            "progress",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise LabError(
                "E_LEDGER",
                f"批次负载缺少字段 {missing[0]}",
                field=f"batch.{missing[0]}",
            )
        normalized = dict(payload)
        normalized["estimates"] = sorted(payload["estimates"], key=lambda item: item["candidate_id"])
        normalized["scheduler_tasks"] = sorted(
            payload["scheduler_tasks"], key=lambda item: item["task_id"]
        )
        return normalized

    def tamper_for_test(self, kind: str = "estimate_cost") -> None:
        if kind == "estimate_cost":
            row = self.connection.execute(
                "SELECT run_id, batch_index, candidate_id, estimate_json FROM estimates ORDER BY run_id, batch_index, candidate_id LIMIT 1"
            ).fetchone()
            if row is None:
                raise LabError("E_LEDGER", "账本为空，无法注入损坏", field="estimates")
            data = json.loads(row["estimate_json"])
            data["total_cost_fx"] = int(data["total_cost_fx"]) + 1
            with self._transaction():
                self.connection.execute(
                    "UPDATE estimates SET estimate_json=?, total_cost_fx=? WHERE run_id=? AND batch_index=? AND candidate_id=?",
                    (
                        _canonical_json(data),
                        data["total_cost_fx"],
                        row["run_id"],
                        row["batch_index"],
                        row["candidate_id"],
                    ),
                )
            return
        if kind == "batch_digest":
            with self._transaction():
                self.connection.execute(
                    "UPDATE batches SET content_digest=? WHERE rowid=(SELECT MIN(rowid) FROM batches)",
                    ("0" * 64,),
                )
            return
        if kind == "replay_operator":
            batch = self.connection.execute(
                "SELECT * FROM batches ORDER BY run_id, batch_index LIMIT 1"
            ).fetchone()
            if batch is None:
                raise LabError("E_LEDGER", "账本为空，无法注入重放差异", field="batches")
            payload = json.loads(batch["payload_json"])
            target = payload["estimates"][0]
            operator = target["operators"][-1]
            operator["output_rows_fx"] = int(operator["output_rows_fx"]) + 1
            estimate_json = _canonical_json(target)
            payload_json = _canonical_json(payload)
            digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
            with self._transaction():
                self.connection.execute(
                    "UPDATE estimates SET estimate_json=? WHERE run_id=? AND batch_index=? AND candidate_id=?",
                    (
                        estimate_json,
                        batch["run_id"],
                        batch["batch_index"],
                        target["candidate_id"],
                    ),
                )
                self.connection.execute(
                    "UPDATE batches SET payload_json=?, content_digest=? WHERE run_id=? AND batch_index=?",
                    (payload_json, digest, batch["run_id"], batch["batch_index"]),
                )
            return
        raise LabError("E_LEDGER", f"未知损坏注入类型 {kind}", field="tamper.kind")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _first_json_mismatch(expected: Any, actual: Any, path: str) -> dict[str, Any]:
    if type(expected) is not type(actual):
        return {"path": path, "expected": expected, "actual": actual}
    if isinstance(expected, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                return {"path": f"{path}.{key}", "expected": None, "actual": actual[key]}
            if key not in actual:
                return {"path": f"{path}.{key}", "expected": expected[key], "actual": None}
            if expected[key] != actual[key]:
                return _first_json_mismatch(expected[key], actual[key], f"{path}.{key}")
        return {"path": path, "expected": expected, "actual": actual}
    if isinstance(expected, list):
        for index in range(min(len(expected), len(actual))):
            if expected[index] != actual[index]:
                return _first_json_mismatch(expected[index], actual[index], f"{path}[{index}]")
        if len(expected) != len(actual):
            return {"path": f"{path}.length", "expected": len(expected), "actual": len(actual)}
        return {"path": path, "expected": expected, "actual": actual}
    return {"path": path, "expected": expected, "actual": actual}
