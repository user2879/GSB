"""块级扫描账本（本地 SQLite）。

账本与权威重算走完全分离的路径：账本只记录调度器实际执行的任务轨迹与
结果摘要；权威重算（:mod:`colbench.replay`）直接从段文件逐块重放，再与
账本逐字节比较。

账本支持整体替换（``replace_all``）：版本演进或快照恢复时清空旧轨迹，
旧数据先写入只读备份表并返回被清空条数。
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .scheduler import ScheduleResult
from .versioning import ParameterVersion

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS scan_ledger (
    task_id          TEXT NOT NULL,
    version_budget   INTEGER NOT NULL,
    version_encoding INTEGER NOT NULL,
    version_format   INTEGER NOT NULL,
    segment_id       TEXT NOT NULL,
    row_group        INTEGER NOT NULL,
    column           TEXT NOT NULL,
    block_index      INTEGER NOT NULL,
    rows_completed   INTEGER NOT NULL,
    memory_cost      INTEGER NOT NULL,
    wait_ms          INTEGER NOT NULL,
    run_ms           INTEGER NOT NULL,
    result_digest    TEXT NOT NULL,
    status           TEXT NOT NULL,
    error_code       TEXT,
    trajectory_json  TEXT NOT NULL,
    stale            INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (task_id, version_budget, version_encoding, version_format)
);
"""


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    def append_batch(self, result: ScheduleResult,
                     version: ParameterVersion) -> int:
        """逐批写入任务轨迹，返回写入条数。"""
        rows = []
        for task in result.tasks:
            rows.append((
                task.task_id, version.budget, version.encoding,
                version.format, task.segment_id, task.row_group,
                task.column, task.block_index, task.rows_completed,
                task.memory_cost, task.wait_ms, task.run_ms,
                task.result_digest, task.status, task.error_code,
                json.dumps(task.to_dict(), ensure_ascii=False,
                           sort_keys=True), 0))
        self.conn.executemany(
            "INSERT OR REPLACE INTO scan_ledger VALUES (?,?,?,?,?,?,?,?,?,"
            "?,?,?,?,?,?,?,?)", rows)
        self.conn.commit()
        return len(rows)

    def mark_stale(self, current: ParameterVersion) -> int:
        """把不属于当前版本的轨迹全部标记为陈旧。"""
        cursor = self.conn.execute(
            "UPDATE scan_ledger SET stale = 1 WHERE NOT ("
            "version_budget = ? AND version_encoding = ? AND "
            "version_format = ?)",
            (current.budget, current.encoding, current.format))
        self.conn.commit()
        return cursor.rowcount

    def active_rows(self, current: ParameterVersion) -> list[sqlite3.Row]:
        cursor = self.conn.execute(
            "SELECT * FROM scan_ledger WHERE stale = 0 AND "
            "version_budget = ? AND version_encoding = ? AND "
            "version_format = ? ORDER BY segment_id, row_group, column, "
            "block_index",
            (current.budget, current.encoding, current.format))
        return cursor.fetchall()

    def all_rows(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM scan_ledger ORDER BY version_budget, "
            "version_encoding, version_format, segment_id, row_group, "
            "column, block_index").fetchall()

    def count(self, *, stale: int | None = None) -> int:
        if stale is None:
            return self.conn.execute(
                "SELECT COUNT(*) FROM scan_ledger").fetchone()[0]
        return self.conn.execute(
            "SELECT COUNT(*) FROM scan_ledger WHERE stale = ?",
            (stale,)).fetchone()[0]

    def replace_all(self, result: ScheduleResult,
                    version: ParameterVersion) -> dict:
        """整体替换：旧轨迹备份到只读备份表后清空当前账本。"""
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS scan_ledger_backup AS "
            "SELECT * FROM scan_ledger WHERE 0")
        self.conn.execute("INSERT INTO scan_ledger_backup SELECT * "
                          "FROM scan_ledger")
        cleared = self.conn.execute(
            "SELECT COUNT(*) FROM scan_ledger").fetchone()[0]
        self.conn.execute("DELETE FROM scan_ledger")
        self.conn.commit()
        written = self.append_batch(result, version)
        return {"cleared": cleared, "written": written,
                "backup_table": "scan_ledger_backup"}

    def digest(self, current: ParameterVersion) -> str:
        import hashlib
        rows = self.active_rows(current)
        blob = "\n".join("|".join(str(row[column]) for column in
                                  ("task_id", "segment_id", "row_group",
                                   "column", "block_index",
                                   "rows_completed", "memory_cost",
                                   "wait_ms", "run_ms", "result_digest",
                                   "status")) for row in rows)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()
