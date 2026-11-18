"""SQLite 账本：持久化调度轨迹、缓存命中、预算占用与层摘要。

写入语义
--------
- 指令进入执行先写 ``running``（占槽位、未提交）；
- 执行/缓存处理结束后在同一事务内写 ``instructions`` 终态并删除 running，
  这是「完整提交」边界。强制终止后 running 中残留的指令即为未提交，
  恢复时只能标记 unknown 并要求重跑。
- 全程使用单一连接 + 线程锁 + WAL，保证跨线程一致与崩溃后可重读。
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Any

LEDGER_FILENAME = "ledger.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS builds (
    build_id TEXT PRIMARY KEY,
    manifest_path TEXT NOT NULL,
    manifest_version TEXT NOT NULL,
    global_slots INTEGER NOT NULL,
    cache_capacity INTEGER NOT NULL,
    started_ms INTEGER NOT NULL,
    ended_ms INTEGER,
    status TEXT NOT NULL,
    report_path TEXT
);
CREATE TABLE IF NOT EXISTS instructions (
    build_id TEXT NOT NULL,
    stage_name TEXT NOT NULL,
    seq INTEGER NOT NULL,
    instruction_id TEXT NOT NULL,
    node_uid TEXT NOT NULL,
    type TEXT NOT NULL,
    params_json TEXT NOT NULL,
    precursor TEXT,
    layer_digest TEXT,
    status TEXT NOT NULL,
    result_code TEXT NOT NULL,
    cache_hit INTEGER NOT NULL,
    exit_code INTEGER,
    stdout_digest TEXT,
    result_digest TEXT,
    wait_start_ms INTEGER,
    wait_end_ms INTEGER,
    wait_elapsed_ms INTEGER,
    elapsed_ms INTEGER,
    skipped_trigger TEXT,
    timeout_residual TEXT,
    timeout_mechanism TEXT,
    PRIMARY KEY (build_id, node_uid)
);
CREATE TABLE IF NOT EXISTS events (
    build_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    node_uid TEXT,
    kind TEXT NOT NULL,
    at_ms INTEGER NOT NULL,
    detail_json TEXT
);
CREATE TABLE IF NOT EXISTS running (
    build_id TEXT NOT NULL,
    node_uid TEXT NOT NULL,
    started_ms INTEGER NOT NULL,
    PRIMARY KEY (build_id, node_uid)
);
CREATE TABLE IF NOT EXISTS evictions (
    build_id TEXT,
    layer_digest TEXT NOT NULL,
    precursor TEXT NOT NULL,
    reason TEXT NOT NULL,
    rule TEXT NOT NULL,
    at_ms INTEGER NOT NULL,
    detail_json TEXT
);
"""


class Ledger:
    def __init__(self, directory: str):
        self.path = os.path.join(directory, LEDGER_FILENAME)
        os.makedirs(directory, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=FULL;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._event_seq = 0

    def checkpoint(self) -> None:
        """把 WAL 合并回主库并截断，保证只复制 ledger.db 即得到完整数据。"""
        with self._lock:
            self._conn.commit()
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.commit()
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            self._conn.close()

    # ---------- 构建级 ----------
    def start_build(self, build_id: str, manifest, started_ms: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO builds VALUES (?,?,?,?,?,?,?,?,?)",
                (build_id, manifest.source_path, manifest.version,
                 manifest.global_slots, manifest.cache_capacity,
                 started_ms, None, "running", None),
            )
            self._conn.commit()

    def finish_build(self, build_id: str, ended_ms: int, status: str,
                     report_path: str | None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE builds SET ended_ms=?, status=?, report_path=? "
                "WHERE build_id=?",
                (ended_ms, status, report_path, build_id),
            )
            self._conn.commit()

    def event(self, build_id: str, kind: str, at_ms: int,
              node_uid: str | None = None, detail: dict | None = None) -> None:
        with self._lock:
            self._event_seq += 1
            self._conn.execute(
                "INSERT INTO events VALUES (?,?,?,?,?,?)",
                (build_id, self._event_seq, node_uid, kind, at_ms,
                 json.dumps(detail or {}, sort_keys=True, ensure_ascii=False)),
            )
            self._conn.commit()

    # ---------- 指令级 ----------
    def mark_running(self, build_id: str, node, started_ms: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO running VALUES (?,?,?)",
                (build_id, node.uid, started_ms),
            )
            self._conn.commit()

    def clear_running(self, build_id: str, node_uid: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM running WHERE build_id=? AND node_uid=?",
                (build_id, node_uid),
            )

    def running_nodes(self, build_id: str) -> list[sqlite3.Row]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM running WHERE build_id=?", (build_id,)
            ).fetchall()
            return list(rows)

    def commit_instruction(self, build_id: str, node, record: dict) -> None:
        """完整提交：写终态指令并清除 running，单事务原子完成。"""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO instructions VALUES ("
                "?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    build_id, node.stage, node.seq, node.instruction.id,
                    node.uid, node.instruction.type,
                    json.dumps(node.instruction.params, sort_keys=True,
                               ensure_ascii=False),
                    record.get("precursor"),
                    record.get("layer_digest"),
                    record.get("status"),
                    record.get("result_code"),
                    1 if record.get("cache_hit") else 0,
                    record.get("exit_code"),
                    record.get("stdout_digest"),
                    record.get("result_digest"),
                    record.get("wait_start_ms"),
                    record.get("wait_end_ms"),
                    record.get("wait_elapsed_ms"),
                    record.get("elapsed_ms"),
                    record.get("skipped_trigger"),
                    record.get("timeout_residual"),
                    record.get("timeout_mechanism"),
                ),
            )
            self._conn.execute(
                "DELETE FROM running WHERE build_id=? AND node_uid=?",
                (build_id, node.uid),
            )
            self._conn.commit()

    def record_eviction(self, build_id: str | None, ev: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO evictions VALUES (?,?,?,?,?,?,?)",
                (build_id, ev["layer_digest"], ev["precursor"], ev["reason"],
                 ev["rule"], ev["at_ms"],
                 json.dumps(ev, sort_keys=True, ensure_ascii=False)),
            )
            self._conn.commit()

    # ---------- 查询 / 恢复 ----------
    def get_build(self, build_id: str) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM builds WHERE build_id=?", (build_id,)
            ).fetchone()

    def get_instructions(self, build_id: str) -> list[sqlite3.Row]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM instructions WHERE build_id=? "
                "ORDER BY rowid", (build_id,)
            ).fetchall()
            return list(rows)

    def get_events(self, build_id: str) -> list[sqlite3.Row]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE build_id=? ORDER BY seq",
                (build_id,),
            ).fetchall()
            return list(rows)

    def get_evictions(self, build_id: str | None = None) -> list[sqlite3.Row]:
        with self._lock:
            if build_id is None:
                rows = self._conn.execute(
                    "SELECT * FROM evictions ORDER BY at_ms, layer_digest"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM evictions WHERE build_id=? "
                    "ORDER BY at_ms, layer_digest", (build_id,)
                ).fetchall()
            return list(rows)

    def completed_node_uids(self, build_id: str) -> set[str]:
        """已完整提交的指令集合（终态：committed/skipped）。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT node_uid, status FROM instructions WHERE build_id=?",
                (build_id,),
            ).fetchall()
            return {r["node_uid"] for r in rows
                    if r["status"] in ("committed", "skipped")}

    def reset_for_replace(self) -> None:
        """整体替换时清空全部业务表（快照恢复路径调用）。"""
        with self._lock:
            for table in ("instructions", "events", "running",
                          "evictions", "builds"):
                self._conn.execute(f"DELETE FROM {table}")
            self._conn.commit()

    def last_finished_build(self) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM builds WHERE status='finished' "
                "ORDER BY started_ms DESC LIMIT 1"
            ).fetchone()
