"""账本持久化与会话快照（需求 7）。

设计：
  - SQLite 账本携带整数格式版本（meta.format_version）与内容摘要（meta.content_digest）。
  - 批次生命周期：先写 pending 行（独立事务，模拟在途），完成时在**同一事务**内
    删除 pending、写入批次完整记录（批次标识、块标识、字节段区间、内容摘要、
    带宽占用、等待时长、误差界）并推进进度游标（progress 表）。
  - 强制终止后重启：仅已完整提交的批次有效；遗留 pending 行标记为 UNKNOWN、
    要求重跑，其占用配额先回收（恢复时从配额占用账本中剔除）。
  - 快照 snapshot.json 记录 format_version / content_digest / spec_version / cursor；
    重新打开时与磁盘当前状态比较，不一致则整体替换（删库重建），不做增量合并。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading

from .errors import E_SNAPSHOT, WorkbenchError

FORMAT_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
  batch_seq INTEGER PRIMARY KEY,
  batch_id TEXT NOT NULL,
  block_id TEXT NOT NULL,
  node_id TEXT NOT NULL,
  start_byte INTEGER NOT NULL,
  end_byte INTEGER NOT NULL,
  digest TEXT NOT NULL,
  bandwidth_kbps INTEGER NOT NULL,
  wait_ms INTEGER NOT NULL,
  err_bound_quanta INTEGER NOT NULL,
  state TEXT NOT NULL,
  spec_version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS pending (
  batch_id TEXT PRIMARY KEY,
  block_id TEXT NOT NULL,
  node_id TEXT NOT NULL,
  start_byte INTEGER NOT NULL,
  end_byte INTEGER NOT NULL,
  bandwidth_kbps INTEGER NOT NULL,
  spec_version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS progress (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  cursor INTEGER NOT NULL,
  consumed_bytes INTEGER NOT NULL,
  spec_version INTEGER NOT NULL
);
"""


class Ledger:
    """SQLite 批次账本（线程安全，完成提交为单事务）。"""

    def __init__(self, db_path: str, content_digest: str, spec_version: int):
        self._path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._init_meta(content_digest, spec_version)

    # ---- 元信息 ----
    def _init_meta(self, content_digest: str, spec_version: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('format_version', ?)",
                (str(FORMAT_VERSION),),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('content_digest', ?)",
                (content_digest,),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('spec_version', ?)",
                (str(spec_version),),
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO progress(id, cursor, consumed_bytes, spec_version)"
                " VALUES(1, 0, 0, ?)",
                (spec_version,),
            )

    def meta(self) -> dict:
        with self._lock:
            rows = self._conn.execute("SELECT key, value FROM meta").fetchall()
        return {k: v for k, v in rows}

    def progress(self) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT cursor, consumed_bytes, spec_version FROM progress WHERE id=1"
            ).fetchone()
        return {"cursor": row[0], "consumed_bytes": row[1], "spec_version": row[2]}

    # ---- 批次生命周期 ----
    def begin_batch(self, record: dict) -> None:
        """登记在途批次（独立事务；崩溃后遗留即未提交）。"""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO pending"
                "(batch_id, block_id, node_id, start_byte, end_byte, bandwidth_kbps, spec_version)"
                " VALUES(?,?,?,?,?,?,?)",
                (
                    record["batch_id"],
                    record["block_id"],
                    record["node_id"],
                    record["start_byte"],
                    record["end_byte"],
                    record["bandwidth_kbps"],
                    record["spec_version"],
                ),
            )

    def commit_batch(self, record: dict, consumed_bytes: int) -> int:
        """批次完成：同一事务内删 pending、写批次、推进进度游标。返回游标。"""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO batches"
                "(batch_id, block_id, node_id, start_byte, end_byte, digest,"
                " bandwidth_kbps, wait_ms, err_bound_quanta, state, spec_version)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record["batch_id"],
                    record["block_id"],
                    record["node_id"],
                    record["start_byte"],
                    record["end_byte"],
                    record["digest"],
                    record["bandwidth_kbps"],
                    record["wait_ms"],
                    record["err_bound_quanta"],
                    record["state"],
                    record["spec_version"],
                ),
            )
            seq = cur.lastrowid
            self._conn.execute("DELETE FROM pending WHERE batch_id=?", (record["batch_id"],))
            self._conn.execute(
                "UPDATE progress SET cursor=?, consumed_bytes=?, spec_version=? WHERE id=1",
                (seq, consumed_bytes, record["spec_version"]),
            )
            return seq

    def committed_batches(self) -> list:
        with self._lock:
            rows = self._conn.execute(
                "SELECT batch_seq, batch_id, block_id, node_id, start_byte, end_byte,"
                " digest, bandwidth_kbps, wait_ms, err_bound_quanta, state, spec_version"
                " FROM batches ORDER BY batch_seq"
            ).fetchall()
        cols = [
            "batch_seq", "batch_id", "block_id", "node_id", "start_byte", "end_byte",
            "digest", "bandwidth_kbps", "wait_ms", "err_bound_quanta", "state",
            "spec_version",
        ]
        return [dict(zip(cols, r)) for r in rows]

    def pending_batches(self) -> list:
        with self._lock:
            rows = self._conn.execute(
                "SELECT batch_id, block_id, node_id, start_byte, end_byte,"
                " bandwidth_kbps, spec_version FROM pending ORDER BY batch_id"
            ).fetchall()
        cols = ["batch_id", "block_id", "node_id", "start_byte", "end_byte",
                "bandwidth_kbps", "spec_version"]
        return [dict(zip(cols, r)) for r in rows]

    def mark_pending_unknown(self) -> list:
        """恢复路径：遗留 pending 标记 UNKNOWN（要求重跑），并清除以便配额回收。"""
        orphans = self.pending_batches()
        with self._lock, self._conn:
            for rec in orphans:
                self._conn.execute(
                    "INSERT INTO batches"
                    "(batch_id, block_id, node_id, start_byte, end_byte, digest,"
                    " bandwidth_kbps, wait_ms, err_bound_quanta, state, spec_version)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        rec["batch_id"], rec["block_id"], rec["node_id"],
                        rec["start_byte"], rec["end_byte"], "UNKNOWN",
                        rec["bandwidth_kbps"], -1, 0, "UNKNOWN", rec["spec_version"],
                    ),
                )
            self._conn.execute("DELETE FROM pending")
        return orphans

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class Snapshot:
    """会话快照：版本识别 + 不一致整体替换。"""

    def __init__(self, workdir: str):
        self._path = os.path.join(workdir, "snapshot.json")

    @property
    def path(self) -> str:
        return self._path

    def write(self, content_digest: str, spec_version: int, cursor: int) -> dict:
        data = {
            "format_version": FORMAT_VERSION,
            "content_digest": content_digest,
            "spec_version": spec_version,
            "cursor": cursor,
        }
        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, sort_keys=True, indent=2)
        os.replace(tmp, self._path)
        return data

    def read(self) -> dict | None:
        if not os.path.exists(self._path):
            return None
        with open(self._path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def check_or_replace(self, content_digest: str, db_path: str) -> str:
        """比较快照版本与内容摘要同磁盘当前状态。

        返回 "fresh"（无快照新建）/ "consistent"（一致可恢复）；
        不一致时整体替换：删除旧账本并返回 "replaced"。
        """
        snap = self.read()
        if snap is None:
            return "fresh"
        mismatch = []
        if snap.get("format_version") != FORMAT_VERSION:
            mismatch.append("format_version")
        if snap.get("content_digest") != content_digest:
            mismatch.append("content_digest")
        ledger_digest = None
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            try:
                row = conn.execute(
                    "SELECT value FROM meta WHERE key='content_digest'"
                ).fetchone()
                ledger_digest = row[0] if row else None
            except sqlite3.Error:
                ledger_digest = None
            finally:
                conn.close()
        if ledger_digest is not None and ledger_digest != content_digest:
            mismatch.append("ledger_content_digest")
        if mismatch:
            for suffix in ("", "-wal", "-shm"):
                if os.path.exists(db_path + suffix):
                    os.remove(db_path + suffix)
            raise WorkbenchError(
                E_SNAPSHOT,
                "快照与磁盘当前状态不一致，已整体替换账本",
                field=",".join(mismatch),
                snapshot=snap,
                expected_digest=content_digest,
            )
        return "consistent"
