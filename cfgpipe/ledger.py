"""SQLite 账本：批次记录、回执、失效轨迹、误差界与进度游标。

- 每个下发批次在**同一事务**内写入全部记录并清除在途标记。
- 批次开始时先在自动提交事务中写入 inflight 行；若进程被强制终止，
  重启时残留的 inflight 行即未提交批次，涉及节点标记为 unknown，须重跑。
  （本模块不会、也无法为被强制终止的进程生成退出报告。）
- 账本携带整数格式版本与内容摘要；重新打开时若与磁盘状态不一致，
  整体替换账本文件而非增量合并。
"""

import os
import sqlite3
from dataclasses import dataclass, field

FORMAT_VERSION = 1

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE batches (
    batch_id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    changeset_version INTEGER NOT NULL,
    state TEXT NOT NULL,
    committed_mono_ms INTEGER NOT NULL
);
CREATE TABLE receipts (
    batch_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    source TEXT NOT NULL,          -- "new" | "carried"
    digest TEXT NOT NULL,
    config_version INTEGER NOT NULL,
    basis_version INTEGER,         -- carried 时的依据版本
    elapsed_ms INTEGER NOT NULL,
    attempts INTEGER NOT NULL,
    wait_ms INTEGER NOT NULL,
    PRIMARY KEY (batch_id, node_id)
);
CREATE TABLE invalidation (
    batch_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    reason TEXT NOT NULL,
    PRIMARY KEY (batch_id, step)
);
CREATE TABLE error_bounds (
    batch_id TEXT NOT NULL,
    item_key TEXT NOT NULL,
    trunc_part INTEGER NOT NULL,
    derived_part INTEGER NOT NULL,
    agg_part INTEGER NOT NULL,
    total INTEGER NOT NULL,
    PRIMARY KEY (batch_id, item_key)
);
CREATE TABLE inflight (
    batch_id TEXT PRIMARY KEY,
    node_ids TEXT NOT NULL,        -- 逗号分隔，字典序
    started_mono_ms INTEGER NOT NULL
);
CREATE TABLE cursor (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    value INTEGER NOT NULL
);
"""


@dataclass
class RecoveryInfo:
    replaced: bool = False          # 账本是否因版本/摘要不一致被整体替换
    cursor: int = -1                # 已提交进度游标（批次序号），-1 表示无
    unknown_nodes: list = field(default_factory=list)  # 未提交批次涉及节点
    inflight_batches: list = field(default_factory=list)
    batches: list = field(default_factory=list)        # 已提交批次摘要


class Ledger:
    def __init__(self, path, content_digest):
        self.path = path
        self.content_digest = content_digest
        self.replaced = False
        fresh = not os.path.exists(path)
        self.db = sqlite3.connect(path)
        if not fresh:
            ok = self._check_meta()
            if not ok:
                self.db.close()
                for suffix in ("", "-wal", "-shm", "-journal"):
                    try:
                        os.remove(path + suffix)
                    except OSError:
                        pass
                self.db = sqlite3.connect(path)
                self.replaced = True
                fresh = True
        if fresh:
            self.db.executescript(SCHEMA)
            self.db.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)",
                ("format_version", str(FORMAT_VERSION)))
            self.db.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)",
                ("content_digest", content_digest))
            self.db.execute("INSERT INTO cursor (id, value) VALUES (1, -1)")
            self.db.commit()

    def _check_meta(self):
        try:
            rows = dict(self.db.execute("SELECT key, value FROM meta"))
        except sqlite3.Error:
            return False
        return (rows.get("format_version") == str(FORMAT_VERSION)
                and rows.get("content_digest") == self.content_digest)

    # -- 写入 ---------------------------------------------------------------

    def begin_batch(self, batch_id, node_ids, started_mono_ms):
        """自动提交写入在途标记（独立于批次事务）。"""
        self.db.execute(
            "INSERT OR REPLACE INTO inflight (batch_id, node_ids, started_mono_ms)"
            " VALUES (?, ?, ?)",
            (batch_id, ",".join(sorted(node_ids)), started_mono_ms))
        self.db.commit()

    def commit_batch(self, batch_id, seq, changeset_version, committed_mono_ms,
                     receipts, invalidation_steps, error_bounds):
        """同一事务写入批次记录、回执、失效轨迹、误差界与进度游标。"""
        with self.db:
            self.db.execute(
                "INSERT INTO batches (batch_id, seq, changeset_version, state,"
                " committed_mono_ms) VALUES (?, ?, ?, ?, ?)",
                (batch_id, seq, changeset_version, "committed", committed_mono_ms))
            for r in receipts:
                self.db.execute(
                    "INSERT INTO receipts (batch_id, node_id, source, digest,"
                    " config_version, basis_version, elapsed_ms, attempts, wait_ms)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (batch_id, r["node_id"], r["source"], r["digest"],
                     r["config_version"], r.get("basis_version"),
                     r["elapsed_ms"], r["attempts"], r["wait_ms"]))
            for step in invalidation_steps:
                self.db.execute(
                    "INSERT INTO invalidation (batch_id, step, kind, name, reason)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (batch_id, step.order, step.kind, step.name, step.reason))
            for key, (tr, de, ag, total) in sorted(error_bounds.items()):
                self.db.execute(
                    "INSERT INTO error_bounds (batch_id, item_key, trunc_part,"
                    " derived_part, agg_part, total) VALUES (?, ?, ?, ?, ?, ?)",
                    (batch_id, key, tr, de, ag, total))
            self.db.execute("UPDATE cursor SET value = ? WHERE id = 1", (seq,))
            self.db.execute("DELETE FROM inflight WHERE batch_id = ?", (batch_id,))

    # -- 读取 ---------------------------------------------------------------

    def recovery_scan(self):
        info = RecoveryInfo(replaced=self.replaced)
        row = self.db.execute("SELECT value FROM cursor WHERE id = 1").fetchone()
        info.cursor = row[0] if row else -1
        for batch_id, node_ids in self.db.execute(
                "SELECT batch_id, node_ids FROM inflight ORDER BY batch_id"):
            info.inflight_batches.append(batch_id)
            for nid in node_ids.split(","):
                if nid and nid not in info.unknown_nodes:
                    info.unknown_nodes.append(nid)
        info.unknown_nodes.sort()
        info.batches = [
            {"batch_id": b, "seq": s, "changeset_version": v}
            for b, s, v in self.db.execute(
                "SELECT batch_id, seq, changeset_version FROM batches ORDER BY seq")
        ]
        return info

    def receipts_of(self, batch_id):
        return [
            {"node_id": r[0], "source": r[1], "digest": r[2],
             "config_version": r[3], "basis_version": r[4]}
            for r in self.db.execute(
                "SELECT node_id, source, digest, config_version, basis_version"
                " FROM receipts WHERE batch_id = ? ORDER BY node_id", (batch_id,))
        ]

    def invalidation_of(self, batch_id):
        return [
            {"step": r[0], "kind": r[1], "name": r[2], "reason": r[3]}
            for r in self.db.execute(
                "SELECT step, kind, name, reason FROM invalidation"
                " WHERE batch_id = ? ORDER BY step", (batch_id,))
        ]

    def error_bounds_of(self, batch_id):
        return {
            r[0]: (r[1], r[2], r[3], r[4])
            for r in self.db.execute(
                "SELECT item_key, trunc_part, derived_part, agg_part, total"
                " FROM error_bounds WHERE batch_id = ? ORDER BY item_key",
                (batch_id,))
        }

    def last_receipts(self):
        """每个节点最近一次已提交回执（按批次序号），用于 carried 依据版本。"""
        rows = self.db.execute(
            "SELECT r.node_id, r.digest, r.config_version, r.batch_id, b.seq"
            " FROM receipts r JOIN batches b ON r.batch_id = b.batch_id"
            " ORDER BY b.seq").fetchall()
        latest = {}
        for node_id, digest, cver, batch_id, seq in rows:
            latest[node_id] = {"digest": digest, "config_version": cver,
                               "batch_id": batch_id}
        return latest

    def close(self):
        self.db.close()

    def clear_inflight(self):
        """恢复流程确认 unknown 节点后清除残留在途标记（准备重跑）。"""
        with self.db:
            self.db.execute("DELETE FROM inflight")
