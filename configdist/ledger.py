"""SQLite 账本：批次、切片、回执、配额消耗与进度游标。

事务约定：
  * begin_batch 在一个事务中写入批次记录（status=running）与全部切片；
  * commit_batch 在同一事务中写入回执摘要、配额消耗、进度游标并
    将批次置为 committed；
  * 强制终止后重启只能恢复 committed 批次；running 批次一律标记
    为 unknown 并要求重跑，不生成任何退出报告。
"""

import os
import sqlite3
import threading
import time

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
    manifest_version INTEGER NOT NULL,
    manifest_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    committed_ms INTEGER,
    slices_total INTEGER NOT NULL,
    slices_done INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS slices (
    batch_id INTEGER NOT NULL,
    node TEXT NOT NULL,
    key TEXT NOT NULL,
    priority INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    expected_summary TEXT NOT NULL,
    expected_digest TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (batch_id, node, key)
);
CREATE TABLE IF NOT EXISTS receipts (
    batch_id INTEGER NOT NULL,
    node TEXT NOT NULL,
    key TEXT NOT NULL,
    view TEXT NOT NULL,
    summary TEXT,
    digest TEXT NOT NULL,
    elapsed_ms INTEGER NOT NULL,
    PRIMARY KEY (batch_id, node, key)
);
CREATE TABLE IF NOT EXISTS quota (
    batch_id INTEGER NOT NULL,
    node TEXT NOT NULL,
    peak_in_flight INTEGER NOT NULL,
    dispatched INTEGER NOT NULL,
    retries INTEGER NOT NULL,
    PRIMARY KEY (batch_id, node)
);
CREATE TABLE IF NOT EXISTS progress (
    batch_id INTEGER PRIMARY KEY,
    next_index INTEGER NOT NULL
);
"""


def _now_ms():
    return time.monotonic_ns() // 1_000_000


class Ledger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path, isolation_level=None,
                                    check_same_thread=False)
        self._lock = threading.RLock()
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.executescript(_SCHEMA)

    @staticmethod
    def _locked(fn):
        def wrapper(self, *args, **kwargs):
            with self._lock:
                return fn(self, *args, **kwargs)
        return wrapper

    def close(self):
        self.conn.close()

    def get_meta(self, key, default=None):
        row = self.conn.execute("SELECT value FROM meta WHERE key=?",
                                (key,)).fetchone()
        return row[0] if row else default

    @_locked
    def set_meta(self, key, value):
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value))

    @_locked
    def begin_batch(self, manifest_version, manifest_json, slices):
        """slices: [{node,key,priority,payload_json,expected_summary,
        expected_digest}]；单事务写入批次与全部切片。"""
        cur = self.conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            cur.execute(
                "INSERT INTO batches(manifest_version,manifest_json,status,"
                "created_ms,slices_total,slices_done) VALUES(?,?,?,?,?,0)",
                (manifest_version, manifest_json, "running", _now_ms(),
                 len(slices)))
            batch_id = cur.lastrowid
            for s in slices:
                cur.execute(
                    "INSERT INTO slices(batch_id,node,key,priority,"
                    "payload_json,expected_summary,expected_digest,status)"
                    " VALUES(?,?,?,?,?,?,?,?)",
                    (batch_id, s["node"], s["key"], s["priority"],
                     s["payload_json"], s["expected_summary"],
                     s["expected_digest"], "pending"))
            cur.execute("INSERT INTO progress(batch_id,next_index)"
                        " VALUES(?,0)", (batch_id,))
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise
        return batch_id

    @_locked
    def commit_batch(self, batch_id, results, quota_stats):
        """results: [{node,key,status,attempts,receipt|None}]；
        单事务写入回执、配额、进度游标并提交批次。"""
        cur = self.conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            done = 0
            retries_by_node = {}
            dispatched_by_node = {}
            for r in results:
                cur.execute(
                    "UPDATE slices SET status=?, attempts=? "
                    "WHERE batch_id=? AND node=? AND key=?",
                    (r["status"], r["attempts"], batch_id, r["node"],
                     r["key"]))
                dispatched_by_node[r["node"]] = \
                    dispatched_by_node.get(r["node"], 0) + 1
                retries_by_node[r["node"]] = retries_by_node.get(r["node"], 0) \
                    + max(0, r["attempts"] - 1)
                if r["status"] == "ok":
                    done += 1
                    rec = r["receipt"]
                    cur.execute(
                        "INSERT OR REPLACE INTO receipts(batch_id,node,key,"
                        "view,summary,digest,elapsed_ms)"
                        " VALUES(?,?,?,?,?,?,?)",
                        (batch_id, r["node"], r["key"], rec["view"],
                         rec.get("summary"), rec["digest"],
                         rec["elapsed_ms"]))
            for node, peak in quota_stats["node_peak"].items():
                cur.execute(
                    "INSERT OR REPLACE INTO quota(batch_id,node,"
                    "peak_in_flight,dispatched,retries) VALUES(?,?,?,?,?)",
                    (batch_id, node, peak, dispatched_by_node.get(node, 0),
                     retries_by_node.get(node, 0)))
            cur.execute("UPDATE progress SET next_index=? WHERE batch_id=?",
                        (len(results), batch_id))
            cur.execute(
                "UPDATE batches SET status='committed', committed_ms=?,"
                " slices_done=? WHERE batch_id=?",
                (_now_ms(), done, batch_id))
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

    @_locked
    def recover(self):
        """重启恢复：running 批次标记为 unknown，返回需重跑的批次号。"""
        cur = self.conn.cursor()
        rows = cur.execute(
            "SELECT batch_id FROM batches WHERE status='running'"
            " ORDER BY batch_id").fetchall()
        unknown = [r[0] for r in rows]
        if unknown:
            cur.execute(
                "UPDATE batches SET status='unknown' WHERE status='running'")
        return unknown

    @_locked
    def supersede_before(self, manifest_version):
        """版本演进：旧版本 committed 批次标记为 superseded（此后只读）。"""
        self.conn.execute(
            "UPDATE batches SET status='superseded' "
            "WHERE status='committed' AND manifest_version<?",
            (manifest_version,))

    def committed_batches(self, manifest_version=None):
        sql = ("SELECT batch_id,manifest_version,manifest_json,slices_total,"
               "slices_done FROM batches WHERE status='committed'")
        args = ()
        if manifest_version is not None:
            sql += " AND manifest_version=?"
            args = (manifest_version,)
        sql += " ORDER BY batch_id"
        return self.conn.execute(sql, args).fetchall()

    def load_slices(self, batch_id):
        return self.conn.execute(
            "SELECT node,key,priority,payload_json,expected_summary,"
            "expected_digest,status,attempts FROM slices WHERE batch_id=?"
            " ORDER BY node,key", (batch_id,)).fetchall()

    def load_receipts(self, batch_id):
        return self.conn.execute(
            "SELECT node,key,view,summary,digest,elapsed_ms FROM receipts"
            " WHERE batch_id=? ORDER BY node,key", (batch_id,)).fetchall()

    def load_quota(self, batch_id):
        return self.conn.execute(
            "SELECT node,peak_in_flight,dispatched,retries FROM quota"
            " WHERE batch_id=? ORDER BY node", (batch_id,)).fetchall()
