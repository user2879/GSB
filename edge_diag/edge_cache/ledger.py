"""SQLite 原子账本与快照版本识别。

设计要点：
- 每次授予/续约/回收在同一个事务内写事件记录、容量余量、租约集合、
  归属裁决与进度游标，要么全部可见要么全部不可见。
- meta 表携带整数 FORMAT_VERSION 与当前声明内容摘要。重新打开时若格式版本
  或摘要与当前声明不一致，整体替换（删除全部业务表重建），绝不增量合并。
- 未确认批次先落 unknown_batches；恢复时只依据完整提交记录重建，未确认
  批次标记 unknown 并要求重跑、先回收占用。
"""

import json
import sqlite3
from pathlib import Path

from .errors import LedgerError

FORMAT_VERSION = 1

SCHEMA = """
CREATE TABLE meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE progress (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_committed_seq INTEGER NOT NULL,
    spec_digest TEXT NOT NULL,
    format_version INTEGER NOT NULL
);
CREATE TABLE event_records (
    seq INTEGER NOT NULL,
    sub_index INTEGER NOT NULL DEFAULT 0,
    action TEXT NOT NULL,
    node_id TEXT,
    shard_id TEXT,
    lease_id TEXT,
    quota_version INTEGER,
    reason TEXT,
    basis TEXT,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (seq, sub_index)
);
CREATE TABLE leases (
    lease_id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    shard_id TEXT NOT NULL,
    content_key TEXT NOT NULL,
    shard_index INTEGER NOT NULL,
    grant_ms INTEGER NOT NULL,
    expire_ms INTEGER NOT NULL,
    renew_count INTEGER NOT NULL,
    lease_version INTEGER NOT NULL,
    quota_version INTEGER NOT NULL,
    slot_index INTEGER NOT NULL,
    state TEXT NOT NULL
);
CREATE TABLE node_water (
    node_id TEXT PRIMARY KEY,
    capacity_slots INTEGER NOT NULL,
    used_slots INTEGER NOT NULL,
    per_node_limit INTEGER NOT NULL,
    active_leases INTEGER NOT NULL,
    quota_version INTEGER NOT NULL
);
CREATE TABLE ownership (
    slot_key TEXT NOT NULL,
    lease_id TEXT,
    node_id TEXT NOT NULL,
    slot_index INTEGER NOT NULL,
    shard_id TEXT,
    reason TEXT NOT NULL,
    basis TEXT NOT NULL,
    seq INTEGER NOT NULL,
    quota_version INTEGER NOT NULL,
    change_ms INTEGER NOT NULL
);
CREATE TABLE waits (
    seq INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    shard_id TEXT NOT NULL,
    wait_start_ms INTEGER NOT NULL,
    wait_end_ms INTEGER,
    granted INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE unknown_batches (
    batch_id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    shard_id TEXT NOT NULL,
    slot_index INTEGER,
    marked_ms INTEGER NOT NULL,
    resolved TEXT NOT NULL
);
CREATE TABLE receipts (
    seq INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    shard_id TEXT NOT NULL,
    digest TEXT NOT NULL,
    fp_sum INTEGER NOT NULL,
    fp_sqsum INTEGER NOT NULL,
    byte_hash TEXT NOT NULL,
    elapsed_ms INTEGER NOT NULL,
    quota_version INTEGER NOT NULL,
    PRIMARY KEY (seq, node_id, shard_id)
);
"""


class Ledger:
    def __init__(self, db_path, spec, spec_digest):
        self.db_path = str(db_path)
        self.spec = spec
        self.spec_digest = spec_digest
        self.conn = sqlite3.connect(self.db_path, timeout=30,
                                    isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.replaced = False
        self._open_or_replace()

    def _open_or_replace(self):
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='meta'")
        if cur.fetchone() is None:
            self._create_fresh()
            return
        try:
            version = int(self._meta("format_version"))
            digest = self._meta("spec_digest")
        except (LedgerError, TypeError, ValueError):
            version, digest = -1, ""
        if version != FORMAT_VERSION or digest != self.spec_digest:
            self._replace_all()
        else:
            self._ensure_progress()

    def _meta(self, key):
        cur = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,))
        row = cur.fetchone()
        if row is None:
            raise LedgerError(f"meta 缺少 {key}", key)
        return row[0]

    def _create_fresh(self):
        self.conn.executescript(SCHEMA)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.executemany(
                "INSERT INTO meta(key,value) VALUES(?,?)",
                [("format_version", str(FORMAT_VERSION)),
                 ("spec_digest", self.spec_digest),
                 ("session_id", self.spec["session"]["session_id"])])
            self.conn.execute(
                "INSERT INTO progress(id,last_committed_seq,spec_digest,"
                "format_version) VALUES(1,0,?,?)",
                (self.spec_digest, FORMAT_VERSION))
            for node in self.spec["nodes"]:
                self.conn.execute(
                    "INSERT INTO node_water(node_id,capacity_slots,used_slots,"
                    "per_node_limit,active_leases,quota_version) "
                    "VALUES(?,?,?,0,0,?)",
                    (node["node_id"], node["capacity_slots"],
                     node["per_node_lease_limit"], node["quota_version"]))
            self.conn.execute("COMMIT")
        except Exception:
            try:
                self.conn.execute("ROLLBACK")
            except Exception:
                pass
            raise

    def _replace_all(self):
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            rows = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            for (name,) in rows:
                self.conn.execute(f"DROP TABLE IF EXISTS {name}")
            self.conn.execute("COMMIT")
        except Exception:
            try:
                self.conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        self.replaced = True
        self._create_fresh()

    def _ensure_progress(self):
        row = self.conn.execute("SELECT id FROM progress WHERE id=1").fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO progress(id,last_committed_seq,spec_digest,"
                "format_version) VALUES(1,0,?,?)",
                (self.spec_digest, FORMAT_VERSION))

    # ---------- 恢复用只读查询 ----------
    def last_committed_seq(self):
        return int(self.conn.execute(
            "SELECT last_committed_seq FROM progress WHERE id=1").fetchone()[0])

    def snapshot(self):
        leases = [dict(zip([d[0] for d in self.conn.execute(
            "SELECT * FROM leases WHERE state='active'").description], row))
            for row in self.conn.execute(
            "SELECT * FROM leases WHERE state='active'").fetchall()]
        water = {row[0]: {"capacity_slots": row[1], "used_slots": row[2],
                          "per_node_limit": row[3], "active_leases": row[4],
                          "quota_version": row[5]}
                 for row in self.conn.execute(
                "SELECT node_id,capacity_slots,used_slots,per_node_limit,"
                "active_leases,quota_version FROM node_water")}
        unknown = [{"batch_id": r[0], "seq": r[1], "node_id": r[2],
                    "shard_id": r[3], "slot_index": r[4], "marked_ms": r[5],
                    "resolved": r[6]}
                   for r in self.conn.execute(
                "SELECT batch_id,seq,node_id,shard_id,slot_index,marked_ms,"
                "resolved FROM unknown_batches").fetchall()]
        budget_used = int(self._meta("budget_used")) if self._meta_exists(
            "budget_used") else 0
        return {"leases": leases, "water": water, "unknown": unknown,
                "budget_used": budget_used}

    def _meta_exists(self, key):
        return self.conn.execute(
            "SELECT 1 FROM meta WHERE key=?", (key,)).fetchone() is not None

    def all_event_records(self):
        return [dict(zip([d[0] for d in self.conn.execute(
            "SELECT * FROM event_records ORDER BY seq,sub_index").description],
            row)) for row in self.conn.execute(
            "SELECT * FROM event_records ORDER BY seq,sub_index").fetchall()]

    def all_receipts(self):
        return [dict(zip([d[0] for d in self.conn.execute(
            "SELECT * FROM receipts ORDER BY seq").description], row))
            for row in self.conn.execute(
            "SELECT * FROM receipts ORDER BY seq").fetchall()]

    def all_ownership(self):
        return [dict(zip([d[0] for d in self.conn.execute(
            "SELECT * FROM ownership ORDER BY rowid").description], row))
            for row in self.conn.execute(
            "SELECT * FROM ownership ORDER BY rowid").fetchall()]

    def all_waits(self):
        return [dict(zip([d[0] for d in self.conn.execute(
            "SELECT * FROM waits ORDER BY seq").description], row))
            for row in self.conn.execute(
            "SELECT * FROM waits ORDER BY seq").fetchall()]

    def set_budget_used(self, conn, value):
        conn.execute(
            "INSERT INTO meta(key,value) VALUES('budget_used',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(value),))

    def get_budget_used(self):
        return int(self._meta("budget_used")) if self._meta_exists(
            "budget_used") else 0

    def close(self):
        self.conn.close()


class LedgerWriter:
    """引擎侧写入器：所有状态变更走显式事务，保证原子可见。"""

    def __init__(self, ledger: Ledger):
        self.ledger = ledger
        self.conn = ledger.conn

    def begin(self):
        self.conn.execute("BEGIN IMMEDIATE")

    def commit(self):
        self.conn.execute("COMMIT")

    def rollback(self):
        self.conn.execute("ROLLBACK")

    def record_event(self, conn, seq, sub_index, action, payload,
                     node_id=None, shard_id=None, lease_id=None,
                     quota_version=None, reason=None, basis=None):
        conn.execute(
            "INSERT INTO event_records(seq,sub_index,action,node_id,shard_id,"
            "lease_id,quota_version,reason,basis,payload_json) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (seq, sub_index, action, node_id, shard_id, lease_id,
             quota_version, reason, basis, json.dumps(payload, sort_keys=True)))

    def upsert_lease(self, conn, lease):
        conn.execute(
            "INSERT INTO leases(lease_id,node_id,shard_id,content_key,"
            "shard_index,grant_ms,expire_ms,renew_count,lease_version,"
            "quota_version,slot_index,state) VALUES(?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(lease_id) DO UPDATE SET expire_ms=excluded.expire_ms,"
            "renew_count=excluded.renew_count,lease_version=excluded.lease_version,"
            "quota_version=excluded.quota_version,state=excluded.state",
            (lease["lease_id"], lease["node_id"], lease["shard_id"],
             lease["content_key"], lease["shard_index"], lease["grant_ms"],
             lease["expire_ms"], lease["renew_count"],
             lease["lease_version"], lease["quota_version"],
             lease["slot_index"], lease["state"]))

    def update_water(self, conn, node_id, used_slots, active_leases,
                     quota_version=None):
        if quota_version is None:
            conn.execute(
                "UPDATE node_water SET used_slots=?,active_leases=? "
                "WHERE node_id=?", (used_slots, active_leases, node_id))
        else:
            conn.execute(
                "UPDATE node_water SET used_slots=?,active_leases=?,"
                "quota_version=? WHERE node_id=?",
                (used_slots, active_leases, quota_version, node_id))

    def insert_ownership(self, conn, slot_key, lease_id, node_id, slot_index,
                         shard_id, reason, basis, seq, quota_version,
                         change_ms):
        conn.execute(
            "INSERT INTO ownership(slot_key,lease_id,node_id,slot_index,"
            "shard_id,reason,basis,seq,quota_version,change_ms) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (slot_key, lease_id, node_id, slot_index, shard_id, reason,
             basis, seq, quota_version, change_ms))

    def insert_wait(self, conn, seq, node_id, shard_id, wait_start_ms):
        conn.execute(
            "INSERT INTO waits(seq,node_id,shard_id,wait_start_ms,wait_end_ms,"
            "granted) VALUES(?,?,?,?,NULL,0)",
            (seq, node_id, shard_id, wait_start_ms))

    def finish_wait(self, conn, seq, node_id, shard_id, wait_end_ms):
        conn.execute(
            "UPDATE waits SET wait_end_ms=?,granted=1 WHERE seq=? AND "
            "node_id=? AND shard_id=? AND granted=0",
            (wait_end_ms, seq, node_id, shard_id))

    def insert_unknown(self, conn, batch_id, seq, node_id, shard_id,
                       slot_index, marked_ms):
        conn.execute(
            "INSERT INTO unknown_batches(batch_id,seq,node_id,shard_id,"
            "slot_index,marked_ms,resolved) VALUES(?,?,?,?,?,?,'unknown')",
            (batch_id, seq, node_id, shard_id, slot_index, marked_ms))

    def resolve_unknown(self, conn, batch_id, resolved):
        conn.execute(
            "UPDATE unknown_batches SET resolved=? WHERE batch_id=?",
            (resolved, batch_id))

    def insert_receipt(self, conn, seq, receipt):
        conn.execute(
            "INSERT INTO receipts(seq,node_id,shard_id,digest,fp_sum,fp_sqsum,"
            "byte_hash,elapsed_ms,quota_version) VALUES(?,?,?,?,?,?,?,?,?)",
            (seq, receipt["node_id"], receipt["shard_id"], receipt["digest"],
             receipt["fp_sum"], receipt["fp_sqsum"], receipt["byte_hash"],
             receipt["elapsed_ms"], receipt["quota_version"]))

    def advance(self, conn, seq):
        conn.execute(
            "UPDATE progress SET last_committed_seq=? WHERE id=1", (seq,))
