"""SQLite 权威账本：原子提交、格式版本与内容摘要识别、快照与只读备份。

- 每完成一个批次，在同一事务内写入批次归属、采纳/否决回执、失效传播轨迹、
  进度游标，并同步刷新内容摘要；
- 账本携带整数格式版本（meta.format_version）与内容摘要（meta.content_digest）；
- 重新打开时格式版本或内容摘要与磁盘状态不一致 -> 整体替换并保留只读备份；
- 进程被强制终止后，只有已完整提交的批次可恢复；未提交批次由调用方标记未知。
"""

import json
import os
import shutil
import sqlite3
import threading
import time

from .core import canonical_json, sha256_hex

FORMAT_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS param_versions(
    worker_id TEXT NOT NULL,
    spec_digest TEXT NOT NULL,
    version INTEGER NOT NULL,
    PRIMARY KEY(worker_id, spec_digest)
);
CREATE TABLE IF NOT EXISTS spec_changes(
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id TEXT NOT NULL,
    old_version INTEGER,
    new_version INTEGER NOT NULL,
    old_digest TEXT,
    new_digest TEXT NOT NULL,
    reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS membership_changes(
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    old_ids TEXT NOT NULL,
    new_ids TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches(
    batch_id TEXT PRIMARY KEY,
    worker_id TEXT NOT NULL,
    param_version INTEGER NOT NULL,
    status TEXT NOT NULL,
    adopted_digest TEXT,
    adopted_format INTEGER,
    request_ids TEXT NOT NULL,
    invalidated INTEGER NOT NULL DEFAULT 0,
    completed_seq INTEGER
);
CREATE TABLE IF NOT EXISTS receipts(
    batch_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    src_format INTEGER NOT NULL,
    src_file TEXT NOT NULL,
    src_line INTEGER NOT NULL,
    raw TEXT NOT NULL,
    output_digest TEXT NOT NULL,
    adopted INTEGER NOT NULL,
    reject_reason TEXT,
    filled TEXT NOT NULL,
    fill_basis TEXT NOT NULL,
    PRIMARY KEY(batch_id, request_id, src_file, src_line)
);
CREATE TABLE IF NOT EXISTS invalidations(
    batch_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    reason TEXT NOT NULL,
    PRIMARY KEY(batch_id, request_id, step)
);
CREATE TABLE IF NOT EXISTS progress(
    id INTEGER PRIMARY KEY CHECK(id = 1),
    cursor INTEGER NOT NULL
);
"""

DIGEST_TABLES = ("param_versions", "spec_changes", "membership_changes",
                 "batches", "receipts", "invalidations", "progress")


class Ledger:
    def __init__(self, path, config_snapshot=None):
        self.path = str(path)
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        replaced = self._open_with_checks()
        self.replaced_on_open = replaced
        if config_snapshot is not None:
            self.set_meta("config_snapshot", canonical_json(config_snapshot))

    def _open_with_checks(self):
        fresh = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=DELETE")
        self._conn.executescript(SCHEMA)
        if fresh or self._meta_get("format_version") is None:
            self._init_meta()
            return False
        ok_version = int(self._meta_get("format_version")) == FORMAT_VERSION
        ok_digest = self._meta_get("content_digest") == self._compute_content_digest()
        if ok_version and ok_digest:
            return False
        self._conn.close()
        self._backup_readonly()
        for suffix in ("", "-journal", "-wal", "-shm"):
            if os.path.exists(self.path + suffix):
                os.remove(self.path + suffix)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._init_meta()
        return True

    def _init_meta(self):
        with self._lock, self._conn:
            self._conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('format_version', ?)",
                               (str(FORMAT_VERSION),))
            self._conn.execute("INSERT OR REPLACE INTO progress(id, cursor) VALUES(1, 0)")
            self._conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('content_digest', ?)",
                               (self._compute_content_digest(),))

    def _backup_readonly(self):
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        backup = f"{self.path}.bak-{stamp}"
        shutil.copy2(self.path, backup)
        os.chmod(backup, 0o444)
        return backup

    def _meta_get(self, key):
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key, value):
        with self._lock, self._conn:
            self._conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                               (key, value))

    def get_meta(self, key):
        with self._lock:
            return self._meta_get(key)

    def _compute_content_digest(self):
        parts = []
        for table in DIGEST_TABLES:
            rows = self._conn.execute(f"SELECT * FROM {table}").fetchall()
            cols = [d[0] for d in self._conn.execute(f"SELECT * FROM {table} LIMIT 0").description]
            canon = [dict(zip(cols, row)) for row in rows]
            canon.sort(key=lambda r: canonical_json(r))
            parts.append({"table": table, "rows": canon})
        return sha256_hex(canonical_json(parts))

    def _refresh_digest(self):
        self._conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('content_digest', ?)",
                           (self._compute_content_digest(),))

    def register_param_version(self, worker_id, spec_digest):
        """返回 (version, changed, old_digest)。规格摘要稳定映射到整数版本。"""
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT version FROM param_versions WHERE worker_id=? AND spec_digest=?",
                (worker_id, spec_digest)).fetchone()
            cur = self._conn.execute(
                "SELECT spec_digest, version FROM param_versions WHERE worker_id=?"
                " ORDER BY version DESC LIMIT 1", (worker_id,)).fetchone()
            old_digest = cur[0] if cur else None
            if row:
                version = row[0]
                changed = old_digest is not None and old_digest != spec_digest
                if changed:
                    self._conn.execute(
                        "INSERT INTO spec_changes(worker_id, old_version, new_version,"
                        " old_digest, new_digest, reason) VALUES(?,?,?,?,?,?)",
                        (worker_id, cur[1], version, old_digest, spec_digest,
                         f"worker-spec-changed:{worker_id}"))
                self._refresh_digest()
                return version, changed, old_digest
            row_max = self._conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM param_versions").fetchone()
            version = row_max[0] + 1
            self._conn.execute(
                "INSERT INTO param_versions(worker_id, spec_digest, version) VALUES(?,?,?)",
                (worker_id, spec_digest, version))
            if old_digest is not None:
                self._conn.execute(
                    "INSERT INTO spec_changes(worker_id, old_version, new_version,"
                    " old_digest, new_digest, reason) VALUES(?,?,?,?,?,?)",
                    (worker_id, cur[1], version, old_digest, spec_digest,
                     f"worker-spec-changed:{worker_id}"))
            self._refresh_digest()
            return version, old_digest is not None, old_digest

    def current_worker_digests(self):
        with self._lock:
            rows = self._conn.execute(
                "SELECT p.worker_id, p.spec_digest FROM param_versions p"
                " JOIN (SELECT worker_id, MAX(version) v FROM param_versions GROUP BY worker_id) m"
                " ON p.worker_id=m.worker_id AND p.version=m.v").fetchall()
            return {wid: digest for wid, digest in rows}

    def load_batches(self):
        with self._lock:
            rows = self._conn.execute(
                "SELECT batch_id, worker_id, param_version, status, adopted_digest,"
                " adopted_format, request_ids FROM batches").fetchall()
            return {r[0]: {"worker_id": r[1], "param_version": r[2], "status": r[3],
                           "adopted_digest": r[4], "adopted_format": r[5],
                           "request_ids": json.loads(r[6])} for r in rows}

    def progress_cursor(self):
        with self._lock:
            return self._conn.execute("SELECT cursor FROM progress WHERE id=1").fetchone()[0]

    def mark_unknown(self, plan_batches):
        """把计划内但未完成的批次标记为未知（独立短事务）。"""
        with self._lock, self._conn:
            for pb in plan_batches:
                self._conn.execute(
                    "INSERT INTO batches(batch_id, worker_id, param_version, status,"
                    " adopted_digest, adopted_format, request_ids, completed_seq)"
                    " VALUES(?,?,?,?,NULL,NULL,?,NULL)"
                    " ON CONFLICT(batch_id) DO NOTHING",
                    (pb.batch_id, pb.worker_id, pb.param_version, "unknown",
                     json.dumps(sorted(pb.request_ids))))
            self._refresh_digest()

    def mark_retired(self, retired_rows):
        """把已从新计划中消失的批次标记为退役并写入失效轨迹（单事务）。"""
        with self._lock, self._conn:
            for bid, rows in retired_rows.items():
                self._conn.execute(
                    "UPDATE batches SET status='retired' WHERE batch_id=? AND status='done'",
                    (bid,))
                self._conn.execute("DELETE FROM invalidations WHERE batch_id=?", (bid,))
                self._conn.executemany(
                    "INSERT INTO invalidations(batch_id, request_id, step, reason)"
                    " VALUES(?,?,?,?)", rows)
            self._refresh_digest()

    def record_reassignment(self, worker_id, old_version, new_version, digest, reason):
        """把派工变更记录为可被重放推导的变更行。"""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO spec_changes(worker_id, old_version, new_version,"
                " old_digest, new_digest, reason) VALUES(?,?,?,?,?,?)",
                (worker_id, old_version, new_version, digest, digest, reason))
            self._refresh_digest()

    def commit_batch(self, pb, receipt_rows, invalidation_rows, membership_change=None):
        """单事务提交一个完成批次：归属、回执、失效轨迹、进度游标、内容摘要。"""
        with self._lock, self._conn:
            cur = self._conn.execute("SELECT cursor FROM progress WHERE id=1").fetchone()[0] + 1
            self._conn.execute("UPDATE progress SET cursor=? WHERE id=1", (cur,))
            self._conn.execute(
                "INSERT OR REPLACE INTO batches(batch_id, worker_id, param_version, status,"
                " adopted_digest, adopted_format, request_ids, invalidated, completed_seq)"
                " VALUES(?,?,?,?,?,?,?,?,?)",
                (pb["batch_id"], pb["worker_id"], pb["param_version"], "done",
                 pb["adopted_digest"], pb["adopted_format"],
                 json.dumps(sorted(pb["request_ids"])),
                 1 if invalidation_rows else 0, cur))
            self._conn.execute("DELETE FROM receipts WHERE batch_id=?", (pb["batch_id"],))
            self._conn.executemany(
                "INSERT INTO receipts(batch_id, request_id, src_format, src_file, src_line,"
                " raw, output_digest, adopted, reject_reason, filled, fill_basis)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?)", receipt_rows)
            self._conn.execute("DELETE FROM invalidations WHERE batch_id=?", (pb["batch_id"],))
            self._conn.executemany(
                "INSERT INTO invalidations(batch_id, request_id, step, reason) VALUES(?,?,?,?)",
                invalidation_rows)
            self._conn.execute("DELETE FROM membership_changes WHERE batch_id=?",
                               (pb["batch_id"],))
            if membership_change is not None:
                old_ids, new_ids = membership_change
                self._conn.execute(
                    "INSERT INTO membership_changes(batch_id, old_ids, new_ids) VALUES(?,?,?)",
                    (pb["batch_id"], json.dumps(sorted(old_ids)), json.dumps(sorted(new_ids))))
            self._refresh_digest()

    def query(self, sql, args=()):
        with self._lock:
            cur = self._conn.execute(sql, args)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def create_snapshot(self, snapshot_dir):
        with self._lock:
            os.makedirs(snapshot_dir, exist_ok=True)
            cursor = self.progress_cursor()
            digest = self._meta_get("content_digest")[:8]
            name = f"ledger-seq{cursor:06d}-{digest}.db"
            dst = os.path.join(snapshot_dir, name)
            backup = sqlite3.connect(dst)
            self._conn.backup(backup)
            backup.close()
            return dst

    @staticmethod
    def list_snapshots(snapshot_dir):
        if not os.path.isdir(snapshot_dir):
            return []
        return sorted(f for f in os.listdir(snapshot_dir) if f.endswith(".db"))

    def close(self):
        with self._lock:
            self._conn.close()
