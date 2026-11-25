"""SQLite 账本：原子提交、快照整体替换、强制终止后恢复（需求 8）。

每批回执与当时租约状态在同一个事务中落盘，并携带格式版本与内容摘要；
重启只承认已提交事务，未提交（inflight 后崩溃）的批标记为 unknown 并
先回收其配额。独立重放逐行重算内容摘要并定位首个差异。
"""

import json
import sqlite3

from .errors import LedgerError
from .plan import canonical_json, stable_hash

LEDGER_FORMAT_VERSION = 1


SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_version INTEGER PRIMARY KEY,
    spec_digest TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL,
    plan_fingerprint TEXT NOT NULL,
    note TEXT
);
CREATE TABLE IF NOT EXISTS batches (
    snapshot_version INTEGER NOT NULL,
    batch_id TEXT NOT NULL,
    batch_seq INTEGER NOT NULL,
    worker_id TEXT NOT NULL,
    request_ids TEXT NOT NULL,
    arrival_seqs TEXT NOT NULL,
    status TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (snapshot_version, batch_id)
);
CREATE TABLE IF NOT EXISTS receipts (
    snapshot_version INTEGER NOT NULL,
    batch_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    fmt INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (snapshot_version, batch_id, request_id, fmt)
);
CREATE TABLE IF NOT EXISTS leases (
    snapshot_version INTEGER NOT NULL,
    lease_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    state TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (snapshot_version, lease_id)
);
"""


def digest_of(obj):
    return stable_hash(obj, digest_size=16)


class Ledger:
    def __init__(self, path):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def begin_snapshot(self, spec_digest, created_at_ms, plan_fingerprint, note=""):
        """快照版本识别：整体替换 = 新建单调递增 snapshot_version。"""
        cur = self.conn.execute("SELECT COALESCE(MAX(snapshot_version), 0) AS v FROM snapshots")
        next_version = int(cur.fetchone()["v"]) + 1
        self.conn.execute(
            "INSERT INTO snapshots VALUES (?, ?, ?, ?, ?)",
            (next_version, spec_digest, int(created_at_ms), plan_fingerprint, note),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('current_snapshot', ?)",
            (str(next_version),),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('format_version', ?)",
            (str(LEDGER_FORMAT_VERSION),),
        )
        self.conn.commit()
        return next_version

    def current_snapshot(self):
        cur = self.conn.execute(
            "SELECT value FROM metadata WHERE key='current_snapshot'")
        row = cur.fetchone()
        return int(row["value"]) if row else None

    def commit_batch(self, snapshot_version, batch_result, lease_dict,
                     receipt_records):
        """批回执 + 租约状态同一事务原子落盘。"""
        batch_payload = {
            "format_version": LEDGER_FORMAT_VERSION,
            "batch": {
                "batch_id": batch_result["batch_id"],
                "batch_seq": batch_result["batch_seq"],
                "worker_id": batch_result["worker_id"],
                "request_ids": batch_result["request_ids"],
                "arrival_seqs": batch_result["arrival_seqs"],
                "param_version": batch_result["param_version"],
                "elapsed_ms": batch_result["elapsed_ms"],
                "results": batch_result["results"],
            },
        }
        batch_digest = digest_of(batch_payload)
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            self.conn.execute(
                "INSERT INTO batches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (snapshot_version, batch_result["batch_id"], batch_result["batch_seq"],
                 batch_result["worker_id"],
                 json.dumps(batch_result["request_ids"], sort_keys=True),
                 json.dumps(batch_result["arrival_seqs"], sort_keys=True),
                 "committed", batch_digest,
                 canonical_json(batch_payload).decode("utf-8")),
            )
            for record in receipt_records:
                receipt_payload = {
                    "format_version": LEDGER_FORMAT_VERSION,
                    "receipt_fmt": record["fmt"],
                    "receipt": record["payload"],
                }
                self.conn.execute(
                    "INSERT INTO receipts VALUES (?, ?, ?, ?, ?, ?)",
                    (snapshot_version, batch_result["batch_id"],
                     record["payload"]["request_id"], record["fmt"],
                     digest_of(receipt_payload),
                     canonical_json(receipt_payload).decode("utf-8")),
                )
            for lease_id, lease_state in lease_dict.items():
                lease_payload = {
                    "format_version": LEDGER_FORMAT_VERSION,
                    "lease": lease_state,
                }
                self.conn.execute(
                    "INSERT OR REPLACE INTO leases VALUES (?, ?, ?, ?, ?, ?)",
                    (snapshot_version, lease_id, lease_state["batch_id"],
                     lease_state["state"], digest_of(lease_payload),
                     canonical_json(lease_payload).decode("utf-8")),
                )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def mark_unknown(self, snapshot_version, plan, at_ms, reason):
        """强制终止后，未提交批只允许以 unknown 状态登记并先回收配额。"""
        payload = {
            "format_version": LEDGER_FORMAT_VERSION,
            "unknown": {"batch_id": plan.batch_id, "reason": reason, "at_ms": at_ms},
        }
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO batches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (snapshot_version, plan.batch_id, plan.batch_seq, plan.worker_id,
                 json.dumps(list(plan.request_ids), sort_keys=True),
                 json.dumps(list(plan.arrival_seqs), sort_keys=True),
                 "unknown", digest_of(payload),
                 canonical_json(payload).decode("utf-8")),
            )

    def load_snapshot(self, snapshot_version=None):
        version = snapshot_version or self.current_snapshot()
        if version is None:
            return None
        snap = self.conn.execute(
            "SELECT * FROM snapshots WHERE snapshot_version=?", (version,)).fetchone()
        if snap is None:
            raise LedgerError(f"快照版本 {version} 不存在", field="snapshot_version")
        batches = [dict(row) for row in self.conn.execute(
            "SELECT * FROM batches WHERE snapshot_version=? ORDER BY batch_seq",
            (version,))]
        receipts = [dict(row) for row in self.conn.execute(
            "SELECT * FROM receipts WHERE snapshot_version=? "
            "ORDER BY batch_id, request_id, fmt",
            (version,))]
        leases_rows = [dict(row) for row in self.conn.execute(
            "SELECT * FROM leases WHERE snapshot_version=? ORDER BY lease_id",
            (version,))]
        return {
            "snapshot_version": version,
            "spec_digest": snap["spec_digest"],
            "plan_fingerprint": snap["plan_fingerprint"],
            "created_at_ms": snap["created_at_ms"],
            "batches": batches,
            "receipts": receipts,
            "leases": leases_rows,
        }

    def list_snapshots(self):
        return [dict(row) for row in self.conn.execute(
            "SELECT * FROM snapshots ORDER BY snapshot_version")]

    def replay_verify(self, snapshot_version=None):
        """独立重放：重算每行 content_digest，返回首个差异或全部通过。"""
        version = snapshot_version or self.current_snapshot()
        checks = [
            ("batches", "SELECT * FROM batches WHERE snapshot_version=? ORDER BY batch_seq"),
            ("receipts", "SELECT * FROM receipts WHERE snapshot_version=? "
                         "ORDER BY batch_id, request_id, fmt"),
            ("leases", "SELECT * FROM leases WHERE snapshot_version=? ORDER BY lease_id"),
        ]
        for table, sql in checks:
            for row in self.conn.execute(sql, (version,)):
                stored = row["content_digest"]
                actual = stable_hash(json.loads(row["payload"]), digest_size=16)
                if stored != actual:
                    return {
                        "ok": False,
                        "first_diff": {
                            "table": table,
                        "key": (row["lease_id"] if table == "leases"
                                else (row["batch_id"] if table == "batches"
                                      else f'{row["batch_id"]}:{row["request_id"]}:r{row["fmt"]}')),
                            "stored_digest": stored,
                            "recomputed_digest": actual,
                        },
                    }
        return {"ok": True, "snapshot_version": version, "checked_tables": [c[0] for c in checks]}
