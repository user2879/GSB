"""SQLite 持久化账本：原子提交、快照整体替换、崩溃恢复与重放校验。

每个分片批次在同一事务内写入：
- 批次归属（plan_id/plan_version/shard_id/executor_id）；
- 采纳回执摘要与代价估算摘要；
- 失效传播轨迹与进度游标；
所有记录携带整数格式版本与内容摘要。

恢复只承认 status='committed' 的完整记录；
未提交事务回滚后其分片标记 unknown 并要求重跑，配额由内存状态自然回收。
"""

import os
import shutil
import sqlite3
import time
from typing import Any, Dict, List, Optional

from .canonical import canonical_text, digest
from .errors import ReplayError

LEDGER_FORMAT_VERSION = 3
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    shard_id TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    plan_version INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    executor_id TEXT NOT NULL,
    status TEXT NOT NULL,
    receipt_digest TEXT NOT NULL,
    cost_digest TEXT NOT NULL,
    shard_digest TEXT NOT NULL,
    receipt_json TEXT NOT NULL,
    cost_json TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    committed_at_ms INTEGER NOT NULL,
    PRIMARY KEY (plan_version, shard_id)
);
CREATE TABLE IF NOT EXISTS invalidations (
    plan_version INTEGER NOT NULL,
    plan_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    basis TEXT NOT NULL,
    block_groups TEXT NOT NULL,
    digest TEXT NOT NULL,
    PRIMARY KEY (plan_version, plan_id, kind)
);
CREATE TABLE IF NOT EXISTS cursors (
    plan_version INTEGER PRIMARY KEY,
    total_shards INTEGER NOT NULL,
    committed_shards INTEGER NOT NULL,
    cursor_digest TEXT NOT NULL,
    updated_at_ms INTEGER NOT NULL
);
"""


def now_ms() -> int:
    return time.monotonic_ns() // 1_000_000


class Ledger:
    def __init__(self, db_path: str, snapshot_dir: str):
        self.db_path = db_path
        self.snapshot_dir = snapshot_dir
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(snapshot_dir, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        with self.conn:
            self.conn.executescript(SCHEMA)
            self.conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES(?, ?)",
                ("format_version", str(LEDGER_FORMAT_VERSION)),
            )

    def close(self):
        self.conn.close()

    def snapshot_fingerprint(self) -> Dict[str, Any]:
        rows = self.conn.execute(
            "SELECT plan_version, shard_id, receipt_digest, cost_digest, shard_digest, status "
            "FROM batches ORDER BY plan_version, shard_id"
        ).fetchall()
        payload = [dict(row) for row in rows]
        return {"format_version": LEDGER_FORMAT_VERSION, "batches": payload}

    def write_snapshot(self, snapshot_version: int) -> str:
        path = os.path.join(self.snapshot_dir, f"snapshot-v{snapshot_version}.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(canonical_text(self.snapshot_fingerprint()))
        return path

    def verify_snapshot(self, snapshot_version: int) -> Optional[str]:
        """快照摘要与当前账本不一致时整体替换并保留只读备份，返回备份路径。"""
        path = os.path.join(self.snapshot_dir, f"snapshot-v{snapshot_version}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            stored = handle.read()
        current = canonical_text(self.snapshot_fingerprint())
        if digest(stored) == digest(current):
            return None
        backup = os.path.join(
            self.snapshot_dir, f"snapshot-v{snapshot_version}.backup-{now_ms()}.readonly"
        )
        shutil.copy2(path, backup)
        os.chmod(backup, 0o444)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(current)
        return backup

    def init_cursor(self, plan_version: int, total_shards: int):
        payload = {
            "plan_version": plan_version,
            "total": total_shards,
            "committed": 0,
        }
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO cursors("
                "plan_version, total_shards, committed_shards, cursor_digest, updated_at_ms) "
                "VALUES(?,?,?,?,?)",
                (plan_version, total_shards, 0, digest(payload), now_ms()),
            )

    def commit_batch(
        self,
        plan_version: int,
        plan_id: str,
        ordinal: int,
        executor_id: str,
        receipt: Dict[str, Any],
        cost_summary: Dict[str, Any],
        shard_digest_value: str,
        invalidation_rows: Optional[List[Dict[str, Any]]] = None,
        total_shards: Optional[int] = None,
    ):
        receipt_json = canonical_text(receipt)
        cost_json = canonical_text(cost_summary)
        receipt_digest_value = digest(receipt)
        cost_digest_value = digest(cost_summary)
        committed = now_ms()
        with self.conn:
            self.conn.execute(
                "INSERT INTO batches("
                "shard_id, plan_id, plan_version, ordinal, executor_id, status, "
                "receipt_digest, cost_digest, shard_digest, receipt_json, cost_json, "
                "format_version, committed_at_ms) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    receipt["shard_id"],
                    plan_id,
                    plan_version,
                    ordinal,
                    executor_id,
                    "committed",
                    receipt_digest_value,
                    cost_digest_value,
                    shard_digest_value,
                    receipt_json,
                    cost_json,
                    LEDGER_FORMAT_VERSION,
                    committed,
                ),
            )
            if invalidation_rows:
                for row in invalidation_rows:
                    self.conn.execute(
                        "INSERT OR REPLACE INTO invalidations("
                        "plan_version, plan_id, kind, basis, block_groups, digest) "
                        "VALUES(?,?,?,?,?,?)",
                        (
                            plan_version,
                            row["plan_id"],
                            row["kind"],
                            row["basis"],
                            canonical_text(row["block_groups"]),
                            digest(row),
                        ),
                    )
            committed_shards = self.conn.execute(
                "SELECT COUNT(*) FROM batches WHERE plan_version=? AND status='committed'",
                (plan_version,),
            ).fetchone()[0]
            if total_shards is not None:
                total = total_shards
            else:
                row = self.conn.execute(
                    "SELECT total_shards FROM cursors WHERE plan_version=?",
                    (plan_version,),
                ).fetchone()
                total = row[0] if row else committed_shards
            cursor_payload = {
                "plan_version": plan_version,
                "total": total,
                "committed": committed_shards,
            }
            self.conn.execute(
                "INSERT OR REPLACE INTO cursors("
                "plan_version, total_shards, committed_shards, cursor_digest, updated_at_ms) "
                "VALUES(?,?,?,?,?)",
                (
                    plan_version,
                    total,
                    committed_shards,
                    digest(cursor_payload),
                    committed,
                ),
            )

    def committed_shards(self, plan_version: int) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM batches WHERE plan_version=? AND status='committed' "
            "ORDER BY ordinal",
            (plan_version,),
        ).fetchall()

    def unknown_shards(self, expected_shard_ids: List[str], plan_version: int):
        known = {
            row["shard_id"]
            for row in self.conn.execute(
                "SELECT shard_id FROM batches WHERE plan_version=?",
                (plan_version,),
            ).fetchall()
        }
        return [shard_id for shard_id in expected_shard_ids if shard_id not in known]

    def replay_verify(
        self,
        plan_version: int,
        expected_costs: Dict[str, Dict[str, Any]],
        expected_receipts: Dict[str, Dict[str, Any]],
    ):
        """逐分片重放比对，定位首个不一致的 shard/plan_version/field 并抛 E_REPLAY。"""
        rows = self.committed_shards(plan_version)
        for row in rows:
            shard_id = row["shard_id"]
            expected_receipt = expected_receipts.get(shard_id)
            if expected_receipt is None:
                raise ReplayError(
                    "账本含未预期分片",
                    object_id=shard_id,
                    field="shard_id",
                )
            if digest(expected_receipt) != row["receipt_digest"]:
                field_name = self._first_diff_field(
                    expected_receipt, _load_json(row["receipt_json"])
                )
                raise ReplayError(
                    "回执重放摘要不一致",
                    object_id=shard_id,
                    field=field_name,
                )
            expected_cost = expected_costs.get(row["plan_id"])
            if expected_cost is not None and digest(expected_cost) != row["cost_digest"]:
                field_name = self._first_diff_field(
                    expected_cost, _load_json(row["cost_json"])
                )
                raise ReplayError(
                    "代价估算重放摘要不一致",
                    object_id=shard_id,
                    field=field_name,
                )
            if row["format_version"] != LEDGER_FORMAT_VERSION:
                raise ReplayError(
                    "账本格式版本不一致", object_id=shard_id, field="format_version"
                )

    @staticmethod
    def _first_diff_field(expected: Dict[str, Any], stored: Dict[str, Any]):
        for key in sorted(set(expected) | set(stored)):
            if expected.get(key) != stored.get(key):
                return key
        return "digest"


def _load_json(text: str):
    import json

    return json.loads(text)
