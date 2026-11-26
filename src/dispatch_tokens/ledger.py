"""SQLite 账本：原子提交、快照整体替换与强制终止恢复。"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .errors import ReplayError
from .spec import Spec

LEDGER_FORMAT_VERSION = 1
UNKNOWN = "UNKNOWN"
COMMITTED = "COMMITTED"


def spec_digest(spec: Spec) -> str:
    """声明内容摘要：对原始 JSON 规范化后取 SHA-256。"""
    canonical = json.dumps(spec.raw, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Ledger:
    def __init__(self, path: str | Path, spec: Spec) -> None:
        self.path = Path(path)
        self.spec = spec
        self.digest = spec_digest(spec)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._ensure_schema()
        self._check_or_replace_snapshot()

    def _ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS shard_outcomes (
                shard_id TEXT PRIMARY KEY,
                node_id TEXT NOT NULL,
                status TEXT NOT NULL,
                param_version INTEGER NOT NULL,
                token_id TEXT,
                token_version INTEGER,
                summary_json TEXT,
                ownership_json TEXT NOT NULL,
                bucket_json TEXT NOT NULL,
                cursor_seq INTEGER NOT NULL,
                updated_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS refill_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                record_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                seq INTEGER NOT NULL,
                record_json TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def _check_or_replace_snapshot(self) -> None:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key='format_version'").fetchone()
        if row is None:
            self.conn.executemany(
                "INSERT INTO meta(key, value) VALUES (?, ?)",
                [("format_version", str(LEDGER_FORMAT_VERSION)),
                 ("spec_digest", self.digest)],
            )
            self.conn.commit()
            return
        stored_version = int(row[0])
        stored_digest = self.conn.execute(
            "SELECT value FROM meta WHERE key='spec_digest'").fetchone()[0]
        if stored_version != LEDGER_FORMAT_VERSION or stored_digest != self.digest:
            # 版本或摘要不一致：整体替换，禁止增量合并。
            self.conn.executescript(
                "DELETE FROM shard_outcomes; DELETE FROM refill_audit; "
                "DELETE FROM decisions; DELETE FROM meta;")
            self.conn.executemany(
                "INSERT INTO meta(key, value) VALUES (?, ?)",
                [("format_version", str(LEDGER_FORMAT_VERSION)),
                 ("spec_digest", self.digest)],
            )
            self.conn.commit()

    def commit_outcome(self, *, shard_id: str, node_id: str, status: str,
                       param_version: int, token_id: str | None,
                       token_version: int | None,
                       summary: dict[str, Any] | None,
                       ownership: dict[str, Any], bucket: dict[str, Any],
                       cursor_seq: int, updated_ms: int,
                       refill_records: list[dict[str, Any]],
                       decision_records: list[dict[str, Any]]) -> None:
        """一个分片结局的全部写入在同一事务内原子提交。"""
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO shard_outcomes(
                    shard_id, node_id, status, param_version, token_id,
                    token_version, summary_json, ownership_json, bucket_json,
                    cursor_seq, updated_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(shard_id) DO UPDATE SET
                    node_id=excluded.node_id,
                    status=excluded.status,
                    param_version=excluded.param_version,
                    token_id=excluded.token_id,
                    token_version=excluded.token_version,
                    summary_json=excluded.summary_json,
                    ownership_json=excluded.ownership_json,
                    bucket_json=excluded.bucket_json,
                    cursor_seq=excluded.cursor_seq,
                    updated_ms=excluded.updated_ms
                """,
                (shard_id, node_id, status, param_version, token_id,
                 token_version,
                 json.dumps(summary, sort_keys=True) if summary else None,
                 json.dumps(ownership, sort_keys=True),
                 json.dumps(bucket, sort_keys=True), cursor_seq, updated_ms),
            )
            for record in refill_records:
                self.conn.execute(
                    "INSERT INTO refill_audit(node_id, seq, record_json) "
                    "VALUES (?, ?, ?)",
                    (node_id, cursor_seq,
                     json.dumps(record, sort_keys=True)),
                )
            for record in decision_records:
                self.conn.execute(
                    "INSERT INTO decisions(seq, record_json) VALUES (?, ?)",
                    (cursor_seq, json.dumps(record, sort_keys=True)),
                )

    def load_outcomes(self) -> dict[str, dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT shard_id, node_id, status, param_version, token_id, "
            "token_version, summary_json, ownership_json, bucket_json, "
            "cursor_seq, updated_ms FROM shard_outcomes").fetchall()
        outcomes: dict[str, dict[str, Any]] = {}
        for row in rows:
            outcomes[row[0]] = {
                "shard_id": row[0], "node_id": row[1], "status": row[2],
                "param_version": row[3], "token_id": row[4],
                "token_version": row[5],
                "summary": json.loads(row[6]) if row[6] else None,
                "ownership": json.loads(row[7]),
                "bucket": json.loads(row[8]),
                "cursor_seq": row[9], "updated_ms": row[10],
            }
        return outcomes

    def mark_unknown(self, shard_id: str, node_id: str, cursor_seq: int,
                     updated_ms: int, bucket: dict[str, Any]) -> None:
        """强制终止后未提交分片标记为 UNKNOWN，并登记额度回收意图。"""
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO shard_outcomes(
                    shard_id, node_id, status, param_version, token_id,
                    token_version, summary_json, ownership_json, bucket_json,
                    cursor_seq, updated_ms)
                VALUES (?, ?, ?, ?, NULL, NULL, NULL, ?, ?, ?, ?)
                ON CONFLICT(shard_id) DO UPDATE SET
                    status='UNKNOWN', summary_json=NULL,
                    ownership_json=excluded.ownership_json,
                    bucket_json=excluded.bucket_json,
                    cursor_seq=excluded.cursor_seq,
                    updated_ms=excluded.updated_ms
                """,
                (shard_id, node_id, UNKNOWN,
                 bucket.get("param_version", 0),
                 json.dumps({"decision": "UNKNOWN_RERUN_RECLAIM_FIRST",
                             "shard_id": shard_id, "node_id": node_id},
                            sort_keys=True),
                 json.dumps(bucket, sort_keys=True), cursor_seq, updated_ms),
            )

    def verify_replay(self, replay_outcomes: dict[str, dict[str, Any]]) -> None:
        """重放结果与已完整提交记录逐项比对，首个不一致抛 E_REPLAY。"""
        committed = self.load_outcomes()
        for shard_id in sorted(set(committed) | set(replay_outcomes)):
            old = committed.get(shard_id)
            new = replay_outcomes.get(shard_id)
            if old is None or new is None:
                raise ReplayError(
                    f"shard {shard_id!r} presence differs "
                    f"(committed={old is not None}, replay={new is not None})",
                    location=f"shards[{shard_id}]", field="<presence>")
            if old["status"] == UNKNOWN:
                continue
            # summary 由真实节点子进程产出，纯引擎重建不产生该字段，
            # 其确定性由样本种子与聚合校验单独保证，不参与账本重放比对。
            for field_name in ("node_id", "status", "param_version",
                               "token_id", "token_version",
                               "ownership", "bucket"):
                if old[field_name] != new[field_name]:
                    raise ReplayError(
                        f"first mismatch at shard {shard_id!r} field "
                        f"{field_name!r}: committed={old[field_name]!r} "
                        f"replay={new[field_name]!r}",
                        location=f"shards[{shard_id}]", field=field_name)

    def close(self) -> None:
        self.conn.close()
