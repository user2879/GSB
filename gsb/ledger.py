"""SQLite 命中账本与快照恢复。

- 每完成一个批次：在同一事务中写入批次回执、归并输入、活跃参数版本与进度游标；
- 账本携带整数格式版本（SNAPSHOT_VERSION）与内容摘要（输入规格摘要）；
- 重新打开时若版本或摘要与当前磁盘状态不一致：整体替换（清空重建），
  并稳定列出差异项，不做增量合并。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

SNAPSHOT_VERSION = 1
LEDGER_FORMAT_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hits (
    request_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    rule_version INTEGER NOT NULL,
    client_key TEXT NOT NULL,
    bucket INTEGER NOT NULL,
    weight INTEGER NOT NULL,
    route TEXT NOT NULL,
    backend_id TEXT NOT NULL,
    param_version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS receipts (
    batch_id TEXT NOT NULL,
    merge_seq INTEGER NOT NULL,
    backend_id TEXT NOT NULL,
    source_format TEXT NOT NULL,
    param_version INTEGER NOT NULL,
    normalized_json TEXT NOT NULL,
    PRIMARY KEY (batch_id, merge_seq)
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    committed INTEGER NOT NULL,
    conclusion_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS waits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    backend_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    wait_start_ms INTEGER NOT NULL,
    wait_end_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS param_versions (
    backend_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    digest TEXT NOT NULL,
    active INTEGER NOT NULL,
    PRIMARY KEY (backend_id, version)
);
CREATE TABLE IF NOT EXISTS cursor (
    key TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
"""

DATA_TABLES = ("hits", "receipts", "batches", "waits", "param_versions", "cursor")


class Ledger:
    def __init__(self, conn: sqlite3.Connection, diffs: list[str]):
        self._conn = conn
        self._lock = threading.Lock()
        self.diffs = diffs

    @classmethod
    def open(cls, path, content_digest: str) -> "Ledger":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        diffs: list[str] = []
        stored_version = conn.execute(
            "SELECT value FROM meta WHERE key='snapshot_version'"
        ).fetchone()
        stored_digest = conn.execute(
            "SELECT value FROM meta WHERE key='content_digest'"
        ).fetchone()
        stale = False
        if stored_version is None or stored_digest is None:
            stale = False  # 全新账本：直接初始化
        else:
            if int(stored_version["value"]) != SNAPSHOT_VERSION:
                diffs.append(
                    f"snapshot_version: {stored_version['value']} -> {SNAPSHOT_VERSION}"
                )
                stale = True
            if stored_digest["value"] != content_digest:
                diffs.append(
                    f"content_digest: {stored_digest['value'][:12]}... -> "
                    f"{content_digest[:12]}..."
                )
                stale = True
        if stale:
            # 整体替换而非增量合并
            with conn:
                for table in DATA_TABLES:
                    conn.execute(f"DELETE FROM {table}")
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('snapshot_version', ?)",
                (str(SNAPSHOT_VERSION),),
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('content_digest', ?)",
                (content_digest,),
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('ledger_format_version', ?)",
                (str(LEDGER_FORMAT_VERSION),),
            )
        return cls(conn, sorted(diffs))

    def commit_batch(
        self,
        *,
        hits,
        receipts,
        conclusion,
        waits,
        active_versions: dict[str, int],
        version_digests: dict[tuple[str, int], str],
        cursor_value: int,
    ) -> None:
        """单事务写入一个已完成批次的全部账本内容。"""
        with self._lock, self._conn:
            for hit in hits:
                self._conn.execute(
                    "INSERT OR REPLACE INTO hits VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        hit.request_id,
                        hit.batch_id,
                        hit.rule_id,
                        hit.rule_version,
                        hit.client_key,
                        hit.bucket,
                        hit.weight,
                        hit.route,
                        hit.backend_id,
                        hit.param_version,
                    ),
                )
            for seq, (backend_id, receipt) in enumerate(receipts):
                self._conn.execute(
                    "INSERT OR REPLACE INTO receipts VALUES (?,?,?,?,?,?)",
                    (
                        receipt.batch_id,
                        seq,
                        backend_id,
                        receipt.source_format,
                        receipt.param_version,
                        json.dumps(receipt.to_dict(), sort_keys=True),
                    ),
                )
            self._conn.execute(
                "INSERT OR REPLACE INTO batches VALUES (?,?,?)",
                (conclusion.batch_id, 1, json.dumps(conclusion.to_dict(), sort_keys=True)),
            )
            for wait in waits:
                self._conn.execute(
                    "INSERT INTO waits(batch_id, backend_id, attempt, wait_start_ms, wait_end_ms)"
                    " VALUES (?,?,?,?,?)",
                    (
                        wait.batch_id,
                        wait.backend_id,
                        wait.attempt,
                        wait.wait_start_ms,
                        wait.wait_end_ms,
                    ),
                )
            for (backend_id, version), digest in sorted(version_digests.items()):
                self._conn.execute(
                    "INSERT OR IGNORE INTO param_versions VALUES (?,?,?,0)",
                    (backend_id, version, digest),
                )
            for backend_id, version in sorted(active_versions.items()):
                self._conn.execute(
                    "UPDATE param_versions SET active=0 WHERE backend_id=?", (backend_id,)
                )
                self._conn.execute(
                    "UPDATE param_versions SET active=1 WHERE backend_id=? AND version=?",
                    (backend_id, version),
                )
            self._conn.execute(
                "INSERT OR REPLACE INTO cursor(key, value) VALUES('next_batch_index', ?)",
                (cursor_value,),
            )

    # ---- 查询接口（供恢复、重放与界面使用） ----

    def committed_batches(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT batch_id FROM batches WHERE committed=1 ORDER BY rowid"
        ).fetchall()
        return [row["batch_id"] for row in rows]

    def hits_for(self, batch_id: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM hits WHERE batch_id=? ORDER BY request_id", (batch_id,)
        ).fetchall()

    def receipts_for(self, batch_id: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM receipts WHERE batch_id=? ORDER BY merge_seq", (batch_id,)
        ).fetchall()

    def conclusion_for(self, batch_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT conclusion_json FROM batches WHERE batch_id=? AND committed=1",
            (batch_id,),
        ).fetchone()
        return json.loads(row["conclusion_json"]) if row else None

    def all_waits(self) -> list[sqlite3.Row]:
        return self._conn.execute("SELECT * FROM waits ORDER BY id").fetchall()

    def route_distribution(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT route, COUNT(*) AS n FROM hits GROUP BY route ORDER BY route"
        ).fetchall()
        return {row["route"]: row["n"] for row in rows}

    def param_version_isolation(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT batch_id, param_version, COUNT(*) AS n FROM hits "
            "GROUP BY batch_id, param_version ORDER BY batch_id, param_version"
        ).fetchall()

    def active_param_versions(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT backend_id, version FROM param_versions WHERE active=1 ORDER BY backend_id"
        ).fetchall()
        return {row["backend_id"]: row["version"] for row in rows}

    def meta(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT key, value FROM meta ORDER BY key").fetchall()
        return {row["key"]: row["value"] for row in rows}

    def cursor_value(self) -> int:
        row = self._conn.execute(
            "SELECT value FROM cursor WHERE key='next_batch_index'"
        ).fetchone()
        return int(row["value"]) if row else 0

    def close(self) -> None:
        self._conn.close()
