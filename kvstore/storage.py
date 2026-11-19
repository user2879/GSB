"""SQLite 持久化层。

正式表：

* ``kv_records``  键值记录（键、值、毫秒过期戳、写入序号、格式版本）
* ``index_entries`` 前缀二级索引（前缀、引用键、索引项值、条目版本）
* ``prefix_counts``  由索引派生的前缀计数视图（仅统计未过期键）
* ``meta``        会话级元数据（键空间版本、轨迹游标、重建进度等）

重建期间使用独立的 ``idx_stage / cnt_stage`` 暂存表，前台查询始终读取正式表，
因此不会读到半写条目；重建完成后在单个事务内 ALTER 换名完成原子切换。
"""

from __future__ import annotations

import os
import sqlite3
import threading
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS kv_records (
    key            TEXT PRIMARY KEY,
    value          TEXT NOT NULL,
    expire_ms      INTEGER NOT NULL,
    write_seq      INTEGER NOT NULL,
    format_version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS index_entries (
    prefix        TEXT NOT NULL,
    ref_key       TEXT NOT NULL,
    index_value   TEXT NOT NULL,
    entry_version INTEGER NOT NULL,
    PRIMARY KEY (prefix, ref_key)
);
CREATE TABLE IF NOT EXISTS prefix_counts (
    prefix      TEXT PRIMARY KEY,
    cnt         INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class Storage:
    """带可重入锁的 SQLite 封装；同一进程内所有访问串行化，保证并发正确性。"""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        directory = os.path.dirname(os.path.abspath(db_path))
        os.makedirs(directory, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(
            db_path, check_same_thread=False, isolation_level=None
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self.conn.execute("PRAGMA synchronous=FULL;")
        self._create_schema()

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def _create_schema(self) -> None:
        with self._lock:
            self.conn.executescript(SCHEMA)

    # ---------------- meta ----------------
    def get_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._lock:
            row = self.conn.execute(
                "SELECT value FROM meta WHERE key=?", (key,)
            ).fetchone()
            return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def all_meta(self) -> dict[str, str]:
        with self._lock:
            return {
                row["key"]: row["value"]
                for row in self.conn.execute("SELECT key, value FROM meta")
            }

    # ---------------- kv ----------------
    def get_record(self, key: str) -> Optional[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(
                "SELECT * FROM kv_records WHERE key=?", (key,)
            ).fetchone()

    def upsert_record(self, record: dict[str, Any]) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO kv_records(key, value, expire_ms, write_seq, format_version) "
                "VALUES(:key, :value, :expire_ms, :write_seq, :format_version) "
                "ON CONFLICT(key) DO UPDATE SET "
                "value=excluded.value, expire_ms=excluded.expire_ms, "
                "write_seq=excluded.write_seq, format_version=excluded.format_version",
                record,
            )

    def delete_record(self, key: str) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM kv_records WHERE key=?", (key,))

    def all_records(self) -> list[sqlite3.Row]:
        with self._lock:
            return list(self.conn.execute("SELECT * FROM kv_records ORDER BY key"))

    def clear_kv(self) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM kv_records")

    # ---------------- index ----------------
    def upsert_index(self, prefix: str, ref_key: str, index_value: str, version: int) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO index_entries(prefix, ref_key, index_value, entry_version) "
                "VALUES(?, ?, ?, ?) "
                "ON CONFLICT(prefix, ref_key) DO UPDATE SET "
                "index_value=excluded.index_value, entry_version=excluded.entry_version",
                (prefix, ref_key, index_value, version),
            )

    def delete_index(self, prefix: str, ref_key: str) -> None:
        with self._lock:
            self.conn.execute(
                "DELETE FROM index_entries WHERE prefix=? AND ref_key=?",
                (prefix, ref_key),
            )

    def get_index_version(self, prefix: str, ref_key: str) -> Optional[int]:
        with self._lock:
            row = self.conn.execute(
                "SELECT entry_version FROM index_entries WHERE prefix=? AND ref_key=?",
                (prefix, ref_key),
            ).fetchone()
            return row["entry_version"] if row else None

    def all_index(self) -> list[sqlite3.Row]:
        with self._lock:
            return list(
                self.conn.execute(
                    "SELECT * FROM index_entries ORDER BY prefix, ref_key"
                )
            )

    def clear_index(self) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM index_entries")

    # ---------------- counts ----------------
    def set_count(self, prefix: str, cnt: int) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO prefix_counts(prefix, cnt) VALUES(?, ?) "
                "ON CONFLICT(prefix) DO UPDATE SET cnt=excluded.cnt",
                (prefix, cnt),
            )

    def get_count(self, prefix: str) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT cnt FROM prefix_counts WHERE prefix=?", (prefix,)
            ).fetchone()
            return row["cnt"] if row else 0

    def delete_count(self, prefix: str) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM prefix_counts WHERE prefix=?", (prefix,))

    def all_counts(self) -> dict[str, int]:
        with self._lock:
            return {
                row["prefix"]: row["cnt"]
                for row in self.conn.execute("SELECT prefix, cnt FROM prefix_counts")
            }

    def clear_counts(self) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM prefix_counts")

    # ---------------- rebuild staging ----------------
    def create_stage(self) -> None:
        with self._lock:
            self.conn.executescript(
                """
                DROP TABLE IF EXISTS idx_stage;
                DROP TABLE IF EXISTS cnt_stage;
                CREATE TABLE idx_stage (
                    prefix TEXT NOT NULL, ref_key TEXT NOT NULL,
                    index_value TEXT NOT NULL, entry_version INTEGER NOT NULL,
                    PRIMARY KEY (prefix, ref_key));
                CREATE TABLE cnt_stage (
                    prefix TEXT PRIMARY KEY, cnt INTEGER NOT NULL);
                """
            )

    def stage_upsert_index(
        self, prefix: str, ref_key: str, index_value: str, version: int
    ) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO idx_stage(prefix, ref_key, index_value, entry_version) "
                "VALUES(?, ?, ?, ?) "
                "ON CONFLICT(prefix, ref_key) DO UPDATE SET "
                "index_value=excluded.index_value, entry_version=excluded.entry_version",
                (prefix, ref_key, index_value, version),
            )

    def stage_delete_index(self, prefix: str, ref_key: str) -> None:
        with self._lock:
            self.conn.execute(
                "DELETE FROM idx_stage WHERE prefix=? AND ref_key=?", (prefix, ref_key)
            )

    def stage_set_count(self, prefix: str, cnt: int) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO cnt_stage(prefix, cnt) VALUES(?, ?) "
                "ON CONFLICT(prefix) DO UPDATE SET cnt=excluded.cnt",
                (prefix, cnt),
            )

    def stage_delete_count(self, prefix: str) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM cnt_stage WHERE prefix=?", (prefix,))

    def drop_stage(self) -> None:
        with self._lock:
            self.conn.executescript("DROP TABLE IF EXISTS idx_stage; DROP TABLE IF EXISTS cnt_stage;")

    def stage_rows(self) -> tuple[list[sqlite3.Row], list[sqlite3.Row]]:
        with self._lock:
            idx = list(self.conn.execute("SELECT * FROM idx_stage ORDER BY prefix, ref_key"))
            cnt = list(self.conn.execute("SELECT * FROM cnt_stage ORDER BY prefix"))
            return idx, cnt

    def atomic_swap_stage(self) -> None:
        """单事务内原子切换索引与计数表；任何读都不会观察到中间态。"""
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                self.conn.execute("DROP TABLE index_entries")
                self.conn.execute("ALTER TABLE idx_stage RENAME TO index_entries")
                self.conn.execute("DROP TABLE prefix_counts")
                self.conn.execute("ALTER TABLE cnt_stage RENAME TO prefix_counts")
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

    def close(self) -> None:
        with self._lock:
            self.conn.close()
