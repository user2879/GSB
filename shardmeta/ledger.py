"""SQLite ledger with atomic commits, snapshot identity and crash recovery.

Every grant / renew / expiry / abort / adoption step is written inside one
transaction together with merge id, shard digest, remaining capacity and the
progress cursor. The ledger carries an integer format version and a content
digest. On reopen, a version or digest mismatch triggers a whole-file
replacement rather than incremental merging.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Optional

from .errors import DivergenceError

LEDGER_FORMAT_VERSION = 3
SCHEMA_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS ledger_meta(
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS events(
        seq INTEGER PRIMARY KEY,
        op TEXT NOT NULL,
        node_id TEXT,
        object_key TEXT,
        shard_index INTEGER,
        lease_id TEXT,
        merge_id TEXT,
        payload TEXT NOT NULL,
        cursor INTEGER NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS leases(
        lease_id TEXT PRIMARY KEY,
        node_id TEXT NOT NULL,
        object_key TEXT NOT NULL,
        shard_index INTEGER NOT NULL,
        granted_ms INTEGER NOT NULL,
        expires_ms INTEGER NOT NULL,
        renew_count INTEGER NOT NULL,
        lease_version INTEGER NOT NULL,
        state TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS shards(
        object_key TEXT NOT NULL,
        shard_index INTEGER NOT NULL,
        node_id TEXT NOT NULL,
        merge_id TEXT NOT NULL,
        shard_digest TEXT NOT NULL,
        shard_bytes INTEGER NOT NULL DEFAULT 0,
        content_check TEXT,
        meta_version INTEGER NOT NULL,
        source_format INTEGER NOT NULL,
        merge_seq INTEGER NOT NULL,
        PRIMARY KEY(object_key, shard_index, merge_id)
    )""",
    """CREATE TABLE IF NOT EXISTS node_counters(
        node_id TEXT PRIMARY KEY,
        capacity_slots INTEGER NOT NULL,
        lease_limit INTEGER NOT NULL,
        active_count INTEGER NOT NULL,
        remaining_slots INTEGER NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS progress(
        singleton INTEGER PRIMARY KEY CHECK(singleton=1),
        cursor INTEGER NOT NULL,
        last_seq INTEGER
    )""",
    """CREATE TABLE IF NOT EXISTS unknown_requests(
        seq INTEGER PRIMARY KEY,
        op TEXT NOT NULL,
        object_key TEXT,
        reason TEXT NOT NULL
    )""",
)


def _row_digest(con: sqlite3.Connection) -> str:
    hasher = hashlib.sha256()
    for table in ("events", "leases", "shards", "node_counters", "progress",
                  "unknown_requests"):
        hasher.update(table.encode())
        columns = [row[1] for row in con.execute(
            f"PRAGMA table_info({table})").fetchall()]
        order = ", ".join(columns)
        rows = con.execute(f"SELECT * FROM {table} ORDER BY {order}").fetchall()
        for row in rows:
            hasher.update(
                json.dumps(list(row), sort_keys=False, default=str).encode("utf-8")
            )
    return hasher.hexdigest()


@dataclass
class OpenResult:
    replaced: bool
    reason: Optional[str]
    recovered_cursor: int


class Ledger:
    def __init__(self, path: str) -> None:
        self.path = path
        self.con = sqlite3.connect(path)
        self.con.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        with self.con:
            for statement in SCHEMA_STATEMENTS:
                self.con.execute(statement)
            existing = self.con.execute(
                "SELECT value FROM ledger_meta WHERE key='format_version'"
            ).fetchone()
            if existing is None:
                self.con.execute(
                    "INSERT INTO ledger_meta(key,value) VALUES('format_version',?)",
                    (str(LEDGER_FORMAT_VERSION),),
                )
                self.con.execute(
                    "INSERT INTO ledger_meta(key,value) VALUES('content_digest',?)",
                    (self._current_digest(),),
                )
                self.con.execute(
                    "INSERT OR IGNORE INTO progress(singleton,cursor,last_seq) VALUES(1,0,NULL)"
                )

    @classmethod
    def open_or_replace(cls, path: str, expected_digest: Optional[str] = None) -> OpenResult:
        """Open ledger; replace wholesale when format/digest identity is wrong."""
        replaced = False
        reason: Optional[str] = None
        cursor = 0
        if os.path.exists(path):
            probe = sqlite3.connect(path)
            try:
                version_row = probe.execute(
                    "SELECT value FROM ledger_meta WHERE key='format_version'"
                ).fetchone()
                stored_version = int(version_row[0]) if version_row else None
                if stored_version != LEDGER_FORMAT_VERSION:
                    replaced, reason = True, f"format-version {stored_version}!={LEDGER_FORMAT_VERSION}"
                elif expected_digest is not None:
                    digest_row = probe.execute(
                        "SELECT value FROM ledger_meta WHERE key='content_digest'"
                    ).fetchone()
                    if digest_row is None or digest_row[0] != expected_digest:
                        replaced, reason = True, "content-digest-mismatch"
                if not replaced:
                    progress = probe.execute(
                        "SELECT cursor FROM progress WHERE singleton=1"
                    ).fetchone()
                    cursor = int(progress[0]) if progress else 0
            except sqlite3.DatabaseError as exc:
                replaced, reason = True, f"corrupt-ledger:{exc}"
            finally:
                probe.close()
            if replaced:
                os.remove(path)
        ledger = cls(path)
        ledger.open_result = OpenResult(replaced=replaced, reason=reason,
                                      recovered_cursor=cursor)
        return ledger

    def _current_digest(self) -> str:
        return _row_digest(self.con)

    def refresh_digest(self) -> str:
        digest = self._current_digest()
        with self.con:
            self.con.execute(
                "UPDATE ledger_meta SET value=? WHERE key='content_digest'", (digest,)
            )
        return digest

    def get_meta(self, key: str) -> Optional[str]:
        row = self.con.execute(
            "SELECT value FROM ledger_meta WHERE key=?", (key,)
        ).fetchone()
        return None if row is None else row[0]

    # ------------------------------------------------------------------ #
    # Atomic writes
    # ------------------------------------------------------------------ #
    def commit_event(self, *, seq: int, op: str, node_id: Optional[str],
                     object_key: Optional[str], shard_index: Optional[int],
                     lease_id: Optional[str], merge_id: Optional[str],
                     payload: dict, cursor: int,
                     lease_row: Optional[dict] = None,
                     shard_row: Optional[dict] = None,
                     node_counter: Optional[dict] = None,
                     unknown: Optional[dict] = None) -> None:
        with self.con:
            self.con.execute(
                """INSERT OR REPLACE INTO events(seq,op,node_id,object_key,shard_index,lease_id,
                   merge_id,payload,cursor) VALUES(?,?,?,?,?,?,?,?,?)""",
                (seq, op, node_id, object_key, shard_index, lease_id, merge_id,
                 json.dumps(payload, sort_keys=True, default=str), cursor),
            )
            if lease_row is not None:
                self.con.execute(
                    """INSERT INTO leases(lease_id,node_id,object_key,shard_index,granted_ms,
                       expires_ms,renew_count,lease_version,state)
                       VALUES(:lease_id,:node_id,:object_key,:shard_index,:granted_ms,
                       :expires_ms,:renew_count,:lease_version,:state)
                    ON CONFLICT(lease_id) DO UPDATE SET
                       expires_ms=excluded.expires_ms,
                       renew_count=excluded.renew_count,
                       lease_version=excluded.lease_version,
                       state=excluded.state""",
                    lease_row,
                )
            if shard_row is not None:
                self.con.execute(
                    """INSERT OR REPLACE INTO shards(object_key,shard_index,node_id,merge_id,
                       shard_digest,shard_bytes,content_check,meta_version,source_format,merge_seq)
                       VALUES(:object_key,:shard_index,:node_id,:merge_id,:shard_digest,
                       :shard_bytes,:content_check,:meta_version,:source_format,:merge_seq)""",
                    shard_row,
                )
            if node_counter is not None:
                self.con.execute(
                    """INSERT INTO node_counters(node_id,capacity_slots,lease_limit,
                       active_count,remaining_slots)
                       VALUES(:node_id,:capacity_slots,:lease_limit,:active_count,
                       :remaining_slots)
                    ON CONFLICT(node_id) DO UPDATE SET
                       active_count=excluded.active_count,
                       remaining_slots=excluded.remaining_slots""",
                    node_counter,
                )
            if unknown is not None:
                self.con.execute(
                    "INSERT OR REPLACE INTO unknown_requests(seq,op,object_key,reason) "
                    "VALUES(:seq,:op,:object_key,:reason)",
                    unknown,
                )
            self.con.execute(
                "UPDATE progress SET cursor=?, last_seq=? WHERE singleton=1",
                (cursor, seq),
            )
            digest = self._current_digest()
            self.con.execute(
                "UPDATE ledger_meta SET value=? WHERE key='content_digest'", (digest,)
            )

    def init_node_counter(self, node_id: str, capacity_slots: int, lease_limit: int) -> None:
        with self.con:
            self.con.execute(
                """INSERT OR IGNORE INTO node_counters(node_id,capacity_slots,lease_limit,
                   active_count,remaining_slots) VALUES(?,?,?,0,?)""",
                (node_id, capacity_slots, lease_limit, capacity_slots),
            )

    # ------------------------------------------------------------------ #
    # Read / replay
    # ------------------------------------------------------------------ #
    def fetch_events(self) -> list[sqlite3.Row]:
        return self.con.execute("SELECT * FROM events ORDER BY seq ASC").fetchall()

    def fetch_leases(self) -> list[sqlite3.Row]:
        return self.con.execute("SELECT * FROM leases ORDER BY lease_id").fetchall()

    def fetch_shards(self) -> list[sqlite3.Row]:
        return self.con.execute(
            "SELECT * FROM shards ORDER BY object_key, shard_index, merge_seq"
        ).fetchall()

    def fetch_counters(self) -> list[sqlite3.Row]:
        return self.con.execute("SELECT * FROM node_counters ORDER BY node_id").fetchall()

    def fetch_unknown(self) -> list[sqlite3.Row]:
        return self.con.execute("SELECT * FROM unknown_requests ORDER BY seq").fetchall()

    def progress_cursor(self) -> int:
        row = self.con.execute(
            "SELECT cursor FROM progress WHERE singleton=1"
        ).fetchone()
        return int(row[0]) if row else 0

    def verify_digest(self) -> None:
        stored = self.get_meta("content_digest")
        actual = self._current_digest()
        if stored != actual:
            raise DivergenceError(
                "ledger content digest mismatch", field="content_digest",
                detail={"stored": stored, "actual": actual},
            )

    def close(self) -> None:
        self.con.close()


__all__ = ["Ledger", "OpenResult", "LEDGER_FORMAT_VERSION"]
