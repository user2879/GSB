"""SQLite 账本持久化、版本识别与强制终止恢复。

每个事件批次在 *同一事务* 内写入：逐事件结局、租约迁移、槽位占用、
等待时长与进度游标。账本头部携带整数格式版本与内容摘要（SHA-256）。

恢复语义：

* 只承认完整提交的批次（commit 原子性），未提交批次在 SQLite 回滚后自然不存在，
  其游标记为 ``unknown`` 并要求重跑；
* 格式版本不符或内容摘要不一致：整体替换账本（删除旧文件重建），不做增量合并；
* 活动租约集合与槽位占用仅依据已提交记录重放重建。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .engine import Outcome, TransitionRecord, WaitRecord
from .errors import LedgerError

LEDGER_FORMAT_VERSION = 1
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outcomes (
    seq INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lease_id TEXT NOT NULL,
    from_state TEXT NOT NULL,
    to_state TEXT NOT NULL,
    reason TEXT NOT NULL,
    clock_ms INTEGER NOT NULL,
    version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS waits (
    seq INTEGER PRIMARY KEY,
    instance_id TEXT NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    wait_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS slots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain_id TEXT NOT NULL,
    slot_index INTEGER NOT NULL,
    lease_id TEXT NOT NULL,
    event_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id INTEGER PRIMARY KEY,
    last_seq INTEGER NOT NULL,
    committed_at_ms INTEGER NOT NULL,
    digest TEXT NOT NULL
);
"""


@dataclass
class RecoveryState:
    last_committed_seq: int
    unknown_from_seq: int | None
    active_leases: list[dict]
    slot_owners: dict[str, dict[int, str]]
    replaced: bool
    digest_ok: bool


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    return conn


class Ledger:
    def __init__(self, path: str | Path, replace: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.replaced_on_open = False
        if replace and self.path.exists():
            self._delete_files()
            self.replaced_on_open = True
        self.conn = _connect(self.path)
        self.conn.executescript(SCHEMA_SQL)
        self._init_meta()
        self._verify_version()

    def _delete_files(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            target = Path(str(self.path) + suffix)
            if target.exists():
                target.unlink()

    def _init_meta(self) -> None:
        row = self.conn.execute("SELECT value FROM meta WHERE key='format_version'").fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES('format_version', ?)",
                (str(LEDGER_FORMAT_VERSION),),
            )
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES('content_digest', '')")
            self.conn.commit()

    def _verify_version(self) -> None:
        row = self.conn.execute("SELECT value FROM meta WHERE key='format_version'").fetchone()
        if int(row[0]) != LEDGER_FORMAT_VERSION:
            # 版本不一致：整体替换而非增量合并
            self.close()
            self._delete_files()
            self.replaced_on_open = True
            self.conn = _connect(self.path)
            self.conn.executescript(SCHEMA_SQL)
            self._init_meta()

    @staticmethod
    def digest_outcomes(outcomes: list[Outcome]) -> str:
        """对给定结局序列计算 SHA-256（全量摘要，校验器可独立复算）。"""
        h = hashlib.sha256()
        for outcome in outcomes:
            h.update(outcome.canonical_json())
            h.update(b"\n")
        return h.hexdigest()

    def rolling_digest(self, new_outcomes: list[Outcome]) -> str:
        """基于已提交结局 + 本批结局重算全量摘要，保证任意时刻都可独立复算。"""
        h = hashlib.sha256()
        rows = self.conn.execute("SELECT payload FROM outcomes ORDER BY seq").fetchall()
        for (payload,) in rows:
            h.update(payload.encode("utf-8"))
            h.update(b"\n")
        for outcome in new_outcomes:
            h.update(outcome.canonical_json())
            h.update(b"\n")
        return h.hexdigest()

    def current_digest(self) -> str:
        row = self.conn.execute("SELECT value FROM meta WHERE key='content_digest'").fetchone()
        return row[0] if row else ""

    def verify_digest(self) -> bool:
        rows = self.conn.execute("SELECT payload FROM outcomes ORDER BY seq").fetchall()
        h = hashlib.sha256()
        for (payload,) in rows:
            h.update(payload.encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest() == self.current_digest()

    def replace_if_digest_bad(self) -> bool:
        if not self.verify_digest():
            self.close()
            self._delete_files()
            self.replaced_on_open = True
            self.conn = _connect(self.path)
            self.conn.executescript(SCHEMA_SQL)
            self._init_meta()
            return True
        return False

    def append_batch(self, batch_id: int, outcomes: list[Outcome],
                     transitions: list[TransitionRecord], waits: list[WaitRecord],
                     slot_snapshot: dict[str, dict[int, str]], clock_ms: int) -> str:
        """一个事件批次的所有写入在同一事务提交；未提交则整体回滚。"""
        digest = self.rolling_digest(outcomes)
        last_seq = max(o.seq for o in outcomes)
        try:
            with self.conn:
                for outcome in outcomes:
                    self.conn.execute(
                        "INSERT OR REPLACE INTO outcomes(seq, path, payload) VALUES(?, ?, ?)",
                        (outcome.seq, "", outcome.canonical_json().decode("utf-8")),
                    )
                for tr in transitions:
                    self.conn.execute(
                        "INSERT INTO transitions(lease_id, from_state, to_state, reason, clock_ms, version)"
                        " VALUES(?, ?, ?, ?, ?, ?)",
                        (tr.lease_id, tr.from_state, tr.to_state, tr.reason, tr.clock_ms, tr.version),
                    )
                for wait in waits:
                    self.conn.execute(
                        "INSERT OR REPLACE INTO waits(seq, instance_id, start_ms, end_ms, wait_ms)"
                        " VALUES(?, ?, ?, ?, ?)",
                        (wait.seq, wait.instance_id, wait.start_ms, wait.end_ms, wait.wait_ms),
                    )
                for domain_id, owners in slot_snapshot.items():
                    for slot_index, lease_id in owners.items():
                        self.conn.execute(
                            "INSERT INTO slots(domain_id, slot_index, lease_id, event_seq)"
                            " VALUES(?, ?, ?, ?)",
                            (domain_id, slot_index, lease_id, last_seq),
                        )
                self.conn.execute(
                    "INSERT INTO batches(batch_id, last_seq, committed_at_ms, digest) VALUES(?, ?, ?, ?)",
                    (batch_id, last_seq, clock_ms, digest),
                )
                self.conn.execute(
                    "UPDATE meta SET value=? WHERE key='content_digest'", (digest,))
        except sqlite3.Error as exc:
            raise LedgerError("账本批次写入失败，事务已回滚", batch_id=batch_id) from exc
        return digest

    def load_outcomes(self) -> list[dict]:
        rows = self.conn.execute("SELECT payload FROM outcomes ORDER BY seq").fetchall()
        return [json.loads(payload) for (payload,) in rows]

    def last_committed_seq(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(last_seq), 0) FROM batches").fetchone()
        return int(row[0])

    def recover(self) -> RecoveryState:
        """仅依据已完整提交的记录重建活动租约与槽位占用。"""
        digest_ok = self.verify_digest()
        outcomes = self.load_outcomes()
        active: dict[str, dict] = {}
        slot_owners: dict[str, dict[int, str]] = {}
        for item in outcomes:
            lid = item["lease_id"]
            if not lid:
                continue
            if item["status"] in ("granted", "renewed"):
                active[lid] = item
                if item["slots"]:
                    slot_owners.setdefault(item["domain_id"], {})
                    for slot in (int(x) for x in item["slots"].split(",") if x != ""):
                        slot_owners[item["domain_id"]][slot] = lid
            elif item["status"] in ("released", "expired"):
                active.pop(lid, None)
                for owners in slot_owners.values():
                    for slot in [s for s, owner in owners.items() if owner == lid]:
                        owners.pop(slot, None)
        return RecoveryState(
            last_committed_seq=self.last_committed_seq(),
            unknown_from_seq=(self.last_committed_seq() + 1),
            active_leases=[active[key] for key in sorted(active)],
            slot_owners={d: dict(sorted(v.items())) for d, v in sorted(slot_owners.items())},
            replaced=self.replaced_on_open,
            digest_ok=digest_ok,
        )

    def close(self) -> None:
        self.conn.close()
