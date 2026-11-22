"""盖章/验章账本：SQLite 持久化、格式版本与内容摘要识别、整体替换。

账本携带整数格式版本 LEDGER_FORMAT_VERSION 与内容摘要 content_digest。
重新打开时若格式版本或内容摘要不一致，绝不增量合并：
先把旧库复制为只读备份，再整体重建盖章记录与验章状态。
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import SnapshotError

LEDGER_FORMAT_VERSION = 1


def _frame(tag: str, payload: bytes) -> bytes:
    tag_bytes = tag.encode("utf-8")
    return (
        len(tag_bytes).to_bytes(2, "big")
        + tag_bytes
        + len(payload).to_bytes(8, "big")
        + payload
    )


def compute_content_digest(stamp_rows: list[tuple[Any, ...]],
                           verification_rows: list[tuple[Any, ...]]) -> str:
    """对全部盖章记录与验章结论做规范化摘要，顺序敏感且稳定。"""
    hasher = hashlib.sha256()

    def consume(prefix: str, rows: list[tuple[Any, ...]]) -> None:
        hasher.update(_frame(prefix + "-count",
                             len(rows).to_bytes(8, "big")))
        for row in rows:
            for cell in row:
                if cell is None:
                    hasher.update(_frame("null", b""))
                elif isinstance(cell, bool):
                    hasher.update(_frame("bool",
                                         b"\x01" if cell else b"\x00"))
                elif isinstance(cell, int):
                    hasher.update(_frame("int",
                                         cell.to_bytes(8, "big",
                                                       signed=True)))
                elif isinstance(cell, float):
                    hasher.update(_frame(
                        "float",
                        np.asarray(cell, dtype=np.float64).tobytes(),
                    ))
                else:
                    hasher.update(_frame(
                        "text", str(cell).encode("utf-8")))

    consume("stamps", sorted(stamp_rows, key=lambda r: str(r[0])))
    consume("verifications",
            sorted(verification_rows, key=lambda r: str(r[0])))
    return hasher.hexdigest()


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stamps (
    token_id TEXT PRIMARY KEY,
    backend_id TEXT NOT NULL,
    identity_digest TEXT NOT NULL,
    identity_version INTEGER NOT NULL,
    token_spec_version INTEGER NOT NULL,
    request_seq INTEGER NOT NULL,
    stamp_seq INTEGER NOT NULL,
    checksum TEXT NOT NULL,
    stamped_at_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS verifications (
    token_id TEXT PRIMARY KEY,
    accepted INTEGER NOT NULL,
    code TEXT NOT NULL,
    reason TEXT NOT NULL,
    field TEXT NOT NULL,
    verified_at_ms INTEGER NOT NULL
);
"""


@dataclass(frozen=True)
class StampRow:
    token_id: str
    backend_id: str
    identity_digest: str
    identity_version: int
    token_spec_version: int
    request_seq: int
    stamp_seq: int
    checksum: str
    stamped_at_ms: int

    def as_tuple(self) -> tuple[Any, ...]:
        return (
            self.token_id, self.backend_id, self.identity_digest,
            self.identity_version, self.token_spec_version,
            self.request_seq, self.stamp_seq, self.checksum,
            self.stamped_at_ms,
        )


@dataclass(frozen=True)
class VerificationRow:
    token_id: str
    accepted: bool
    code: str
    reason: str
    field: str
    verified_at_ms: int

    def as_tuple(self) -> tuple[Any, ...]:
        return (self.token_id, self.accepted, self.code, self.reason,
                self.field, self.verified_at_ms)


@dataclass
class ReplacementReport:
    replaced: bool
    reason: str
    backup_path: str
    format_version: int
    expected_digest: str
    stored_digest: str
    stamp_token_ids: tuple[str, ...]
    verification_token_ids: tuple[str, ...]


class LedgerSession:
    """单连接账本，显式事务保证逐条完整提交。"""

    def __init__(self, db_path: Path, connection: sqlite3.Connection) -> None:
        self.db_path = db_path
        self.conn = connection
        self._lock = threading.RLock()

    def close(self) -> None:
        self._refresh_meta_digest()
        self.conn.close()

    def __enter__(self) -> "LedgerSession":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def get_stamp(self, token_id: str) -> StampRow | None:
        with self._lock:
            cur = self.conn.execute(
                "SELECT token_id, backend_id, identity_digest, identity_version,"
                " token_spec_version, request_seq, stamp_seq, checksum,"
                " stamped_at_ms FROM stamps WHERE token_id = ?",
                (token_id,),
            )
            row = cur.fetchone()
            return StampRow(*row) if row else None

    def max_stamp_seq(self, backend_id: str) -> int:
        with self._lock:
            cur = self.conn.execute(
                "SELECT COALESCE(MAX(stamp_seq), 0) FROM stamps "
                "WHERE backend_id = ?",
                (backend_id,),
            )
            return int(cur.fetchone()[0])

    def list_stamps(self) -> list[StampRow]:
        with self._lock:
            cur = self.conn.execute(
                "SELECT token_id, backend_id, identity_digest, identity_version,"
                " token_spec_version, request_seq, stamp_seq, checksum,"
                " stamped_at_ms FROM stamps ORDER BY token_id"
            )
            return [StampRow(*row) for row in cur.fetchall()]

    def list_verifications(self) -> list[VerificationRow]:
        with self._lock:
            cur = self.conn.execute(
                "SELECT token_id, accepted, code, reason, field, verified_at_ms"
                " FROM verifications ORDER BY token_id"
            )
            return [VerificationRow(*row) for row in cur.fetchall()]

    def upsert_stamp(self, row: StampRow) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO stamps (token_id, backend_id, identity_digest,"
                " identity_version, token_spec_version, request_seq,"
                " stamp_seq, checksum, stamped_at_ms) VALUES (?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(token_id) DO UPDATE SET backend_id=excluded.backend_id,"
                " identity_digest=excluded.identity_digest,"
                " identity_version=excluded.identity_version,"
                " token_spec_version=excluded.token_spec_version,"
                " request_seq=excluded.request_seq,"
                " stamp_seq=excluded.stamp_seq, checksum=excluded.checksum,"
                " stamped_at_ms=excluded.stamped_at_ms",
                row.as_tuple(),
            )

    def replace_verification(self, row: VerificationRow) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO verifications (token_id, accepted, code, reason,"
                " field, verified_at_ms) VALUES (?,?,?,?,?,?)"
                " ON CONFLICT(token_id) DO UPDATE SET accepted=excluded.accepted,"
                " code=excluded.code, reason=excluded.reason,"
                " field=excluded.field, verified_at_ms=excluded.verified_at_ms",
                row.as_tuple(),
            )

    def delete_stamp(self, token_id: str) -> None:
        with self._lock, self.conn:
            self.conn.execute("DELETE FROM stamps WHERE token_id = ?",
                              (token_id,))

    def snapshot_digest(self) -> str:
        rows = self._raw_rows()
        return compute_content_digest(rows[0], rows[1])

    def _raw_rows(self) -> tuple[list[tuple[Any, ...]],
                                 list[tuple[Any, ...]]]:
        stamp_rows = [row.as_tuple() for row in self.list_stamps()]
        verification_rows = [row.as_tuple()
                             for row in self.list_verifications()]
        return stamp_rows, verification_rows

    def _refresh_meta_digest(self) -> None:
        with self._lock:
            stamp_rows, verification_rows = self._raw_rows()
            digest = compute_content_digest(stamp_rows, verification_rows)
            with self.conn:
                self.conn.execute(
                "INSERT INTO meta(key, value) VALUES('content_digest', ?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (digest,),
            )
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES('format_version', ?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(LEDGER_FORMAT_VERSION),),
            )


def _read_meta(connection: sqlite3.Connection) -> dict[str, str]:
    try:
        cur = connection.execute("SELECT key, value FROM meta")
    except sqlite3.DatabaseError as exc:
        raise SnapshotError(f"账本结构损坏: {exc}", field="schema") from exc
    return dict(cur.fetchall())


def _make_readonly_backup(db_path: Path) -> Path:
    stem = db_path.stem
    index = 1
    while True:
        backup = db_path.with_name(f"{stem}.backup-{index:03d}.db")
        if not backup.exists():
            break
        index += 1
    shutil.copy2(db_path, backup)
    os.chmod(backup, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return backup


def open_ledger(db_path: str | Path, *,
                on_mismatch: str = "replace") \
        -> tuple[LedgerSession, ReplacementReport | None]:
    """打开账本；格式/摘要不符时整体替换。

    on_mismatch:
      - "replace"（默认）：备份旧库后整体重建；
      - "raise"：抛 E_SNAPSHOT，供异常注入与核对场景使用。
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        connection = sqlite3.connect(str(db_path), check_same_thread=False)
        connection.executescript(SCHEMA)
        session = LedgerSession(db_path, connection)
        session._refresh_meta_digest()
        return session, None

    probe = sqlite3.connect(str(db_path))
    try:
        meta = _read_meta(probe)
        old_stamps = probe.execute(
            "SELECT token_id, backend_id, identity_digest, identity_version,"
            " token_spec_version, request_seq, stamp_seq, checksum,"
            " stamped_at_ms FROM stamps ORDER BY token_id"
        ).fetchall()
        old_verifications = probe.execute(
            "SELECT token_id, accepted, code, reason, field, verified_at_ms"
            " FROM verifications ORDER BY token_id"
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        probe.close()
        if on_mismatch == "raise":
            raise SnapshotError(f"账本结构损坏: {exc}", field="schema")
        backup = _make_readonly_backup(db_path)
        connection = _rebuild_fresh(db_path)
        report = ReplacementReport(
            replaced=True, reason="schema-corrupt", backup_path=str(backup),
            format_version=LEDGER_FORMAT_VERSION, expected_digest="",
            stored_digest="", stamp_token_ids=tuple(r[0] for r in old_stamps),
            verification_token_ids=tuple(r[0] for r in old_verifications),
        )
        return LedgerSession(db_path, connection), report

    stored_version = int(meta.get("format_version", "0"))
    stored_digest = meta.get("content_digest", "")
    actual_digest = compute_content_digest(
        list(old_stamps), list(old_verifications))
    probe.close()

    version_ok = stored_version == LEDGER_FORMAT_VERSION
    digest_ok = stored_digest == actual_digest
    if version_ok and digest_ok:
        connection = sqlite3.connect(str(db_path), check_same_thread=False)
        return LedgerSession(db_path, connection), None

    reason = "format-version" if not version_ok else "content-digest"
    if on_mismatch == "raise":
        raise SnapshotError(
            f"账本{reason}不一致，拒绝打开",
            field=reason,
            detail={"stored_digest": stored_digest,
                    "actual_digest": actual_digest,
                    "stored_version": stored_version,
                    "expected_version": LEDGER_FORMAT_VERSION},
        )

    backup = _make_readonly_backup(db_path)
    connection = _rebuild_fresh(db_path)
    report = ReplacementReport(
        replaced=True, reason=reason, backup_path=str(backup),
        format_version=LEDGER_FORMAT_VERSION,
        expected_digest=actual_digest, stored_digest=stored_digest,
        stamp_token_ids=tuple(r[0] for r in old_stamps),
        verification_token_ids=tuple(r[0] for r in old_verifications),
    )
    return LedgerSession(db_path, connection), report


def _rebuild_fresh(db_path: Path) -> sqlite3.Connection:
    db_path.unlink()
    connection = sqlite3.connect(str(db_path), check_same_thread=False)
    connection.executescript(SCHEMA)
    with connection:
        connection.execute(
            "INSERT INTO meta(key, value) VALUES('format_version', ?)",
            (str(LEDGER_FORMAT_VERSION),),
        )
        connection.execute(
            "INSERT INTO meta(key, value) VALUES('content_digest', ?)",
            (compute_content_digest([], []),),
        )
    return connection


def tamper_stamp_checksum(db_path: str | Path, token_id: str,
                          bogus: str = "0" * 64) -> None:
    """异常注入：直接篡改库内盖章校验值，不更新内容摘要。"""
    connection = sqlite3.connect(str(db_path))
    try:
        connection.execute(
            "UPDATE stamps SET checksum = ? WHERE token_id = ?",
            (bogus, token_id),
        )
        connection.commit()
    finally:
        connection.close()


def tamper_content_digest(db_path: str | Path) -> None:
    """异常注入：破坏账本 meta 中的内容摘要。"""
    connection = sqlite3.connect(str(db_path))
    try:
        connection.execute(
            "UPDATE meta SET value = ? WHERE key = 'content_digest'",
            ("deadbeef" * 8,),
        )
        connection.commit()
    finally:
        connection.close()
