"""SQLite ledger, digest verification and whole-document snapshot recovery."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from . import FORMAT_VERSION
from .canonical import canonical_text, sha256_hex
from .errors import SnapshotError


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ledger_header (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    format_version INTEGER NOT NULL,
    content_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ledger_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS intents (
    intent_id TEXT PRIMARY KEY,
    event_seq INTEGER NOT NULL,
    lease_id TEXT NOT NULL,
    op TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_monotonic_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_seq INTEGER NOT NULL,
    record_kind TEXT NOT NULL,
    lease_id TEXT,
    generation INTEGER,
    payload_json TEXT NOT NULL,
    payload_digest TEXT NOT NULL
);
"""


class Ledger:
    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        if read_only:
            if not self.path.exists():
                raise SnapshotError(
                    "ledger file does not exist", field="ledger", details={"path": str(path)}
                )
            uri = self.path.resolve().as_uri() + "?mode=ro"
            self.conn = sqlite3.connect(uri, uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        if not read_only:
            self.conn.execute("PRAGMA synchronous=FULL")
            self.conn.execute("PRAGMA journal_mode=DELETE")
            self.conn.executescript(SCHEMA_SQL)
            self._initialize_meta()
        self.verify_digest()

    @classmethod
    def open(cls, path: str | Path) -> "Ledger":
        return cls(path)

    @classmethod
    def open_readonly(cls, path: str | Path) -> "Ledger":
        return cls(path, read_only=True)

    def _initialize_meta(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO ledger_header(id, format_version, content_digest)
                VALUES(1, ?, '')
                """,
                (FORMAT_VERSION,),
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO ledger_meta(key, value) VALUES('format_version', ?)",
                (str(FORMAT_VERSION),),
            )
            digest = self._calculate_digest()
            header = self.conn.execute(
                "SELECT content_digest FROM ledger_header WHERE id=1"
            ).fetchone()
            meta = self.conn.execute(
                "SELECT value FROM ledger_meta WHERE key='content_digest'"
            ).fetchone()
            if not header["content_digest"]:
                self.conn.execute(
                    "UPDATE ledger_header SET content_digest=? WHERE id=1",
                    (digest,),
                )
            if meta is None:
                self.conn.execute(
                    "INSERT INTO ledger_meta(key, value) VALUES('content_digest', ?)",
                    (digest,),
                )

    def _row_digest(self, row_id: int, payload_digest: str) -> str:
        return sha256_hex(f"{FORMAT_VERSION}:{row_id}:{payload_digest}")

    def _calculate_digest(self) -> str:
        rows = self.conn.execute(
            "SELECT id, payload_digest FROM entries ORDER BY id ASC"
        ).fetchall()
        chain = sha256_hex(f"ledger-format-{FORMAT_VERSION}")
        for row in rows:
            chain = sha256_hex(chain + ":" + self._row_digest(row["id"], row["payload_digest"]))
        return "sha256:" + chain

    def verify_digest(self) -> str:
        header = self.conn.execute(
            "SELECT format_version, content_digest FROM ledger_header WHERE id=1"
        ).fetchone()
        if header is None or not isinstance(header["format_version"], int):
            raise SnapshotError(
                "ledger integer format_version header is missing",
                field="ledger_header.format_version",
                details={"expected": FORMAT_VERSION},
            )
        if header["format_version"] != FORMAT_VERSION:
            raise SnapshotError(
                "ledger format_version is missing or unsupported",
                field="ledger_header.format_version",
                details={"expected": FORMAT_VERSION, "actual": header["format_version"]},
            )
        meta_version = self.conn.execute(
            "SELECT value FROM ledger_meta WHERE key='format_version'"
        ).fetchone()
        if meta_version is None or meta_version["value"] != str(FORMAT_VERSION):
            raise SnapshotError(
                "ledger format_version is missing or unsupported",
                field="ledger_meta.format_version",
                details={"expected": FORMAT_VERSION},
            )
        stored = self.conn.execute(
            "SELECT value FROM ledger_meta WHERE key='content_digest'"
        ).fetchone()
        stored_digest = header["content_digest"]
        for row in self.conn.execute("SELECT id, payload_json, payload_digest FROM entries"):
            actual = "sha256:" + sha256_hex(canonical_text(json.loads(row["payload_json"])))
            if actual != row["payload_digest"]:
                raise SnapshotError(
                    "ledger record content digest mismatch",
                    field=f"entries[{row['id']}].payload_json",
                    details={"entry_id": row["id"]},
                )
        actual_digest = self._calculate_digest()
        if stored_digest != actual_digest or (
            stored is not None and stored["value"] != actual_digest
        ):
            raise SnapshotError(
                "ledger content digest does not match committed records",
                field="ledger_header.content_digest",
                details={
                    "stored": stored_digest,
                    "meta": stored["value"] if stored else None,
                    "actual": actual_digest,
                },
            )
        return actual_digest

    @property
    def content_digest(self) -> str:
        row = self.conn.execute(
            "SELECT content_digest FROM ledger_header WHERE id=1"
        ).fetchone()
        return str(row["content_digest"])

    def record_intent(
        self,
        *,
        intent_id: str,
        event_seq: int,
        lease_id: str,
        op: str,
        payload: dict[str, Any],
        monotonic_ms: int,
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO intents(
                    intent_id, event_seq, lease_id, op, payload_json,
                    status, created_monotonic_ms
                ) VALUES(?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    intent_id,
                    event_seq,
                    lease_id,
                    op,
                    canonical_text(payload),
                    monotonic_ms,
                ),
            )

    def append_records(
        self,
        records: list[dict[str, Any]],
        *,
        intent_id: str | None = None,
        intent_ids: list[str] | tuple[str, ...] | None = None,
    ) -> list[int]:
        ids: list[int] = []
        complete_intent_ids = list(intent_ids or [])
        if intent_id is not None:
            complete_intent_ids.append(intent_id)
        with self.conn:
            for record in records:
                payload = canonical_text(record)
                digest = "sha256:" + sha256_hex(payload)
                lease_id = record.get("lease_id")
                generation = record.get("generation_after")
                if generation is None:
                    generation = record.get("generation")
                cursor = self.conn.execute(
                    """
                    INSERT INTO entries(
                        event_seq, record_kind, lease_id, generation,
                        payload_json, payload_digest
                    ) VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(record.get("event_seq", 0)),
                        str(record.get("entry_kind", "unknown")),
                        lease_id,
                        generation,
                        payload,
                        digest,
                    ),
                )
                ids.append(int(cursor.lastrowid))
            if complete_intent_ids:
                self.conn.execute(
                    "UPDATE intents SET status='committed' WHERE intent_id IN (%s)"
                    % ",".join("?" for _ in complete_intent_ids),
                    tuple(complete_intent_ids),
                )
            digest_value = self._calculate_digest()
            self.conn.execute(
                "UPDATE ledger_header SET content_digest=? WHERE id=1",
                (digest_value,),
            )
            self.conn.execute(
                "UPDATE ledger_meta SET value=? WHERE key='content_digest'",
                (digest_value,),
            )
        return ids

    def records(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT payload_json FROM entries ORDER BY id ASC").fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def pending_intents(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM intents WHERE status='pending' ORDER BY event_seq ASC, intent_id ASC"
        ).fetchall()
        return [dict(row) for row in rows]

    def recovery_report(self) -> dict[str, Any]:
        pending = self.pending_intents()
        unknown = sorted(
            {
                "lease_id": item["lease_id"],
                "intent_id": item["intent_id"],
                "op": item["op"],
                "event_seq": item["event_seq"],
            }
            for item in pending
        )
        return {
            "format_version": FORMAT_VERSION,
            "content_digest": self.content_digest,
            "committed_record_count": len(self.records()),
            "unknown_leases": unknown,
            "recovery_rule": "only complete SQLite transactions are authoritative",
        }

    def close(self) -> None:
        self.conn.close()


def write_snapshot(path: str | Path, state: dict[str, Any]) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = "sha256:" + sha256_hex(canonical_text(state))
    envelope = {
        "format_version": FORMAT_VERSION,
        "content_digest": digest,
        "state": state,
    }
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(canonical_text(envelope) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return envelope


def load_valid_snapshot(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotError("snapshot is unreadable or invalid JSON", field=str(path)) from exc
    if not isinstance(envelope, dict):
        raise SnapshotError("snapshot must be an object", field=str(path))
    version = envelope.get("format_version")
    if not isinstance(version, int) or isinstance(version, bool) or version != FORMAT_VERSION:
        raise SnapshotError(
            "snapshot format_version is unsupported",
            field="format_version",
            details={"actual": version, "expected": FORMAT_VERSION},
        )
    state = envelope.get("state")
    expected_digest = envelope.get("content_digest")
    actual_digest = "sha256:" + sha256_hex(canonical_text(state))
    if not isinstance(expected_digest, str) or expected_digest != actual_digest:
        raise SnapshotError(
            "snapshot content_digest mismatch",
            field="content_digest",
            details={"expected": actual_digest, "actual": expected_digest},
        )
    return state


def recover_snapshot(path: str | Path, ledger: Ledger) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a valid snapshot or replace it wholesale from committed ledger."""

    try:
        state = load_valid_snapshot(path)
        return state, {"replaced": False, "path": str(path)}
    except SnapshotError as exc:
        state = {
            "format_version": FORMAT_VERSION,
            "source": "sqlite_ledger_whole_replacement",
            "records": ledger.records(),
            "recovery": ledger.recovery_report(),
        }
        write_snapshot(path, state)
        return state, {
            "replaced": True,
            "path": str(path),
            "reason_code": exc.code,
            "reason_field": exc.field,
        }
