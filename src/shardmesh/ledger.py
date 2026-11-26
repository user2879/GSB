"""Atomic SQLite ledger and recovery primitives."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_dumps, record_digest, sha256_hex
from .timeutil import now_ns

LEDGER_FORMAT_VERSION = 1


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self.lock, self.conn:
            self.conn.execute("PRAGMA user_version=1")
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS shard_records (
                    shard_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    format_version INTEGER NOT NULL,
                    cursor_seq INTEGER NOT NULL,
                    content_digest TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    updated_at_ns INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS lease_grants (
                    lease_id TEXT PRIMARY KEY,
                    shard_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    lease_json TEXT NOT NULL,
                    updated_at_ns INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ledger_events (
                    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    content_digest TEXT NOT NULL,
                    created_at_ns INTEGER NOT NULL
                );
                """
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO meta(key,value) VALUES('format_version', ?)",
                (str(LEDGER_FORMAT_VERSION),),
            )
            self.conn.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('cursor_seq','0')")
            self.conn.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('snapshot_version','0')")

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    def meta_get(self, key: str, default: str | None = None) -> str | None:
        with self.lock:
            row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row is not None else default

    def meta_set(self, key: str, value: Any) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(value)),
            )

    def initialize_spec(self, spec_dict: dict[str, Any]) -> str:
        digest = sha256_hex(spec_dict)
        self.meta_set("spec_digest", digest)
        self.meta_set("spec_json", canonical_dumps(spec_dict))
        return digest

    def write_lease_grant(self, lease_dict: dict[str, Any], state: str = "active") -> None:
        payload = canonical_dumps(lease_dict)
        with self.lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO lease_grants(lease_id,shard_id,node_id,state,lease_json,updated_at_ns)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(lease_id) DO UPDATE SET
                  shard_id=excluded.shard_id,
                  node_id=excluded.node_id,
                  state=excluded.state,
                  lease_json=excluded.lease_json,
                  updated_at_ns=excluded.updated_at_ns
                """,
                (
                    lease_dict["lease_id"],
                    lease_dict["shard_id"],
                    lease_dict["node_id"],
                    state,
                    payload,
                    now_ns(),
                ),
            )

    def update_lease_state(self, lease_id: str, state: str) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                "UPDATE lease_grants SET state=?, updated_at_ns=? WHERE lease_id=?",
                (state, now_ns(), lease_id),
            )

    def list_open_lease_grants(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT lease_json FROM lease_grants WHERE state='active' ORDER BY lease_id"
            ).fetchall()
        return [json.loads(row["lease_json"]) for row in rows]

    def reclaim_open_leases(self, reason: str = "process_restart") -> list[dict[str, Any]]:
        leases = self.list_open_lease_grants()
        with self.lock, self.conn:
            self.conn.execute(
                "UPDATE lease_grants SET state=?, updated_at_ns=? WHERE state='active'",
                (f"reclaimed:{reason}", now_ns()),
            )
            self.append_event("leases_reclaimed", {"leases": leases, "reason": reason}, commit=False)
        return leases

    def next_cursor(self) -> int:
        with self.lock, self.conn:
            row = self.conn.execute("SELECT value FROM meta WHERE key='cursor_seq'").fetchone()
            current = int(row["value"])
            nxt = current + 1
            self.conn.execute("UPDATE meta SET value=? WHERE key='cursor_seq'", (str(nxt),))
            return nxt

    def append_event(self, event_type: str, payload: dict[str, Any], *, commit: bool = True) -> int:
        body = canonical_dumps(payload)
        digest = sha256_hex(payload)

        def insert(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                "INSERT INTO ledger_events(event_type,payload_json,content_digest,created_at_ns) VALUES(?,?,?,?)",
                (event_type, body, digest, now_ns()),
            )
            return int(cursor.lastrowid)

        if commit:
            with self.lock, self.conn:
                return insert(self.conn)
        return insert(self.conn)

    def commit_outcome(self, record: dict[str, Any]) -> dict[str, Any]:
        record = dict(record)
        record["format_version"] = LEDGER_FORMAT_VERSION
        record["content_digest"] = record_digest(record)
        payload = canonical_dumps(record)
        shard_id = record["shard_id"]
        state = record.get("state", "committed")
        lease_id = record.get("lease", {}).get("lease_id")
        with self.lock, self.conn:
            cursor_row = self.conn.execute("SELECT value FROM meta WHERE key='cursor_seq'").fetchone()
            cursor = int(cursor_row["value"]) + 1
            self.conn.execute("UPDATE meta SET value=? WHERE key='cursor_seq'", (str(cursor),))
            record["cursor_seq"] = cursor
            record["content_digest"] = record_digest(record)
            payload = canonical_dumps(record)
            self.conn.execute(
                """
                INSERT INTO shard_records(shard_id,state,format_version,cursor_seq,content_digest,record_json,updated_at_ns)
                VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(shard_id) DO UPDATE SET
                  state=excluded.state,
                  format_version=excluded.format_version,
                  cursor_seq=excluded.cursor_seq,
                  content_digest=excluded.content_digest,
                  record_json=excluded.record_json,
                  updated_at_ns=excluded.updated_at_ns
                """,
                (shard_id, state, LEDGER_FORMAT_VERSION, cursor, record["content_digest"], payload, now_ns()),
            )
            if lease_id:
                self.conn.execute(
                    "UPDATE lease_grants SET state='released', updated_at_ns=? WHERE lease_id=?",
                    (now_ns(), lease_id),
                )
            self.conn.execute(
                "INSERT INTO ledger_events(event_type,payload_json,content_digest,created_at_ns) VALUES(?,?,?,?)",
                ("shard_outcome", payload, record["content_digest"], now_ns()),
            )
        return record

    def mark_shard_state(self, shard_id: str, state: str, marker: dict[str, Any] | None = None) -> None:
        record = {
            "shard_id": shard_id,
            "state": state,
            "marker": marker or {},
            "format_version": LEDGER_FORMAT_VERSION,
            "cursor_seq": int(self.meta_get("cursor_seq", "0") or "0"),
        }
        record["content_digest"] = record_digest(record)
        with self.lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO shard_records(shard_id,state,format_version,cursor_seq,content_digest,record_json,updated_at_ns)
                VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(shard_id) DO UPDATE SET state=excluded.state, record_json=excluded.record_json,
                  content_digest=excluded.content_digest, updated_at_ns=excluded.updated_at_ns
                """,
                (shard_id, state, LEDGER_FORMAT_VERSION, record["cursor_seq"], record["content_digest"], canonical_dumps(record), now_ns()),
            )

    def flag_records(self, shard_ids: Iterable[str], state: str, **flags: Any) -> list[str]:
        changed: list[str] = []
        with self.lock, self.conn:
            for shard_id in shard_ids:
                row = self.conn.execute("SELECT record_json FROM shard_records WHERE shard_id=?", (shard_id,)).fetchone()
                if row is None:
                    continue
                record = json.loads(row["record_json"])
                record.update(flags)
                record["state"] = state
                record["format_version"] = LEDGER_FORMAT_VERSION
                record["content_digest"] = record_digest(record)
                self.conn.execute(
                    "UPDATE shard_records SET state=?, content_digest=?, record_json=?, updated_at_ns=? WHERE shard_id=?",
                    (state, record["content_digest"], canonical_dumps(record), now_ns(), shard_id),
                )
                changed.append(shard_id)
        return changed

    def get_raw_row(self, shard_id: str) -> sqlite3.Row | None:
        with self.lock:
            return self.conn.execute("SELECT * FROM shard_records WHERE shard_id=?", (shard_id,)).fetchone()

    def get_record(self, shard_id: str) -> dict[str, Any] | None:
        row = self.get_raw_row(shard_id)
        return json.loads(row["record_json"]) if row is not None else None

    def list_rows(self) -> list[sqlite3.Row]:
        with self.lock:
            return list(self.conn.execute("SELECT * FROM shard_records ORDER BY shard_id").fetchall())

    def list_records(self) -> list[dict[str, Any]]:
        records = []
        for row in self.list_rows():
            record = json.loads(row["record_json"])
            record["_row_state"] = row["state"]
            records.append(record)
        return records

    def committed_outcome_ids(self) -> set[str]:
        ids: set[str] = set()
        for row in self.list_rows():
            record = json.loads(row["record_json"])
            if row["state"] in {"committed", "current", "stale"} and record.get("outcome"):
                ids.add(row["shard_id"])
        return ids

    def verify_row_digests(self) -> list[dict[str, str]]:
        mismatches = []
        for row in self.list_rows():
            record = json.loads(row["record_json"])
            expected = record_digest(record)
            if expected != row["content_digest"] or expected != record.get("content_digest"):
                mismatches.append({"shard_id": row["shard_id"], "field": "content_digest"})
        return mismatches

    def increment_snapshot_version(self) -> int:
        with self.lock, self.conn:
            row = self.conn.execute("SELECT value FROM meta WHERE key='snapshot_version'").fetchone()
            version = int(row["value"]) + 1
            self.conn.execute("UPDATE meta SET value=? WHERE key='snapshot_version'", (str(version),))
            return version

    def all_meta(self) -> dict[str, str]:
        with self.lock:
            rows = self.conn.execute("SELECT key,value FROM meta ORDER BY key").fetchall()
        return {row["key"]: row["value"] for row in rows}


def verify_records_field_by_field(
    actual_records: Iterable[dict[str, Any]],
    expected_records: Iterable[dict[str, Any]],
    *,
    compare_fields: Iterable[str] | None = None,
) -> None:
    actual = {record["shard_id"]: record for record in actual_records}
    expected = {record["shard_id"]: record for record in expected_records}
    for shard_id in sorted(set(actual) | set(expected)):
        if shard_id not in actual:
            from .errors import ShardMeshError

            raise ShardMeshError("E_DIVERGE", f"missing shard {shard_id} in incremental result", category="shard", field_name="shard_id")
        if shard_id not in expected:
            from .errors import ShardMeshError

            raise ShardMeshError("E_DIVERGE", f"unexpected shard {shard_id} in incremental result", category="shard", field_name="shard_id")
        fields = compare_fields or sorted(set(actual[shard_id]) | set(expected[shard_id]))
        for field_name in fields:
            left = actual[shard_id].get(field_name)
            right = expected[shard_id].get(field_name)
            if left != right:
                from .errors import ShardMeshError

                node_id = None
                if isinstance(right, dict):
                    node_id = right.get("node_id")
                raise ShardMeshError(
                    "E_DIVERGE",
                    f"field mismatch for shard {shard_id}: {field_name}",
                    category="shard",
                    field_name=field_name,
                    details={"shard_id": shard_id, "node_id": node_id, "incremental": left, "full_rebuild": right},
                )
