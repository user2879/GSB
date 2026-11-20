"""SQLite-backed, version-scoped cache consistency ledger."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .cache import monotonic_ms
from .errors import EdgeError
from .model import ContentSnapshot, LoadedSpec, Policy, cache_key, canonical_json_bytes
from .spec import parse_content_records, parse_policy


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ledger_versions (
    ledger_version INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL CHECK(status IN ('pending','complete','unknown')),
    round_kind TEXT NOT NULL,
    progress_cursor INTEGER NOT NULL DEFAULT 0,
    content_version TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    policy_digest TEXT NOT NULL,
    spec_digest TEXT NOT NULL,
    content_json TEXT NOT NULL,
    policy_json TEXT NOT NULL,
    error_detail TEXT
);
CREATE TABLE IF NOT EXISTS entries (
    ledger_version INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    path TEXT NOT NULL,
    etag TEXT NOT NULL,
    content_version TEXT NOT NULL,
    written_mono_ms INTEGER NOT NULL,
    ttl_ms INTEGER NOT NULL,
    remaining_ttl_ms INTEGER NOT NULL,
    entry_version INTEGER NOT NULL,
    content_size INTEGER NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (ledger_version, node_id, cache_key)
);
CREATE TABLE IF NOT EXISTS evictions (
    ledger_version INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    event_seq INTEGER NOT NULL,
    cache_key TEXT NOT NULL,
    path TEXT NOT NULL,
    etag TEXT NOT NULL,
    reason TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (ledger_version, node_id, event_seq)
);
CREATE TABLE IF NOT EXISTS propagation (
    ledger_version INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    trigger_path TEXT NOT NULL,
    affected_path TEXT NOT NULL,
    depth INTEGER NOT NULL,
    content_version TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    PRIMARY KEY (ledger_version, seq)
);
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    ledger_version INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    policy_digest TEXT NOT NULL,
    created_at TEXT NOT NULL,
    read_only INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entries_version_node ON entries(ledger_version, node_id, cache_key);
CREATE INDEX IF NOT EXISTS idx_evictions_version ON evictions(ledger_version, node_id, event_seq);
CREATE INDEX IF NOT EXISTS idx_propagation_version ON propagation(ledger_version, seq);
"""


class Ledger:
    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        if read_only:
            if not self.path.exists():
                raise EdgeError(
                    code="E_STATE",
                    message="ledger database does not exist",
                    details={"path": str(self.path)},
                )
            self.connection = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10.0)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(str(self.path), timeout=10.0)
        self.connection.row_factory = sqlite3.Row
        if read_only:
            self.connection.execute("PRAGMA query_only=ON")
        else:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.executescript(SCHEMA)
            self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def recover_pending(self) -> list[int]:
        """Mark interrupted pending rounds unknown; they are never trusted."""

        with self.connection:
            rows = self.connection.execute(
                "SELECT ledger_version FROM ledger_versions WHERE status='pending' ORDER BY ledger_version"
            ).fetchall()
            versions = [int(row["ledger_version"]) for row in rows]
            for version in versions:
                self.connection.execute(
                    "UPDATE ledger_versions SET status='unknown', error_detail=?, completed_at=? WHERE ledger_version=? AND status='pending'",
                    (
                        "process terminated before the round transaction completed; rerun required",
                        utc_now(),
                        version,
                    ),
                )
        return versions

    def begin_round(self, spec: LoadedSpec, *, round_kind: str) -> int:
        with self.connection:
            row = self.connection.execute("SELECT COALESCE(MAX(ledger_version), 0) AS value FROM ledger_versions").fetchone()
            ledger_version = int(row["value"]) + 1
            self.connection.execute(
                """
                INSERT INTO ledger_versions(
                    ledger_version, created_at, status, round_kind, progress_cursor,
                    content_version, policy_version, content_digest, policy_digest, spec_digest,
                    content_json, policy_json
                ) VALUES (?, ?, 'pending', ?, 0, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ledger_version,
                    utc_now(),
                    round_kind,
                    spec.content_version,
                    spec.policy_version,
                    spec.content_digest,
                    spec.policy_digest,
                    spec.digest,
                    _json(spec.content.to_declared_list()),
                    _json(spec.policy.to_dict()),
                ),
            )
        return ledger_version

    def mark_unknown(self, ledger_version: int, reason: str) -> None:
        with self.connection:
            self.connection.execute(
                """
                UPDATE ledger_versions
                   SET status='unknown', completed_at=?, error_detail=?
                 WHERE ledger_version=? AND status='pending'
                """,
                (utc_now(), reason, ledger_version),
            )

    def complete_round(
        self,
        ledger_version: int,
        *,
        entries: Sequence[dict[str, Any]],
        evictions: Sequence[dict[str, Any]],
        propagation: Sequence[dict[str, Any]],
        progress_cursor: int,
    ) -> None:
        row = self.connection.execute(
            """
            SELECT status, content_version, policy_version, content_json, policy_json
              FROM ledger_versions
             WHERE ledger_version=?
            """,
            (ledger_version,),
        ).fetchone()
        if row is None:
            raise EdgeError(code="E_STATE", message="ledger round does not exist", details={"ledger_version": ledger_version})
        if row["status"] != "pending":
            raise EdgeError(
                code="E_STATE",
                message="only a pending ledger round can be completed",
                details={"ledger_version": ledger_version, "status": row["status"]},
            )
        content_version = str(row["content_version"])
        policy_version = str(row["policy_version"])
        normalized_entries = _normalize_entries(entries)
        for entry in normalized_entries:
            if entry["content_version"] != content_version:
                raise EdgeError(
                    code="E_STALE",
                    message="round entry state references a mixed content version",
                    path=entry["path"],
                    details={
                        "expected": content_version,
                        "actual": entry["content_version"],
                        "node_id": entry["node_id"],
                    },
                )
        try:
            round_content = ContentSnapshot(tuple(parse_content_records(json.loads(row["content_json"])).records))
            round_policy = parse_policy(json.loads(row["policy_json"]))
        except (TypeError, ValueError, EdgeError) as exc:
            raise EdgeError(
                code="E_STATE",
                message="pending round version references are invalid",
                details={"ledger_version": ledger_version, "error": str(exc)},
            ) from exc
        _validate_round_entries(round_content, round_policy, content_version, normalized_entries)
        normalized_evictions = _normalize_evictions(evictions)
        normalized_propagation = _normalize_propagation(propagation, content_version, policy_version)

        try:
            with self.connection:
                self.connection.execute("BEGIN IMMEDIATE")
                self.connection.execute("DELETE FROM entries WHERE ledger_version=?", (ledger_version,))
                self.connection.execute("DELETE FROM evictions WHERE ledger_version=?", (ledger_version,))
                self.connection.execute("DELETE FROM propagation WHERE ledger_version=?", (ledger_version,))
                for position, entry in enumerate(normalized_entries, start=1):
                    self.connection.execute(
                        """
                        INSERT INTO entries(
                            ledger_version, node_id, cache_key, path, etag, content_version,
                            written_mono_ms, ttl_ms, remaining_ttl_ms, entry_version, content_size, position
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            ledger_version,
                            entry["node_id"],
                            entry["cache_key"],
                            entry["path"],
                            entry["etag"],
                            entry["content_version"],
                            entry["written_mono_ms"],
                            entry["ttl_ms"],
                            entry["remaining_ttl_ms"],
                            entry["entry_version"],
                            entry["content_size"],
                            position,
                        ),
                    )
                for event in normalized_evictions:
                    payload = {key: value for key, value in event.items() if key not in _EVICTION_COLUMNS}
                    self.connection.execute(
                        """
                        INSERT INTO evictions(
                            ledger_version, node_id, event_seq, cache_key, path, etag, reason, payload_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            ledger_version,
                            event["node_id"],
                            event["event_seq"],
                            event["cache_key"],
                            event["path"],
                            event["etag"],
                            event["reason"],
                            _json(payload),
                        ),
                    )
                for row_propagation in normalized_propagation:
                    self.connection.execute(
                        """
                        INSERT INTO propagation(
                            ledger_version, seq, trigger_path, affected_path, depth, content_version, policy_version
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            ledger_version,
                            row_propagation["seq"],
                            row_propagation["trigger_path"],
                            row_propagation["affected_path"],
                            row_propagation["depth"],
                            content_version,
                            policy_version,
                        ),
                    )
                self.connection.execute(
                    """
                    UPDATE ledger_versions
                       SET status='complete', completed_at=?, progress_cursor=?
                     WHERE ledger_version=?
                    """,
                    (utc_now(), int(progress_cursor), ledger_version),
                )
                self.connection.execute(
                    "INSERT INTO meta(key, value) VALUES('current_ledger_version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(ledger_version),),
                )
        except sqlite3.Error as exc:
            raise EdgeError(
                code="E_STATE",
                message="failed to commit a complete ledger round",
                details={"ledger_version": ledger_version, "error": str(exc)},
            ) from exc

    def current_version(self) -> int | None:
        row = self.connection.execute("SELECT value FROM meta WHERE key='current_ledger_version'").fetchone()
        if row is None:
            latest = self.connection.execute(
                "SELECT MAX(ledger_version) AS value FROM ledger_versions WHERE status='complete'"
            ).fetchone()
            return int(latest["value"]) if latest and latest["value"] is not None else None
        return int(row["value"])

    def version_row(self, ledger_version: int) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM ledger_versions WHERE ledger_version=?",
            (ledger_version,),
        ).fetchone()
        if row is None:
            raise EdgeError(
                code="E_STATE",
                message="ledger version does not exist",
                details={"ledger_version": ledger_version},
            )
        return _version_row(row)

    def current_row(self) -> dict[str, Any] | None:
        current = self.current_version()
        if current is None:
            return None
        return self.version_row(current)

    def current_entries(self, node_id: str | None = None) -> list[dict[str, Any]]:
        current = self.current_version()
        if current is None:
            return []
        return self.entries_for_version(current, node_id=node_id)

    def entries_for_version(self, ledger_version: int, node_id: str | None = None) -> list[dict[str, Any]]:
        if node_id is None:
            rows = self.connection.execute(
                "SELECT * FROM entries WHERE ledger_version=? ORDER BY node_id, cache_key",
                (ledger_version,),
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT * FROM entries WHERE ledger_version=? AND node_id=? ORDER BY cache_key",
                (ledger_version, node_id),
            ).fetchall()
        return [_entry_row(row, current=(ledger_version == self.current_version())) for row in rows]

    def current_evictions(self) -> list[dict[str, Any]]:
        current = self.current_version()
        if current is None:
            return []
        return self.evictions_for_version(current)

    def evictions_for_version(self, ledger_version: int) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM evictions WHERE ledger_version=? ORDER BY node_id, event_seq",
            (ledger_version,),
        ).fetchall()
        return [_eviction_row(row) for row in rows]

    def current_propagation(self) -> list[dict[str, Any]]:
        current = self.current_version()
        if current is None:
            return []
        return self.propagation_for_version(current)

    def propagation_for_version(self, ledger_version: int) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM propagation WHERE ledger_version=? ORDER BY seq",
            (ledger_version,),
        ).fetchall()
        return [
            {
                "ledger_version": int(row["ledger_version"]),
                "seq": int(row["seq"]),
                "trigger_path": str(row["trigger_path"]),
                "affected_path": str(row["affected_path"]),
                "depth": int(row["depth"]),
                "content_version": str(row["content_version"]),
                "policy_version": str(row["policy_version"]),
            }
            for row in rows
        ]

    def current_spec(self) -> LoadedSpec | None:
        row = self.current_row()
        if row is None:
            return None
        content = ContentSnapshot(tuple(parse_content_records(json.loads(row["content_json"])).records))
        policy = parse_policy(json.loads(row["policy_json"]))
        return LoadedSpec(content=content, policy=policy, source_path=None)

    def version_set(self) -> dict[str, Any] | None:
        row = self.current_row()
        if row is None:
            return None
        content = ContentSnapshot(tuple(parse_content_records(json.loads(row["content_json"])).records))
        return {
            "ledger_version": row["ledger_version"],
            "content_version": row["content_version"],
            "policy_version": row["policy_version"],
            "content_versions": content.etag_set,
            "content_digest": row["content_digest"],
            "policy_digest": row["policy_digest"],
        }

    def version_list(self) -> list[dict[str, Any]]:
        current = self.current_version()
        rows = self.connection.execute("SELECT * FROM ledger_versions ORDER BY ledger_version").fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = _version_row(row)
            item["stale"] = item["ledger_version"] != current
            item.pop("content_json", None)
            item.pop("policy_json", None)
            result.append(item)
        return result

    def unknown_rounds(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT ledger_version, round_kind, created_at, error_detail FROM ledger_versions "
            "WHERE status='unknown' ORDER BY ledger_version"
        ).fetchall()
        return [dict(row) for row in rows]

    def view_model(self) -> dict[str, Any]:
        current = self.current_row()
        versions = self.version_list()
        if current is None:
            return {
                "ok": True,
                "ledger_version": None,
                "content_version": None,
                "policy_version": None,
                "entries": [],
                "evictions": [],
                "propagation": [],
                "versions": versions,
                "unknown_rounds": self.unknown_rounds(),
                "current_stats": {"entry_count": 0, "eviction_count": 0, "propagation_count": 0},
            }
        entries = self.current_entries()
        evictions = self.current_evictions()
        propagation = self.current_propagation()
        return {
            "ok": True,
            "ledger_version": current["ledger_version"],
            "content_version": current["content_version"],
            "policy_version": current["policy_version"],
            "progress_cursor": current["progress_cursor"],
            "entries": entries,
            "evictions": evictions,
            "propagation": propagation,
            "versions": versions,
            "unknown_rounds": self.unknown_rounds(),
            "current_stats": {
                "entry_count": len(entries),
                "eviction_count": len(evictions),
                "propagation_count": len(propagation),
            },
        }

    def restore_snapshot_document(self, document: dict[str, Any], *, round_kind: str = "snapshot_restore") -> int:
        """Install a snapshot as a new current ledger version.

        Historical versions remain auditable but are stale.  The new version
        contains only the snapshot's entries and derivation graph.
        """

        try:
            content_records = parse_content_records(document["content_records"])
            policy = parse_policy(document["policy"])
        except (KeyError, TypeError, ValueError, EdgeError) as exc:
            raise EdgeError(
                code="E_SNAPSHOT",
                message="snapshot document contains invalid content or policy records",
                field_path="content_records",
                details={"error": str(exc)},
            ) from exc
        content = ContentSnapshot(tuple(content_records.records))
        content_version = str(document["content_version"])
        policy_version = str(document["policy_version"])
        if content.version != content_version:
            raise EdgeError(
                code="E_SNAPSHOT",
                message="snapshot content version does not match its content records",
                field_path="content_version",
                details={"declared": content_version, "computed": content.version},
            )
        if policy.version != policy_version:
            raise EdgeError(
                code="E_SNAPSHOT",
                message="snapshot policy version does not match its policy document",
                field_path="policy_version",
                details={"declared": policy_version, "computed": policy.version},
            )
        raw_entries = document.get("entries", [])
        if not isinstance(raw_entries, list):
            raise EdgeError(code="E_SNAPSHOT", message="snapshot entries must be an array", field_path="entries")
        normalized = _normalize_entries(raw_entries)
        _validate_entries_against(content, policy, content_version, normalized)
        now = monotonic_ms()
        restored_entries: list[dict[str, Any]] = []
        for entry in normalized:
            remaining = max(0, int(entry["remaining_ttl_ms"]))
            if remaining <= 0:
                continue
            restored = dict(entry)
            restored["written_mono_ms"] = now
            restored["ttl_ms"] = max(1, remaining)
            restored["remaining_ttl_ms"] = remaining
            restored_entries.append(restored)
        normalized_evictions = _normalize_evictions(document.get("evictions", []))
        normalized_propagation = _normalize_propagation(
            document.get("propagation", []),
            content_version,
            policy_version,
        )

        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            row = self.connection.execute(
                "SELECT COALESCE(MAX(ledger_version), 0) AS value FROM ledger_versions"
            ).fetchone()
            ledger_version = int(row["value"]) + 1
            self.connection.execute(
                """
                INSERT INTO ledger_versions(
                    ledger_version, created_at, completed_at, status, round_kind, progress_cursor,
                    content_version, policy_version, content_digest, policy_digest, spec_digest,
                    content_json, policy_json
                ) VALUES (?, ?, ?, 'complete', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ledger_version,
                    utc_now(),
                    utc_now(),
                    round_kind,
                    int(document.get("progress_cursor", len(content.records))),
                    content_version,
                    policy_version,
                    content.digest,
                    policy.digest,
                    _json({"content_digest": content.digest, "policy_digest": policy.digest}),
                    _json(content.to_declared_list()),
                    _json(policy.to_dict()),
                ),
            )
            for position, entry in enumerate(restored_entries, start=1):
                self.connection.execute(
                    """
                    INSERT INTO entries(
                        ledger_version, node_id, cache_key, path, etag, content_version,
                        written_mono_ms, ttl_ms, remaining_ttl_ms, entry_version, content_size, position
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ledger_version,
                        entry["node_id"],
                        entry["cache_key"],
                        entry["path"],
                        entry["etag"],
                        entry["content_version"],
                        entry["written_mono_ms"],
                        entry["ttl_ms"],
                        entry["remaining_ttl_ms"],
                        entry["entry_version"],
                        entry["content_size"],
                        position,
                    ),
                )
            for event in normalized_evictions:
                payload = {key: value for key, value in event.items() if key not in _EVICTION_COLUMNS}
                self.connection.execute(
                    """
                    INSERT INTO evictions(
                        ledger_version, node_id, event_seq, cache_key, path, etag, reason, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ledger_version,
                        event["node_id"],
                        event["event_seq"],
                        event["cache_key"],
                        event["path"],
                        event["etag"],
                        event["reason"],
                        _json(payload),
                    ),
                )
            for propagation_row in normalized_propagation:
                self.connection.execute(
                    """
                    INSERT INTO propagation(
                        ledger_version, seq, trigger_path, affected_path, depth, content_version, policy_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ledger_version,
                        propagation_row["seq"],
                        propagation_row["trigger_path"],
                        propagation_row["affected_path"],
                        propagation_row["depth"],
                        content_version,
                        policy_version,
                    ),
                )
            self.connection.execute(
                "INSERT INTO meta(key, value) VALUES('current_ledger_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(ledger_version),),
            )
        return ledger_version

    def register_snapshot(
        self,
        *,
        snapshot_id: str,
        path: str,
        ledger_version: int,
        content_digest: str,
        policy_digest: str,
        read_only: bool,
    ) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO snapshots(
                    snapshot_id, path, ledger_version, content_digest, policy_digest, created_at, read_only
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (snapshot_id, path, ledger_version, content_digest, policy_digest, utc_now(), 1 if read_only else 0),
            )

    def close_connection(self) -> None:
        self.close()


_EVICTION_COLUMNS = {"ledger_version", "node_id", "event_seq", "cache_key", "path", "etag", "reason"}


def _json(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _version_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "ledger_version": int(row["ledger_version"]),
        "created_at": str(row["created_at"]),
        "completed_at": row["completed_at"],
        "status": str(row["status"]),
        "round_kind": str(row["round_kind"]),
        "progress_cursor": int(row["progress_cursor"]),
        "content_version": str(row["content_version"]),
        "policy_version": str(row["policy_version"]),
        "content_digest": str(row["content_digest"]),
        "policy_digest": str(row["policy_digest"]),
        "spec_digest": str(row["spec_digest"]),
        "content_json": str(row["content_json"]),
        "policy_json": str(row["policy_json"]),
        "error_detail": row["error_detail"],
    }


def _entry_row(row: sqlite3.Row, *, current: bool) -> dict[str, Any]:
    return {
        "ledger_version": int(row["ledger_version"]),
        "node_id": str(row["node_id"]),
        "cache_key": str(row["cache_key"]),
        "path": str(row["path"]),
        "etag": str(row["etag"]),
        "content_version": str(row["content_version"]),
        "written_mono_ms": int(row["written_mono_ms"]),
        "ttl_ms": int(row["ttl_ms"]),
        "remaining_ttl_ms": int(row["remaining_ttl_ms"]),
        "entry_version": int(row["entry_version"]),
        "content_size": int(row["content_size"]),
        "stale": not current,
    }


def _eviction_row(row: sqlite3.Row) -> dict[str, Any]:
    payload = json.loads(row["payload_json"])
    return {
        "ledger_version": int(row["ledger_version"]),
        "node_id": str(row["node_id"]),
        "event_seq": int(row["event_seq"]),
        "cache_key": str(row["cache_key"]),
        "path": str(row["path"]),
        "etag": str(row["etag"]),
        "reason": str(row["reason"]),
        **payload,
    }


def _normalize_entries(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in entries:
        try:
            entry = {
                "node_id": str(raw["node_id"]),
                "cache_key": str(raw["cache_key"]),
                "path": str(raw["path"]),
                "etag": str(raw["etag"]),
                "content_version": str(raw["content_version"]),
                "written_mono_ms": int(raw["written_mono_ms"]),
                "ttl_ms": int(raw["ttl_ms"]),
                "remaining_ttl_ms": int(raw.get("remaining_ttl_ms", raw["ttl_ms"])),
                "entry_version": int(raw["entry_version"]),
                "content_size": int(raw.get("content_size", 0)),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise EdgeError(
                code="E_STATE",
                message="entry state has invalid or missing fields",
                details={"entry": raw, "error": str(exc)},
            ) from exc
        if entry["ttl_ms"] <= 0 or entry["entry_version"] <= 0:
            raise EdgeError(code="E_STATE", message="entry ttl_ms and entry_version must be positive", details={"entry": entry})
        identity = (entry["node_id"], entry["cache_key"])
        if identity in seen:
            raise EdgeError(
                code="E_STATE",
                message="duplicate entry identity in a ledger round",
                details={"node_id": entry["node_id"], "cache_key": entry["cache_key"]},
            )
        seen.add(identity)
        result.append(entry)
    return sorted(result, key=lambda item: (item["node_id"], item["cache_key"]))


def _normalize_evictions(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for raw in events:
        try:
            event = {
                "ledger_version": int(raw.get("ledger_version", 0)),
                "node_id": str(raw["node_id"]),
                "event_seq": int(raw["event_seq"]),
                "cache_key": str(raw["cache_key"]),
                "path": str(raw["path"]),
                "etag": str(raw["etag"]),
                "reason": str(raw["reason"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise EdgeError(code="E_STATE", message="eviction event is malformed", details={"event": raw}) from exc
        extra = {key: value for key, value in raw.items() if key not in _EVICTION_COLUMNS}
        event.update(extra)
        identity = (event["node_id"], event["event_seq"])
        if identity in seen:
            raise EdgeError(code="E_STATE", message="duplicate eviction event sequence", details={"event": event})
        seen.add(identity)
        result.append(event)
    return sorted(result, key=lambda item: (item["node_id"], item["event_seq"]))


def _normalize_propagation(
    rows: Iterable[dict[str, Any]],
    content_version: str,
    policy_version: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen_seq: set[int] = set()
    for raw in rows:
        try:
            row = {
                "seq": int(raw["seq"]),
                "trigger_path": str(raw["trigger_path"]),
                "affected_path": str(raw["affected_path"]),
                "depth": int(raw["depth"]),
                "content_version": str(raw.get("content_version", content_version)),
                "policy_version": str(raw.get("policy_version", policy_version)),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise EdgeError(code="E_STATE", message="propagation row is malformed", details={"row": raw}) from exc
        if row["seq"] in seen_seq:
            raise EdgeError(code="E_STATE", message="duplicate propagation sequence", details={"row": row})
        seen_seq.add(row["seq"])
        if row["content_version"] != content_version or row["policy_version"] != policy_version:
            raise EdgeError(
                code="E_STALE",
                message="propagation row references mixed content or policy versions",
                details={"row": row, "round_content_version": content_version, "round_policy_version": policy_version},
            )
        result.append(row)
    return sorted(result, key=lambda item: item["seq"])


def _validate_entries_against(
    content: ContentSnapshot,
    policy: Policy,
    content_version: str,
    entries: Sequence[dict[str, Any]],
) -> None:
    by_path = content.by_path
    node_ids = {node.node_id for node in policy.nodes}
    for entry in entries:
        node_id = entry["node_id"]
        path = entry["path"]
        etag = entry["etag"]
        if node_id not in node_ids:
            raise EdgeError(
                code="E_SNAPSHOT",
                message="snapshot entry refers to a node absent from its policy",
                path=path,
                field_path=f"entries[node={node_id}].node_id",
            )
        if path not in by_path:
            raise EdgeError(
                code="E_SNAPSHOT",
                message="snapshot entry refers to a missing content path",
                path=path,
                field_path=f"entries[node={node_id},key={entry['cache_key']}].path",
            )
        if etag != by_path[path].etag:
            raise EdgeError(
                code="E_SNAPSHOT",
                message="snapshot entry ETag does not belong to its content version",
                path=path,
                field_path=f"entries[node={node_id},key={entry['cache_key']}].etag",
                details={"expected": by_path[path].etag, "actual": etag},
            )
        if entry["content_version"] != content_version:
            raise EdgeError(
                code="E_SNAPSHOT",
                message="snapshot entry references a stale content version",
                path=path,
                field_path=f"entries[node={node_id},key={entry['cache_key']}].content_version",
                details={"expected": content_version, "actual": entry["content_version"]},
            )


def _validate_round_entries(
    content: ContentSnapshot,
    policy: Policy,
    content_version: str,
    entries: Sequence[dict[str, Any]],
) -> None:
    by_path = content.by_path
    node_ids = {node.node_id for node in policy.nodes}
    for entry in entries:
        node_id = entry["node_id"]
        path = entry["path"]
        etag = entry["etag"]
        if node_id not in node_ids:
            raise EdgeError(
                code="E_STALE",
                message="round entry refers to a node outside the round policy",
                path=path,
                details={"node_id": node_id, "ledger_version_content": content_version},
            )
        if path not in by_path:
            raise EdgeError(
                code="E_STALE",
                message="round entry refers to a path outside the round content version",
                path=path,
                details={"node_id": node_id},
            )
        if etag != by_path[path].etag:
            raise EdgeError(
                code="E_STALE",
                message="round entry ETag does not belong to the round content version",
                path=path,
                details={"node_id": node_id, "expected": by_path[path].etag, "actual": etag},
            )
        expected_key = cache_key(policy, node_id=node_id, path=path, etag=etag)
        if entry["cache_key"] != expected_key:
            raise EdgeError(
                code="E_STALE",
                message="round entry cache key does not match the round policy",
                path=path,
                details={"node_id": node_id, "expected": expected_key, "actual": entry["cache_key"]},
            )
