"""SQLite truth store with pending/complete event commits and item versions."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from .errors import E_REPLAY, E_STATE, E_VERSION, BusinessError
from .models import (
    Event,
    ItemResult,
    PeerScript,
    SessionConfig,
    canonical_json,
    digest_of,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS session (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    config_version INTEGER NOT NULL,
    script_version INTEGER NOT NULL,
    config_digest TEXT NOT NULL,
    script_digest TEXT NOT NULL,
    config_json TEXT NOT NULL,
    script_json TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    current_event_seq INTEGER NOT NULL,
    created_mono_ms INTEGER NOT NULL,
    updated_mono_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_seq INTEGER PRIMARY KEY,
    event_type TEXT NOT NULL,
    recorded_mono_ms INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'complete', 'unknown')),
    config_version INTEGER NOT NULL,
    script_version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS conclusions (
    item_key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    value_digest TEXT NOT NULL,
    revision INTEGER NOT NULL,
    basis_event_seq INTEGER NOT NULL,
    depends_on_items_json TEXT NOT NULL,
    depends_on_events_json TEXT NOT NULL,
    config_version INTEGER NOT NULL,
    script_version INTEGER NOT NULL,
    epoch INTEGER NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0,
    last_affected_event_seq INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS conclusion_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    value_digest TEXT NOT NULL,
    revision INTEGER NOT NULL,
    basis_event_seq INTEGER NOT NULL,
    depends_on_items_json TEXT NOT NULL,
    depends_on_events_json TEXT NOT NULL,
    config_version INTEGER NOT NULL,
    script_version INTEGER NOT NULL,
    epoch INTEGER NOT NULL,
    stale INTEGER NOT NULL,
    last_affected_event_seq INTEGER NOT NULL,
    replaced_mono_ms INTEGER NOT NULL,
    replace_reason TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_mono_ms INTEGER NOT NULL,
    config_version INTEGER,
    script_version INTEGER,
    config_digest TEXT,
    script_digest TEXT,
    event_count INTEGER NOT NULL,
    state_json TEXT NOT NULL,
    conclusions_json TEXT NOT NULL,
    events_json TEXT NOT NULL,
    replay_points_json TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    config_json TEXT NOT NULL,
    script_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS version_history (
    kind TEXT NOT NULL,
    digest TEXT NOT NULL,
    version INTEGER NOT NULL,
    first_seen_mono_ms INTEGER NOT NULL,
    PRIMARY KEY (kind, digest)
);

CREATE INDEX IF NOT EXISTS idx_events_status ON events(status, event_seq);
CREATE INDEX IF NOT EXISTS idx_snapshots_name ON snapshots(name, snapshot_id);
CREATE INDEX IF NOT EXISTS idx_version_history_kind_version ON version_history(kind, version);
"""


def _item_from_row(row: sqlite3.Row) -> ItemResult:
    value_json = row["value_json"]
    if digest_of(json.loads(value_json)) != row["value_digest"]:
        raise BusinessError(
            E_REPLAY,
            "conclusion content digest mismatch",
            [f"conclusions.{row['item_key']}"],
        )
    return ItemResult(
        key=row["item_key"],
        value=json.loads(value_json),
        revision=int(row["revision"]),
        basis_event_seq=int(row["basis_event_seq"]),
        depends_on_items=tuple(json.loads(row["depends_on_items_json"])),
        depends_on_events=tuple(
            int(value) for value in json.loads(row["depends_on_events_json"])
        ),
        config_version=int(row["config_version"]),
        script_version=int(row["script_version"]),
        stale=bool(row["stale"]),
        last_affected_event_seq=int(row["last_affected_event_seq"]),
    )


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def reset(self) -> None:
        with self.conn:
            for table in (
                "session",
                "events",
                "conclusions",
                "conclusion_history",
                "snapshots",
                "version_history",
            ):
                self.conn.execute(f"DELETE FROM {table}")

    def get_session(self) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM session WHERE id = 1").fetchone()

    def create_session(
        self, config: SessionConfig, script: PeerScript, mono_ms: int
    ) -> None:
        if self.get_session() is not None:
            raise BusinessError(E_STATE, "session already exists", ["session"])
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO session (
                    id, config_version, script_version, config_digest, script_digest,
                    config_json, script_json, epoch, current_event_seq,
                    created_mono_ms, updated_mono_ms
                ) VALUES (1, ?, ?, ?, ?, ?, ?, 1, 0, ?, ?)
                """,
                (
                    config.config_version,
                    script.script_version,
                    config.digest,
                    script.digest,
                    canonical_json(config.raw),
                    canonical_json(script.raw),
                    mono_ms,
                    mono_ms,
                ),
            )

    def replace_session(
        self,
        config: SessionConfig,
        script: PeerScript,
        mono_ms: int,
        *,
        current_event_seq: int | None = None,
    ) -> int:
        current = self.get_session()
        epoch = 1 if current is None else int(current["epoch"]) + 1
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO session (
                    id, config_version, script_version, config_digest, script_digest,
                    config_json, script_json, epoch, current_event_seq,
                    created_mono_ms, updated_mono_ms
                ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    config_version=excluded.config_version,
                    script_version=excluded.script_version,
                    config_digest=excluded.config_digest,
                    script_digest=excluded.script_digest,
                    config_json=excluded.config_json,
                    script_json=excluded.script_json,
                    epoch=excluded.epoch,
                    current_event_seq=excluded.current_event_seq,
                    updated_mono_ms=excluded.updated_mono_ms
                """,
                (
                    config.config_version,
                    script.script_version,
                    config.digest,
                    script.digest,
                    canonical_json(config.raw),
                    canonical_json(script.raw),
                    epoch,
                    int(current_event_seq or 0),
                    mono_ms,
                    mono_ms,
                ),
            )
        return epoch

    def touch_session(self, mono_ms: int) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE session SET updated_mono_ms = ? WHERE id = 1", (mono_ms,)
            )

    def current_epoch(self) -> int:
        row = self.get_session()
        return 1 if row is None else int(row["epoch"])

    def register_version(
        self, kind: str, digest: str, version: int, mono_ms: int
    ) -> dict[str, int | str] | None:
        existing_digest = self.conn.execute(
            "SELECT * FROM version_history WHERE kind = ? AND digest = ?",
            (kind, digest),
        ).fetchone()
        if existing_digest is not None:
            return dict(existing_digest)
        existing_version = self.conn.execute(
            "SELECT * FROM version_history WHERE kind = ? AND version = ?",
            (kind, version),
        ).fetchone()
        if existing_version is not None:
            return dict(existing_version)
        with self.conn:
            self.conn.execute(
                "INSERT INTO version_history(kind, digest, version, first_seen_mono_ms) VALUES (?, ?, ?, ?)",
                (kind, digest, version, mono_ms),
            )
        return None

    def version_record(self, kind: str, digest: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM version_history WHERE kind = ? AND digest = ?",
            (kind, digest),
        ).fetchone()

    def max_version(self, kind: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(version), 0) AS v FROM version_history WHERE kind = ?",
            (kind,),
        ).fetchone()
        return int(row["v"]) if row is not None else 0

    def next_event_seq(self) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(event_seq), 0) + 1 AS next_seq FROM events"
        ).fetchone()
        return int(row["next_seq"])

    def begin_event(
        self,
        event: Event,
        config_version: int,
        script_version: int,
        *,
        reserve_seq: int | None = None,
    ) -> Event:
        event_seq = reserve_seq if reserve_seq is not None else self.next_event_seq()
        assigned = event.with_seq(event_seq)
        payload_json = canonical_json(assigned.as_dict())
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO events(
                    event_seq, event_type, recorded_mono_ms, payload_json, payload_digest,
                    status, config_version, script_version
                ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    event_seq,
                    assigned.event_type,
                    assigned.recorded_mono_ms,
                    payload_json,
                    digest_of(assigned.as_dict()),
                    config_version,
                    script_version,
                ),
            )
        return assigned

    def commit_event(
        self, event: Event, items: Iterable[ItemResult], mono_ms: int
    ) -> None:
        item_list = list(items)
        with self.conn:
            cursor = self.conn.execute(
                "UPDATE events SET status = 'complete' WHERE event_seq = ? AND status = 'pending'",
                (event.event_seq,),
            )
            if cursor.rowcount != 1:
                raise BusinessError(
                    E_STATE,
                    "event is not pending",
                    ["event_seq"],
                    {"event_seq": event.event_seq},
                )
            epoch = self.current_epoch()
            for item in item_list:
                self.conn.execute(
                    """
                    INSERT INTO conclusions(
                        item_key, value_json, value_digest, revision, basis_event_seq,
                        depends_on_items_json, depends_on_events_json, config_version,
                        script_version, epoch, stale, last_affected_event_seq
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                    ON CONFLICT(item_key) DO UPDATE SET
                        value_json=excluded.value_json,
                        value_digest=excluded.value_digest,
                        revision=excluded.revision,
                        basis_event_seq=excluded.basis_event_seq,
                        depends_on_items_json=excluded.depends_on_items_json,
                        depends_on_events_json=excluded.depends_on_events_json,
                        config_version=excluded.config_version,
                        script_version=excluded.script_version,
                        epoch=excluded.epoch,
                        stale=0,
                        last_affected_event_seq=excluded.last_affected_event_seq
                    """,
                    (
                        item.key,
                        item.value_json,
                        item.value_digest,
                        item.revision,
                        item.basis_event_seq,
                        canonical_json(list(item.depends_on_items)),
                        canonical_json(list(item.depends_on_events)),
                        item.config_version,
                        item.script_version,
                        epoch,
                        item.last_affected_event_seq,
                    ),
                )
            self.conn.execute(
                "UPDATE session SET current_event_seq = ?, updated_mono_ms = ? WHERE id = 1",
                (event.event_seq, mono_ms),
            )

    def mark_pending_unknown(self, reason: str) -> tuple[int, ...]:
        rows = self.conn.execute(
            "SELECT event_seq FROM events WHERE status = 'pending' ORDER BY event_seq"
        ).fetchall()
        event_seqs = tuple(int(row["event_seq"]) for row in rows)
        if event_seqs:
            with self.conn:
                self.conn.execute(
                    "UPDATE events SET status = 'unknown' WHERE status = 'pending'"
                )
        return event_seqs

    def mark_event_unknown(self, event_seq: int, reason: str = "") -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE events SET status = 'unknown' WHERE event_seq = ? AND status = 'pending'",
                (event_seq,),
            )

    def version_by_number(self, kind: str, version: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM version_history WHERE kind = ? AND version = ?",
            (kind, version),
        ).fetchone()

    def event_by_seq(
        self, event_seq: int, *, status: str | None = None
    ) -> Event | None:
        sql = "SELECT * FROM events WHERE event_seq = ?"
        params: tuple[Any, ...] = (event_seq,)
        if status is not None:
            sql += " AND status = ?"
            params = (event_seq, status)
        row = self.conn.execute(sql, params).fetchone()
        if row is None:
            return None
        raw = json.loads(row["payload_json"])
        if digest_of(raw) != row["payload_digest"]:
            raise BusinessError(
                E_REPLAY, "event payload digest mismatch", [f"events[{event_seq}]"]
            )
        return Event.from_dict(raw)

    def unknown_to_pending(self, event_seq: int) -> Event:
        with self.conn:
            cursor = self.conn.execute(
                "UPDATE events SET status = 'pending' WHERE event_seq = ? AND status = 'unknown'",
                (event_seq,),
            )
            if cursor.rowcount != 1:
                raise BusinessError(
                    E_STATE,
                    "event is not unknown",
                    ["event_seq"],
                    {"event_seq": event_seq},
                )
        event = self.event_by_seq(event_seq, status="pending")
        if event is None:
            raise BusinessError(
                E_STATE, "unknown event payload is missing", ["event_seq"]
            )
        return event

    def events(self, *, include_unknown: bool = False) -> tuple[Event, ...]:
        statuses = ("complete", "unknown") if include_unknown else ("complete",)
        placeholders = ",".join("?" for _ in statuses)
        rows = self.conn.execute(
            f"SELECT * FROM events WHERE status IN ({placeholders}) ORDER BY event_seq",
            statuses,
        ).fetchall()
        result: list[Event] = []
        for row in rows:
            raw = json.loads(row["payload_json"])
            if digest_of(raw) != row["payload_digest"]:
                raise BusinessError(
                    E_REPLAY,
                    "event payload digest mismatch",
                    [f"events[{row['event_seq']}]"],
                )
            result.append(Event.from_dict(raw))
        return tuple(result)

    def pending_event_seqs(self) -> tuple[int, ...]:
        rows = self.conn.execute(
            "SELECT event_seq FROM events WHERE status = 'pending' ORDER BY event_seq"
        ).fetchall()
        return tuple(int(row["event_seq"]) for row in rows)

    def unknown_event_seqs(self) -> tuple[int, ...]:
        rows = self.conn.execute(
            "SELECT event_seq FROM events WHERE status = 'unknown' ORDER BY event_seq"
        ).fetchall()
        return tuple(int(row["event_seq"]) for row in rows)

    def event_count(self, status: str = "complete") -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS count FROM events WHERE status = ?", (status,)
        ).fetchone()
        return int(row["count"])

    def conclusions(self, *, include_stale: bool = False) -> dict[str, ItemResult]:
        sql = "SELECT * FROM conclusions"
        if not include_stale:
            sql += " WHERE stale = 0"
        sql += " ORDER BY item_key"
        return {
            row["item_key"]: _item_from_row(row)
            for row in self.conn.execute(sql).fetchall()
        }

    def conclusion_history(self) -> tuple[ItemResult, ...]:
        rows = self.conn.execute(
            "SELECT * FROM conclusion_history ORDER BY history_id"
        ).fetchall()
        return tuple(_item_from_row(row) for row in rows)

    def replace_conclusions(
        self,
        items: dict[str, ItemResult],
        *,
        reason: str,
        mono_ms: int,
        archive: bool = True,
    ) -> int:
        archived = 0
        with self.conn:
            if archive:
                current = self.conn.execute(
                    "SELECT * FROM conclusions ORDER BY item_key"
                ).fetchall()
                archived = len(current)
                for row in current:
                    self.conn.execute(
                        """
                        INSERT INTO conclusion_history(
                            item_key, value_json, value_digest, revision, basis_event_seq,
                            depends_on_items_json, depends_on_events_json, config_version,
                            script_version, epoch, stale, last_affected_event_seq,
                            replaced_mono_ms, replace_reason
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                        """,
                        (
                            row["item_key"],
                            row["value_json"],
                            row["value_digest"],
                            row["revision"],
                            row["basis_event_seq"],
                            row["depends_on_items_json"],
                            row["depends_on_events_json"],
                            row["config_version"],
                            row["script_version"],
                            row["epoch"],
                            row["last_affected_event_seq"],
                            mono_ms,
                            reason,
                        ),
                    )
                self.conn.execute("DELETE FROM conclusions")
            epoch = self.current_epoch()
            for item in items.values():
                self.conn.execute(
                    """
                    INSERT INTO conclusions(
                        item_key, value_json, value_digest, revision, basis_event_seq,
                        depends_on_items_json, depends_on_events_json, config_version,
                        script_version, epoch, stale, last_affected_event_seq
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                    """,
                    (
                        item.key,
                        item.value_json,
                        item.value_digest,
                        item.revision,
                        item.basis_event_seq,
                        canonical_json(list(item.depends_on_items)),
                        canonical_json(list(item.depends_on_events)),
                        item.config_version,
                        item.script_version,
                        epoch,
                        item.last_affected_event_seq,
                    ),
                )
        return archived

    def replace_events(
        self, events: Iterable[Event], *, config_version: int, script_version: int
    ) -> None:
        event_list = list(events)
        with self.conn:
            self.conn.execute("DELETE FROM events")
            for event in event_list:
                raw = event.as_dict()
                self.conn.execute(
                    """
                    INSERT INTO events(
                        event_seq, event_type, recorded_mono_ms, payload_json, payload_digest,
                        status, config_version, script_version
                    ) VALUES (?, ?, ?, ?, ?, 'complete', ?, ?)
                    """,
                    (
                        event.event_seq,
                        event.event_type,
                        event.recorded_mono_ms,
                        canonical_json(raw),
                        digest_of(raw),
                        config_version,
                        script_version,
                    ),
                )
            self.conn.execute(
                "UPDATE session SET current_event_seq = ? WHERE id = 1",
                (event_list[-1].event_seq if event_list else 0,),
            )

    def save_snapshot(self, payload: dict[str, Any]) -> int:
        content_payload = dict(payload)
        content_digest = digest_of(content_payload)
        with self.conn:
            cursor = self.conn.execute(
                """
                INSERT INTO snapshots(
                    name, created_mono_ms, config_version, script_version, config_digest,
                    script_digest, event_count, state_json, conclusions_json, events_json,
                    replay_points_json, content_digest, config_json, script_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["name"],
                    payload["created_mono_ms"],
                    payload.get("config_version"),
                    payload.get("script_version"),
                    payload.get("config_digest"),
                    payload.get("script_digest"),
                    payload["event_count"],
                    canonical_json(payload["state"]),
                    canonical_json(payload["conclusions"]),
                    canonical_json(payload["events"]),
                    canonical_json(payload["replay_points"]),
                    content_digest,
                    canonical_json(payload["config"]),
                    canonical_json(payload["script"]),
                ),
            )
        return int(cursor.lastrowid)

    def snapshot(self, snapshot_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
        ).fetchone()

    def snapshots(self) -> tuple[sqlite3.Row, ...]:
        return tuple(
            self.conn.execute(
                "SELECT * FROM snapshots ORDER BY snapshot_id DESC"
            ).fetchall()
        )

    def snapshot_payload(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = {
            "name": row["name"],
            "created_mono_ms": int(row["created_mono_ms"]),
            "config_version": None
            if row["config_version"] is None
            else int(row["config_version"]),
            "script_version": None
            if row["script_version"] is None
            else int(row["script_version"]),
            "config_digest": row["config_digest"],
            "script_digest": row["script_digest"],
            "event_count": int(row["event_count"]),
            "state": json.loads(row["state_json"]),
            "conclusions": json.loads(row["conclusions_json"]),
            "events": json.loads(row["events_json"]),
            "replay_points": json.loads(row["replay_points_json"]),
            "config": json.loads(row["config_json"]),
            "script": json.loads(row["script_json"]),
        }
        if digest_of(payload) != row["content_digest"]:
            raise BusinessError(
                E_VERSION,
                "snapshot content digest mismatch",
                ["snapshot.content_digest"],
                {"snapshot_id": int(row["snapshot_id"])},
            )
        return payload
