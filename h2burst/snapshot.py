"""SQLite-backed immutable session snapshots and whole-state replacement."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .engine import ENGINE_FORMAT_VERSION, Engine
from .errors import WorkbenchError
from .model import SessionSpec, canonical_json, digest_value, parse_session


SNAPSHOT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class SnapshotInfo:
    snapshot_id: int
    label: str
    session_id: str
    declared_version: int
    content_digest: str
    snapshot_format_version: int
    engine_format_version: int
    state_digest: str
    next_event_seq: int
    read_only: bool
    created_order: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "label": self.label,
            "session_id": self.session_id,
            "declared_version": self.declared_version,
            "content_digest": self.content_digest,
            "snapshot_format_version": self.snapshot_format_version,
            "engine_format_version": self.engine_format_version,
            "state_digest": self.state_digest,
            "next_event_seq": self.next_event_seq,
            "read_only": self.read_only,
            "created_order": self.created_order,
        }


@dataclass
class RestoreResult:
    snapshot: SnapshotInfo
    spec: SessionSpec
    engine: Engine
    whole_replacement: bool
    version_mismatch: bool
    content_digest_mismatch: bool
    diff: dict[str, list[Any]]
    backup_snapshot_id: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot": self.snapshot.to_dict(),
            "whole_replacement": self.whole_replacement,
            "version_mismatch": self.version_mismatch,
            "content_digest_mismatch": self.content_digest_mismatch,
            "diff": self.diff,
            "backup_snapshot_id": self.backup_snapshot_id,
            "next_event_seq": self.engine.next_event_seq,
        }


def _snapshot_error(message: str, field: str | None = None) -> WorkbenchError:
    return WorkbenchError("E_SNAPSHOT", message, field=field)


def _stable_json(value: Any) -> str:
    return canonical_json(value)


class SnapshotStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        try:
            self.connection = sqlite3.connect(self.path)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys = ON")
            self._init_schema()
        except sqlite3.Error as exc:
            raise WorkbenchError("E_IO", f"cannot open snapshot database: {exc}", field="path") from exc

    def _init_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                session_id TEXT NOT NULL,
                declared_version INTEGER NOT NULL,
                content_digest TEXT NOT NULL,
                snapshot_format_version INTEGER NOT NULL,
                engine_format_version INTEGER NOT NULL,
                state_digest TEXT NOT NULL,
                state_json TEXT NOT NULL,
                spec_json TEXT NOT NULL,
                next_event_seq INTEGER NOT NULL,
                read_only INTEGER NOT NULL DEFAULT 0 CHECK (read_only IN (0, 1)),
                created_order INTEGER NOT NULL UNIQUE
            );
            CREATE INDEX IF NOT EXISTS snapshots_session_idx
                ON snapshots(session_id, created_order);
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SnapshotStore":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _insert(
        self,
        engine: Engine,
        label: str,
        *,
        read_only: bool = False,
    ) -> SnapshotInfo:
        state = engine.to_state_dict()
        state_digest = str(state["state_digest"])
        spec_json = _stable_json(engine.spec.to_dict())
        state_json = _stable_json(state)
        order = self.connection.execute(
            "SELECT COALESCE(MAX(created_order), 0) + 1 FROM snapshots"
        ).fetchone()[0]
        try:
            cursor = self.connection.execute(
                """
                INSERT INTO snapshots (
                    label, session_id, declared_version, content_digest,
                    snapshot_format_version, engine_format_version, state_digest,
                    state_json, spec_json, next_event_seq, read_only, created_order
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    label,
                    engine.spec.session_id,
                    engine.spec.declared_version,
                    engine.spec.content_digest,
                    SNAPSHOT_FORMAT_VERSION,
                    ENGINE_FORMAT_VERSION,
                    state_digest,
                    state_json,
                    spec_json,
                    engine.next_event_seq,
                    1 if read_only else 0,
                    order,
                ),
            )
            self.connection.commit()
        except sqlite3.Error as exc:
            self.connection.rollback()
            raise _snapshot_error(f"cannot persist snapshot: {exc}") from exc
        return SnapshotInfo(
            snapshot_id=int(cursor.lastrowid),
            label=label,
            session_id=engine.spec.session_id,
            declared_version=engine.spec.declared_version,
            content_digest=engine.spec.content_digest,
            snapshot_format_version=SNAPSHOT_FORMAT_VERSION,
            engine_format_version=ENGINE_FORMAT_VERSION,
            state_digest=state_digest,
            next_event_seq=engine.next_event_seq,
            read_only=read_only,
            created_order=order,
        )

    def save(self, engine: Engine, label: str) -> SnapshotInfo:
        if not label:
            raise _snapshot_error("snapshot label must not be empty", "label")
        return self._insert(engine, label)

    def _row(self, snapshot_id: int) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM snapshots WHERE id = ?", (snapshot_id,)
        ).fetchone()
        if row is None:
            raise _snapshot_error(f"snapshot {snapshot_id} does not exist", "snapshot_id")
        return row

    @staticmethod
    def _info(row: Mapping[str, Any] | sqlite3.Row) -> SnapshotInfo:
        return SnapshotInfo(
            snapshot_id=int(row["id"]),
            label=str(row["label"]),
            session_id=str(row["session_id"]),
            declared_version=int(row["declared_version"]),
            content_digest=str(row["content_digest"]),
            snapshot_format_version=int(row["snapshot_format_version"]),
            engine_format_version=int(row["engine_format_version"]),
            state_digest=str(row["state_digest"]),
            next_event_seq=int(row["next_event_seq"]),
            read_only=bool(row["read_only"]),
            created_order=int(row["created_order"]),
        )

    def list(self) -> list[SnapshotInfo]:
        rows = self.connection.execute("SELECT * FROM snapshots ORDER BY created_order").fetchall()
        return [self._info(row) for row in rows]

    def list_snapshots(self) -> list[dict[str, Any]]:
        return [info.to_dict() for info in self.list()]

    def load(self, snapshot_id: int) -> tuple[SnapshotInfo, SessionSpec, dict[str, Any]]:
        row = self._row(snapshot_id)
        info = self._info(row)
        if info.snapshot_format_version != SNAPSHOT_FORMAT_VERSION:
            raise _snapshot_error(
                f"unsupported snapshot format version {info.snapshot_format_version}",
                "snapshot_format_version",
            )
        if info.engine_format_version != ENGINE_FORMAT_VERSION:
            raise _snapshot_error(
                f"unsupported engine format version {info.engine_format_version}",
                "engine_format_version",
            )
        try:
            spec_raw = json.loads(row["spec_json"])
            state_raw = json.loads(row["state_json"])
        except (json.JSONDecodeError, TypeError) as exc:
            raise _snapshot_error(f"snapshot JSON is corrupt: {exc}", "state_json") from exc
        spec = parse_session(spec_raw)
        if info.session_id != spec.session_id:
            raise _snapshot_error("snapshot session id does not match its declaration", "session_id")
        if info.declared_version != spec.declared_version:
            raise _snapshot_error("snapshot declared version does not match its declaration", "declared_version")
        if info.content_digest != spec.content_digest:
            raise _snapshot_error("snapshot session digest does not match its declaration", "session_digest")
        if state_raw.get("content_digest") != info.content_digest:
            raise _snapshot_error("snapshot state content digest is inconsistent", "content_digest")
        if state_raw.get("state_digest") != info.state_digest:
            raise _snapshot_error("snapshot state digest column is inconsistent", "state_digest")
        # Engine reconstruction verifies the state digest and format version too.
        engine = Engine.from_state_dict(spec, state_raw)
        info_state = engine.to_state_dict()
        if info_state["state_digest"] != info.state_digest:
            raise _snapshot_error("snapshot state failed canonical re-digest", "state_digest")
        return info, spec, state_raw

    @staticmethod
    def _flatten(spec: SessionSpec, state: Mapping[str, Any]) -> dict[str, Any]:
        flat: dict[str, Any] = {
            "session.session_id": spec.session_id,
            "session.declared_version": spec.declared_version,
            "session.content_digest": spec.content_digest,
            "session.format_version": spec.format_version,
            "connection.initial_window": spec.connection.initial_window,
            "connection.initial_stream_window": spec.connection.initial_stream_window,
            "connection.max_concurrent_streams": spec.connection.max_concurrent_streams,
            "connection.max_frame_size": spec.connection.max_frame_size,
            "connection.scheduling_quantum": spec.connection.scheduling_quantum,
            "runtime.connection_window": state.get("connection_window"),
            "runtime.current_strategy": state.get("current_strategy"),
            "runtime.pinned_strategy": state.get("pinned_strategy"),
            "runtime.event_cursor": state.get("event_cursor"),
            "runtime.round_index": state.get("round_index"),
            "runtime.frame_count": len(state.get("frames", [])),
            "runtime.transition_count": len(state.get("transitions", [])),
        }
        for stream in state.get("streams", []):
            stream_id = int(stream["stream_id"])
            prefix = f"streams.{stream_id}"
            flat[f"{prefix}.priority"] = stream.get("priority")
            flat[f"{prefix}.window"] = stream.get("window")
            flat[f"{prefix}.state"] = stream.get("state")
            flat[f"{prefix}.body_remaining"] = stream.get("body_remaining")
            flat[f"{prefix}.granted_bytes"] = stream.get("granted_bytes")
        for index, event in enumerate(spec.events):
            prefix = f"events.{index}"
            flat[f"{prefix}.seq"] = event.seq
            flat[f"{prefix}.type"] = event.kind
            flat[f"{prefix}.arrival_ms"] = event.arrival_ms
        return flat

    def _diff(
        self,
        current_spec: SessionSpec | None,
        current_state: Mapping[str, Any] | None,
        snapshot_spec: SessionSpec,
        snapshot_state: Mapping[str, Any],
    ) -> dict[str, list[Any]]:
        if current_spec is None or current_state is None:
            return {"added": [], "removed": [], "changed": []}
        before = self._flatten(current_spec, current_state)
        after = self._flatten(snapshot_spec, snapshot_state)
        added = sorted(set(after) - set(before))
        removed = sorted(set(before) - set(after))
        changed = [
            {"path": path, "before": before[path], "after": after[path]}
            for path in sorted(set(before) & set(after))
            if before[path] != after[path]
        ]
        return {"added": added, "removed": removed, "changed": changed}

    def restore(
        self,
        snapshot_id: int,
        current_spec: SessionSpec | None,
        current_engine: Engine | None = None,
    ) -> RestoreResult:
        info, snapshot_spec, snapshot_state = self.load(snapshot_id)
        version_mismatch = current_spec is not None and info.declared_version != current_spec.declared_version
        digest_mismatch = current_spec is not None and info.content_digest != current_spec.content_digest
        whole_replacement = bool(current_spec is not None and (version_mismatch or digest_mismatch))
        current_state = current_engine.to_state_dict() if current_engine is not None else None
        diff = self._diff(current_spec, current_state, snapshot_spec, snapshot_state)
        backup_id: int | None = None
        if whole_replacement:
            if current_engine is not None:
                backup = self._insert(
                    current_engine,
                    f"backup-before-restore-{snapshot_id}",
                    read_only=True,
                )
                backup_id = backup.snapshot_id
            self.connection.execute("UPDATE snapshots SET read_only = 1")
            self.connection.commit()
        target = Engine.from_state_dict(snapshot_spec, snapshot_state)
        return RestoreResult(
            snapshot=info,
            spec=snapshot_spec,
            engine=target,
            whole_replacement=whole_replacement,
            version_mismatch=version_mismatch,
            content_digest_mismatch=digest_mismatch,
            diff=diff,
            backup_snapshot_id=backup_id,
        )
