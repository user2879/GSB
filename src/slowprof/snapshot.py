"""Versioned snapshots, whole-state restoration and read-only archives."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import pretty_json_bytes
from .errors import E_IO, E_VERSION, SlowProfError, VersionError
from .models import CURRENT_METRIC_VERSION, SNAPSHOT_SCHEMA_VERSION
from .session import ProfileSession

_REQUIRED_TOP_LEVEL = (
    "interval",
    "sample",
    "aggregation",
    "folded_result",
    "hotspot_ranking",
    "error_annotation",
)


@dataclass(frozen=True, slots=True)
class RestoreOutcome:
    active_session: ProfileSession
    strategy: str
    mismatches: tuple[dict[str, Any], ...]
    changed_frames: tuple[dict[str, Any], ...]
    rank_moves: tuple[dict[str, Any], ...]
    archived_snapshot_path: str | None
    archived_snapshot_read_only: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "strategy": self.strategy,
            "mismatches": list(self.mismatches),
            "changed_frames": list(self.changed_frames),
            "rank_moves": list(self.rank_moves),
            "archived_snapshot_path": self.archived_snapshot_path,
            "archived_snapshot_read_only": self.archived_snapshot_read_only,
            "active_conclusion_sha256": self.active_session.conclusion_sha256,
        }


def _version_error(message: str, *, field: str, details: dict[str, Any] | None = None) -> VersionError:
    return VersionError(message, field=field, details=details)


def load_snapshot(path: str | Path) -> dict[str, Any]:
    """Read and gate a snapshot before any state replacement is attempted."""

    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise SlowProfError(
            E_IO,
            "cannot read snapshot",
            field="snapshot_path",
            details={"path": str(path), "error": str(exc)},
        ) from exc
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _version_error(
            "snapshot is not valid UTF-8 JSON",
            field="$snapshot",
            details={"error": str(exc)},
        ) from exc
    if not isinstance(document, dict):
        raise _version_error("snapshot root must be an object", field="$snapshot")

    schema = document.get("snapshot_schema_version")
    if not isinstance(schema, str) or not schema:
        raise _version_error(
            "snapshot_schema_version is missing",
            field="snapshot_schema_version",
        )
    if schema != SNAPSHOT_SCHEMA_VERSION:
        raise _version_error(
            "unsupported snapshot schema version",
            field="snapshot_schema_version",
            details={"expected": SNAPSHOT_SCHEMA_VERSION, "actual": schema},
        )

    metric_version = document.get("metric_version")
    if not isinstance(metric_version, str) or not metric_version:
        raise _version_error(
            "metric_version (metric methodology version) is missing",
            field="metric_version",
        )

    for key in _REQUIRED_TOP_LEVEL:
        if key not in document:
            raise _version_error(
                f"snapshot is missing required section {key}",
                field=key,
            )
    if not isinstance(document["interval"], dict):
        raise _version_error("snapshot interval must be an object", field="interval")
    if not isinstance(document["sample"], dict):
        raise _version_error("snapshot sample metadata must be an object", field="sample")
    if not isinstance(document["aggregation"], dict):
        raise _version_error("snapshot aggregation must be an object", field="aggregation")
    if not isinstance(document["folded_result"], dict):
        raise _version_error("snapshot folded_result must be an object", field="folded_result")
    if not isinstance(document["hotspot_ranking"], list):
        raise _version_error("snapshot hotspot_ranking must be a list", field="hotspot_ranking")
    if not isinstance(document["error_annotation"], dict):
        raise _version_error(
            "snapshot error_annotation must be an object",
            field="error_annotation",
        )
    return document


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        temp_name = None
    except OSError as exc:
        raise SlowProfError(
            E_IO,
            "cannot write snapshot atomically",
            field="snapshot_path",
            details={"path": str(path), "error": str(exc)},
        ) from exc
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except OSError:
                pass


def save_snapshot(session: ProfileSession, path: str | Path) -> str:
    document = session.to_snapshot_document()
    _atomic_write(Path(path), pretty_json_bytes(document))
    return str(path)


def _metadata_mismatches(snapshot: dict[str, Any], current: ProfileSession) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []

    def compare(field: str, expected: Any, actual: Any) -> None:
        if expected != actual:
            mismatches.append({"field": field, "snapshot": expected, "current": actual})

    compare("metric_version", snapshot.get("metric_version"), CURRENT_METRIC_VERSION)
    interval = snapshot.get("interval", {})
    compare(
        "interval.interval_version",
        interval.get("interval_version"),
        current.interval.interval_version,
    )
    compare("interval.start_ms", interval.get("start_ms"), current.interval.start_ms)
    compare("interval.end_ms", interval.get("end_ms"), current.interval.end_ms)

    metadata = current.metadata
    sample = snapshot.get("sample", {})
    if metadata is not None:
        compare("sample.file_sha256", sample.get("file_sha256"), metadata.file_sha256)
        compare(
            "sample.selected_samples_sha256",
            sample.get("selected_samples_sha256"),
            metadata.selected_samples_sha256,
        )

    aggregation = snapshot.get("aggregation", {})
    compare("aggregation.path", aggregation.get("path"), current.aggregation.selection.path)
    compare("policy.scale", snapshot.get("policy", {}).get("scale"), current.policy.scale)
    compare("policy.top_k", snapshot.get("policy", {}).get("top_k"), current.policy.top_k)
    compare(
        "policy.max_depth",
        snapshot.get("policy", {}).get("max_depth"),
        current.policy.max_depth,
    )
    compare(
        "policy.full_unique_stack_threshold",
        snapshot.get("policy", {}).get("full_unique_stack_threshold"),
        current.policy.full_unique_stack_threshold,
    )
    compare(
        "policy.full_total_weight_threshold",
        snapshot.get("policy", {}).get("full_total_weight_threshold"),
        current.policy.full_total_weight_threshold,
    )
    compare(
        "conclusion_sha256",
        snapshot.get("conclusion_sha256"),
        current.conclusion_sha256,
    )
    return mismatches


def _changed_frames(snapshot: dict[str, Any], current: ProfileSession) -> list[dict[str, Any]]:
    old_frames = {
        item.get("frame"): item
        for item in snapshot.get("folded_result", {}).get("frames", [])
        if isinstance(item, dict) and isinstance(item.get("frame"), str)
    }
    new_frames = {stat.frame: stat.to_dict() for stat in current.aggregation.frame_stats}
    changed: list[dict[str, Any]] = []
    fields = (
        "self_time_ms",
        "cumulative_time_ms",
        "participant_weight",
        "occurrence_count",
    )
    for frame in sorted(set(old_frames) | set(new_frames)):
        old = old_frames.get(frame)
        new = new_frames.get(frame)
        changes: dict[str, Any] = {}
        for field in fields:
            old_value = old.get(field) if old is not None else None
            new_value = new.get(field) if new is not None else None
            if old_value != new_value:
                changes[field] = {"snapshot": old_value, "current": new_value}
        if changes:
            changed.append({"frame": frame, "changes": changes})
    return changed


def _rank_moves(snapshot: dict[str, Any], current: ProfileSession) -> list[dict[str, Any]]:
    old_positions = {
        item.get("frame"): item.get("rank")
        for item in snapshot.get("hotspot_ranking", [])
        if isinstance(item, dict) and isinstance(item.get("frame"), str)
    }
    new_positions = {
        item["frame"]: item["rank"] for item in current.ranking_report["ranking"]
    }
    moves: list[dict[str, Any]] = []
    for frame in set(old_positions) | set(new_positions):
        old_pos = old_positions.get(frame)
        new_pos = new_positions.get(frame)
        if old_pos != new_pos:
            moves.append(
                {
                    "frame": frame,
                    "snapshot_rank": old_pos,
                    "current_rank": new_pos,
                }
            )
    moves.sort(
        key=lambda item: (
            item["current_rank"] if item["current_rank"] is not None else 10**9,
            item["snapshot_rank"] if item["snapshot_rank"] is not None else 10**9,
            item["frame"],
        )
    )
    return moves


def _archive_read_only(snapshot_path: Path) -> tuple[str, bool]:
    try:
        with open(snapshot_path, "rb") as source:
            payload = source.read()
    except OSError as exc:
        raise SlowProfError(
            E_IO,
            "cannot read old snapshot for archiving",
            field="snapshot_path",
            details={"path": str(snapshot_path), "error": str(exc)},
        ) from exc
    suffix = snapshot_path.suffix or ".json"
    stem = snapshot_path.stem
    token = hashlib.sha256(payload).hexdigest()[:16]
    archive = snapshot_path.with_name(f"{stem}.readonly-{token}{suffix}")
    try:
        if archive.exists():
            existing = archive.read_bytes()
            if existing != payload:
                raise SlowProfError(
                    E_IO,
                    "content-addressed snapshot archive already exists with different bytes",
                    field="snapshot_path",
                    details={"path": str(archive)},
                )
        else:
            with open(archive, "xb") as target:
                target.write(payload)
                target.flush()
                os.fsync(target.fileno())
        mode = os.stat(archive).st_mode
        os.chmod(archive, mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        return str(archive), True
    except SlowProfError:
        raise
    except OSError as exc:
        raise SlowProfError(
            E_IO,
            "cannot preserve old snapshot as a read-only archive",
            field="snapshot_path",
            details={"path": str(snapshot_path), "error": str(exc)},
        ) from exc


def restore_snapshot(snapshot_path: str | Path, current_session: ProfileSession) -> RestoreOutcome:
    """Compare metadata first, then use either the snapshot or whole replacement.

    A mismatch never merges individual frames.  The current session is already
    a complete state; it becomes the active state as a unit, while the old
    snapshot is copied to a read-only archive before this function returns.
    """

    path = Path(snapshot_path)
    snapshot = load_snapshot(path)
    mismatches = _metadata_mismatches(snapshot, current_session)
    if not mismatches:
        return RestoreOutcome(
            active_session=current_session,
            strategy="restore",
            mismatches=(),
            changed_frames=(),
            rank_moves=(),
            archived_snapshot_path=None,
            archived_snapshot_read_only=False,
        )

    changed = _changed_frames(snapshot, current_session)
    moves = _rank_moves(snapshot, current_session)
    archived_path, is_read_only = _archive_read_only(path)
    return RestoreOutcome(
        active_session=current_session,
        strategy="replace",
        mismatches=tuple(mismatches),
        changed_frames=tuple(changed),
        rank_moves=tuple(moves),
        archived_snapshot_path=archived_path,
        archived_snapshot_read_only=is_read_only,
    )


class SnapshotStore:
    """Small object facade for callers that prefer an injected store."""

    def save(self, session: ProfileSession, path: str | Path) -> str:
        return save_snapshot(session, path)

    def load(self, path: str | Path) -> dict[str, Any]:
        return load_snapshot(path)

    def restore(self, path: str | Path, current_session: ProfileSession) -> RestoreOutcome:
        return restore_snapshot(path, current_session)
