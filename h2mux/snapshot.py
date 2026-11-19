"""Versioned, digest-protected session snapshots and whole-state restore."""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .declaration import POLICIES, FrameScript, SessionDeclaration, parse_declaration
from .errors import SnapshotError
from .priority import PriorityTree
from .session import H2Session, MAX_WINDOW
from .util import declaration_diff, digest_value


SNAPSHOT_FORMAT_VERSION = 1


@dataclass
class SnapshotDocument:
    path: Path
    raw: dict[str, Any]
    declaration: SessionDeclaration
    strategy: str
    script_index: int
    script_event_count: int
    script_prefix_digest: str
    script_digest: str
    state: dict[str, Any]
    label: str
    content_digest: str
    created_clock_ms: int

    @property
    def declaration_version(self) -> int:
        return self.declaration.declaration_version


@dataclass
class RestorationResult:
    session: H2Session
    source_path: str
    changed_fields: list[dict[str, Any]]
    version_basis: dict[str, Any]
    replacement_required: bool
    replacement_reasons: list[str]
    script_prefix_match: bool
    script_digest_match: bool
    snapshot_content_digest: str
    snapshot_declaration_version: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "changed_fields": self.changed_fields,
            "version_basis": self.version_basis,
            "replacement_required": self.replacement_required,
            "replacement_reasons": self.replacement_reasons,
            "script_prefix_match": self.script_prefix_match,
            "script_digest_match": self.script_digest_match,
            "snapshot_content_digest": self.snapshot_content_digest,
            "snapshot_declaration_version": self.snapshot_declaration_version,
        }


def build_snapshot(session: H2Session, label: str = "") -> dict[str, Any]:
    declaration_dict = session.declaration.to_dict()
    content = {
        "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
        "declaration_version": session.declaration.declaration_version,
        "declaration_digest": digest_value(declaration_dict),
        "declaration": declaration_dict,
        "strategy": session.strategy,
        "script": {
            "event_count": len(session.script.events),
            "digest": session.script.source_digest,
            "prefix_event_count": session.script_index,
            "prefix_digest": session.script.prefix_digest(session.script_index),
        },
        "label": label,
        "created_clock_ms": session.clock_ms,
        "state": session.state_dict(),
    }
    content["content_digest"] = digest_value(content)
    return content


def _read_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SnapshotError(f"cannot read snapshot: {exc}", path=str(path)) from exc
    except json.JSONDecodeError as exc:
        raise SnapshotError(
            f"snapshot is not valid JSON: {exc.msg}",
            path=str(path),
            details={"line": exc.lineno, "column": exc.colno},
        ) from exc


def load_snapshot_document(path: str | Path) -> SnapshotDocument:
    source = Path(path)
    raw = _read_json_file(source)
    if not isinstance(raw, dict):
        raise SnapshotError("snapshot root must be an object", path=str(source))
    format_version = raw.get("snapshot_format_version")
    if not isinstance(format_version, int) or isinstance(format_version, bool):
        raise SnapshotError("snapshot_format_version must be an integer", path=str(source))
    if format_version != SNAPSHOT_FORMAT_VERSION:
        raise SnapshotError(
            "unsupported snapshot_format_version",
            path=str(source),
            details={"supported": SNAPSHOT_FORMAT_VERSION, "actual": format_version},
        )
    digest = raw.get("content_digest")
    if not isinstance(digest, str):
        raise SnapshotError("snapshot is missing content_digest", path=str(source))
    content = {key: value for key, value in raw.items() if key != "content_digest"}
    computed = digest_value(content)
    if computed != digest:
        raise SnapshotError(
            "snapshot content digest mismatch",
            path=str(source),
            details={"expected": digest, "computed": computed},
        )
    declaration_value = raw.get("declaration")
    if not isinstance(declaration_value, dict):
        raise SnapshotError("snapshot declaration must be an object", path=str(source))
    try:
        declaration = parse_declaration(declaration_value)
    except Exception as exc:
        raise SnapshotError(
            "snapshot declaration is invalid",
            path=str(source),
            details={"cause": getattr(exc, "to_dict", lambda: {"message": str(exc)})()},
        ) from exc
    declaration_digest = raw.get("declaration_digest")
    actual_declaration_digest = digest_value(declaration.to_dict())
    if declaration_digest != actual_declaration_digest:
        raise SnapshotError(
            "snapshot declaration digest mismatch",
            path=str(source),
            details={"expected": declaration_digest, "computed": actual_declaration_digest},
        )
    state = raw.get("state")
    if not isinstance(state, dict):
        raise SnapshotError("snapshot state must be an object", path=str(source))
    _validate_state(state, source, declaration)
    script = raw.get("script")
    if not isinstance(script, dict):
        raise SnapshotError("snapshot script metadata must be an object", path=str(source))
    for key in ("event_count", "digest", "prefix_event_count", "prefix_digest"):
        if key not in script:
            raise SnapshotError(f"snapshot script metadata is missing {key}", path=str(source))
    if not isinstance(script["event_count"], int) or isinstance(script["event_count"], bool):
        raise SnapshotError("snapshot script event_count must be an integer", path=str(source))
    if not isinstance(script["prefix_event_count"], int) or isinstance(script["prefix_event_count"], bool):
        raise SnapshotError("snapshot script prefix_event_count must be an integer", path=str(source))
    if not isinstance(script["digest"], str) or not isinstance(script["prefix_digest"], str):
        raise SnapshotError("snapshot script digests must be strings", path=str(source))
    if script["prefix_event_count"] != int(state["script_index"]):
        raise SnapshotError(
            "snapshot script cursor and state script_index disagree",
            path=str(source),
        )
    if not 0 <= script["prefix_event_count"] <= script["event_count"]:
        raise SnapshotError("snapshot script cursor is outside event_count", path=str(source))
    strategy = raw.get("strategy")
    if not isinstance(strategy, str) or strategy != state.get("strategy"):
        raise SnapshotError("snapshot strategy metadata is inconsistent", path=str(source))
    return SnapshotDocument(
        path=source,
        raw=raw,
        declaration=declaration,
        strategy=strategy,
        script_index=int(state["script_index"]),
        script_event_count=int(script["event_count"]),
        script_prefix_digest=str(script["prefix_digest"]),
        script_digest=str(script["digest"]),
        state=state,
        label=str(raw.get("label", "")),
        content_digest=digest,
        created_clock_ms=int(raw.get("created_clock_ms", 0)),
    )


def _state_int(state: dict[str, Any], key: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    value = state.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise SnapshotError(f"snapshot state {key} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise SnapshotError(f"snapshot state {key} is outside its valid range")
    return value


def _validate_state(
    state: dict[str, Any],
    source: Path,
    declaration: SessionDeclaration,
) -> None:
    if state.get("strategy") not in POLICIES:
        raise SnapshotError("snapshot strategy is not supported", path=str(source))
    if state.get("initial_strategy", state.get("strategy")) not in POLICIES:
        raise SnapshotError("snapshot initial strategy is not supported", path=str(source))
    if state.get("clock_mode", "virtual_monotonic_ms") != "virtual_monotonic_ms":
        raise SnapshotError("snapshot clock mode is not supported", path=str(source))
    for key in ("clock_ms", "frame_seq", "script_index", "schedule_round", "send_conn_window", "recv_conn_window"):
        _state_int(state, key)
    _state_int(state, "send_conn_window", maximum=MAX_WINDOW)
    _state_int(state, "recv_conn_window", maximum=MAX_WINDOW)
    _state_int(state, "peer_initial_stream_window", maximum=MAX_WINDOW)
    _state_int(state, "peer_max_frame_size", minimum=16384, maximum=16777215)
    _state_int(state, "peer_max_concurrent", minimum=0, maximum=(1 << 32) - 1)
    _state_int(state, "priority_update_seq")
    streams = state.get("streams")
    if not isinstance(streams, list):
        raise SnapshotError("snapshot state streams must be an array", path=str(source))
    seen: set[int] = set()
    for item in streams:
        if not isinstance(item, dict):
            raise SnapshotError("snapshot stream state must be an object", path=str(source))
        stream_id = item.get("stream_id")
        if not isinstance(stream_id, int) or stream_id in seen:
            raise SnapshotError("snapshot stream ids must be unique integers", path=str(source))
        seen.add(stream_id)
        if item.get("state") not in {
            "idle",
            "open",
            "half_closed_local",
            "half_closed_remote",
            "closed",
        }:
            raise SnapshotError("snapshot stream state is invalid", path=str(source))
        for window_key in ("send_window", "recv_window"):
            value = item.get(window_key)
            if not isinstance(value, int) or value < 0 or value > MAX_WINDOW:
                raise SnapshotError("snapshot stream window is invalid", path=str(source))
    expected_stream_ids = {item.stream_id for item in declaration.requests}
    if seen != expected_stream_ids:
        raise SnapshotError(
            "snapshot streams do not match the declaration requests",
            path=str(source),
            details={"expected": sorted(expected_stream_ids), "actual": sorted(seen)},
        )
    queue = state.get("queue")
    if not isinstance(queue, list) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in queue
    ):
        raise SnapshotError("snapshot queue must be an integer array", path=str(source))
    if any(stream_id not in expected_stream_ids for stream_id in queue):
        raise SnapshotError("snapshot queue contains an undeclared stream", path=str(source))
    ledger = state.get("ledger")
    if not isinstance(ledger, dict):
        raise SnapshotError("snapshot ledger must be an object", path=str(source))
    for component in (
        "records",
        "transitions",
        "empty_waits",
        "queue_waits",
        "priority_events",
        "schedule_rounds",
        "errors",
        "replacements",
        "controls",
    ):
        if not isinstance(ledger.get(component), list):
            raise SnapshotError(f"snapshot ledger {component} must be an array", path=str(source))
    priority_value = state.get("priority")
    if not isinstance(priority_value, dict):
        raise SnapshotError("snapshot priority tree must be an object", path=str(source))
    try:
        priority = PriorityTree.from_dict(priority_value)
    except Exception as exc:
        raise SnapshotError(
            "snapshot priority tree is invalid",
            path=str(source),
            details={"cause": str(exc)},
        ) from exc
    if set(priority.stream_ids) != expected_stream_ids:
        raise SnapshotError("snapshot priority nodes do not match declared streams", path=str(source))


class SnapshotStore:
    def __init__(self, directory: str | Path = "snapshots") -> None:
        self.directory = Path(directory)

    def _path(self, name: str | Path) -> Path:
        candidate = Path(name)
        return candidate if candidate.is_absolute() else self.directory / candidate

    def save(self, session: H2Session, name: str | Path, *, label: str = "") -> Path:
        target = self._path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        document = build_snapshot(session, label=label)
        if target.exists():
            backup = Path(str(target) + ".bak")
            backup.parent.mkdir(parents=True, exist_ok=True)
            if backup.exists():
                try:
                    os.chmod(backup, stat.S_IWRITE | stat.S_IREAD)
                except OSError:
                    pass
            shutil.copyfile(target, backup)
            try:
                os.chmod(backup, stat.S_IREAD)
            except OSError:
                pass
        fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, target)
        except Exception:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise
        return target

    def load(self, name: str | Path) -> SnapshotDocument:
        return load_snapshot_document(self._path(name))

    def list_documents(self) -> list[dict[str, Any]]:
        if not self.directory.exists():
            return []
        rows: list[dict[str, Any]] = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                document = load_snapshot_document(path)
                rows.append(
                    {
                        "name": path.name,
                        "path": str(path),
                        "label": document.label,
                        "declaration_version": document.declaration_version,
                        "strategy": document.strategy,
                        "script_index": document.script_index,
                        "content_digest": document.content_digest,
                        "created_clock_ms": document.created_clock_ms,
                    }
                )
            except SnapshotError as exc:
                rows.append(
                    {
                        "name": path.name,
                        "path": str(path),
                        "error": exc.to_dict(),
                    }
                )
        return rows


def restore_snapshot(
    snapshot_path: str | Path,
    script: FrameScript,
    *,
    current_session: H2Session | None = None,
    current_declaration: SessionDeclaration | None = None,
    current_strategy: str | None = None,
) -> RestorationResult:
    document = load_snapshot_document(snapshot_path)
    if document.script_index > len(script.events):
        raise SnapshotError(
            "snapshot cursor is beyond the supplied script",
            path=str(snapshot_path),
            details={"snapshot_index": document.script_index, "script_events": len(script.events)},
        )
    actual_prefix_digest = script.prefix_digest(document.script_index)
    prefix_match = actual_prefix_digest == document.script_prefix_digest
    if not prefix_match:
        raise SnapshotError(
            "snapshot script prefix does not match the supplied script",
            path=str(snapshot_path),
            details={"expected": document.script_prefix_digest, "actual": actual_prefix_digest},
        )
    comparison_declaration = (
        current_session.declaration if current_session is not None else current_declaration
    )
    comparison_strategy = (
        current_session.strategy if current_session is not None else current_strategy
    )
    changed_fields: list[dict[str, Any]] = []
    version_basis: dict[str, Any] = {
        "snapshot_declaration_version": document.declaration.declaration_version,
        "snapshot_declaration_digest": digest_value(document.declaration.to_dict()),
        "snapshot_content_digest": document.content_digest,
    }
    reasons: list[str] = []
    if comparison_declaration is not None:
        changed_fields = declaration_diff(
            comparison_declaration.to_dict(), document.declaration.to_dict()
        )
        version_basis.update(
            {
                "current_declaration_version": comparison_declaration.declaration_version,
                "current_declaration_digest": comparison_declaration.content_digest,
            }
        )
        if comparison_declaration.declaration_version != document.declaration.declaration_version:
            reasons.append("DECLARATION_VERSION_CHANGED")
        if comparison_declaration.content_digest != document.declaration.content_digest:
            reasons.append("DECLARATION_CONTENT_CHANGED")
    if comparison_strategy is not None:
        version_basis["current_strategy"] = comparison_strategy
        version_basis["snapshot_strategy"] = document.strategy
        if comparison_strategy != document.strategy:
            reasons.append("STRATEGY_CHANGED")
    replacement_required = bool(reasons)
    session = H2Session(
        document.declaration,
        script,
        strategy=document.strategy,
        starvation_rounds=int(document.state.get("starvation_rounds", 3)),
    )
    try:
        session.restore_state(document.state)
    except Exception as exc:
        raise SnapshotError(
            "snapshot state could not be restored",
            path=str(snapshot_path),
            details={"cause": str(exc)},
        ) from exc
    session.ledger.replacements.append(
        {
            "kind": "whole_state_restore",
            "source_path": str(snapshot_path),
            "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
            "snapshot_declaration_version": document.declaration.declaration_version,
            "snapshot_content_digest": document.content_digest,
            "replacement_required": replacement_required,
            "reasons": reasons,
            "changed_fields": changed_fields,
            "version_basis": version_basis,
            "restored_script_index": document.script_index,
        }
    )
    return RestorationResult(
        session=session,
        source_path=str(snapshot_path),
        changed_fields=changed_fields,
        version_basis=version_basis,
        replacement_required=replacement_required,
        replacement_reasons=reasons,
        script_prefix_match=prefix_match,
        script_digest_match=script.source_digest == document.script_digest,
        snapshot_content_digest=document.content_digest,
        snapshot_declaration_version=document.declaration.declaration_version,
    )
