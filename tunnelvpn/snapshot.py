"""Checksummed, atomically written snapshots and whole-state restoration."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import GlobalLimits, TunnelConfig, canonical_json_bytes, parse_config
from .errors import ErrorCode, TunnelError
from .metrics import Ledger
from .session import SessionState

SNAPSHOT_FORMAT = "TUNNELVPN_SNAPSHOT"
SNAPSHOT_VERSION = 1


@dataclass(frozen=True)
class SnapshotInfo:
    path: Path
    complete: bool
    route_version: str | None
    created_utc: str | None
    ledger_digest: str | None
    session_count: int | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "name": self.path.name,
            "complete": self.complete,
            "route_version": self.route_version,
            "created_utc": self.created_utc,
            "ledger_digest": self.ledger_digest,
            "session_count": self.session_count,
            "error": self.error,
        }


@dataclass(frozen=True)
class Snapshot:
    path: Path
    payload: dict[str, Any]
    payload_digest: str

    @property
    def route_version(self) -> str:
        return str(self.payload["route_version"])

    @property
    def created_utc(self) -> str:
        return str(self.payload["created_utc"])

    def config(self) -> TunnelConfig:
        return parse_config(self.payload["config"])

    def ledger(self) -> Ledger:
        raw = self.payload.get("ledger", {})
        if not isinstance(raw, dict):
            raise TunnelError(ErrorCode.E_STATE, "快照账本结构非法")
        records = raw.get("records", [])
        if not isinstance(records, list):
            raise TunnelError(ErrorCode.E_STATE, "快照账本记录必须为数组")
        ledger = Ledger.from_dicts(records)
        expected_digest = raw.get("digest")
        if expected_digest != ledger.digest():
            raise TunnelError(ErrorCode.E_STATE, "快照账本摘要校验失败")
        return ledger

    def raw_sessions(self) -> list[dict[str, Any]]:
        sessions = self.payload.get("sessions", [])
        if not isinstance(sessions, list):
            raise TunnelError(ErrorCode.E_STATE, "快照会话状态必须为数组")
        return sessions

    def restored_sessions(self) -> list[SessionState]:
        restored: list[SessionState] = []
        for raw in self.raw_sessions():
            if not isinstance(raw, dict):
                raise TunnelError(ErrorCode.E_STATE, "快照会话记录必须为对象")
            state = SessionState.from_dict(raw)
            if state.status in {"RUNNING", "QUEUED"} or state.end_ms is None:
                state.status = "UNKNOWN"
                state.business_code = "UNKNOWN"
                state.requires_rerun = True
                state.end_ms = None
            restored.append(state)
        return sorted(restored, key=lambda item: (item.start_ms, item.session_id))

    def statistics(self) -> dict[str, Any]:
        raw = self.payload.get("statistics")
        if not isinstance(raw, dict):
            raise TunnelError(ErrorCode.E_STATE, "快照统计结论缺失")
        return raw


@dataclass(frozen=True)
class RestoreReport:
    whole_replaced: bool
    version_changed: bool
    snapshot_route_version: str
    current_route_version: str
    route_changes: list[dict[str, Any]]
    global_changes: list[dict[str, Any]]
    changed_tunnel_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "whole_replaced": self.whole_replaced,
            "version_changed": self.version_changed,
            "snapshot_route_version": self.snapshot_route_version,
            "current_route_version": self.current_route_version,
            "route_changes": self.route_changes,
            "global_changes": self.global_changes,
            "changed_tunnel_ids": self.changed_tunnel_ids,
        }


@dataclass(frozen=True)
class RestoredState:
    snapshot: Snapshot
    config: TunnelConfig
    ledger: Ledger
    sessions: list[SessionState]
    sequence_watermarks: dict[str, int]
    budget_usage: dict[str, Any]
    statistics: dict[str, Any]
    report: RestoreReport


class SnapshotStore:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def latest_path(self) -> Path:
        return self.directory / "latest.json"

    def save_payload(self, payload: dict[str, Any]) -> Path:
        payload_bytes = canonical_json_bytes(payload)
        digest = hashlib.sha256(payload_bytes).hexdigest()
        envelope = {
            "format": SNAPSHOT_FORMAT,
            "snapshot_version": SNAPSHOT_VERSION,
            "payload_sha256": digest,
            "payload": payload,
        }
        content = canonical_json_bytes(envelope) + b"\n"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        final_path = self.directory / f"snapshot-{stamp}-{os.getpid()}.json"
        self._atomic_write(final_path, content)
        self._atomic_write(self.latest_path, content)
        return final_path

    def save_engine(self, engine: Any) -> Snapshot:
        route_version = engine.config.route_version
        created = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        sessions = engine.sessions_snapshot()
        payload = {
            "created_utc": created,
            "route_version": route_version,
            "config": engine.config.to_dict(),
            "sequence_watermarks": engine.sequences.watermarks(),
            "budget_usage": engine.budget.budget_usage(),
            "sessions": sessions,
            "ledger": {
                "records": engine.ledger.to_dicts(),
                "digest": engine.ledger.digest(),
            },
            "statistics": engine.ledger.statistics(),
            "peer_behaviors": [
                peer.to_dict() for peer in sorted(engine.config.peers, key=lambda item: item.id)
            ],
            "session_budget_usage": [
                {
                    "session_id": item["session_id"],
                    "route_id": item["route_id"],
                    "payload_bytes": item["payload_bytes"],
                    "forwarded_bytes": item["forwarded_bytes"],
                    "wire_bytes": item["wire_bytes"],
                }
                for item in sessions
            ],
        }
        path = self.save_payload(payload)
        envelope_digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        return Snapshot(path=path, payload=payload, payload_digest=envelope_digest)

    def _atomic_write(self, path: Path, content: bytes) -> None:
        fd, temp_name = tempfile.mkstemp(
            prefix=".snapshot-tmp-", suffix=".part", dir=self.directory
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
            self._fsync_directory()
        except BaseException:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def _fsync_directory(self) -> None:
        if os.name == "nt":
            return
        try:
            fd = os.open(self.directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    def load(self, path: str | Path) -> Snapshot:
        file_path = Path(path)
        if not file_path.is_absolute():
            candidate = self.directory / file_path
            if candidate.exists():
                file_path = candidate
        try:
            raw_bytes = file_path.read_bytes()
        except OSError as exc:
            raise TunnelError(
                ErrorCode.E_STATE, f"无法读取快照: {exc}", field=str(file_path)
            ) from exc
        try:
            envelope = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TunnelError(
                ErrorCode.E_STATE, "快照 JSON 不完整或损坏", field=str(file_path)
            ) from exc
        if not isinstance(envelope, dict) or envelope.get("format") != SNAPSHOT_FORMAT:
            raise TunnelError(ErrorCode.E_STATE, "快照格式标识不匹配", field="format")
        if envelope.get("snapshot_version") != SNAPSHOT_VERSION:
            raise TunnelError(
                ErrorCode.E_STATE, "不支持的快照版本", field="snapshot_version"
            )
        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            raise TunnelError(ErrorCode.E_STATE, "快照 payload 必须是对象")
        digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        if envelope.get("payload_sha256") != digest:
            raise TunnelError(ErrorCode.E_STATE, "快照校验摘要不一致", field="payload_sha256")
        snapshot = Snapshot(path=file_path, payload=payload, payload_digest=digest)
        # Validate the critical nested structures eagerly, not on a later UI read.
        snapshot.config()
        snapshot.ledger()
        snapshot.restored_sessions()
        snapshot.statistics()
        return snapshot

    def load_latest(self) -> Snapshot:
        if not self.latest_path.exists():
            raise TunnelError(ErrorCode.E_STATE, "没有可恢复的 latest 快照")
        return self.load(self.latest_path)

    def list_snapshots(self) -> list[SnapshotInfo]:
        entries: list[SnapshotInfo] = []
        paths = sorted(
            self.directory.glob("snapshot-*.json"), key=lambda item: item.name, reverse=True
        )
        for path in paths:
            try:
                with path.open("r", encoding="utf-8") as handle:
                    envelope = json.load(handle)
                payload = envelope.get("payload", {})
                digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
                complete = (
                    envelope.get("format") == SNAPSHOT_FORMAT
                    and envelope.get("snapshot_version") == SNAPSHOT_VERSION
                    and envelope.get("payload_sha256") == digest
                )
                entries.append(
                    SnapshotInfo(
                        path=path,
                        complete=complete,
                        route_version=payload.get("route_version"),
                        created_utc=payload.get("created_utc"),
                        ledger_digest=payload.get("ledger", {}).get("digest"),
                        session_count=len(payload.get("sessions", [])),
                        error=None if complete else "checksum_mismatch",
                    )
                )
            except Exception as exc:
                entries.append(
                    SnapshotInfo(
                        path=path,
                        complete=False,
                        route_version=None,
                        created_utc=None,
                        ledger_digest=None,
                        session_count=None,
                        error=str(exc),
                    )
                )
        return entries


def restore_snapshot(snapshot: Snapshot, current_config: TunnelConfig) -> RestoredState:
    snapshot_config = snapshot.config()
    ledger = snapshot.ledger()
    sessions = snapshot.restored_sessions()
    statistics = snapshot.statistics()
    recomputed = ledger.statistics()
    if canonical_json_bytes(statistics) != canonical_json_bytes(recomputed):
        raise TunnelError(ErrorCode.E_STATE, "快照统计结论与账本重算结果不一致")
    raw_watermarks = snapshot.payload.get("sequence_watermarks", {})
    if not isinstance(raw_watermarks, dict):
        raise TunnelError(ErrorCode.E_STATE, "快照序号水位结构非法")
    watermarks: dict[str, int] = {}
    for tunnel_id, value in raw_watermarks.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TunnelError(
                ErrorCode.E_STATE,
                "快照序号水位必须为非负整数",
                tunnel_id=str(tunnel_id),
            )
        watermarks[str(tunnel_id)] = value
    raw_budget = snapshot.payload.get("budget_usage", {})
    if not isinstance(raw_budget, dict):
        raise TunnelError(ErrorCode.E_STATE, "快照预算用量结构非法")

    session_ids = {state.session_id for state in sessions}
    for state in sessions:
        in_ledger = ledger.contains(state.session_id)
        if state.requires_rerun and in_ledger:
            raise TunnelError(
                ErrorCode.E_STATE,
                "未知会话不得已有账本记录",
                tunnel_id=state.route_id,
                session_id=state.session_id,
            )
        if not state.requires_rerun and not in_ledger:
            raise TunnelError(
                ErrorCode.E_STATE,
                "已结束会话缺少账本记录",
                tunnel_id=state.route_id,
                session_id=state.session_id,
            )
    for record in ledger.records():
        if record.session_id not in session_ids:
            raise TunnelError(
                ErrorCode.E_STATE,
                "账本记录缺少会话状态",
                tunnel_id=record.route_id,
                session_id=record.session_id,
            )
        watermark = watermarks.get(record.route_id)
        if watermark is not None and record.sequence > watermark:
            raise TunnelError(
                ErrorCode.E_STATE,
                "账本序号超过快照水位",
                tunnel_id=record.route_id,
                session_id=record.session_id,
            )

    route_changes, changed_ids = _route_changes(current_config, snapshot_config)
    global_changes = _global_changes(
        current_config.global_limits, snapshot_config.global_limits
    )
    report = RestoreReport(
        whole_replaced=True,
        version_changed=current_config.route_version != snapshot_config.route_version,
        snapshot_route_version=snapshot_config.route_version,
        current_route_version=current_config.route_version,
        route_changes=route_changes,
        global_changes=global_changes,
        changed_tunnel_ids=changed_ids,
    )
    return RestoredState(
        snapshot=snapshot,
        config=snapshot_config,
        ledger=ledger,
        sessions=sessions,
        sequence_watermarks=watermarks,
        budget_usage=raw_budget,
        statistics=statistics,
        report=report,
    )


def _route_changes(
    current: TunnelConfig, restored: TunnelConfig
) -> tuple[list[dict[str, Any]], list[str]]:
    current_routes = current.route_map()
    restored_routes = restored.route_map()
    changes: list[dict[str, Any]] = []
    changed_ids: set[str] = set()
    for tunnel_id in sorted(set(current_routes) | set(restored_routes)):
        current_route = current_routes.get(tunnel_id)
        restored_route = restored_routes.get(tunnel_id)
        if current_route is None:
            changes.append(
                {
                    "tunnel_id": tunnel_id,
                    "change": "only_in_snapshot",
                    "fields": ["route"],
                }
            )
            changed_ids.add(tunnel_id)
            continue
        if restored_route is None:
            changes.append(
                {
                    "tunnel_id": tunnel_id,
                    "change": "only_in_current",
                    "fields": ["route"],
                }
            )
            changed_ids.add(tunnel_id)
            continue
        current_dict = current_route.to_dict()
        restored_dict = restored_route.to_dict()
        fields: list[dict[str, Any]] = []
        for field in sorted(current_dict):
            if current_dict[field] != restored_dict[field]:
                fields.append(
                    {
                        "field": field,
                        "current": current_dict[field],
                        "snapshot": restored_dict[field],
                    }
                )
        if fields:
            changes.append(
                {"tunnel_id": tunnel_id, "change": "changed", "fields": fields}
            )
            changed_ids.add(tunnel_id)
    return changes, sorted(changed_ids)


def _global_changes(
    current: GlobalLimits, restored: GlobalLimits
) -> list[dict[str, Any]]:
    current_dict = current.to_dict()
    restored_dict = restored.to_dict()
    return [
        {
            "field": field,
            "current": current_dict[field],
            "snapshot": restored_dict[field],
        }
        for field in sorted(current_dict)
        if current_dict[field] != restored_dict[field]
    ]
