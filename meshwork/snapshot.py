from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .canonical import canonical_json, sha256_hex
from .errors import MeshError
from .logging_utils import StructuredLogger

SNAPSHOT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class SnapshotState:
    route_revision: int
    instance_version: str
    budget_used_global: int
    budget_used_by_service: dict[str, int]
    breakers: dict[str, dict[str, Any]]
    completed_cursor: int
    ledger_format_version: int
    ledger_digest: str
    request_digest: str = ""

    def payload(self) -> dict[str, Any]:
        return {
            "route_revision": self.route_revision,
            "instance_version": self.instance_version,
            "budget_used_global": self.budget_used_global,
            "budget_used_by_service": {
                key: int(self.budget_used_by_service[key])
                for key in sorted(self.budget_used_by_service)
            },
            "breakers": {key: self.breakers[key] for key in sorted(self.breakers)},
            "completed_cursor": self.completed_cursor,
            "ledger_format_version": self.ledger_format_version,
            "ledger_digest": self.ledger_digest,
            "request_digest": self.request_digest,
        }


@dataclass(frozen=True)
class SnapshotEnvelope:
    path: str
    created_wall_ms: int
    format_version: int
    digest: str
    state: SnapshotState


def snapshot_digest(payload: dict[str, Any]) -> str:
    return sha256_hex(payload)


class SnapshotStore:
    def __init__(
        self,
        directory: str | Path,
        logger: StructuredLogger | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._logger = logger

    def _paths(self) -> list[Path]:
        return sorted(self.directory.glob("snapshot-*.json"), key=lambda item: item.name)

    def write(self, state: SnapshotState) -> SnapshotEnvelope:
        payload = state.payload()
        digest = snapshot_digest(payload)
        created = int(time.time() * 1000)
        envelope = {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "created_wall_ms": created,
            "digest": digest,
            "payload": payload,
        }
        path = self.directory / f"snapshot-{created:013d}-{digest[:10]}.json"
        while path.exists():
            created += 1
            path = self.directory / f"snapshot-{created:013d}-{digest[:10]}.json"
        temp = path.with_suffix(".json.tmp")
        temp.write_text(canonical_json(envelope), encoding="utf-8")
        os.replace(temp, path)
        if self._logger is not None:
            self._logger.info(
                "snapshot_written",
                path=str(path),
                digest=digest,
                route_revision=state.route_revision,
                completed_cursor=state.completed_cursor,
            )
        return SnapshotEnvelope(str(path), created, SNAPSHOT_FORMAT_VERSION, digest, state)

    def load(self, path: str | Path, *, verify_digest: bool = True) -> SnapshotEnvelope:
        source = Path(path)
        try:
            text = source.read_text(encoding="utf-8")
        except OSError as exc:
            raise MeshError(
                "E_SNAPSHOT",
                f"cannot read snapshot: {exc}",
                field="snapshot.file",
                location=str(source),
            ) from exc
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MeshError(
                "E_SNAPSHOT",
                f"snapshot is not valid JSON: {exc.msg}",
                field="snapshot.json",
                location=f"{source}:{exc.lineno}:{exc.colno}",
            ) from exc
        if not isinstance(raw, dict):
            raise MeshError(
                "E_SNAPSHOT",
                "snapshot envelope must be a JSON object",
                field="snapshot",
                location=str(source),
            )
        version = raw.get("format_version")
        if version != SNAPSHOT_FORMAT_VERSION:
            raise MeshError(
                "E_SNAPSHOT",
                f"snapshot format version {version!r} != {SNAPSHOT_FORMAT_VERSION}",
                field="snapshot.format_version",
                location=str(source),
            )
        payload = raw.get("payload")
        if not isinstance(payload, dict):
            raise MeshError(
                "E_SNAPSHOT",
                "snapshot payload must be an object",
                field="snapshot.payload",
                location=str(source),
            )
        stored_digest = raw.get("digest")
        actual_digest = snapshot_digest(payload)
        if verify_digest and stored_digest != actual_digest:
            raise MeshError(
                "E_SNAPSHOT",
                "snapshot content digest mismatch",
                field="snapshot.digest",
                location=str(source),
                details={"stored": stored_digest, "actual": actual_digest},
            )
        try:
            state = SnapshotState(
                route_revision=int(payload["route_revision"]),
                instance_version=str(payload["instance_version"]),
                budget_used_global=int(payload["budget_used_global"]),
                budget_used_by_service={
                    str(key): int(value)
                    for key, value in dict(payload["budget_used_by_service"]).items()
                },
                breakers={
                    str(key): dict(value)
                    for key, value in dict(payload["breakers"]).items()
                },
                completed_cursor=int(payload["completed_cursor"]),
                ledger_format_version=int(payload["ledger_format_version"]),
                ledger_digest=str(payload["ledger_digest"]),
                request_digest=str(payload.get("request_digest", "")),
            )
            created = int(raw.get("created_wall_ms", 0))
        except (KeyError, TypeError, ValueError) as exc:
            raise MeshError(
                "E_SNAPSHOT",
                f"snapshot payload field is invalid: {exc}",
                field="snapshot.payload",
                location=str(source),
            ) from exc
        return SnapshotEnvelope(str(source), created, int(version), actual_digest, state)

    def load_latest(self) -> SnapshotEnvelope | None:
        paths = self._paths()
        if not paths:
            return None
        return self.load(paths[-1])

    def load_latest_unverified(self) -> SnapshotEnvelope | None:
        """Parse the newest structurally valid envelope without trusting its digest.

        This is used only to compute a readable replacement diff. The result is
        never treated as authoritative state.
        """

        paths = self._paths()
        if not paths:
            return None
        try:
            return self.load(paths[-1], verify_digest=False)
        except MeshError:
            return None

    def list(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for path in self._paths():
            try:
                envelope = self.load(path)
                items.append(
                    {
                        "path": str(path),
                        "created_wall_ms": envelope.created_wall_ms,
                        "format_version": envelope.format_version,
                        "digest": envelope.digest,
                        "route_revision": envelope.state.route_revision,
                        "instance_version": envelope.state.instance_version,
                        "completed_cursor": envelope.state.completed_cursor,
                        "valid": True,
                    }
                )
            except MeshError as exc:
                items.append(
                    {
                        "path": str(path),
                        "valid": False,
                        "code": exc.code,
                        "message": exc.message,
                    }
                )
        return sorted(items, key=lambda item: (str(item["path"]),))

    def validate_against(
        self,
        envelope: SnapshotEnvelope,
        *,
        route_revision: int,
        instance_version: str,
        ledger_digest: str,
    ) -> None:
        if envelope.state.route_revision != route_revision:
            raise MeshError(
                "E_SNAPSHOT",
                f"snapshot route revision {envelope.state.route_revision} != active {route_revision}",
                field="snapshot.route_revision",
                location=envelope.path,
            )
        if envelope.state.instance_version != instance_version:
            raise MeshError(
                "E_SNAPSHOT",
                "snapshot instance-set version mismatch",
                field="snapshot.instance_version",
                location=envelope.path,
                details={
                    "snapshot": envelope.state.instance_version,
                    "active": instance_version,
                },
            )
        if envelope.state.ledger_digest != ledger_digest:
            raise MeshError(
                "E_SNAPSHOT",
                "snapshot ledger digest mismatch",
                field="snapshot.ledger_digest",
                location=envelope.path,
                details={
                    "snapshot": envelope.state.ledger_digest,
                    "ledger": ledger_digest,
                },
            )

    def replace_with(
        self,
        authoritative: SnapshotState,
        previous: SnapshotEnvelope | None = None,
    ) -> tuple[SnapshotEnvelope, dict[str, Any]]:
        diff = diff_snapshot_states(previous.state, authoritative) if previous else empty_diff()
        envelope = self.write(authoritative)
        if self._logger is not None:
            self._logger.warning(
                "snapshot_replaced",
                code="E_SNAPSHOT",
                reason="version_or_digest_mismatch",
                previous=previous.path if previous else None,
                replacement=envelope.path,
                diff=diff,
            )
        return envelope, diff


def empty_diff() -> dict[str, Any]:
    return {
        "added": [],
        "removed": [],
        "changed": [],
        "scalar_changed": [],
    }


def diff_snapshot_states(previous: SnapshotState, current: SnapshotState) -> dict[str, Any]:
    added: list[str] = []
    removed: list[str] = []
    changed: list[str] = []
    scalar_changed: list[str] = []
    scalars = [
        "route_revision",
        "instance_version",
        "budget_used_global",
        "completed_cursor",
        "ledger_format_version",
        "ledger_digest",
        "request_digest",
    ]
    previous_dict = asdict(previous)
    current_dict = asdict(current)
    for field in scalars:
        if previous_dict[field] != current_dict[field]:
            scalar_changed.append(field)
    for group in ("budget_used_by_service", "breakers"):
        old_map = getattr(previous, group)
        new_map = getattr(current, group)
        for key in sorted(set(old_map) | set(new_map)):
            label = f"{group}.{key}"
            if key not in old_map:
                added.append(label)
            elif key not in new_map:
                removed.append(label)
            elif canonical_json(old_map[key]) != canonical_json(new_map[key]):
                changed.append(label)
    return {
        "added": sorted(added),
        "removed": sorted(removed),
        "changed": sorted(changed),
        "scalar_changed": sorted(scalar_changed),
    }
