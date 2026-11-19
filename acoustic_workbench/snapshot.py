"""Immutable diagnostic snapshots, version stamps and atomic persistence."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .errors import ErrorCode, WorkbenchError
from .hashing import canonical_json, content_hash, write_bytes_atomic, write_text_atomic

SCHEMA_VERSION = "acoustic-workbench.snapshot/1.0"


def deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(deep_freeze(item) for item in value)
    return value


def deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [deep_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class Snapshot:
    schema_version: str
    revision: int
    content_hash: str
    version_stamp: str
    data: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(deep_thaw(self.data))

    def assert_valid(self) -> None:
        payload = self.to_dict()
        expected_hash = payload.get("content_hash")
        expected_stamp = payload.get("version_stamp")
        payload.pop("content_hash", None)
        payload.pop("version_stamp", None)
        actual_hash = content_hash(payload)
        expected_version = f"{SCHEMA_VERSION}#r{int(payload.get('revision', 0))}:{actual_hash[:16]}"
        if expected_hash != actual_hash or expected_stamp != expected_version:
            raise WorkbenchError(
                ErrorCode.SNAPSHOT_STALE,
                "snapshot content hash/version stamp does not match",
                stage="snapshot",
                details={
                    "expected_hash": expected_hash,
                    "actual_hash": actual_hash,
                    "expected_version": expected_stamp,
                    "actual_version": expected_version,
                },
            )


def build_snapshot(data: Mapping[str, Any], revision: int) -> Snapshot:
    payload = copy.deepcopy(dict(data))
    payload["schema_version"] = SCHEMA_VERSION
    payload["revision"] = int(revision)
    digest = content_hash(payload)
    stamp = f"{SCHEMA_VERSION}#r{int(revision)}:{digest[:16]}"
    payload["content_hash"] = digest
    payload["version_stamp"] = stamp
    frozen = deep_freeze(payload)
    snapshot = Snapshot(
        schema_version=SCHEMA_VERSION,
        revision=int(revision),
        content_hash=digest,
        version_stamp=stamp,
        data=frozen,
    )
    snapshot.assert_valid()
    return snapshot


def snapshot_from_dict(payload: Mapping[str, Any]) -> Snapshot:
    data = copy.deepcopy(dict(payload))
    schema_version = str(data.get("schema_version", ""))
    revision = int(data.get("revision", 0))
    digest = data.get("content_hash")
    stamp = data.get("version_stamp")
    if schema_version != SCHEMA_VERSION:
        raise WorkbenchError(
            ErrorCode.SNAPSHOT_STALE,
            "snapshot schema version is unsupported or stale",
            stage="snapshot",
            details={"expected": SCHEMA_VERSION, "actual": schema_version},
        )
    snapshot = Snapshot(
        schema_version=schema_version,
        revision=revision,
        content_hash=str(digest),
        version_stamp=str(stamp),
        data=deep_freeze(data),
    )
    snapshot.assert_valid()
    return snapshot


class SnapshotStore:
    """Atomic on-disk snapshot with explicit predecessor checks."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists()

    def load(self, *, expected_hash: str | None = None) -> Snapshot:
        if not self.path.exists():
            raise WorkbenchError(
                ErrorCode.SNAPSHOT_STALE,
                "snapshot file does not exist",
                stage="snapshot",
                details={"path": self.path.name},
            )
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            snapshot = snapshot_from_dict(payload)
        except WorkbenchError:
            raise
        except Exception as exc:
            raise WorkbenchError(
                ErrorCode.SNAPSHOT_STALE,
                "snapshot file is malformed",
                stage="snapshot",
                details={"error": str(exc)},
            ) from exc
        if expected_hash is not None and snapshot.content_hash != expected_hash:
            raise WorkbenchError(
                ErrorCode.SNAPSHOT_STALE,
                "snapshot changed relative to the expected predecessor hash",
                stage="snapshot",
                details={"expected": expected_hash, "actual": snapshot.content_hash},
            )
        return snapshot

    def commit(self, snapshot: Snapshot, *, expected_previous_hash: str | None = None) -> None:
        snapshot.assert_valid()
        if self.path.exists():
            current = self.load()
            if expected_previous_hash is None:
                raise WorkbenchError(
                    ErrorCode.SNAPSHOT_STALE,
                    "replacing an existing snapshot requires its predecessor hash",
                    stage="snapshot",
                    details={"actual_previous": current.content_hash},
                )
            if current.content_hash != expected_previous_hash:
                raise WorkbenchError(
                    ErrorCode.SNAPSHOT_STALE,
                    "refusing to replace a snapshot with an unexpected predecessor",
                    stage="snapshot",
                    details={
                        "expected_previous": expected_previous_hash,
                        "actual_previous": current.content_hash,
                    },
                )
        write_text_atomic(self.path, canonical_json(snapshot.to_dict(), pretty=True))

    def verify_version(self, expected_version_stamp: str) -> Snapshot:
        snapshot = self.load()
        if snapshot.version_stamp != expected_version_stamp:
            raise WorkbenchError(
                ErrorCode.SNAPSHOT_STALE,
                "snapshot version stamp differs from the expected version",
                stage="snapshot",
                details={"expected": expected_version_stamp, "actual": snapshot.version_stamp},
            )
        return snapshot


class ArtifactCache:
    """Persistent stage cache whose entries are rejected on hash mismatch."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._memory: dict[tuple[str, str], tuple[Any, str, str]] = {}

    def _path(self, stage: str, key: str) -> Path:
        return self.directory / f"{stage}-{key}.json"

    def get_or_compute(self, stage: str, key: str, compute) -> tuple[Any, str, str]:
        memory_key = (stage, key)
        if memory_key in self._memory:
            payload, digest, _ = self._memory[memory_key]
            actual = content_hash(payload)
            if actual != digest:
                raise WorkbenchError(
                    ErrorCode.SNAPSHOT_STALE,
                    "in-memory cached artifact hash no longer matches",
                    stage=stage,
                    details={"expected": digest, "actual": actual, "cache_key": key},
                )
            return payload, "memory", digest

        path = self._path(stage, key)
        if path.exists():
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise WorkbenchError(
                    ErrorCode.SNAPSHOT_STALE,
                    "cached artifact envelope is malformed",
                    stage=stage,
                    details={"error": str(exc), "cache_key": key},
                ) from exc
            import base64
            import pickle

            try:
                payload = pickle.loads(base64.b64decode(envelope["payload"]))
            except Exception as exc:
                raise WorkbenchError(
                    ErrorCode.SNAPSHOT_STALE,
                    "cached artifact payload cannot be decoded",
                    stage=stage,
                    details={"error": str(exc), "cache_key": key},
                ) from exc
            digest = content_hash(payload)
            if (
                envelope.get("schema") != "acoustic-workbench.cache/1.0"
                or envelope.get("stage") != stage
                or envelope.get("key") != key
                or envelope.get("content_hash") != digest
            ):
                raise WorkbenchError(
                    ErrorCode.SNAPSHOT_STALE,
                    "cached artifact key/hash mismatch",
                    stage=stage,
                    details={
                        "cache_key": key,
                        "expected_hash": envelope.get("content_hash"),
                        "actual_hash": digest,
                    },
                )
            self._memory[memory_key] = (payload, digest, "disk")
            return payload, "disk", digest

        payload = compute()
        digest = content_hash(payload)
        import base64
        import pickle

        envelope = {
            "schema": "acoustic-workbench.cache/1.0",
            "stage": stage,
            "key": key,
            "content_hash": digest,
            "payload": base64.b64encode(pickle.dumps(payload, protocol=5)).decode("ascii"),
        }
        write_text_atomic(path, canonical_json(envelope))
        self._memory[memory_key] = (payload, digest, "computed")
        return payload, "computed", digest
