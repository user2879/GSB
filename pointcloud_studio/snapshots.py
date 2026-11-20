"""Immutable, atomically replaced point/mesh scene snapshots."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .engine import EngineSnapshotData, StudioEngine
from .errors import ErrorCode, StudioError
from .mesh import CombinedMesh, PatchMesh, ReconstructionParams
from .util import canonical_json, sha256_bytes


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class SceneSnapshot:
    version: int
    positions: np.ndarray
    normals: np.ndarray
    confidence: np.ndarray
    neighbor_counts: np.ndarray
    residuals: np.ndarray
    error_bounds: np.ndarray
    normal_radius: float
    params: ReconstructionParams
    patches: Mapping[str, PatchMesh]
    mesh: CombinedMesh
    ledger: Mapping[str, Any]
    point_content_sha256: str
    mesh_content_sha256: str
    ledger_sha256: str
    content_sha256: str
    operation_count: int

    @property
    def point_count(self) -> int:
        return int(self.positions.shape[0])


def scene_from_engine(engine: StudioEngine) -> SceneSnapshot:
    data = engine.snapshot_data()
    return make_scene_snapshot(data)


def make_scene_snapshot(data: EngineSnapshotData) -> SceneSnapshot:
    ledger_copy = _deep_freeze(dict(data.ledger))
    ledger_sha = str(ledger_copy.get("ledger_sha256"))  # type: ignore[union-attr]
    digest = hashlib.sha256()
    digest.update(b"PCSTSCENE1\0")
    digest.update(int(data.version).to_bytes(8, "little"))
    digest.update(bytes.fromhex(data.cloud_sha256))
    digest.update(bytes.fromhex(data.mesh.content_sha256))
    digest.update(ledger_sha.encode("ascii"))
    scene_hash = digest.hexdigest()
    return SceneSnapshot(
        version=int(data.version),
        positions=data.positions,
        normals=data.normals,
        confidence=data.confidence,
        neighbor_counts=data.neighbor_counts,
        residuals=data.residuals,
        error_bounds=data.error_bounds,
        normal_radius=float(data.normal_radius),
        params=data.params,
        patches=MappingProxyType(dict(data.patches)),
        mesh=data.mesh,
        ledger=ledger_copy,
        point_content_sha256=data.cloud_sha256,
        mesh_content_sha256=data.mesh.content_sha256,
        ledger_sha256=ledger_sha,
        content_sha256=scene_hash,
        operation_count=int(data.operation_count),
    )


class SnapshotStore:
    """Single atomic pointer; readers can never combine two versions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: SceneSnapshot | None = None
        self._publish_count = 0

    def publish(self, snapshot: SceneSnapshot) -> SceneSnapshot:
        if snapshot.positions.flags.writeable or snapshot.normals.flags.writeable:
            raise StudioError(
                ErrorCode.E_SNAPSHOT_STALE,
                "snapshot arrays must be immutable before publication",
                context={"version": snapshot.version},
            )
        with self._lock:
            if self._current is not None and snapshot.version <= self._current.version:
                raise StudioError(
                    ErrorCode.E_SNAPSHOT_STALE,
                    "snapshot version is not newer than the current version",
                    context={
                        "candidate_version": snapshot.version,
                        "current_version": self._current.version,
                    },
                )
            self._current = snapshot
            self._publish_count += 1
            return snapshot

    def get(self, expected_version: int | None = None) -> SceneSnapshot:
        with self._lock:
            current = self._current
        if current is None:
            raise StudioError(
                ErrorCode.E_SNAPSHOT_STALE,
                "no scene snapshot has been published",
                context={"expected_version": expected_version},
            )
        if expected_version is not None and current.version != int(expected_version):
            raise StudioError(
                ErrorCode.E_SNAPSHOT_STALE,
                "requested snapshot version is stale",
                context={
                    "expected_version": int(expected_version),
                    "current_version": current.version,
                    "current_content_sha256": current.content_sha256,
                },
            )
        return current

    @property
    def publish_count(self) -> int:
        with self._lock:
            return self._publish_count
