"""Immutable render snapshots and stale-version protection."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .errors import E_SNAPSHOT_STALE, NPRError
from .geometry import FlatElement
from .model import Scene
from .util import canonical_json_bytes, sha256_bytes, sha256_json, sorted_hash_pairs

STYLE_MODES = ("outline", "sketch", "cartoon")


@dataclass(frozen=True, slots=True)
class ElementRaster:
    element_id: str
    element_hash: str
    geometry_hash: str
    render_key: str
    render_hash: str
    mode: str
    x: int
    y: int
    width: int
    height: int
    pixels: bytes

    def metadata(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "element_hash": self.element_hash,
            "geometry_hash": self.geometry_hash,
            "render_key": self.render_key,
            "render_hash": self.render_hash,
            "mode": self.mode,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "pixel_hash": sha256_bytes(self.pixels),
        }


def _freeze(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True, slots=True)
class RenderSnapshot:
    version: int
    mode: str
    viewport: tuple[int, int]
    scene_hash: str
    image: bytes
    layer_buffers: Mapping[str, bytes]
    layer_hashes: Mapping[str, str]
    element_rasters: Mapping[str, ElementRaster]
    element_hashes: Mapping[str, str]
    skeleton_hashes: Mapping[str, str]
    geometry_hashes: Mapping[str, str]
    flat_geometry: Mapping[str, FlatElement]
    strategy: Mapping[str, Any]
    snapshot_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_buffers", MappingProxyType(dict(self.layer_buffers)))
        object.__setattr__(self, "layer_hashes", MappingProxyType(dict(self.layer_hashes)))
        object.__setattr__(self, "element_rasters", MappingProxyType(dict(self.element_rasters)))
        object.__setattr__(self, "element_hashes", MappingProxyType(dict(self.element_hashes)))
        object.__setattr__(self, "skeleton_hashes", MappingProxyType(dict(self.skeleton_hashes)))
        object.__setattr__(self, "geometry_hashes", MappingProxyType(dict(self.geometry_hashes)))
        object.__setattr__(self, "flat_geometry", MappingProxyType(dict(self.flat_geometry)))
        object.__setattr__(self, "strategy", MappingProxyType(dict(self.strategy)))
        expected = self.compute_hash()
        if self.snapshot_hash and self.snapshot_hash != expected:
            raise NPRError(E_SNAPSHOT_STALE, "snapshot content hash does not match its metadata")
        object.__setattr__(self, "snapshot_hash", expected)

    @property
    def image_hash(self) -> str:
        return sha256_bytes(self.image)

    def metadata(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "mode": self.mode,
            "viewport": [int(self.viewport[0]), int(self.viewport[1])],
            "scene_hash": self.scene_hash,
            "image_hash": self.image_hash,
            "layer_hashes": sorted_hash_pairs(dict(self.layer_hashes)),
            "render_hashes": sorted_hash_pairs(dict(self.element_hashes)),
            "skeleton_hashes": sorted_hash_pairs(dict(self.skeleton_hashes)),
            "geometry_hashes": sorted_hash_pairs(dict(self.geometry_hashes)),
        }

    def compute_hash(self) -> str:
        return sha256_json(self.metadata())

    def verify(self) -> None:
        if self.compute_hash() != self.snapshot_hash:
            raise NPRError(E_SNAPSHOT_STALE, "snapshot hash verification failed", version=self.version)
        expected_size = self.viewport[0] * self.viewport[1] * 4
        if len(self.image) != expected_size:
            raise NPRError(E_SNAPSHOT_STALE, "snapshot image does not match the viewport")
        for layer_id, buffer in self.layer_buffers.items():
            if len(buffer) != expected_size:
                raise NPRError(E_SNAPSHOT_STALE, "layer buffer does not match the viewport", layer_id=layer_id)


@dataclass(frozen=True, slots=True)
class RenderState:
    """Immutable pair handed to a render/read thread."""

    scene: Scene
    snapshot: RenderSnapshot

    def __post_init__(self) -> None:
        if self.scene.content_hash != self.snapshot.scene_hash:
            raise NPRError(E_SNAPSHOT_STALE, "render state scene and snapshot do not match")

    @property
    def version(self) -> int:
        return self.snapshot.version



class SnapshotStore:
    """A compare-and-swap store for whole snapshots.

    The replacement is a single Python reference assignment, so readers either
    observe the old immutable snapshot or the new one, never a partial mix.
    """

    def __init__(self, initial: RenderSnapshot | None = None) -> None:
        self._lock = threading.Lock()
        self._current = initial
        if initial is not None:
            initial.verify()

    @property
    def current(self) -> RenderSnapshot | None:
        return self._current

    def initialize(self, snapshot: RenderSnapshot) -> RenderSnapshot:
        snapshot.verify()
        with self._lock:
            if self._current is not None:
                raise NPRError(E_SNAPSHOT_STALE, "snapshot store is already initialised")
            self._current = snapshot
        return snapshot

    def publish(self, snapshot: RenderSnapshot, expected_version: int | None = None) -> RenderSnapshot:
        snapshot.verify()
        with self._lock:
            current = self._current
            if current is not None:
                if expected_version is not None and current.version != expected_version:
                    raise NPRError(
                        E_SNAPSHOT_STALE,
                        "expected snapshot version is stale",
                        expected=expected_version,
                        current=current.version,
                    )
                if snapshot.version <= current.version:
                    raise NPRError(
                        E_SNAPSHOT_STALE,
                        "snapshot version is not newer than the current version",
                        incoming=snapshot.version,
                        current=current.version,
                    )
            elif expected_version is not None:
                raise NPRError(E_SNAPSHOT_STALE, "no snapshot exists for the expected version", expected=expected_version)
            self._current = snapshot
        return snapshot

    def read(self, expected_version: int | None = None, expected_hash: str | None = None) -> RenderSnapshot:
        snapshot = self._current
        if snapshot is None:
            raise NPRError(E_SNAPSHOT_STALE, "no render snapshot is available")
        if expected_version is not None and snapshot.version != expected_version:
            raise NPRError(
                E_SNAPSHOT_STALE,
                "requested snapshot version is stale",
                expected=expected_version,
                current=snapshot.version,
            )
        if expected_hash is not None and snapshot.snapshot_hash != expected_hash:
            raise NPRError(E_SNAPSHOT_STALE, "requested snapshot content hash is stale")
        return snapshot


def build_snapshot(
    *,
    version: int,
    mode: str,
    viewport: tuple[int, int],
    scene_hash: str,
    image: bytes,
    layer_buffers: Mapping[str, bytes],
    layer_hashes: Mapping[str, str],
    element_rasters: Mapping[str, ElementRaster],
    skeleton_hashes: Mapping[str, str],
    geometry_hashes: Mapping[str, str],
    flat_geometry: Mapping[str, FlatElement],
    strategy: Mapping[str, Any],
) -> RenderSnapshot:
    if mode not in STYLE_MODES:
        raise NPRError(E_SNAPSHOT_STALE, "unknown render style mode", mode=mode)
    render_hashes = {element_id: raster.render_hash for element_id, raster in element_rasters.items()}
    return RenderSnapshot(
        version=int(version),
        mode=mode,
        viewport=(int(viewport[0]), int(viewport[1])),
        scene_hash=scene_hash,
        image=bytes(image),
        layer_buffers=_freeze(layer_buffers),
        layer_hashes=_freeze(layer_hashes),
        element_rasters=_freeze(element_rasters),
        element_hashes=_freeze(render_hashes),
        skeleton_hashes=_freeze(skeleton_hashes),
        geometry_hashes=_freeze(geometry_hashes),
        flat_geometry=_freeze(flat_geometry),
        strategy=_freeze(strategy),
    )


def snapshot_bytes(snapshot: RenderSnapshot) -> bytes:
    """Canonical metadata block for ledgers and persistence manifests."""

    return canonical_json_bytes(snapshot.metadata())
