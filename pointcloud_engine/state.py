"""Block-level persistence, digest validation and versioned recovery."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

import numpy as np

from .errors import EngineError
from .io_utils import (
    atomic_write_json,
    canonical_json,
    ensure_dir,
    hash_array,
    hash_canonical,
    new_hasher,
)
from .render import PreparedPoints, render_tile
from .scene import Scene

TILE_PREFIX = "tile_"
TILE_SUFFIX = ".npz"
META_NAME = "version.json"
INTERRUPT_NAME = "interrupted.json"


def tile_filename(tile_index: int) -> str:
    return f"{TILE_PREFIX}{tile_index:05d}{TILE_SUFFIX}"


def _tile_input_digest(scene: Scene, prepared: PreparedPoints, tile_x0: int, tile_y0: int) -> str:
    """Digest of everything that can change this tile's rendered output.

    It captures the exact subset of points whose footprint overlaps the tile
    (in file order), their projection parameters and the tile geometry. A
    local point edit therefore invalidates *only* tiles covered by that
    point's footprint; untouched tiles keep an identical digest.
    """
    tw, th = scene.tile_width, scene.tile_height
    overlaps = (
        (prepared.max_x >= tile_x0)
        & (prepared.min_x < tile_x0 + tw)
        & (prepared.max_y >= tile_y0)
        & (prepared.min_y < tile_y0 + th)
    )
    point_indices = prepared.index[overlaps]
    world_all = scene.world_points()
    positions = world_all[point_indices]
    grays = scene.cloud.grays[point_indices]
    payload = {
        "point_indices": point_indices.tolist(),
        "positions": positions.tolist(),
        "grays": grays.tolist(),
        "center_x": prepared.center_x[overlaps].tolist(),
        "center_y": prepared.center_y[overlaps].tolist(),
        "radius_px": prepared.radius_px[overlaps].tolist(),
        "depth": prepared.depth[overlaps].tolist(),
        "origin": [tile_x0, tile_y0],
        "tile_size": [tw, th],
        "point_radius": scene.point_radius,
        "decay": {"kind": scene.decay.kind, "coefficient": scene.decay.coefficient},
        "grid": [scene.grid_width, scene.grid_height],
    }
    return hash_canonical(payload)


def block_digest(gray: np.ndarray, depth: np.ndarray, hit_count: int, input_digest: str) -> str:
    hasher = new_hasher()
    hasher.update(hash_array(gray).encode("ascii"))
    hasher.update(b"|")
    hasher.update(hash_array(depth).encode("ascii"))
    hasher.update(b"|")
    hasher.update(canonical_json({"hit_count": int(hit_count), "input": input_digest}))
    return hasher.hexdigest()


@dataclass
class TileRecord:
    tile_index: int
    version_id: str
    input_digest: str
    block_digest: str
    declared_hits: int
    shape: tuple[int, int]
    gray: np.ndarray
    depth: np.ndarray


class TileStore:
    def __init__(self, scene: Scene, version_id: str) -> None:
        self.scene = scene
        self.version_id = version_id
        self.state_dir = scene.state_dir
        ensure_dir(self.state_dir)
        self.archive_dir = os.path.join(self.state_dir, "archive", version_id[:12])

    def meta_path(self) -> str:
        return os.path.join(self.state_dir, META_NAME)

    def path_for(self, tile_index: int) -> str:
        return os.path.join(self.state_dir, tile_filename(tile_index))

    def write_meta(self) -> None:
        atomic_write_json(
            self.meta_path(),
            {"version_id": self.version_id, "tile_count": self.scene.tile_count},
        )

    def save_tile(self, tile_index: int, result: dict, input_digest: str) -> str:
        gray = np.ascontiguousarray(result["gray"], dtype=np.float32)
        depth = np.ascontiguousarray(result["depth"], dtype=np.float32)
        digest = block_digest(gray, depth, result["hit_count"], input_digest)
        target = self.path_for(tile_index)
        tmp = target + ".tmp.npz"
        ensure_dir(os.path.dirname(target))
        np.savez(
            tmp,
            gray=gray,
            depth=depth,
            version_id=np.asarray(self.version_id),
            input_digest=np.asarray(input_digest),
            block_digest=np.asarray(digest),
            declared_hits=np.asarray(result["hit_count"], dtype=np.int64),
            tile_index=np.asarray(tile_index, dtype=np.int64),
            shape=np.asarray(gray.shape, dtype=np.int64),
        )
        os.replace(tmp, target)
        return digest

    def load_tile(self, tile_index: int) -> TileRecord | None:
        path = self.path_for(tile_index)
        if not os.path.isfile(path):
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                record = TileRecord(
                    tile_index=int(data["tile_index"]),
                    version_id=str(data["version_id"]),
                    input_digest=str(data["input_digest"]),
                    block_digest=str(data["block_digest"]),
                    declared_hits=int(data["declared_hits"]),
                    shape=tuple(int(v) for v in data["shape"]),
                    gray=np.asarray(data["gray"], dtype=np.float32),
                    depth=np.asarray(data["depth"], dtype=np.float32),
                )
        except (OSError, KeyError, ValueError) as exc:
            raise EngineError("E_DIGEST", f"tile {tile_index} state is unreadable: {exc}",
                              tile_index=tile_index)
        return record

    def verify_record(self, record: TileRecord, expected_input_digest: str) -> tuple[bool, str]:
        if record.version_id != self.version_id:
            return False, "version mismatch"
        if record.input_digest != expected_input_digest:
            return False, "input digest mismatch"
        if record.shape != (self.scene.tile_height, self.scene.tile_width):
            return False, "tile shape mismatch"
        # Declared pixel count is cross-checked directly against the buffer,
        # independent of the stored block digest.
        actual_hits = int(np.count_nonzero(record.depth > 0.0))
        if actual_hits != record.declared_hits:
            return False, "declared pixel count mismatch"
        actual = block_digest(record.gray, record.depth, record.declared_hits, record.input_digest)
        if actual != record.block_digest:
            return False, "block digest mismatch"
        return True, "ok"

    def archive_stale_files(self, stale_paths: list[str]) -> list[str]:
        """Move old-version/current-version stale blocks into an archive tree."""
        if not stale_paths:
            return []
        ensure_dir(self.archive_dir)
        moved: list[str] = []
        for path in stale_paths:
            if not os.path.exists(path):
                continue
            destination = os.path.join(self.archive_dir, os.path.basename(path))
            counter = 1
            while os.path.exists(destination):
                destination = os.path.join(
                    self.archive_dir, f"{counter}_{os.path.basename(path)}"
                )
                counter += 1
            shutil.move(path, destination)
            moved.append(destination)
        return moved

    def list_tile_files(self) -> list[str]:
        if not os.path.isdir(self.state_dir):
            return []
        return sorted(
            os.path.join(self.state_dir, name)
            for name in os.listdir(self.state_dir)
            if name.startswith(TILE_PREFIX) and name.endswith(TILE_SUFFIX)
        )


def compute_input_digest(scene: Scene, prepared: PreparedPoints, tile_index: int) -> str:
    from .render import tile_origin

    x0, y0 = tile_origin(scene, tile_index)
    return _tile_input_digest(scene, prepared, x0, y0)


def render_and_persist(scene: Scene, prepared: PreparedPoints, store: TileStore, tile_index: int) -> dict:
    from .render import tile_origin, inject_runtime_error

    inject_runtime_error(scene, tile_index)
    x0, y0 = tile_origin(scene, tile_index)
    result = render_tile(scene, prepared, x0, y0)
    input_digest = _tile_input_digest(scene, prepared, x0, y0)
    store.save_tile(tile_index, result, input_digest)
    return result
