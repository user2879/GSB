"""Reconstruction service: edits, incremental rebuilds and snapshot publishing."""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from . import config
from .consistency import compare_meshes
from .errors import (
    NeighborRadiusError,
    PatchDependencyError,
    SnapshotStaleError,
)
from .hashing import digest_array
from .mesh import (
    ReconstructionParams,
    all_tiles,
    build_node_fields,
    build_tile,
    classify_surfaces,
    full_rebuild,
    merge_tiles,
    tiles_touching_points,
)
from .normals import affected_neighbourhood, reestimate_local
from .spatial import VoxelGrid
from .state import Snapshot


def tile_hash(tile: dict) -> str:
    return digest_array(tile["vertices"]) + ":" + digest_array(tile["faces"])


@dataclass
class EditResult:
    snapshot: Snapshot
    invalidated: tuple
    rebuilt: tuple
    reused: tuple
    consistency: dict | None
    audit: dict


class ReconstructionService:
    """Owns the committed snapshot and serializes edits on a worker lock."""

    def __init__(self, initial: Snapshot):
        self._lock = threading.RLock()
        self._snapshot = initial
        self._fields_cache = None

    @property
    def snapshot(self) -> Snapshot:
        with self._lock:
            return self._snapshot

    def apply_edit(self, edit: dict, *, base_version: int, verify: bool = True) -> EditResult:
        with self._lock:
            if base_version != self._snapshot.version:
                raise SnapshotStaleError(
                    f"edit targets v{base_version} but head is v{self._snapshot.version}",
                    context={"base_version": base_version, "head_version": self._snapshot.version},
                )
            etype = edit["type"]
            current = self._snapshot
            if etype == "cull_region":
                new_points, new_normals, culled = self._apply_cull(current, edit)
                snapshot, invalidated, rebuilt, audit = self._incremental_rebuild(
                    current, new_points, new_normals, current.params, edit, culled
                )
            elif etype == "reestimate_normals":
                new_normals, audit = self._apply_normals(current, edit)
                snapshot = self._republish(current, current.points, new_normals, current.params, edit, audit)
                invalidated, rebuilt = (), ()
            elif etype == "rebuild_params":
                params = ReconstructionParams(
                    voxel_size=current.params.voxel_size,
                    support_radius=float(edit["support_radius"]),
                    mesh_grid_step=float(edit["mesh_grid_step"]),
                )
                snapshot, invalidated, rebuilt, audit = self._full_params_rebuild(current, params, edit)
            else:
                raise NeighborRadiusError(f"unknown edit type {etype}")

            consistency = None
            if verify:
                consistency = self._verify(snapshot, edit.get("op_id"))
            result = EditResult(
                snapshot=snapshot,
                invalidated=tuple(sorted(invalidated)),
                rebuilt=tuple(sorted(rebuilt)),
                reused=tuple(sorted(set(snapshot.tiles) - set(rebuilt))),
                consistency=consistency,
                audit=audit,
            )
            self._snapshot = snapshot
            return result

    # ------------------------------------------------------------------ edits
    def _apply_cull(self, snap: Snapshot, edit: dict):
        lo = np.asarray(edit["region_min"], dtype=np.float64)
        hi = np.asarray(edit["region_max"], dtype=np.float64)
        if np.any(hi < lo):
            raise NeighborRadiusError("cull region_max < region_min")
        mask = np.any((snap.points < lo) | (snap.points > hi), axis=1)
        kept = np.flatnonzero(mask)
        culled_local = np.flatnonzero(~mask)
        audit = {
            "op_id": edit.get("op_id"),
            "type": "cull_region",
            "culled_count": int(culled_local.size),
            "remaining_count": int(kept.size),
        }
        return (
            np.ascontiguousarray(snap.points[kept]),
            np.ascontiguousarray(snap.normals[kept]),
            culled_local,
        )

    def _apply_normals(self, snap: Snapshot, edit: dict):
        radius = float(edit["radius"])
        if radius <= 0.0 or not np.isfinite(radius):
            raise NeighborRadiusError(f"invalid radius {radius}")
        lo = np.asarray(edit["region_min"], dtype=np.float64)
        hi = np.asarray(edit["region_max"], dtype=np.float64)
        region_idx = VoxelGrid(snap.points, max(radius, 1e-6)).points_within_region(lo, hi)
        grid = VoxelGrid(snap.points, max(radius, 1e-6))
        targets = affected_neighbourhood(grid, snap.points[region_idx], radius)
        new_normals, counts, audit = reestimate_local(snap.points, snap.normals, targets, radius)
        audit.update({"op_id": edit.get("op_id"), "type": "reestimate_normals"})
        return new_normals, audit

    def _full_params_rebuild(self, snap: Snapshot, params: ReconstructionParams, edit: dict):
        result = full_rebuild(snap.points, params)
        tiles = result["tiles"]
        hashes = {tid: tile_hash(tile) for tid, tile in tiles.items()}
        merged = merge_tiles(tiles, params)
        new_snap = Snapshot(
            version=snap.version + 1,
            points=snap.points,
            normals=snap.normals,
            params=params,
            tiles=tiles,
            merged=merged,
            tile_hashes=hashes,
            op_history=snap.op_history + (edit,),
            edit_audit=snap.edit_audit + (
                {"op_id": edit.get("op_id"), "type": "rebuild_params", **params.to_dict()},
            ),
        )
        invalidated = tuple(sorted(all_tiles(snap.params)))
        return new_snap, invalidated, tuple(sorted(all_tiles(params))), {
            "op_id": edit.get("op_id"),
            "type": "rebuild_params",
        }

    # ------------------------------------------------------------- rebuilding
    def _incremental_rebuild(self, snap, points, normals, params, edit, culled_local):
        closure = tiles_touching_points(snap.points, culled_local, params)
        if not closure:
            audit = {
                "op_id": edit.get("op_id"),
                "type": edit["type"],
                "invalidated": [],
                "rebuilt": [],
                "note": "no tile depends on the edited points",
            }
            new_snap = Snapshot(
                version=snap.version + 1,
                points=points,
                normals=normals,
                params=params,
                tiles=snap.tiles,
                merged=snap.merged,
                tile_hashes=snap.tile_hashes,
                op_history=snap.op_history + (edit,),
                edit_audit=snap.edit_audit + (audit,),
            )
            return new_snap, closure, set(), audit

        surface_ids = classify_surfaces(points)
        fields = build_node_fields(points, surface_ids, params)
        new_tiles = dict(snap.tiles)
        rebuilt: set = set()
        for tile_id in sorted(closure):
            new_tiles[tile_id] = build_tile(tile_id, fields, params)
            rebuilt.add(tile_id)
        hashes = {tid: tile_hash(tile) for tid, tile in new_tiles.items()}
        merged = merge_tiles(new_tiles, params)
        audit = {
            "op_id": edit.get("op_id"),
            "type": edit["type"],
            "invalidated": [list(t) for t in sorted(closure)],
            "rebuilt": [list(t) for t in sorted(rebuilt)],
            "reused_count": len(new_tiles) - len(rebuilt),
        }
        new_snap = Snapshot(
            version=snap.version + 1,
            points=points,
            normals=normals,
            params=params,
            tiles=new_tiles,
            merged=merged,
            tile_hashes=hashes,
            op_history=snap.op_history + (edit,),
            edit_audit=snap.edit_audit + (audit,),
        )
        return new_snap, closure, rebuilt, audit

    def _republish(self, snap, points, normals, params, edit, audit):
        return Snapshot(
            version=snap.version + 1,
            points=points,
            normals=normals,
            params=params,
            tiles=snap.tiles,
            merged=snap.merged,
            tile_hashes=snap.tile_hashes,
            op_history=snap.op_history + (edit,),
            edit_audit=snap.edit_audit + (audit,),
        )

    def _verify(self, snapshot: Snapshot, op_id: int | None) -> dict:
        full = full_rebuild(snapshot.points, snapshot.params)
        return compare_meshes(
            snapshot.merged,
            full["merged"],
            op_id=op_id,
        )


def build_initial_snapshot(points: np.ndarray, normals: np.ndarray) -> Snapshot:
    params = ReconstructionParams()
    result = full_rebuild(points, params)
    tiles = result["tiles"]
    hashes = {tid: tile_hash(tile) for tid, tile in tiles.items()}
    return Snapshot(
        version=0,
        points=np.ascontiguousarray(points),
        normals=np.ascontiguousarray(normals),
        params=params,
        tiles=tiles,
        merged=result["merged"],
        tile_hashes=hashes,
        op_history=(),
        edit_audit=(),
    )
