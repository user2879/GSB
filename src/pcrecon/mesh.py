"""Deterministic tiled surface reconstruction.

The room consists of six axis-aligned planar surfaces (floor, ceiling, four
walls).  Each surface owns a *global* 2-D node grid; every live point is
assigned to its nearest node, and a node value is the mean signed offset of
its assigned points (NaN when empty).  Tiles (patches) are fixed-size windows
over that global grid, so:

* a tile's vertices depend only on the points assigned to the tile and its
  eight neighbours (the declared support halo) — the same point set is seen by
  a full rebuild and an incremental rebuild;
* untouched tiles therefore reproduce byte-identical vertices;
* the invalidation closure (edited tiles + edge-sharing neighbours) equals the
  set of tiles actually recomputed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .errors import PatchDependencyError

SURFACE_TABLE = (
    # name, fixed axis, fixed value, inward normal, 2-D axis pair
    ("floor", 2, 0.0, np.array([0.0, 0.0, 1.0]), (0, 1)),
    ("ceiling", 2, 10.0, np.array([0.0, 0.0, -1.0]), (0, 1)),
    ("wall_x0", 0, 0.0, np.array([1.0, 0.0, 0.0]), (1, 2)),
    ("wall_x1", 0, 10.0, np.array([-1.0, 0.0, 0.0]), (1, 2)),
    ("wall_y0", 1, 0.0, np.array([0.0, 1.0, 0.0]), (0, 2)),
    ("wall_y1", 1, 10.0, np.array([0.0, -1.0, 0.0]), (0, 2)),
)


@dataclass(frozen=True)
class ReconstructionParams:
    voxel_size: float = config.VOXEL_SIZE
    support_radius: float = config.SUPPORT_RADIUS
    mesh_grid_step: float = config.MESH_GRID_STEP

    def to_dict(self) -> dict:
        return {
            "voxel_size": self.voxel_size,
            "support_radius": self.support_radius,
            "mesh_grid_step": self.mesh_grid_step,
        }


def classify_surfaces(points: np.ndarray) -> np.ndarray:
    """Assign every point to its nearest defining plane (tie -> lower id)."""
    distances = np.empty((points.shape[0], len(SURFACE_TABLE)), dtype=np.float64)
    for sid, (_, axis, value, _, _) in enumerate(SURFACE_TABLE):
        distances[:, sid] = np.abs(points[:, axis] - value)
    return np.argmin(distances, axis=1).astype(np.int32)


def _nodes_per_tile(step: float, voxel: float) -> int:
    n = round(voxel / step)
    if abs(n * step - voxel) > 1e-9:
        raise PatchDependencyError(
            f"mesh_grid_step {step} must subdivide voxel_size {voxel} evenly"
        )
    return int(n)


def _global_node_grid(sid: int, step: float) -> np.ndarray:
    room = config.ROOM
    count = int(round(room / step)) + 1
    return np.linspace(0.0, room, count)


def build_node_fields(
    points: np.ndarray,
    surface_ids: np.ndarray,
    params: ReconstructionParams,
) -> list[dict]:
    """Global deterministic node value fields for all six surfaces."""
    fields: list[dict] = []
    for sid, (name, axis, value, normal, axes) in enumerate(SURFACE_TABLE):
        coords = _global_node_grid(sid, params.mesh_grid_step)
        n_nodes = coords.shape[0]
        sums = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        counts = np.zeros((n_nodes, n_nodes), dtype=np.int64)
        sel = np.flatnonzero(surface_ids == sid)
        if sel.size:
            pts = points[sel]
            u = pts[:, axes[0]]
            v = pts[:, axes[1]]
            iu = np.clip(np.floor(u / params.mesh_grid_step + 0.5).astype(np.int64), 0, n_nodes - 1)
            iv = np.clip(np.floor(v / params.mesh_grid_step + 0.5).astype(np.int64), 0, n_nodes - 1)
            offsets = (pts - np.array([value if a == axis else 0.0 for a in range(3)])) @ normal
            np.add.at(sums, (iu, iv), offsets)
            np.add.at(counts, (iu, iv), 1)
        with np.errstate(invalid="ignore"):
            values = sums / counts
        fields.append(
            {
                "surface_id": sid,
                "name": name,
                "coords": coords,
                "values": values,
                "counts": counts,
            }
        )
    return fields


def _tile_list(params: ReconstructionParams) -> list[tuple[int, int, int]]:
    tiles: list[tuple[int, int, int]] = []
    tiles_per_surface = int(round(config.ROOM / params.voxel_size))
    for sid in range(len(SURFACE_TABLE)):
        for ti in range(tiles_per_surface):
            for tj in range(tiles_per_surface):
                tiles.append((sid, ti, tj))
    return tiles


def build_tile(
    tile_id: tuple[int, int, int],
    fields: list[dict],
    params: ReconstructionParams,
) -> dict:
    """Build one patch from the global node fields (deterministic)."""
    sid, ti, tj = tile_id
    field = fields[sid]
    name, axis, value, normal, axes = SURFACE_TABLE[sid]
    sub = _nodes_per_tile(params.mesh_grid_step, params.voxel_size)
    base_i = ti * sub
    base_j = tj * sub
    coords = field["coords"]
    values = field["values"]

    nodes: list[tuple[int, int, int]] = []
    node_index: dict[tuple[int, int], int] = {}
    for a in range(sub + 1):
        for b in range(sub + 1):
            gi, gj = base_i + a, base_j + b
            if np.isfinite(values[gi, gj]):
                node_index[(a, b)] = len(nodes)
                nodes.append((gi, gj, len(nodes)))

    vertices = np.zeros((len(nodes), 3), dtype=np.float64)
    for a, b, local in nodes:
        gi, gj, _ = (a, b, local)
        u = coords[gi]
        v = coords[gj]
        offset = float(values[gi, gj])
        xyz = np.zeros(3, dtype=np.float64)
        xyz[axes[0]] = u
        xyz[axes[1]] = v
        xyz[axis] = value
        xyz = xyz + normal * offset
        vertices[local] = xyz

    faces: list[tuple[int, int, int]] = []
    for a in range(sub):
        for b in range(sub):
            corners = ((a, b), (a + 1, b), (a + 1, b + 1), (a, b + 1))
            present = [c for c in corners if c in node_index]
            if len(present) == 4:
                ids = [node_index[c] for c in corners]
                # Fixed winding; flip when the surface normal basis is mirrored.
                faces.append((ids[0], ids[1], ids[2]))
                faces.append((ids[0], ids[2], ids[3]))
    face_arr = np.array(faces, dtype=np.int64) if faces else np.zeros((0, 3), dtype=np.int64)
    return {
        "tile_id": list(tile_id),
        "surface": name,
        "vertices": np.ascontiguousarray(vertices),
        "faces": np.ascontiguousarray(face_arr),
    }


def full_rebuild(points: np.ndarray, params: ReconstructionParams) -> dict:
    surface_ids = classify_surfaces(points)
    fields = build_node_fields(points, surface_ids, params)
    tiles = {}
    for tile_id in _tile_list(params):
        tiles[tile_id] = build_tile(tile_id, fields, params)
    merged = merge_tiles(tiles, params)
    return {"fields": fields, "tiles": tiles, "merged": merged, "params": params}


def merge_tiles(tiles: dict, params: ReconstructionParams) -> dict:
    """Concatenate tiles in canonical tile order into one mesh."""
    vert_parts: list[np.ndarray] = []
    face_parts: list[np.ndarray] = []
    tile_ranges = {}
    offset = 0
    for tile_id in _tile_list(params):
        tile = tiles[tile_id]
        start = offset
        vert_parts.append(tile["vertices"])
        if tile["faces"].shape[0]:
            face_parts.append(tile["faces"] + offset)
        offset += tile["vertices"].shape[0]
        tile_ranges["|".join(map(str, tile_id))] = [start, offset]
    vertices = (
        np.ascontiguousarray(np.concatenate(vert_parts, axis=0))
        if vert_parts
        else np.zeros((0, 3), dtype=np.float64)
    )
    faces = (
        np.ascontiguousarray(np.concatenate(face_parts, axis=0))
        if face_parts
        else np.zeros((0, 3), dtype=np.int64)
    )
    return {"vertices": vertices, "faces": faces, "tile_ranges": tile_ranges}


def tiles_touching_points(
    points: np.ndarray,
    edited_local: np.ndarray,
    params: ReconstructionParams,
) -> set[tuple[int, int, int]]:
    """Tiles whose support halo contains any edited point, plus neighbours."""
    if edited_local.shape[0] == 0:
        return set()
    surface_ids = classify_surfaces(points)
    tiles_per_surface = int(round(config.ROOM / params.voxel_size))
    halo = int(np.ceil(params.support_radius / params.voxel_size))
    closure: set[tuple[int, int, int]] = set()
    edited_sel = surface_ids[edited_local]
    for sid, (_, axis, _, _, axes) in enumerate(SURFACE_TABLE):
        rows = np.flatnonzero(edited_sel == sid)
        if rows.size == 0:
            continue
        pts = points[edited_local[rows]]
        u = pts[:, axes[0]]
        v = pts[:, axes[1]]
        iu = np.floor(u / params.voxel_size).astype(np.int64)
        iv = np.floor(v / params.voxel_size).astype(np.int64)
        for base in np.unique(np.column_stack((iu, iv)), axis=0):
            for di in range(-halo, halo + 1):
                for dj in range(-halo, halo + 1):
                    ti, tj = int(base[0]) + di, int(base[1]) + dj
                    if 0 <= ti < tiles_per_surface and 0 <= tj < tiles_per_surface:
                        closure.add((sid, ti, tj))
    return closure


def all_tiles(params: ReconstructionParams) -> set[tuple[int, int, int]]:
    return set(_tile_list(params))
