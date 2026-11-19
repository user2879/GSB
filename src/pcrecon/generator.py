"""Deterministic synthetic point-cloud generator.

A single :class:`numpy.random.Generator` seeded with
:data:`pcrecon.config.SEED` (PCG64) drives every random decision, in a fixed
call order.  Regenerating with the same configuration therefore produces
byte-identical binary files.
"""
from __future__ import annotations

import os

import numpy as np

from . import config
from .binary_format import write_array
from .hashing import atomic_write_bytes, canonical_json_bytes, blake2b_hex, write_canonical_json

# Surface id -> (name, area).  Floor is counted once (z=0).
SURFACES = (
    ("floor", config.ROOM * config.ROOM),
    ("ceiling", config.ROOM * config.ROOM),
    ("wall_x0", config.ROOM * config.ROOM),
    ("wall_x1", config.ROOM * config.ROOM),
    ("wall_y0", config.ROOM * config.ROOM),
    ("wall_y1", config.ROOM * config.ROOM),
)


def _largest_remainder_allocation(total: int, weights: np.ndarray) -> np.ndarray:
    raw = weights * total
    base = np.floor(raw).astype(np.int64)
    remainder = total - int(base.sum())
    order = np.argsort(-(raw - base), kind="stable")
    for idx in order[:remainder]:
        base[idx] += 1
    return base


def _surface_points(rng: np.random.Generator, name: str, count: int) -> np.ndarray:
    r = config.ROOM
    u = rng.random(count, dtype=np.float64) * r
    v = rng.random(count, dtype=np.float64) * r
    pts = np.empty((count, 3), dtype=np.float64)
    if name == "floor":
        pts[:, 0], pts[:, 1], pts[:, 2] = u, v, np.full(count, config.FLOOR_Z)
    elif name == "ceiling":
        pts[:, 0], pts[:, 1], pts[:, 2] = u, v, np.full(count, r)
    elif name == "wall_x0":
        pts[:, 0], pts[:, 1], pts[:, 2] = np.full(count, 0.0), u, v
    elif name == "wall_x1":
        pts[:, 0], pts[:, 1], pts[:, 2] = np.full(count, r), u, v
    elif name == "wall_y0":
        pts[:, 0], pts[:, 1], pts[:, 2] = u, np.full(count, 0.0), v
    else:  # wall_y1
        pts[:, 0], pts[:, 1], pts[:, 2] = u, np.full(count, r), v
    return pts


def _surface_normals(name: str, count: int) -> np.ndarray:
    n = np.zeros((count, 3), dtype=np.float64)
    table = {
        "floor": (0.0, 0.0, 1.0),
        "ceiling": (0.0, 0.0, -1.0),
        "wall_x0": (1.0, 0.0, 0.0),
        "wall_x1": (-1.0, 0.0, 0.0),
        "wall_y0": (0.0, 1.0, 0.0),
        "wall_y1": (0.0, -1.0, 0.0),
    }
    n[:] = table[name]
    return n


def _camera_trajectory() -> list[dict]:
    center = config.ROOM / 2.0
    poses = []
    n = 24
    radius = config.ROOM * 0.42
    for i in range(n):
        angle = 2.0 * np.pi * i / n
        x = center + radius * np.cos(angle)
        y = center + radius * np.sin(angle)
        z = config.ROOM * (0.35 + 0.25 * (np.sin(angle) + 1.0) / 2.0)
        eye = [round(float(v), 9) for v in (x, y, z)]
        target = [center, center, config.ROOM * 0.45]
        up = [0.0, 0.0, 1.0]
        poses.append({"id": i, "eye": eye, "target": target, "up": up})
    return poses


def _edit_sequence() -> list[dict]:
    r = config.ROOM
    return [
        {
            "op_id": 0,
            "type": "cull_region",
            "region_min": [r * 0.2, r * 0.2, r * 0.0],
            "region_max": [r * 0.5, r * 0.5, r * 0.35],
        },
        {
            "op_id": 1,
            "type": "reestimate_normals",
            "region_min": [r * 0.1, r * 0.1, r * 0.0],
            "region_max": [r * 0.6, r * 0.6, r * 0.5],
            "radius": 0.45,
        },
        {
            "op_id": 2,
            "type": "cull_region",
            "region_min": [r * 0.55, r * 0.55, r * 0.0],
            "region_max": [r * 0.85, r * 0.85, r * 0.4],
        },
        {
            "op_id": 3,
            "type": "rebuild_params",
            "support_radius": 0.95,
            "mesh_grid_step": 0.2,
        },
    ]


def generate_sample(out_dir: str, *, seed: int = config.SEED) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    names = [s[0] for s in SURFACES]
    weights = np.array([s[1] for s in SURFACES], dtype=np.float64)
    weights /= weights.sum()
    counts = _largest_remainder_allocation(config.TOTAL_POINTS, weights)

    points_parts: list[np.ndarray] = []
    normal_parts: list[np.ndarray] = []
    surface_ids_parts: list[np.ndarray] = []
    for name, count in zip(names, counts):
        clean = _surface_points(rng, name, int(count))
        ideal_n = _surface_normals(name, int(count))
        tangential = rng.normal(0.0, config.POINT_JITTER, size=clean.shape)
        # Remove the normal component of tangential noise so depth noise stays separate.
        tangential -= tangential * ideal_n * ideal_n
        depth = rng.normal(0.0, config.DEPTH_NOISE, size=(int(count), 1))
        noisy = clean + tangential + depth * ideal_n
        # Confidence depends deterministically on local depth-noise magnitude.
        points_parts.append(noisy)
        normal_parts.append(ideal_n)
        surface_ids_parts.append(np.full(int(count), names.index(name), dtype=np.int32))

    points = np.ascontiguousarray(np.concatenate(points_parts, axis=0), dtype=np.float64)
    normals_xyz = np.ascontiguousarray(np.concatenate(normal_parts, axis=0), dtype=np.float64)
    surface_ids = np.concatenate(surface_ids_parts, axis=0)

    # Deterministic per-point confidence from a dedicated, fixed-order RNG stream.
    noise_mag = np.abs(rng.normal(0.0, config.DEPTH_NOISE, size=points.shape[0]))
    confidence = np.clip(1.0 - noise_mag / (config.DEPTH_NOISE * 4.0), 0.0, 1.0)
    normals = np.column_stack([normals_xyz, confidence]).astype(np.float64, copy=False)

    # Stable ordering: sort by (surface_id, x, y, z) so file layout is canonical.
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0], surface_ids))
    points = np.ascontiguousarray(points[order])
    normals = np.ascontiguousarray(normals[order])
    surface_ids = np.ascontiguousarray(surface_ids[order])

    points_path = os.path.join(out_dir, config.POINTS_FILE)
    normals_path = os.path.join(out_dir, config.NORMALS_FILE)
    points_meta = write_array(points_path, points)
    normals_meta = write_array(normals_path, normals)
    points_meta["path"] = config.POINTS_FILE
    normals_meta["path"] = config.NORMALS_FILE

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    hist, edges = np.histogram(confidence, bins=np.linspace(0.0, 1.0, 11))
    stats = {
        "point_total": int(points.shape[0]),
        "surface_counts": {name: int(cnt) for name, cnt in zip(names, counts)},
        "bbox_min": [float(v) for v in bbox_min],
        "bbox_max": [float(v) for v in bbox_max],
        "confidence_distribution": {
            "bins_edges": [float(v) for v in edges],
            "counts": [int(v) for v in hist],
            "mean": float(confidence.mean()),
            "min": float(confidence.min()),
            "max": float(confidence.max()),
        },
    }

    manifest = {
        "format": "pcrecon-sample",
        "format_version": 1,
        "seed": seed,
        "rng": "numpy.PCG64",
        "config": {
            "room": config.ROOM,
            "point_jitter": config.POINT_JITTER,
            "depth_noise": config.DEPTH_NOISE,
            "confidence_k": config.CONFIDENCE_K,
            "total_points": config.TOTAL_POINTS,
        },
        "points": points_meta,
        "normals": normals_meta,
        "stats": stats,
    }
    manifest_path = os.path.join(out_dir, config.MANIFEST_FILE)
    manifest_hash = write_canonical_json(manifest_path, manifest)

    camera_path = os.path.join(out_dir, config.CAMERA_FILE)
    camera = {"format": "pcrecon-cameras", "format_version": 1, "poses": _camera_trajectory()}
    camera_hash = write_canonical_json(camera_path, camera)

    edits_path = os.path.join(out_dir, config.EDITS_FILE)
    edits = {"format": "pcrecon-edits", "format_version": 1, "edits": _edit_sequence()}
    edits_hash = write_canonical_json(edits_path, edits)

    return {
        "out_dir": out_dir,
        "manifest": manifest,
        "files": {
            config.POINTS_FILE: points_meta["blake2b256"],
            config.NORMALS_FILE: normals_meta["blake2b256"],
            config.MANIFEST_FILE: manifest_hash,
            config.CAMERA_FILE: camera_hash,
            config.EDITS_FILE: edits_hash,
        },
    }
