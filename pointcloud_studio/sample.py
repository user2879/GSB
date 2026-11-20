"""Deterministic synthetic room/ground point-cloud bundle."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .formats import encode_normals, encode_points, load_normals, load_points
from .util import canonical_json, json_bytes, point_quantization_bound, read_json, sha256_bytes


@dataclass(frozen=True)
class SampleConfig:
    seed: int = 20261120
    point_count: int = 120_000
    room_points: int = 72_000
    ground_points: int = 48_000
    room_min: tuple[float, float, float] = (-4.0, 0.0, -4.0)
    room_max: tuple[float, float, float] = (4.0, 3.2, 4.0)
    ground_y: float = -0.6
    ground_extent: float = 7.5
    surface_jitter: float = 0.004
    normal_jitter: float = 0.0015
    row_columns: int = 110
    patch_size: float = 0.8
    triangulation_cell_size: float = 0.20
    layout_origin: tuple[float, float, float] = (-8.0, -2.0, -8.0)

    def validate(self) -> None:
        if self.point_count != self.room_points + self.ground_points:
            raise ValueError("point_count must equal room_points + ground_points")
        if self.point_count != 120_000:
            raise ValueError("the acceptance sample must contain exactly 120000 points")
        if self.room_points % 6:
            raise ValueError("room_points must be divisible by six faces")
        if self.ground_points <= 0:
            raise ValueError("ground_points must be positive")

    def normalized(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(lengths, np.finfo(np.float64).tiny)


def _face_points(
    rng: np.random.Generator,
    count: int,
    *,
    axis: int,
    fixed: float,
    u_axis: int,
    u_range: tuple[float, float],
    v_axis: int,
    v_range: tuple[float, float],
    normal: tuple[float, float, float],
    jitter: float,
    normal_jitter: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.empty((count, 3), dtype=np.float64)
    points[:, axis] = fixed
    points[:, u_axis] = rng.uniform(u_range[0], u_range[1], size=count)
    points[:, v_axis] = rng.uniform(v_range[0], v_range[1], size=count)
    points += rng.uniform(-jitter, jitter, size=(count, 3))

    normals = np.tile(np.asarray(normal, dtype=np.float64), (count, 1))
    normals += rng.normal(0.0, normal_jitter, size=(count, 3))
    normals = _normalize_rows(normals)

    edge_u = np.minimum(points[:, u_axis] - u_range[0], u_range[1] - points[:, u_axis])
    edge_v = np.minimum(points[:, v_axis] - v_range[0], v_range[1] - points[:, v_axis])
    edge_distance = np.maximum(0.0, np.minimum(edge_u, edge_v))
    return points, normals, edge_distance


def generate_cloud(config: SampleConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate positions, normals, confidence and class ids deterministically."""

    config.validate()
    rng = np.random.Generator(np.random.PCG64(config.seed))
    room_min = np.asarray(config.room_min, dtype=np.float64)
    room_max = np.asarray(config.room_max, dtype=np.float64)
    spans = room_max - room_min
    per_face = config.room_points // 6

    face_specs = [
        # axis, fixed, u-axis, u-range, v-axis, v-range, inward normal
        (0, room_min[0], 1, (room_min[1], room_max[1]), 2, (room_min[2], room_max[2]), (1.0, 0.0, 0.0)),
        (0, room_max[0], 1, (room_min[1], room_max[1]), 2, (room_min[2], room_max[2]), (-1.0, 0.0, 0.0)),
        (1, room_min[1], 0, (room_min[0], room_max[0]), 2, (room_min[2], room_max[2]), (0.0, 1.0, 0.0)),
        (1, room_max[1], 0, (room_min[0], room_max[0]), 2, (room_min[2], room_max[2]), (0.0, -1.0, 0.0)),
        (2, room_min[2], 0, (room_min[0], room_max[0]), 1, (room_min[1], room_max[1]), (0.0, 0.0, 1.0)),
        (2, room_max[2], 0, (room_min[0], room_max[0]), 1, (room_min[1], room_max[1]), (0.0, 0.0, -1.0)),
    ]

    rooms: list[np.ndarray] = []
    room_normals: list[np.ndarray] = []
    room_edges: list[np.ndarray] = []
    for spec in face_specs:
        points, normals, edges = _face_points(
            rng,
            per_face,
            axis=spec[0],
            fixed=spec[1],
            u_axis=spec[2],
            u_range=spec[3],
            v_axis=spec[4],
            v_range=spec[5],
            normal=spec[6],
            jitter=config.surface_jitter,
            normal_jitter=config.normal_jitter,
        )
        rooms.append(points)
        room_normals.append(normals)
        room_edges.append(edges)

    room_points = np.concatenate(rooms, axis=0)
    room_normals = np.concatenate(room_normals, axis=0)
    room_edge_distance = np.concatenate(room_edges, axis=0)

    ground = np.empty((config.ground_points, 3), dtype=np.float64)
    ground[:, 0] = rng.uniform(-config.ground_extent, config.ground_extent, config.ground_points)
    ground[:, 1] = config.ground_y
    ground[:, 2] = rng.uniform(-config.ground_extent, config.ground_extent, config.ground_points)
    ground += rng.uniform(-config.surface_jitter, config.surface_jitter, size=ground.shape)
    ground_normals = np.tile(np.asarray((0.0, 1.0, 0.0)), (config.ground_points, 1))
    ground_normals += rng.normal(0.0, config.normal_jitter, size=ground_normals.shape)
    ground_normals = _normalize_rows(ground_normals)
    ground_edge = np.minimum(
        config.ground_extent - np.abs(ground[:, 0]),
        config.ground_extent - np.abs(ground[:, 2]),
    )

    positions = np.concatenate((room_points, ground), axis=0)
    normals = np.concatenate((room_normals, ground_normals), axis=0)
    edge_distance = np.concatenate((room_edge_distance, ground_edge), axis=0)
    class_ids = np.concatenate(
        (np.zeros(config.room_points, dtype=np.int32), np.ones(config.ground_points, dtype=np.int32))
    )

    confidence_noise = rng.uniform(0.0, 0.012, size=config.point_count)
    confidence = 0.69 + 0.27 * np.exp(-np.maximum(edge_distance, 0.0) / 0.42) + confidence_noise
    confidence = np.clip(confidence, 0.02, 0.999).astype(np.float32)

    positions = positions.astype(np.float32)
    normals = _normalize_rows(normals).astype(np.float32)
    return positions, normals, confidence, class_ids


def camera_trajectory(config: SampleConfig) -> dict[str, Any]:
    poses: list[dict[str, Any]] = []
    for index in range(16):
        angle = 2.0 * math.pi * index / 16.0
        radius = 8.2 + 0.35 * math.sin(index * 0.7)
        eye = [radius * math.cos(angle), 2.1 + 0.55 * math.sin(index * 0.9), radius * math.sin(angle)]
        target = [0.0, 1.2, 0.0]
        poses.append(
            {
                "index": index,
                "eye": [round(float(value), 9) for value in eye],
                "target": target,
                "up": [0.0, 1.0, 0.0],
                "fov_y_degrees": 52.0,
                "near": 0.05,
                "far": 80.0,
            }
        )
    return {
        "schema": "pointcloud-studio.camera-trajectory/1",
        "seed": config.seed,
        "pose_count": len(poses),
        "loop": True,
        "poses": poses,
    }


def edit_sequence(config: SampleConfig) -> dict[str, Any]:
    operations = [
        {
            "seq": 1,
            "type": "remove_sphere",
            "description": "剔除右侧墙面局部点",
            "params": {"center": [2.7, 1.3, 3.985], "radius": 0.36},
        },
        {
            "seq": 2,
            "type": "remove_box",
            "description": "视口框选剔除左侧墙面局部点",
            "params": {"bounds_min": [-4.03, 0.15, -1.25], "bounds_max": [-3.77, 1.55, 0.45]},
        },
        {
            "seq": 3,
            "type": "reestimate_normals",
            "description": "按邻域半径重估右侧墙面受影响区域法线",
            "params": {
                "region": {"kind": "sphere", "center": [2.7, 1.3, 3.96], "radius": 0.72},
                "neighbor_radius": 0.34,
            },
        },
        {
            "seq": 4,
            "type": "reestimate_normals",
            "description": "重估地面框选区域法线",
            "params": {
                "region": {
                    "kind": "box",
                    "bounds_min": [-1.1, -0.72, 0.4],
                    "bounds_max": [0.6, -0.48, 1.9],
                },
                "neighbor_radius": 0.42,
            },
        },
        {
            "seq": 5,
            "type": "set_recon_param",
            "description": "调整局部三角化栅格；该全局参数显式触发全量替换",
            "params": {"name": "triangulation_cell_size", "from": 0.20, "value": 0.22},
        },
    ]
    return {
        "schema": "pointcloud-studio.edit-sequence/1",
        "seed": config.seed,
        "operation_count": len(operations),
        "operations": operations,
    }


def _confidence_manifest(confidence: np.ndarray) -> dict[str, Any]:
    values = confidence.astype(np.float64)
    edges = np.linspace(0.0, 1.0, 11)
    histogram, _ = np.histogram(values, bins=edges, range=(0.0, 1.0))
    percentiles = np.percentile(values, [5, 50, 95])
    return {
        "minimum": round(float(values.min()), 9),
        "maximum": round(float(values.max()), 9),
        "mean": round(float(values.mean()), 9),
        "stddev": round(float(values.std()), 9),
        "p05": round(float(percentiles[0]), 9),
        "p50": round(float(percentiles[1]), 9),
        "p95": round(float(percentiles[2]), 9),
        "histogram": {
            "bin_edges": [round(float(item), 9) for item in edges],
            "counts": [int(item) for item in histogram],
        },
    }


def _write_if_changed(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == payload:
        return
    path.write_bytes(payload)


def write_sample_bundle(output_dir: str | Path, config: SampleConfig | None = None) -> dict[str, Any]:
    config = config or SampleConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    positions, normals, confidence, class_ids = generate_cloud(config)

    points_bytes = encode_points(positions)
    normals_bytes = encode_normals(normals, confidence)
    _write_if_changed(output / "points.bin", points_bytes)
    _write_if_changed(output / "normals.bin", normals_bytes)

    camera = camera_trajectory(config)
    edits = edit_sequence(config)
    _write_if_changed(output / "camera.json", json_bytes(camera))
    _write_if_changed(output / "edits.json", json_bytes(edits))

    parameters = config.normalized()
    summary = sha256_bytes(canonical_json(parameters).encode("utf-8"))
    manifest = {
        "schema": "pointcloud-studio.sample-manifest/1",
        "sample_name": "deterministic-room-and-ground-120k",
        "point_count": int(positions.shape[0]),
        "class_counts": {"cube_room": int(config.room_points), "ground": int(config.ground_points)},
        "class_codes": {"0": "cube_room", "1": "ground"},
        "bounds": {
            "min": [round(float(value), 9) for value in positions.min(axis=0)],
            "max": [round(float(value), 9) for value in positions.max(axis=0)],
            "center": [round(float(value), 9) for value in ((positions.min(axis=0) + positions.max(axis=0)) / 2.0)],
            "extent": [round(float(value), 9) for value in (positions.max(axis=0) - positions.min(axis=0))],
        },
        "normal_confidence": _confidence_manifest(confidence),
        "normal_length_tolerance": 2.0e-5,
        "point_quantization_bound_max": round(float(point_quantization_bound(positions).max()), 12),
        "generation_parameters": parameters,
        "generation_parameter_summary_sha256": summary,
        "recommended_reconstruction": {
            "layout_origin": list(config.layout_origin),
            "patch_size": config.patch_size,
            "triangulation_cell_size": config.triangulation_cell_size,
            "support_radius": round(config.patch_size * 0.72, 9),
            "neighbor_radius_default": 0.35,
            "position_tolerance": 1.0e-5,
        },
        "files": {
            "points.bin": {
                "sha256": sha256_bytes(points_bytes),
                "size": len(points_bytes),
                "format": "PCSTPTS1 little-endian float32 (N,3)",
            },
            "normals.bin": {
                "sha256": sha256_bytes(normals_bytes),
                "size": len(normals_bytes),
                "format": "PCSTNRM1 little-endian float32 (N,4: nx,ny,nz,confidence)",
            },
            "camera.json": {"sha256": sha256_bytes(json_bytes(camera))},
            "edits.json": {"sha256": sha256_bytes(json_bytes(edits))},
        },
    }
    _write_if_changed(output / "manifest.json", json_bytes(manifest))
    return manifest


def load_sample_bundle(input_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    root = Path(input_dir)
    points = load_points(root / "points.bin")
    normals, confidence = load_normals(root / "normals.bin")
    if points.shape[0] != normals.shape[0]:
        from .errors import ErrorCode, StudioError

        raise StudioError(
            ErrorCode.E_POINT_CLOUD_FORMAT,
            "point and normal counts differ",
            context={"points": int(points.shape[0]), "normals": int(normals.shape[0]), "path": str(root)},
        )
    manifest = read_json(root / "manifest.json")
    return points, normals, confidence, manifest


def validate_bundle_hashes(input_dir: str | Path) -> dict[str, str]:
    root = Path(input_dir)
    manifest = read_json(root / "manifest.json")
    verified: dict[str, str] = {}
    for relative, expected in manifest["files"].items():
        payload = (root / relative).read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected["sha256"]:
            from .errors import ErrorCode, StudioError

            raise StudioError(
                ErrorCode.E_POINT_CLOUD_FORMAT,
                "sample file hash mismatch",
                context={"path": str(root / relative), "expected": expected["sha256"], "actual": actual},
            )
        verified[relative] = actual
    return verified

