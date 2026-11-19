"""Shared helpers to build deterministic in-memory scenes and temp workspaces."""

from __future__ import annotations

import json
import os
import shutil
import tempfile

import numpy as np

from pointcloud_engine.pointcloud import PointCloud, save_point_cloud


def make_cloud(n: int = 60, seed: int = 7) -> PointCloud:
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float64)
    positions[:, 2] *= 0.5
    grays = np.linspace(0.05, 0.95, n, dtype=np.float64)
    normals = rng.normal(size=(n, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return PointCloud(positions, grays, normals, source_path="")


def base_scene(workdir: str) -> dict:
    return {
        "point_cloud": "cloud.pts3",
        "world_scale": 1.0,
        "origin_offset": [0.0, 0.0, 0.0],
        "camera": {
            "position": [0.0, 0.0, 4.0],
            "target": [0.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
            "vfov_deg": 60.0,
            "aspect": 1.5,
        },
        "width": 24,
        "height": 16,
        "supersample": 3,
        "point_radius": 0.12,
        "depth_decay": {"kind": "exp", "coefficient": 0.05},
        "tile_width": 18,
        "tile_height": 12,
        "concurrency": 4,
        "memory_budget_bytes": 18 * 12 * 8 * 4,
        "state_dir": "state",
        "output_dir": "output",
        "near": 1e-6,
        "tile_delay_seconds": 0.0,
        "fault_injections": [],
    }


class Workspace:
    def __init__(self) -> None:
        self.dir = tempfile.mkdtemp(prefix="pce_test_")

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, cloud: PointCloud | None = None, scene: dict | None = None) -> str:
        cloud = cloud or make_cloud()
        cloud.source_path = os.path.join(self.dir, "cloud.pts3")
        save_point_cloud(cloud.source_path, cloud)
        scene = scene or base_scene(self.dir)
        path = os.path.join(self.dir, "scene.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(scene, handle, ensure_ascii=False, indent=2, sort_keys=True)
        return path

    def path(self, *parts: str) -> str:
        return os.path.join(self.dir, *parts)
