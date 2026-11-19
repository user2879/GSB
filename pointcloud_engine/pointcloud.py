"""Point cloud container, PTS3 parsing and deterministic sample generation.

On-disk PTS3 text format (one header line, then one point per line)::

    PTS3 <count>
    x y z gray nx ny nz
    ...

Only whitespace-separated ASCII floats are used, so the format needs no
external dependency and is easy to inspect or hand-edit for failure tests.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

from .errors import EngineError

FILE_MAGIC = "PTS3"


@dataclass
class PointCloud:
    positions: np.ndarray  # (N, 3) float64 world-space coordinates
    grays: np.ndarray      # (N,)   float64 in [0, 1]
    normals: np.ndarray    # (N, 3) float64 non-zero direction vectors
    source_path: str

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])

    def content_digest(self) -> str:
        from .io_utils import new_hasher

        hasher = new_hasher()
        hasher.update(b"pts3|")
        for array in (self.positions, self.grays, self.normals):
            hasher.update(np.ascontiguousarray(array).tobytes(order="C"))
            hasher.update(b"|")
        return hasher.hexdigest()


def _parse_float(tokens: list[str], index: int, line_no: int) -> float:
    try:
        return float(tokens[index])
    except (ValueError, IndexError):
        raise EngineError(
            "E_POINTCLOUD",
            f"line {line_no}: expected numeric field at column {index + 1}",
            line=line_no,
        )


def load_point_cloud(path: str) -> PointCloud:
    if not os.path.isfile(path):
        raise EngineError("E_POINTCLOUD", f"point cloud file not found: {path}", path=path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]
    except OSError as exc:
        raise EngineError("E_POINTCLOUD", f"cannot read point cloud: {exc}", path=path)

    if not lines:
        raise EngineError("E_POINTCLOUD", "point cloud file is empty", path=path)

    header = lines[0].split()
    if len(header) != 2 or header[0] != FILE_MAGIC:
        raise EngineError(
            "E_POINTCLOUD",
            f"first line must be '{FILE_MAGIC} <count>'",
            path=path,
        )
    try:
        declared = int(header[1])
    except ValueError:
        raise EngineError("E_POINTCLOUD", "PTS3 count must be an integer", path=path)
    if declared < 0 or len(lines) - 1 != declared:
        raise EngineError(
            "E_POINTCLOUD",
            f"declared count {declared} does not match data rows {len(lines) - 1}",
            path=path,
        )

    positions = np.zeros((declared, 3), dtype=np.float64)
    grays = np.zeros(declared, dtype=np.float64)
    normals = np.zeros((declared, 3), dtype=np.float64)

    for row in range(declared):
        line_no = row + 2
        tokens = lines[row + 1].split()
        if len(tokens) != 7:
            raise EngineError(
                "E_POINTCLOUD",
                f"line {line_no}: expected 7 fields (x y z gray nx ny nz), got {len(tokens)}",
                line=line_no,
            )
        values = [_parse_float(tokens, col, line_no) for col in range(7)]
        positions[row] = values[0:3]
        grays[row] = values[3]
        normals[row] = values[4:7]

    cloud = PointCloud(positions, grays, normals, os.path.abspath(path))
    validate_point_cloud(cloud)
    return cloud


def validate_point_cloud(cloud: PointCloud) -> None:
    n = cloud.count
    if n == 0:
        raise EngineError("E_POINTCLOUD", "point cloud contains no points")
    if cloud.positions.shape != (n, 3) or cloud.normals.shape != (n, 3) or cloud.grays.shape != (n,):
        raise EngineError("E_POINTCLOUD", "point cloud arrays have inconsistent shapes")

    if not np.all(np.isfinite(cloud.positions)):
        raise EngineError("E_FINITE", "one or more point coordinates are not finite")
    if not np.all(np.isfinite(cloud.grays)):
        raise EngineError("E_FINITE", "one or more grayscale values are not finite")
    if not np.all(np.isfinite(cloud.normals)):
        raise EngineError("E_FINITE", "one or more normal components are not finite")

    if np.any(cloud.grays < 0.0) or np.any(cloud.grays > 1.0):
        bad = int(np.argmax((cloud.grays < 0.0) | (cloud.grays > 1.0)))
        raise EngineError("E_GRAY", f"grayscale at point {bad} is outside [0, 1]", point_index=bad)

    normal_lengths = np.linalg.norm(cloud.normals, axis=1)
    if np.any(normal_lengths <= 0.0):
        bad = int(np.argmax(normal_lengths <= 0.0))
        raise EngineError("E_NORMAL", f"normal at point {bad} is the zero vector", point_index=bad)


def save_point_cloud(path: str, cloud: PointCloud) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    parts = [f"{FILE_MAGIC} {cloud.count}"]
    for i in range(cloud.count):
        x, y, z = cloud.positions[i]
        g = cloud.grays[i]
        nx, ny, nz = cloud.normals[i]
        parts.append(
            f"{x:.10f} {y:.10f} {z:.10f} {g:.10f} {nx:.10f} {ny:.10f} {nz:.10f}"
        )
    from .io_utils import atomic_write_text

    atomic_write_text(path, "\n".join(parts) + "\n")


def generate_sample_cloud(path: str, *, rings: int = 18, sectors: int = 24, seed: int = 20260101) -> PointCloud:
    """Generate a deterministic colored sphere-like point cloud locally.

    Points lie on a noisy unit sphere so that self-occlusion is visible.
    Grayscale is a smooth deterministic function of the direction; normals
    point radially outward. No external asset or network resource is used.
    """
    rng = np.random.default_rng(seed)
    positions: list[list[float]] = []
    grays: list[float] = []
    normals: list[list[float]] = []

    for r in range(rings):
        theta = math.pi * (r + 1) / (rings + 1)
        for s in range(sectors):
            phi = 2.0 * math.pi * s / sectors
            nx = math.sin(theta) * math.cos(phi)
            ny = math.sin(theta) * math.sin(phi)
            nz = math.cos(theta)
            jitter = 1.0 + 0.03 * math.sin(3.0 * phi + 2.0 * theta) + 0.01 * float(rng.standard_normal())
            positions.append([nx * jitter, ny * jitter, nz * jitter])
            grays.append(float(np.clip(0.5 + 0.5 * nx * nz + 0.15 * ny, 0.0, 1.0)))
            normals.append([nx, ny, nz])

    # Deterministic interior points to exercise depth ordering.
    for k in range(12):
        t = 2.0 * math.pi * k / 12.0
        positions.append([0.4 * math.cos(t), 0.4 * math.sin(t), -0.2])
        grays.append(0.25 + 0.02 * k)
        normals.append([math.cos(t), math.sin(t), 0.0])

    cloud = PointCloud(
        np.asarray(positions, dtype=np.float64),
        np.asarray(grays, dtype=np.float64),
        np.asarray(normals, dtype=np.float64),
        os.path.abspath(path),
    )
    save_point_cloud(path, cloud)
    return cloud
