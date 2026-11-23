"""体素级点云合并：按 voxel_size 网格融合，输出稳定的代表点序列。"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .errors import E_VOXEL_ZERO_SIZE, FailureList
import math

from .utils import combine_fingerprint, snap_to_grid


@dataclass
class VoxelRecord:
    vx: int
    vy: int
    vz: int
    x: float
    y: float
    z: float
    landmark_ids: list
    point_count: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VoxelRecord":
        return cls(**d)


def merge_point_cloud(points: list, voxel_size: float, fixed_bits: int,
                      failures: FailureList) -> list:
    if voxel_size <= 0:
        failures.fail(
            E_VOXEL_ZERO_SIZE,
            f"voxel size must be positive, got {voxel_size}",
            stage="voxel",
            sort_key=f"voxel|0000|{E_VOXEL_ZERO_SIZE}",
        )
        return []

    buckets: dict = {}
    for p in points:
        vx = math.floor(p.x / voxel_size)
        vy = math.floor(p.y / voxel_size)
        vz = math.floor(p.z / voxel_size)
        buckets.setdefault((vx, vy, vz), []).append(p)

    records = []
    for key in sorted(buckets):
        group = buckets[key]
        vx, vy, vz = key
        # 代表点固定为该体素自身网格点（索引 * voxel_size），
        # 再做定点量化，保证索引与坐标在负半轴也严格一致
        gx = snap_to_grid(vx * voxel_size, voxel_size, fixed_bits)
        gy = snap_to_grid(vy * voxel_size, voxel_size, fixed_bits)
        gz = snap_to_grid(vz * voxel_size, voxel_size, fixed_bits)
        records.append(
            VoxelRecord(
                vx=key[0], vy=key[1], vz=key[2],
                x=gx, y=gy, z=gz,
                landmark_ids=sorted(g.landmark_id for g in group),
                point_count=len(group),
            )
        )
    return records



def voxel_cache_key(points_fp: str, cfg_fp: str) -> str:
    return combine_fingerprint("voxel", points_fp, cfg_fp)
