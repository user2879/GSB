"""重建配置：阈值、深度档位、体素、定点位宽、并发与预算。"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .utils import fingerprint, read_json

DEFAULTS = dict(
    feature_threshold=160,      # 特征检测灰度阈值
    match_min_inliers=3,        # 匹配最小内点数
    min_disparity=2.0,          # 三角化最小视差（像素）
    depth_resolution=(32, 32),  # 深度图分辨率 (H, W)
    depth_quant_levels=64,      # 深度量化档位
    voxel_size=0.5,             # 点云合并体素尺寸
    fixed_point_bits=8,         # 定点小数位宽
    max_workers=2,              # 线程池最大并发
    point_budget=10_000,        # 全局重建点数预算
    view_point_quota=10_000,    # 单视图重建点数配额
)


@dataclass(frozen=True)
class ReconConfig:
    feature_threshold: int
    match_min_inliers: int
    min_disparity: float
    depth_resolution: tuple
    depth_quant_levels: int
    voxel_size: float
    fixed_point_bits: int
    max_workers: int
    point_budget: int
    view_point_quota: int

    @classmethod
    def from_dict(cls, d: dict) -> "ReconConfig":
        merged = {**DEFAULTS, **(d or {})}
        merged["depth_resolution"] = tuple(merged["depth_resolution"])
        return cls(**merged)

    @classmethod
    def load(cls, path: str) -> "ReconConfig":
        return cls.from_dict(read_json(path))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["depth_resolution"] = list(self.depth_resolution)
        return d

    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())

    def with_updates(self, **changes) -> "ReconConfig":
        data = self.to_dict()
        data.update(changes)
        return ReconConfig.from_dict(data)
