"""重建配置模型与校验。"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict

from .errors import BizCode, WorkbenchError
from .utils import content_hash, read_json


@dataclass(frozen=True)
class RebuildConfig:
    # 特征检测
    corner_threshold: float = 0.0015
    max_features_per_view: int = 120
    descriptor_radius: int = 4
    # 匹配
    match_distance_threshold: float = 0.6
    min_inliers: int = 6
    # 三角化
    min_disparity: float = 1.5
    # 深度图
    depth_width: int = 48
    depth_height: int = 36
    depth_levels: int = 32
    smooth_iterations: int = 2
    smooth_weight: float = 0.25
    # 点云合并
    voxel_size: float = 0.25
    # 定点量化
    fixed_point_bits: int = 12
    # 容差
    abs_tolerance: float = 1e-4
    rel_tolerance: float = 1e-4
    # 并发
    thread_count: int = 2
    # 快照
    snapshot_interval_views: int = 2
    # 全局重建点预算（-1 表示不限）
    rebuild_point_budget: int = -1

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RebuildConfig":
        allowed = {f.name for f in RebuildConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(data) - allowed
        if unknown:
            raise WorkbenchError(
                f"未知配置项: {sorted(unknown)}",
                stage="config",
                code=BizCode.E_CONFIG_INVALID,
            )
        cfg = RebuildConfig(**{k: data[k] for k in data if k in allowed})
        cfg.validate()
        return cfg

    @staticmethod
    def load(path: str) -> "RebuildConfig":
        return RebuildConfig.from_dict(read_json(path))

    def validate(self) -> None:
        checks = [
            (self.corner_threshold > 0, "corner_threshold 必须为正"),
            (self.max_features_per_view > 0, "max_features_per_view 必须为正"),
            (self.descriptor_radius >= 1, "descriptor_radius 至少为 1"),
            (self.match_distance_threshold > 0, "match_distance_threshold 必须为正"),
            (self.min_inliers >= 1, "min_inliers 至少为 1"),
            (self.min_disparity >= 0, "min_disparity 不能为负"),
            (self.depth_width > 0 and self.depth_height > 0, "深度图分辨率非法"),
            (self.depth_levels >= 2, "depth_levels 至少为 2"),
            (self.smooth_iterations >= 0, "smooth_iterations 不能为负"),
            (0.0 <= self.smooth_weight < 1.0, "smooth_weight 须位于 [0,1)"),
            (self.voxel_size >= 0, "voxel_size 不能为负"),
            (self.fixed_point_bits >= 1, "fixed_point_bits 至少为 1"),
            (self.abs_tolerance > 0 and self.rel_tolerance > 0, "容差必须为正"),
            (self.thread_count >= 1, "thread_count 至少为 1"),
            (self.snapshot_interval_views >= 1, "snapshot_interval_views 至少为 1"),
        ]
        for ok, reason in checks:
            if not ok:
                raise WorkbenchError(reason, stage="config", code=BizCode.E_CONFIG_INVALID)
        if self.voxel_size == 0.0:
            raise WorkbenchError(
                "voxel_size 为 0，无法进行体素合并",
                stage="point_cloud",
                code=BizCode.E_VOXEL_ZERO,
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        return content_hash(self.to_dict())


def default_config_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "configs", "rebuild_config.json"))
