"""重建配置的解析、合法性校验与参数指纹。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .errors import E_VIEW_INVALID, StageError
from .util import canonical_json, content_fingerprint


@dataclass(frozen=True)
class ReconstructionConfig:
    feature_threshold: float
    match_min_inliers: int
    triang_min_parallax_deg: float
    depth_resolution: tuple[int, int]
    depth_quant_levels: int
    depth_smooth_passes: int
    voxel_size: float
    fixed_point_frac_bits: int
    thread_workers: int
    global_point_budget: int
    per_view_quota: int
    snapshot_commit_interval_views: int
    view_pairs: tuple[tuple[int, int], ...] = field(default_factory=tuple)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "ReconstructionConfig":
        if not isinstance(raw, dict):
            raise StageError(E_VIEW_INVALID, "重建配置必须是 JSON 对象", stage="config")
        required = {
            "feature_threshold": float,
            "match_min_inliers": int,
            "triang_min_parallax_deg": float,
            "depth_quant_levels": int,
            "depth_smooth_passes": int,
            "voxel_size": float,
            "fixed_point_frac_bits": int,
            "thread_workers": int,
            "global_point_budget": int,
            "per_view_quota": int,
            "snapshot_commit_interval_views": int,
        }
        values: dict[str, Any] = {}
        for key, caster in required.items():
            if key not in raw:
                raise StageError(E_VIEW_INVALID, f"配置缺少字段 {key}", stage="config")
            value = raw[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise StageError(E_VIEW_INVALID, f"配置字段 {key} 必须是数值", stage="config")
            values[key] = caster(value)

        depth_resolution_raw = raw.get("depth_resolution")
        if (
            not isinstance(depth_resolution_raw, list)
            or len(depth_resolution_raw) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in depth_resolution_raw)
        ):
            raise StageError(
                E_VIEW_INVALID, "depth_resolution 必须是两个整数组成的数组", stage="config"
            )
        depth_resolution = (int(depth_resolution_raw[0]), int(depth_resolution_raw[1]))

        pairs: list[tuple[int, int]] = []
        for pair in raw.get("view_pairs", []):
            if not isinstance(pair, list) or len(pair) != 2:
                raise StageError(E_VIEW_INVALID, "view_pairs 项必须是两个视图序号", stage="config")
            left, right = int(pair[0]), int(pair[1])
            if left == right:
                raise StageError(E_VIEW_INVALID, "视图对不能引用同一视图", stage="config")
            pairs.append((left, right) if left < right else (right, left))
        # 去重并按视图对标识稳定排序
        pairs = sorted(set(pairs))

        cfg = ReconstructionConfig(
            feature_threshold=values["feature_threshold"],
            match_min_inliers=values["match_min_inliers"],
            triang_min_parallax_deg=values["triang_min_parallax_deg"],
            depth_resolution=depth_resolution,
            depth_quant_levels=values["depth_quant_levels"],
            depth_smooth_passes=values["depth_smooth_passes"],
            voxel_size=values["voxel_size"],
            fixed_point_frac_bits=values["fixed_point_frac_bits"],
            thread_workers=values["thread_workers"],
            global_point_budget=values["global_point_budget"],
            per_view_quota=values["per_view_quota"],
            snapshot_commit_interval_views=values["snapshot_commit_interval_views"],
            view_pairs=tuple(pairs),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not math.isfinite(self.feature_threshold) or self.feature_threshold <= 0:
            raise StageError(E_VIEW_INVALID, "feature_threshold 必须为正数", stage="config")
        if self.match_min_inliers < 1:
            raise StageError(E_VIEW_INVALID, "match_min_inliers 必须 >= 1", stage="config")
        if not (0.0 < self.triang_min_parallax_deg < 90.0):
            raise StageError(E_VIEW_INVALID, "triang_min_parallax_deg 必须落在 (0, 90) 度", stage="config")
        if min(self.depth_resolution) < 2:
            raise StageError(E_VIEW_INVALID, "depth_resolution 边长必须 >= 2", stage="config")
        if self.depth_quant_levels < 2:
            raise StageError(E_VIEW_INVALID, "depth_quant_levels 必须 >= 2", stage="config")
        if self.depth_smooth_passes < 0:
            raise StageError(E_VIEW_INVALID, "depth_smooth_passes 必须 >= 0", stage="config")
        if not math.isfinite(self.voxel_size) or self.voxel_size <= 0:
            raise StageError(E_VIEW_INVALID, "voxel_size 必须为正数", stage="config")
        if not (0 <= self.fixed_point_frac_bits <= 30):
            raise StageError(E_VIEW_INVALID, "fixed_point_frac_bits 必须落在 [0, 30]", stage="config")
        if self.thread_workers < 1:
            raise StageError(E_VIEW_INVALID, "thread_workers 必须 >= 1", stage="config")
        if self.global_point_budget < 1:
            raise StageError(E_VIEW_INVALID, "global_point_budget 必须 >= 1", stage="config")
        if self.per_view_quota < 1:
            raise StageError(E_VIEW_INVALID, "per_view_quota 必须 >= 1", stage="config")
        if self.snapshot_commit_interval_views < 1:
            raise StageError(
                E_VIEW_INVALID, "snapshot_commit_interval_views 必须 >= 1", stage="config"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_threshold": self.feature_threshold,
            "match_min_inliers": self.match_min_inliers,
            "triang_min_parallax_deg": self.triang_min_parallax_deg,
            "depth_resolution": list(self.depth_resolution),
            "depth_quant_levels": self.depth_quant_levels,
            "depth_smooth_passes": self.depth_smooth_passes,
            "voxel_size": self.voxel_size,
            "fixed_point_frac_bits": self.fixed_point_frac_bits,
            "thread_workers": self.thread_workers,
            "global_point_budget": self.global_point_budget,
            "per_view_quota": self.per_view_quota,
            "snapshot_commit_interval_views": self.snapshot_commit_interval_views,
            "view_pairs": [list(pair) for pair in self.view_pairs],
        }

    def fingerprint(self) -> str:
        return content_fingerprint(canonical_json(self.to_dict()))

    def section_fingerprint(self, *names: str) -> str:
        return content_fingerprint(canonical_json({name: self.to_dict()[name] for name in names}))
