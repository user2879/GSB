"""异常注入与分类（业务要求 7）。

注入配置是纯 JSON 可序列化数据，随运行记录落盘以保证精确复现。
所有注入都在输入/参数的 *副本* 上进行，原始观测集与调用方参数不被修改。

参数层：
    nonfinite_albedo  -> E_NONFINITE_ALBEDO（受影响像素）
    zero_light        -> E_ZERO_LIGHT（受影响光源/视点）
    degenerate_views  -> E_DEGENERATE_VIEWS（受影响像素与有效视点数）
    ill_conditioned   -> 越过条件数阈值（受影响像素，状态型而非中断）
管线层：
    skipped_views     -> E_VIEW_SKIPPED（渲染阶段跳过的视点编号）
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .errors import E_VIEW_SKIPPED, BusinessError


@dataclass
class FaultConfig:
    """异常注入配置（空配置表示无注入）。"""

    nonfinite_albedo: bool = False
    nonfinite_kind: str = "nan"  # "nan" 或 "inf"
    nonfinite_pixels: list[list[int]] = field(default_factory=list)  # [[row,col], ...]
    zero_light: bool = False
    zero_light_index: int = 0
    degenerate_views: bool = False
    degenerate_skip_views: list[int] = field(default_factory=list)
    ill_conditioned: bool = False
    ill_conditioned_threshold: float = 0.99  # 压到良态条件数（约 1.0）之下，强制全部像素越界
    skipped_views: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FaultConfig":
        return cls(**data)

    @property
    def active(self) -> bool:
        return bool(
            self.nonfinite_albedo
            or self.zero_light
            or self.degenerate_views
            or self.ill_conditioned
            or self.skipped_views
        )


def inject_nonfinite_albedo(albedo: np.ndarray, fault: FaultConfig) -> np.ndarray:
    """在反照率副本上写入 NaN/Inf，返回受污染副本。"""
    out = np.array(albedo, copy=True)
    fill = np.nan if fault.nonfinite_kind == "nan" else np.inf
    for row, col in fault.nonfinite_pixels:
        out[int(row), int(col)] = fill
    return out


def inject_zero_light(light_directions: list[list[float]], fault: FaultConfig) -> list[list[float]]:
    """在光源方向副本上把指定光源替换为零向量。"""
    out = [list(v) for v in light_directions]
    idx = int(fault.zero_light_index)
    if not 0 <= idx < len(out):
        raise IndexError(f"zero_light_index 越界: {idx}")
    out[idx] = [0.0, 0.0, 0.0]
    return out


def build_valid_mask(
    num_views: int,
    height: int,
    width: int,
    skip_views: list[int],
) -> np.ndarray:
    """构造 (M,H,W) 有效视点头像，跳过指定视点（副本数组）。"""
    mask = np.ones((num_views, height, width), dtype=bool)
    for view_id in skip_views:
        mask[int(view_id)] = False
    return mask


def validate_skipped_views(skipped: list[int], num_views: int) -> None:
    """管线层视点跳过的显式分类与受影响视点说明。"""
    bad = [int(v) for v in skipped if not 0 <= int(v) < num_views]
    if bad:
        raise BusinessError(
            E_VIEW_SKIPPED,
            "渲染阶段被要求跳过不存在的视点",
            layer="pipeline",
            details={"skipped_views": bad, "num_views": num_views},
        )


def fault_summary(fault: FaultConfig) -> dict[str, Any]:
    """供运行记录使用的注入摘要。"""
    summary = fault.to_dict()
    summary["active"] = fault.active
    return summary
