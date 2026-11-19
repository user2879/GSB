"""确定性配置：观测生成、光度立体求解、可微渲染优化与容差。

所有配置均为纯数据（dataclass + JSON 可序列化），不包含随机状态，
以保证同一份配置在任意机器、任意迭代顺序下产出可复算的结果。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


def default_light_directions() -> list[list[float]]:
    """6 个光源的原始方向（单位化前），跨半球充分分布以保证设计矩阵良态。"""
    return [
        [1.0, 0.35, 0.65],
        [-1.0, -0.25, 0.7],
        [0.3, 1.0, 0.55],
        [-0.25, -1.0, 0.6],
        [0.6, -0.5, 1.0],
        [-0.7, 0.55, 0.9],
    ]


def _rotation_y(deg: float) -> list[list[float]]:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ]


def default_camera_poses() -> list[dict[str, Any]]:
    """2 个正交相机位姿：正视图 + 绕 Y 轴倾斜 10 度的视图。

    相机局部坐标中视线方向为 +Z；世界系视线方向为 ``R @ [0,0,1]``。
    """
    return [
        {"name": "cam_front", "rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0]},
        {"name": "cam_tilt", "rotation": _rotation_y(20.0), "translation": [0.0, 0.0, 0.0]},
    ]


@dataclass
class ObservationConfig:
    """观测集生成配置。"""

    seed: int = 20261120
    size: int = 256
    num_lights: int = 6
    # 单位化前的原始光源方向；运行时按 L2 范数单位化
    light_directions: list[list[float]] = field(default_factory=default_light_directions)
    light_normalization: str = "L2: l_hat = l / sqrt(l.x^2+l.y^2+l.z^2)，零向量拒绝"
    camera_poses: list[dict[str, Any]] = field(default_factory=default_camera_poses)
    # Blinn-Phong 镜面项参数（前向模型固定，不参与优化）
    specular_gain: float = 0.15
    shininess: float = 12.0
    # 线性光观测的浮点表示口径：计算用 float64，落盘用 float32（小端）
    storage_dtype: str = "float32"
    compute_dtype: str = "float64"
    linear_light_note: str = "线性光（无伽马/无色调映射），观测=max(diffuse+specular,0)"


@dataclass
class SolveConfig:
    """多视点光度立体（加权最小二乘 + IRLS）配置。"""

    irls_iterations: int = 8
    # Huber 权重调谐常数（残差尺度）
    irls_tuning: float = 1.345
    # 权重下限，保证镜面外点被降权但设计矩阵不塌缩
    weight_floor: float = 0.05
    # 设计矩阵 SVD 条件数阈值，超过即判病态
    condition_threshold: float = 30.0
    # 有效视点少于该数即退化
    min_views: int = 4
    singular_floor: float = 1e-10
    # 病态像素误差区间使用的置信尺度（95% 正态分位近似）
    confidence_z: float = 1.96


@dataclass
class OptConfig:
    """交替优化、冲突容差与瓦片核对配置。"""

    max_iterations: int = 12
    lr_normal: float = 0.3
    lr_albedo: float = 0.3
    # 首轮之后，PS 独立解与当前 DR 参数的融合系数（首轮用 1.0 纯 PS 初始化）
    ps_blend: float = 0.5
    # 损失下降量低于该值即收敛
    loss_delta_tol: float = 1e-12
    # 阶段间冲突容差
    conflict_angle_deg: float = 5.0
    conflict_albedo_abs: float = 0.05
    # 瓦片尺寸（行, 列）
    tile_rows: int = 32
    tile_cols: int = 32
    # 增量 vs 全量重渲染核对容差（不允许在分歧时放宽）
    render_check_atol: float = 1e-10
    render_check_rtol: float = 1e-12
    # 解析梯度有限差分核对
    finite_diff_eps: float = 1e-7
    gradcheck_tol: float = 1e-6


@dataclass
class PipelineConfig:
    """管线总配置。"""

    observation: ObservationConfig = field(default_factory=ObservationConfig)
    solve: SolveConfig = field(default_factory=SolveConfig)
    opt: OptConfig = field(default_factory=OptConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        return cls(
            observation=ObservationConfig(**data["observation"]),
            solve=SolveConfig(**data["solve"]),
            opt=OptConfig(**data["opt"]),
        )
