"""斯涅尔折射、全反射与临界角判定。

全部使用 float64、固定运算顺序。入射约定：
- d 为入射光线单位方向（指向界面）；
- n 为朝向入射侧的界面单位法线，因此 cos_i = -dot(d, n) > 0；
- eta = ior_incident / ior_transmitted。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .errors import IorRangeError
from .vectors import dot, reflect

# 声明的合法折射率范围（含空气 1.0）。
IOR_MIN = 1.0
IOR_MAX = 3.0
IOR_EPS = 1e-12
CRITICAL_EPS = 1e-12


@dataclass(frozen=True)
class RefractResult:
    direction: np.ndarray
    classification: str  # "REFRACT" | "TIR" | "NO_CRITICAL_ANGLE"
    cos_theta_i: float
    sin2_theta_t: float
    critical_angle: Optional[float]


def validate_ior(ior: float, *, body_id: str, role: str) -> float:
    """折射率合法性：必须为正且落在声明范围 [IOR_MIN, IOR_MAX]。"""
    value = float(ior)
    if not math.isfinite(value) or value <= 0.0 or value < IOR_MIN - IOR_EPS or value > IOR_MAX + IOR_EPS:
        raise IorRangeError(
            f"{role} ior of '{body_id}' is outside [{IOR_MIN}, {IOR_MAX}]",
            {"body_id": body_id, "role": role, "ior": value,
             "declared_range": [IOR_MIN, IOR_MAX]},
        )
    return value


def critical_angle(ior_incident: float, ior_transmitted: float) -> Optional[float]:
    """临界角（弧度）。光疏->光密不存在临界角，返回 None。"""
    ratio = np.float64(ior_transmitted) / np.float64(ior_incident)
    if ratio >= 1.0:
        return None
    return float(np.arcsin(ratio))


def refract_or_reflect(
    direction: np.ndarray,
    normal: np.ndarray,
    ior_incident: float,
    ior_transmitted: float,
) -> RefractResult:
    """按斯涅尔定律计算折射方向；超临界角则全反射改向。

    判定归类可复算：
    - REFRACT：存在折射传播方向；
    - TIR：光密->光疏且入射角超过临界角（sin2_t > 1）；
    - NO_CRITICAL_ANGLE：光疏->光密，临界角不存在，永远不会全反射。
    退化折射率由 validate_ior 在调用处先拦截；此处对非正值额外防御。
    """
    n_i = np.float64(ior_incident)
    n_t = np.float64(ior_transmitted)
    if not (np.isfinite(n_i) and np.isfinite(n_t)) or n_i <= 0.0 or n_t <= 0.0:
        raise IorRangeError(
            "non-positive or non-finite ior encountered at interface",
            {"ior_incident": float(n_i), "ior_transmitted": float(n_t)},
        )

    eta = n_i / n_t
    cos_i = -dot(direction, normal)
    # 数值修正：掠入射时 cos_i 可能出现极小负值。
    if cos_i < 0.0:
        cos_i = np.float64(0.0)
    sin2_t = float(eta * eta * (np.float64(1.0) - cos_i * cos_i))
    theta_c = critical_angle(float(n_i), float(n_t))

    if sin2_t > 1.0 + CRITICAL_EPS:
        # 全反射：沿镜面反射方向继续传播，光线不丢弃。
        reflected = reflect(direction, normal)
        return RefractResult(reflected, "TIR", float(cos_i), sin2_t, theta_c)

    if sin2_t > 1.0:  # 临界角邻域的数值钳制：贴边按折射处理
        sin2_t = 1.0
    cos_t = np.sqrt(np.float64(1.0) - np.float64(sin2_t))
    factor = eta * cos_i - cos_t
    refracted = np.array(
        [
            eta * direction[0] + factor * normal[0],
            eta * direction[1] + factor * normal[1],
            eta * direction[2] + factor * normal[2],
        ],
        dtype=np.float64,
    )
    classification = "NO_CRITICAL_ANGLE" if theta_c is None else "REFRACT"
    return RefractResult(refracted, classification, float(cos_i), sin2_t, theta_c)
