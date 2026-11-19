"""介质内部吸收与比尔—朗伯（Beer-Lambert）路径衰减。

每段介质内线段长度 L、吸收系数 sigma（RGB 三分量，单位 1/长度）：
- 解析式：T_analytic = exp(-sigma * L)
- 声明采样数 S 的确定性数值积分：将线段等分为 S 段，
  T_quad = exp(-sum_s sigma * (L/S))，求和顺序固定（s=0..S-1）。
sigma 为常量时两者在声明容差内一致；S 只改变依赖该采样的像素。
"""

from __future__ import annotations

import numpy as np

from .errors import AbsorbNegativeError
from .vectors import as_vec3

TRANSMITTANCE_CUTOFF = np.float64(1e-6)
ANALYTIC_TOL = np.float64(1e-10)


def validate_absorption(sigma, *, body_id: str) -> np.ndarray:
    value = as_vec3(sigma)
    if not np.all(np.isfinite(value)) or np.any(value < 0.0):
        raise AbsorbNegativeError(
            f"absorption coefficient of '{body_id}' contains a negative component",
            {"body_id": body_id, "sigma": [float(x) for x in value]},
        )
    return value


def analytical_transmittance(sigma: np.ndarray, length: float) -> np.ndarray:
    return np.exp(-sigma * np.float64(length))


def integrate_transmittance(
    sigma,
    length: float,
    sample_count: int,
    *,
    body_id: str = "",
) -> tuple[np.ndarray, np.ndarray, float]:
    """返回 (采样透过率, 解析透过率, 相对/绝对偏差最大值)。"""
    coeff = validate_absorption(sigma, body_id=body_id)
    if not isinstance(sample_count, int) or sample_count < 1:
        raise AbsorbNegativeError(
            "sample_count must be a positive integer",
            {"body_id": body_id, "sample_count": sample_count},
        )
    seg_len = np.float64(length) / np.float64(sample_count)
    # 固定顺序累加光学厚度，再取指数；运算顺序不随调用环境变化。
    optical = np.zeros(3, dtype=np.float64)
    for _ in range(sample_count):
        optical = optical + coeff * seg_len
    sampled = np.exp(-optical)
    analytic = analytical_transmittance(coeff, length)
    deviation = float(np.max(np.abs(sampled - analytic)))
    return sampled, analytic, deviation


def is_fully_absorbed(transmittance: np.ndarray) -> bool:
    return bool(np.min(transmittance) < TRANSMITTANCE_CUTOFF)
