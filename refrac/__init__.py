"""透明介质折射合成领域组件。

纯 NumPy 实现：确定性程序化场景、光线/透明面求交、斯涅尔折射、
介质吸收路径衰减、全反射处理、逐像素路径报告与可复现异常注入。
"""

from .errors import (
    RefracError,
    IorRangeError,
    RayDegenerateError,
    DepthBoundsError,
    AbsorbNegativeError,
    IntersectDegenerateError,
    PixelIsolatedError,
    InjectConfigError,
    ERROR_CODES,
)
from .rng import DeterministicRng

__all__ = [
    "RefracError",
    "IorRangeError",
    "RayDegenerateError",
    "DepthBoundsError",
    "AbsorbNegativeError",
    "IntersectDegenerateError",
    "PixelIsolatedError",
    "InjectConfigError",
    "ERROR_CODES",
    "DeterministicRng",
]
