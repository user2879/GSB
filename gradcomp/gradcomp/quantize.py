"""整数定点缩放量化、截断与逐元素误差界推导。

量化映射：q = trunc(x * s)，解压 x_hat = q / s。
截断方向 truncation：
  floor        向负无穷截断，量化误差 e = x_hat - x ∈ (-1/s, 0]
  nearest      四舍五入（.5 向偶数），|e| <= 0.5/s
  toward_zero  向零截断，|e| < 1/s
所有误差界推导只依赖缩放因子 s 与梯度范围观测值。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import E_CONFIG, GradCompError

# 浮点舍入相对余量：用于“实测误差 <= 理论界”的核验判定。
FLOAT_SLACK = 1.0e-9


def _pick_int_dtype(max_abs_q: int) -> np.dtype:
    """按量化后绝对值选择可容纳的最小整数 dtype。"""
    for dtype in (np.int8, np.int16, np.int32, np.int64):
        info = np.iinfo(dtype)
        if info.min <= -max_abs_q and max_abs_q <= info.max:
            return np.dtype(dtype)
    raise GradCompError(E_CONFIG, "量化后整数超出 int64 范围，请增大缩放因子",
                        field="scale_exponent")


@dataclass(frozen=True)
class Quantizer:
    scale_exponent: int
    truncation: str

    @property
    def scale(self) -> float:
        return float(10 ** self.scale_exponent)

    def _trunc(self, scaled: np.ndarray) -> np.ndarray:
        if self.truncation == "floor":
            return np.floor(scaled)
        if self.truncation == "nearest":
            return np.rint(scaled)  # 银行家舍入，确定性
        if self.truncation == "toward_zero":
            return np.sign(scaled) * np.floor(np.abs(scaled))
        raise GradCompError(E_CONFIG, f"未知截断方向 {self.truncation}",
                            field="truncation")

    def quantize(self, x: np.ndarray) -> tuple[np.ndarray, np.dtype]:
        """返回量化整数数组与所使用的整数 dtype。"""
        scaled = np.asarray(x, dtype=np.float64) * self.scale
        q = self._trunc(scaled).astype(np.int64)
        max_abs = int(np.max(np.abs(q))) if q.size else 0
        dtype = _pick_int_dtype(max_abs)
        return q.astype(dtype), dtype

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        return q.astype(np.float64) / self.scale

    def single_value_bound(self) -> float:
        """单值量化误差上界（绝对值）。"""
        step = 1.0 / self.scale
        return step * (0.5 if self.truncation == "nearest" else 1.0)

    def roundtrip(self, x: np.ndarray) -> dict[str, Any]:
        """量化-解压往返，返回误差、观测范围与可核验误差界。"""
        x = np.asarray(x, dtype=np.float64)
        q, dtype = self.quantize(x)
        x_hat = self.dequantize(q)
        err = x_hat - x
        bound1 = self.single_value_bound()
        observed = {
            "min": float(np.min(x)) if x.size else 0.0,
            "max": float(np.max(x)) if x.size else 0.0,
            "max_abs": float(np.max(np.abs(x))) if x.size else 0.0,
        }
        max_abs_err = float(np.max(np.abs(err))) if err.size else 0.0
        return {
            "quantized_dtype": str(dtype),
            "scale": self.scale,
            "scale_exponent": self.scale_exponent,
            "truncation": self.truncation,
            "observed_range": observed,
            "single_value_bound": bound1,
            "measured_max_abs_error": max_abs_err,
            "bound_respected": bool(
                max_abs_err <= bound1 * (1.0 + FLOAT_SLACK) + FLOAT_SLACK
            ),
            "errors": err,
            "dequantized": x_hat,
        }


def vector_relative_bound(scale: float, dimension: int,
                          gradient_norm: float,
                          truncation: str = "floor") -> dict[str, float]:
    """整向量范数相对误差上界。

    由 |e_i| <= c/s（nearest 时 c=0.5，否则 c=1）逐元素界推导：
      ||e||_2 <= c/s * sqrt(d)
      相对误差 <= c*sqrt(d) / (s * ||x||_2)
    同时给出合成 k 个元素后的绝对/相对上界。
    """
    c = 0.5 if truncation == "nearest" else 1.0
    step = 1.0 / scale
    vec_abs = c * step * math.sqrt(dimension)
    rel = vec_abs / gradient_norm if gradient_norm > 0 else math.inf
    return {
        "c_per_element": c,
        "step": step,
        "vector_abs_bound_2norm": vec_abs,
        "gradient_2norm": gradient_norm,
        "vector_relative_bound": rel,
    }


def combined_elements_bound(scale: float, k: int,
                            truncation: str = "floor") -> dict[str, float]:
    """k 个元素各自带量化误差合成后的范数上界。"""
    c = 0.5 if truncation == "nearest" else 1.0
    step = 1.0 / scale
    return {
        "elements": k,
        "combined_abs_bound_2norm": c * step * math.sqrt(k),
        "combined_abs_bound_l1": c * step * k,
    }
