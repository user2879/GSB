"""统一精度口径的定点量化。

格度约定
--------
* ``scale`` 必须是 10 的整数幂 ``10**exponent``（exponent 为非负整数）。
* 物理值 ``x`` 量化为格点整数 ``q = trunc_toward_zero(x * scale)``。
* 所有乘加在 Python 任意精度整数 / numpy.int64 上完成，中间不回到浮点。
* 截断方向统一为 **向零截断 (truncation toward zero)**，由 :func:`trunc_zero`
  显式实现，不依赖浮点转 int 的隐式行为差异。
* 每个独立量化动作的格内误差界为 1 个整数格 ``|x - q/scale| < 1/scale``，
  误差界全部以整数格表示，随 scale 改变可直接重新解释。

派生量纲
--------
派生项 ``y = sum(c_i * x_i)`` 中系数与值使用同一 scale，因此
``q(y) = trunc_zero( sum(q(c_i) * q(x_i)) / scale )``，
除以 scale 使结果回到同一格度，截断方向仍为向零。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from .errors import RangeError

TRUNCATION_DIRECTION = "toward_zero"


def trunc_zero(numerator: int, denominator: int) -> int:
    """整数除法并显式向零截断。

    Python 的 ``//`` 是向下取整，负数时会偏离向零语义，这里用
    ``abs`` 配合符号位实现与平台无关的向零截断。
    """
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    q, _r = divmod(abs(numerator), denominator)
    return q if numerator >= 0 else -q


def ceil_div(numerator: int, denominator: int) -> int:
    """非负整数的向上取整除法（保守误差界用）。"""
    if numerator < 0:
        raise ValueError("ceil_div 仅用于非负分子")
    return -((-numerator) // denominator)


@dataclass(frozen=True)
class GridSpec:
    """定点格度参数。

    exponent : 缩放因子 10 的幂次，scale = 10**exponent
    int_bits : 带符号整数位数（含符号位），决定可表示物理范围
    """

    exponent: int
    int_bits: int = 63

    @property
    def scale(self) -> int:
        return 10 ** self.exponent

    @property
    def qmax(self) -> int:
        return 2 ** (self.int_bits - 1) - 1

    @property
    def qmin(self) -> int:
        return -(2 ** (self.int_bits - 1))

    def physical_max(self) -> float:
        return self.qmax / self.scale

    def validate_power_of_ten(self) -> None:
        if not isinstance(self.exponent, int) or self.exponent < 0:
            raise ValueError("exponent 必须是非负整数，scale 才能是 10 的整数幂")
        scale = self.scale
        digits = str(scale)
        if not (digits == "1" or (digits[0] == "1" and set(digits[1:]) == {"0"})):
            raise ValueError(f"scale={scale} 不是 10 的整数幂")

    def quantize(self, x: float, *, key: str | None = None) -> int:
        """把物理值量化为格点整数并做范围检查。"""
        scaled = x * self.scale
        q = trunc_zero(int(math.floor(abs(scaled))), 1)
        q = q if scaled >= 0 else -q
        if q > self.qmax or q < self.qmin:
            raise RangeError(
                f"配置项 {key} 量化值 {q} 超出 {self.int_bits} 位定点范围 "
                f"[{self.qmin}, {self.qmax}]（scale={self.scale}）",
                {
                    "key": key,
                    "quantized": q,
                    "qmin": self.qmin,
                    "qmax": self.qmax,
                    "scale": self.scale,
                    "physical_value": x,
                },
            )
        return q

    def quantize_batch(self, values: Sequence[float]) -> np.ndarray:
        """NumPy 批量量化（向零截断，dtype=int64）。"""
        arr = np.asarray(values, dtype=np.float64) * np.float64(self.scale)
        q = np.trunc(arr).astype(np.int64)  # np.trunc 向零，与 trunc_zero 一致
        if np.any(q > self.qmax) or np.any(q < self.qmin):
            bad = int(np.where((q > self.qmax) | (q < self.qmin))[0][0])
            raise RangeError(
                "批量量化出现越界值",
                {"index": bad, "quantized": int(q[bad]), "scale": self.scale},
            )
        return q

    def to_json(self) -> dict:
        return {"exponent": self.exponent, "int_bits": self.int_bits}

    @staticmethod
    def from_json(data: dict) -> "GridSpec":
        if data is None or "exponent" not in data:
            from .errors import GridError

            raise GridError(
                "快照缺失精度格度参数 grid",
                {"missing_field": "grid.exponent"},
            )
        return GridSpec(
            exponent=int(data["exponent"]),
            int_bits=int(data.get("int_bits", 63)),
        )


def integer_mac(term_pairs: Sequence[Tuple[int, int]]) -> int:
    """纯整数乘加 (multiply-accumulate)：返回 sum(a*b)，全程无浮点。"""
    acc = 0
    for a, b in term_pairs:
        acc += int(a) * int(b)
    return acc


def derive_quantized(
    coef_quantized: Sequence[Tuple[str, int]],
    upstream_quantized: Dict[str, int],
    scale: int,
) -> int:
    """整数系数与上游格点值乘加后除以 scale，回到同一格度（向零截断）。"""
    terms = [(cq, upstream_quantized[key]) for key, cq in coef_quantized]
    return trunc_zero(integer_mac(terms), scale)


def propagated_error_bound(
    coef_quantized: Sequence[Tuple[str, int]],
    upstream_bounds: Dict[str, int],
    upstream_quantized: Dict[str, int],
    scale: int,
) -> Tuple[int, Dict[str, int]]:
    """线性派生项的误差界（整数格表示，保守上界）。

    对 ``y = sum(c_i x_i)``，物理误差来源有三类，统一换算到 y 的整数格：

    1. 上游传播：``ceil( |q(c_i)| * B(x_i) / scale )``
    2. 系数量化（1 格）被上游值放大：``ceil( |q(x_i)| / scale )``
    3. 最终 rescale 的向零截断：常数 1 格

    总界 ``B(y) = 1 + sum_i(贡献1_i + 贡献2_i)``，
    返回 (总误差界, 每个上游键的分项贡献)，全部为整数格。
    """
    contributions: Dict[str, int] = {}
    total = 1  # 最终向零截断的 1 格
    for key, cq in coef_quantized:
        upstream_part = ceil_div(abs(cq) * int(upstream_bounds[key]), scale)
        coef_part = ceil_div(abs(int(upstream_quantized[key])), scale)
        contributions[key] = upstream_part + coef_part
        total += upstream_part + coef_part
    return total, contributions
