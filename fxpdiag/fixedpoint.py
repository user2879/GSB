"""定点整数引擎。

所有中间量都以整数定点数 ``q``（真值 = ``q * 2**-frac_bits``）参与运算，
每次量化/右移都会记录舍入方向；位宽变化会真实改变饱和上下界与整数中间值。
仅依赖标准库（``math``）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .errors import ErrorCode, PipelineError

ALLOWED_BIT_WIDTHS = (8, 16, 32)
ALLOWED_FRAC_BITS = range(1, 16)
TIER_GUARD_BITS = {"coarse": 0, "standard": 1, "fine": 2}


class RoundingMode(str, Enum):
    HALF_EVEN = "HALF_EVEN"
    HALF_UP = "HALF_UP"
    TRUNC = "TRUNC"
    FLOOR = "FLOOR"


def signed_bounds(bit_width: int, guard_bits: int) -> tuple[int, int]:
    """返回带符号整数饱和上下界（guard 位为量化档位预留的额外精度）。"""

    mag = bit_width - 1 + guard_bits
    return -(1 << mag), (1 << mag) - 1


@dataclass(frozen=True)
class QuantSpec:
    """量化档位不可变描述，参与缓存键派生。"""

    bit_width: int
    frac_bits: int
    tier: str
    rounding_mode: str

    def key(self) -> str:
        return f"bw{self.bit_width}:fb{self.frac_bits}:{self.tier}:{self.rounding_mode}"


@dataclass
class RoundingEvent:
    """单次量化/舍入事件。``direction`` 取 EXACT/UP/DOWN/SAT_UP/SAT_DOWN。"""

    op: str
    index: int
    direction: str
    before: float
    after_q: int
    step: float


class Quantizer:
    """单档位定点量化器（按行实例化，自带事件轨迹）。"""

    def __init__(self, spec: QuantSpec, trace: list[RoundingEvent] | None = None):
        if spec.bit_width not in ALLOWED_BIT_WIDTHS:
            raise PipelineError(
                ErrorCode.E_BITWIDTH_ENUM,
                f"位宽 {spec.bit_width} 不在允许枚举 {ALLOWED_BIT_WIDTHS} 内",
                {"bit_width": spec.bit_width},
            )
        if spec.frac_bits not in ALLOWED_FRAC_BITS:
            raise PipelineError(
                ErrorCode.E_SHAPE_INVALID,
                f"定点小数位宽 {spec.frac_bits} 超出允许范围 1..15",
                {"frac_bits": spec.frac_bits},
            )
        if spec.tier not in TIER_GUARD_BITS:
            raise PipelineError(
                ErrorCode.E_SHAPE_INVALID,
                f"未知量化档位 {spec.tier}",
                {"tier": spec.tier},
            )
        self.spec = spec
        self.frac_bits = spec.frac_bits
        self.guard_bits = TIER_GUARD_BITS[spec.tier]
        self.qmin, self.qmax = signed_bounds(spec.bit_width, self.guard_bits)
        self.step = 2.0 ** (-spec.frac_bits)
        self.rounding_mode = RoundingMode(spec.rounding_mode)
        self.trace = trace if trace is not None else []
        self._seq = 0

    # ------------------------------------------------------------------ #
    # 基础量化
    # ------------------------------------------------------------------ #
    def _round_scaled(self, scaled: float) -> int:
        mode = self.rounding_mode
        if mode is RoundingMode.TRUNC:
            return int(math.copysign(math.floor(abs(scaled)), scaled))
        if mode is RoundingMode.FLOOR:
            return math.floor(scaled)
        if mode is RoundingMode.HALF_UP:
            return int(math.copysign(math.floor(abs(scaled) + 0.5), scaled))
        # HALF_EVEN（银行家舍入）
        return int(round(scaled))

    def quantize(self, x: float, op: str = "quantize", index: int | None = None) -> int:
        """浮点 -> 定点整数，含舍入方向与饱和事件记录。"""

        if not math.isfinite(x):
            raise PipelineError(
                ErrorCode.E_NON_FINITE, f"非有限值无法量化: {x}", {"op": op}
            )
        scaled = x * (1 << self.frac_bits)
        q = self._round_scaled(scaled)
        direction = "EXACT" if q == scaled else ("UP" if q > scaled else "DOWN")
        if q > self.qmax:
            direction = "SAT_UP"
            q = self.qmax
        elif q < self.qmin:
            direction = "SAT_DOWN"
            q = self.qmin
        if index is None:
            index = self._seq
            self._seq += 1
        self.trace.append(RoundingEvent(op, index, direction, x, q, self.step))
        return q

    def dequant(self, q: int) -> float:
        return q * self.step

    def requant(self, q: int, src_frac: int, op: str, index: int) -> int:
        """把另一小数位宽下的整数重新量化到本档位。"""

        delta = src_frac - self.frac_bits
        if delta == 0:
            out = q
        elif delta > 0:
            num = q
            for _ in range(delta):
                if num >= 0:
                    num = (num + 1) >> 1 if self.rounding_mode is RoundingMode.HALF_UP else num >> 1
                else:
                    num = -(((-num) + 1) >> 1) if self.rounding_mode is RoundingMode.HALF_UP else -((-num) >> 1)
            out = num
        else:
            out = q << (-delta)
        out = self._saturate(out)
        before = q * 2.0 ** (-src_frac)
        direction = "EXACT" if self.dequant(out) == before else (
            "UP" if self.dequant(out) > before else "DOWN"
        )
        self.trace.append(RoundingEvent(op, index, direction, before, out, self.step))
        return out

    def _saturate(self, q: int) -> int:
        if q > self.qmax:
            return self.qmax
        if q < self.qmin:
            return self.qmin
        return q

    # ------------------------------------------------------------------ #
    # 定点四则运算（结果重新量化回本档位，保证位宽真实生效）
    # ------------------------------------------------------------------ #
    def add(self, a: int, b: int, op: str, index: int) -> int:
        out = self._saturate(a + b)
        self.trace.append(
            RoundingEvent(op, index, "EXACT" if out == a + b else "SAT_UP",
                          self.dequant(a) + self.dequant(b), out, self.step)
        )
        return out

    def mul(self, a: int, b: int, op: str, index: int) -> int:
        """定点乘法：真值 a*b*step，结果按舍入模式回到本档位。"""

        raw = a * b
        if self.frac_bits == 0:
            q = raw
        else:
            if self.rounding_mode is RoundingMode.TRUNC:
                q = raw >> self.frac_bits if raw >= 0 else -((-raw) >> self.frac_bits)
            elif self.rounding_mode is RoundingMode.FLOOR:
                q = raw >> self.frac_bits
            else:
                half = 1 << (self.frac_bits - 1)
                q = (raw + half) >> self.frac_bits
                if self.rounding_mode is RoundingMode.HALF_EVEN and raw & ((1 << self.frac_bits) - 1) == half:
                    q -= q & 1
        q_sat = self._saturate(q)
        before = self.dequant(a) * self.dequant(b)
        direction = "EXACT" if self.dequant(q_sat) == before else (
            "UP" if self.dequant(q_sat) > before else "DOWN"
        )
        self.trace.append(RoundingEvent(op, index, direction, before, q_sat, self.step))
        return q_sat

    def div(self, a: int, b: int, op: str, index: int) -> int:
        """定点除法 a/b，b 为同档位整数。"""

        num = a << self.frac_bits
        if self.rounding_mode is RoundingMode.TRUNC:
            q = abs(num) // abs(b)
            q = q if (num >= 0) == (b > 0) else -q
        elif self.rounding_mode is RoundingMode.FLOOR:
            q = num // b
        else:
            q = (2 * num + b) // (2 * b) if b > 0 else -((2 * num + (-b)) // (2 * (-b)))
            if (num < 0) != (b < 0):
                q = -abs(q)
        q_sat = self._saturate(q)
        before = self.dequant(a) / self.dequant(b)
        direction = "EXACT" if self.dequant(q_sat) == before else (
            "UP" if self.dequant(q_sat) > before else "DOWN"
        )
        self.trace.append(RoundingEvent(op, index, direction, before, q_sat, self.step))
        return q_sat


def rsqrt_int(q: int, frac_bits: int, iterations: int = 32) -> int:
    """纯整数 rsqrt：输入 q（2^-f 定点），返回 rsqrt 真值的同档定点整数。

    定点比例：rsqrt(q/2^f) = r/2^f 等价于 r^2 * q = 2^(3f)。
    先用位长给出数量级整数初值，再走固定次数的 rsqrt 整数牛顿迭代
    ``r <- r*(3*S - q*r^2)/(2*S)``（S=2^(3f)），中间乘积为任意精度整数；
    退化（q<=0）由调用方处理。
    """

    if q <= 0:
        raise ValueError("rsqrt_int 仅接受正输入，退化行须由调用方处理")
    scale3 = 1 << (3 * frac_bits)
    ratio = scale3 // q
    r = max(1, 1 << max(0, (ratio.bit_length() - 1) // 2))
    for _ in range(iterations):
        r2_q = q * r * r
        numerator = r * (3 * scale3 - r2_q)
        r_next = numerator // (2 * scale3)
        if r_next <= 0 or r_next == r:
            break
        r = r_next
    return r


def exp_fixed(q: int, quantizer: Quantizer, index: int) -> int:
    """定点指数近似：范围归约 + 四阶泰勒多项式。

    返回 exp(q * step) 在同一档位下的定点整数。
    近似形式 ``EXP_RANGE_REDUCTION_TAYLOR4`` 会写入产物元数据。
    """

    f = quantizer.frac_bits
    one = 1 << f
    x = quantizer.dequant(q)
    # 范围归约：x = k*ln2 + r，|r| <= ln2/2
    ln2 = 0.6931471805599453
    k = math.floor(x / ln2 + 0.5)
    r = x - k * ln2
    rq = quantizer.quantize(r, "exp.reduce", index)
    # 泰勒四阶：1 + r + r^2/2 + r^3/6 + r^4/24（全程定点，逐项重量化）
    poly = one
    term = one
    for power in range(1, 5):
        term = quantizer.mul(term, rq, f"exp.taylor{power}", index)
        term = quantizer.div(term, quantizer.quantize(float(power), "exp.coef", index),
                             f"exp.taylor{power}.div", index)
        poly = quantizer.add(poly, term, f"exp.taylor{power}.sum", index)
    # 2^k：k 通常为负整数，用 1/2^(-k) 的定点右移实现
    if k >= 0:
        scale = one << k
    else:
        scale = one >> (-k)
    out = quantizer.mul(poly, scale, "exp.scale2k", index)
    return out
