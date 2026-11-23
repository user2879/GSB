"""定点小数与量化档位。

为彻底消除浮点不确定性，全部打分在整数域进行：
- 权重/阈值这类有限小数按 DECIMAL_PLACES 解析为定点整数（“基值”）。
- 量化档位（quant mode）只决定最终定点整数的缩放/舍入方式。

允许枚举：
  decimal_places : {4, 6, 8}
  quant_mode     : {"trunc", "half_even", "half_up"}
  weight_scale   : 10 ** decimal_places（基值单位）
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation, localcontext

ALLOWED_DECIMAL_PLACES = (4, 6, 8)
ALLOWED_QUANT_MODES = ("trunc", "half_even", "half_up")


def validate_decimal_places(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("decimal_places must be int")
    if value not in ALLOWED_DECIMAL_PLACES:
        raise ValueError(
            f"decimal_places {value} not in {ALLOWED_DECIMAL_PLACES}"
        )
    return value


def validate_quant_mode(value: object) -> str:
    if not isinstance(value, str) or value not in ALLOWED_QUANT_MODES:
        raise ValueError(f"quant_mode {value!r} not in {ALLOWED_QUANT_MODES}")
    return value


def to_fixed(value: object, decimal_places: int) -> int:
    """把有限小数/整数解析为定点整数基值（scale = 10**decimal_places）。

    使用 Decimal 做精确字符串解析，禁止 float 的二进制误差进入流水线。
    """
    validate_decimal_places(decimal_places)
    if isinstance(value, bool):  # bool 是 int 子类，显式拒绝
        raise ValueError("boolean is not a valid fixed-point number")
    if isinstance(value, int):
        return value * (10 ** decimal_places)
    if isinstance(value, float):
        # 允许 float 但通过 repr 转 Decimal，保证可重现；非有限值在此拒绝
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("non-finite float cannot be fixed-point encoded")
        dec = Decimal(repr(value))
    elif isinstance(value, str):
        try:
            dec = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError(f"cannot parse decimal {value!r}") from exc
    elif isinstance(value, Decimal):
        dec = value
    else:
        raise ValueError(f"unsupported numeric type {type(value)!r}")
    if not dec.is_finite():
        raise ValueError("non-finite value cannot be fixed-point encoded")
    scale = 10 ** decimal_places
    with localcontext() as ctx:
        ctx.prec = 50
        scaled = dec * scale
        return int(scaled.to_integral_value(rounding="ROUND_HALF_EVEN"))


def requant(fixed_base: int, from_places: int, to_places: int, mode: str) -> int:
    """在不同小数位宽间切换，并按量化档位舍入。

    fixed_base 是以 10**from_places 为单位的定点整数，
    返回以 10**to_places 为单位的定点整数。
    """
    validate_quant_mode(mode)
    if from_places == to_places:
        return fixed_base
    factor = 10 ** abs(to_places - from_places)
    if to_places > from_places:
        return fixed_base * factor
    # 缩小时需要舍入
    return _divide_round(fixed_base, factor, mode)


def _divide_round(num: int, divisor: int, mode: str) -> int:
    q, r = divmod(num, divisor)
    if r == 0:
        return q
    # 统一按正数语义处理（分数权重为正，但保留符号通用处理）
    sign = -1 if num < 0 else 1
    r_abs = abs(r)
    half = divisor // 2
    if mode == "trunc":
        return q
    if mode == "half_up":
        if r_abs * 2 >= divisor:
            return q + sign
        return q
    if mode == "half_even":
        if r_abs * 2 > divisor:
            return q + sign
        if r_abs * 2 == divisor and abs(q) % 2 == 1:
            return q + sign
        return q
    raise ValueError(f"unknown quant mode {mode!r}")


def fixed_to_str(fixed_base: int, decimal_places: int) -> str:
    """审计展示用：把定点整数还原为十进制字符串。"""
    scale = 10 ** decimal_places
    sign = "-" if fixed_base < 0 else ""
    whole, frac = divmod(abs(fixed_base), scale)
    return f"{sign}{whole}.{str(frac).zfill(decimal_places)}"
