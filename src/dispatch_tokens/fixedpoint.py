"""整数定点口径。

系统内不允许使用浮点参与令牌水位裁决。所有定点量统一约定：

* 逻辑令牌单位 ``L`` 以缩放因子 ``S``（10 的 0..3 次幂）量化为整数单位 ``Q``，
  即 ``q = L * S``；反向换算采用向零截断（Python ``//`` 对非负数即向下截断）。
* 所有除法为整数除，截断方向显式声明为 **向零截断（TRUNC_TOWARD_ZERO）**，
  参与运算的量均为非负整数，因此与向下截断等价。
* 溢出（含修订换因子后的再量化、补充夹取）采用 **饱和规则（SATURATE）**：
  超出上界取上界，低于下界 0 取 0，不做回绕。

在线补算与重启重放必须导入本模块的同一组函数，禁止各写一份算术，
以保证逐位一致。
"""
from __future__ import annotations

TRUNC_TOWARD_ZERO = "TRUNC_TOWARD_ZERO"
SATURATE = "SATURATE"
ALLOWED_SCALES = (1, 10, 100, 1000)


def validate_scale(scale: int) -> None:
    if not isinstance(scale, int) or isinstance(scale, bool) or scale not in ALLOWED_SCALES:
        raise ValueError(f"scale must be one of {ALLOWED_SCALES}, got {scale!r}")


def quantize(logical: int, scale: int) -> int:
    """逻辑令牌 -> 定点整数（精确乘法，无截断误差）。"""
    validate_scale(scale)
    if logical < 0:
        raise ValueError("logical token amount must be non-negative")
    return logical * scale


def dequantize(quantized: int, scale: int) -> int:
    """定点整数 -> 逻辑令牌，向零截断。"""
    validate_scale(scale)
    if quantized < 0:
        raise ValueError("quantized amount must be non-negative")
    return quantized // scale


def rescale(quantized: int, old_scale: int, new_scale: int) -> int:
    """修订边界上的再量化：先反量化再量化，截断误差记入量化误差输出。

    因为缩放因子都是 10 的幂，只有当 ``new_scale < old_scale`` 时才会产生
    截断，截断掉的余数由调用方计入误差统计。
    """
    validate_scale(old_scale)
    validate_scale(new_scale)
    if quantized < 0:
        raise ValueError("quantized amount must be non-negative")
    if new_scale == old_scale:
        return quantized
    logical = quantized // old_scale
    return logical * new_scale


def trunc_div(numerator: int, denominator: int) -> int:
    """显式向零截断的整数除法（输入均为非负）。"""
    if denominator <= 0:
        raise ZeroDivisionError("denominator must be positive")
    if numerator < 0:
        raise ValueError("numerator must be non-negative")
    return numerator // denominator


def saturate(value: int, lower: int = 0, upper: int | None = None) -> int:
    """饱和夹取。"""
    if value < lower:
        return lower
    if upper is not None and value > upper:
        return upper
    return value


def ratio_floor(numerator: int, denominator: int) -> int:
    """定点水位比例的整数分量（分子分母同量纲时为 0..1 的万分比口径由调用方乘基）。"""
    return trunc_div(numerator, denominator)


def ratio_basis(scale: int) -> int:
    """水位比例的整数基：以 ``scale`` 为满刻度，比例 = balance*scale//capacity。"""
    validate_scale(scale)
    return scale
