"""权限数值口径转换、舍入误差上界与传播链累计。

内部统一使用 fractions.Fraction 做精确运算，对外输出 float，
保证八进制小数（1/8、1/64 ...）在二进制环境下不引入额外浮点误差。
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Iterable

from .errors import AuditError, E_PARSE
from .models import (
    ALL_PERM_BITS,
    REP_INT_MASK,
    REP_OCTAL_STRING,
    ROUND_NEAREST,
    ROUND_TRUNCATE,
)

_OCT_DIGITS = set("01234567")


def _fraction_to_float(fr: Fraction) -> float:
    """Fraction -> float；本引擎涉及的分母均为 2 的幂，转换精确。"""
    return float(fr)


def parse_int_mask(value: Any) -> tuple[int, Fraction]:
    """整数位掩码口径：解析无误差，误差上界为 0。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuditError(f"int_mask 口径要求整数值，得到 {type(value).__name__}", E_PARSE)
    if value < 0:
        raise AuditError("权限位掩码不允许为负数", E_PARSE)
    if value & ~ALL_PERM_BITS:
        raise AuditError(f"权限位 0o{value:o} 含未知位（合法位掩码 0..0o17）", E_PARSE)
    return value, Fraction(0)


def parse_octal_string(value: Any, rounding: str = ROUND_TRUNCATE) -> tuple[int, Fraction, Fraction]:
    """八进制字符串口径。

    支持形如 "751" 与 "751.24" 的表示；小数点后按八进制进位
    （2/8 + 4/64 ...）。返回 (整数掩码, 点估计残差, 误差上界)。

    - truncate：直接截断小数部分，误差上界 = 被丢弃的八进制小数；
    - nearest ：四舍五入到最近整数，误差上界 = 1/2 单位。
    """
    if not isinstance(value, str) or not value:
        raise AuditError("octal_string 口径要求非空字符串", E_PARSE)
    if rounding not in (ROUND_TRUNCATE, ROUND_NEAREST):
        raise AuditError(f"未知舍入口径 {rounding}", E_PARSE)

    body = value
    sign = 1
    if body[0] in "+-":
        if body[0] == "-":
            sign = -1
        body = body[1:]
    if not body:
        raise AuditError(f"非法八进制字符串 {value!r}", E_PARSE)

    int_part, dot, frac_part = body.partition(".")
    if not int_part and not frac_part:
        raise AuditError(f"非法八进制字符串 {value!r}", E_PARSE)
    if int_part and not set(int_part) <= _OCT_DIGITS:
        raise AuditError(f"八进制整数部分含非八进制数字: {value!r}", E_PARSE)
    if dot and (not frac_part or not set(frac_part) <= _OCT_DIGITS):
        raise AuditError(f"八进制小数部分非法: {value!r}", E_PARSE)

    integer_value = int(int_part or "0", 8) * sign
    fraction = Fraction(0)
    for position, digit in enumerate(frac_part, start=1):
        fraction += Fraction(int(digit), 8 ** position)
    fraction *= sign

    exact = Fraction(integer_value) + fraction
    if rounding == ROUND_TRUNCATE:
        mask = integer_value
        residual = fraction  # 截断丢弃的部分
        bound = abs(fraction)
    else:
        rounded = math.floor(exact + Fraction(1, 2)) if exact >= 0 else math.ceil(exact - Fraction(1, 2))
        mask = rounded
        residual = exact - rounded
        bound = Fraction(1, 2) if fraction != 0 else Fraction(0)

    if mask < 0:
        raise AuditError("八进制字符串解析结果为负权限值", E_PARSE)
    if mask & ~ALL_PERM_BITS:
        raise AuditError(f"解析权限位 0o{mask:o} 含未知位（合法位掩码 0..0o17）", E_PARSE)
    return mask, residual, bound


def parse_permission(representation: str, value: Any, rounding: str) -> dict[str, Any]:
    """统一入口：两种口径解析为规范化权限位 + 误差描述。"""
    if representation == REP_INT_MASK:
        mask, bound = parse_int_mask(value)
        return {
            "mask": mask,
            "epsilon": Fraction(0),
            "residual": Fraction(0),
            "bound": Fraction(0),
            "exact": Fraction(mask),
            "representation": representation,
            "rounding": rounding,
        }
    if representation == REP_OCTAL_STRING:
        mask, residual, bound = parse_octal_string(value, rounding)
        exact = mask + residual
        return {
            "mask": mask,
            "epsilon": bound,
            "residual": residual,
            "bound": bound,
            "exact": exact,
            "representation": representation,
            "rounding": rounding,
        }
    raise AuditError(f"权限口径标记只能为 int_mask / octal_string，得到 {representation!r}", E_PARSE)


def accumulate_chain(sources: Iterable[tuple[int, Fraction]]) -> tuple[int, Fraction]:
    """父目录权限 + ACL 继承链的传播累计。

    点估计按位或合并（权限传递为授权叠加），误差上界按各来源绝对值
    线性求和（最坏情形同号叠加，给出保守界）。
    """
    combined_mask = 0
    total_bound = Fraction(0)
    for mask, bound in sources:
        combined_mask |= int(mask)
        total_bound += abs(bound)
    return combined_mask, total_bound


def is_critical(mask: int, epsilon: Fraction, required_bit: int) -> bool:
    """判定累计误差界是否跨越允许/拒绝阈值（按位稳定性）。

    权限数值的不确定性体现为真实值在 [v-eps, v+eps] 内偏离点估计。
    对所需权限的每个位独立判断：该位在点估计中的取值，是否可能因
    邻域内的偏离而翻转；任一位可能翻转即标记临界不确定，
    禁止给出确定越权结论。
    """
    if required_bit == 0 or epsilon <= 0:
        return False
    bit = 1
    while bit <= required_bit:
        if required_bit & bit:
            current = bool(mask & bit)
            # 翻转该位所需的最小数值偏差：位为 0 时需加上 bit，
            # 位为 1 时需减去 bit
            flip_distance = Fraction(bit)
            if flip_distance <= epsilon:
                return True
        bit <<= 1
    return False


def bound_as_float(epsilon: Fraction) -> float:
    return _fraction_to_float(epsilon)


def describe_bound(epsilon: Fraction) -> dict[str, float]:
    """误差界的可 JSON 序列化描述。"""
    return {
        "epsilon": _fraction_to_float(epsilon),
        "epsilon_numerator": epsilon.numerator,
        "epsilon_denominator": epsilon.denominator,
    }
