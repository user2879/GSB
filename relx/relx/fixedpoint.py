"""定点整数域运算与量化档位。

量化档位 q 表示每个权重/阈值保留的小数位（2/4/8），定点位宽 width
决定有符号整数饱和范围（8/16/32）。量化使用整数四舍五入（半数向上，
对负值采用对称舍入），完全在整数域完成，不引入浮点缓存键。
"""

from .errors import E_CONFIG_INVALID, RelxError

ALLOWED_WIDTHS = (8, 16, 32)
ALLOWED_Q_LEVELS = (2, 4, 8)


def integer_bounds(width: int):
    if width not in ALLOWED_WIDTHS:
        raise RelxError(E_CONFIG_INVALID, f"定点位宽不在允许枚举内: {width}")
    upper = (1 << (width - 1)) - 1
    lower = -(1 << (width - 1))
    return lower, upper


def saturate(value: int, width: int) -> int:
    lower, upper = integer_bounds(width)
    if value > upper:
        return upper
    if value < lower:
        return lower
    return value


def quantize(decimal_value, width: int, q: int) -> int:
    """把十进制小数按 q 位小数量化，并按位宽饱和。"""
    if q not in ALLOWED_Q_LEVELS:
        raise RelxError(E_CONFIG_INVALID, f"量化档位不在允许枚举内: {q}")
    scaled = float(decimal_value) * (10 ** q)
    sign = -1 if scaled < 0 else 1
    rounded = sign * int(abs(scaled) + 0.5)
    return saturate(rounded, width)


def half_up_divide(numerator: int, denominator: int) -> int:
    """带符号整数四舍五入除法，分母必须为正。"""
    if denominator <= 0:
        raise ValueError("分母必须为正整数")
    sign = -1 if numerator < 0 else 1
    return sign * ((abs(numerator) + denominator // 2) // denominator)
