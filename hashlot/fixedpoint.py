"""整数定点运算与饱和规则。

会话缩放因子 ``scale`` 为正整数（10 的幂），所有浮点输入先量化为 int64
定点整数：``q = trunc(x * scale)``，截断方向为“向零截断”。

- 单值量化误差上界：|x*scale - q| < 1，还原后 |x - q/scale| < 1/scale。
- 求和采用 int64 饱和加法，溢出钳制到 INT64_MIN/INT64_MAX。
- n 个值聚合的绝对量化误差上界：n / scale（还原后），定点域为 n。
- 均值采用定点除法（向零截断），误差上界额外加 1/scale。
"""

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
SUPPORTED_SCALES = (1, 10, 100, 1000, 10000, 100000, 1000000)


def validate_scale(scale) -> int:
    if not isinstance(scale, int) or isinstance(scale, bool):
        raise ValueError("scale 必须为整数")
    if scale not in SUPPORTED_SCALES:
        raise ValueError(f"scale 必须属于受支持集合 {SUPPORTED_SCALES}")
    return scale


def quantize(value: float, scale: int) -> int:
    """浮点值按向零截断量化为 int64 定点整数。"""
    if value is None:
        raise ValueError("空值不得参与量化")
    scaled = float(value) * scale
    truncated = int(scaled)  # int() 对浮点即向零截断
    if truncated > INT64_MAX:
        return INT64_MAX
    if truncated < INT64_MIN:
        return INT64_MIN
    return truncated


def sat_add(a: int, b: int) -> int:
    """int64 饱和加法。"""
    total = int(a) + int(b)
    if total > INT64_MAX:
        return INT64_MAX
    if total < INT64_MIN:
        return INT64_MIN
    return total


def sat_div(numerator: int, divisor: int) -> int:
    """定点整数除法，向零截断；除数为零时返回 None。"""
    if divisor == 0:
        return None
    q = abs(numerator) // abs(divisor)
    if (numerator < 0) != (divisor < 0):
        q = -q
    if q > INT64_MAX:
        return INT64_MAX
    if q < INT64_MIN:
        return INT64_MIN
    return q


def quantization_error_bound(n: int, scale: int) -> str:
    """返回 n 个输入聚合后还原域的绝对误差上界（字符串保留精确分数）。"""
    return f"{n + 1}/{scale}"


def value_error_bound(scale: int) -> str:
    return f"1/{scale}"
