"""Fixed-point quantization口径.

口径声明（显式规则）：
- 缩放因子为 10 的整数幂：scale = 10 ** scale_exp，scale_exp 允许 0..6。
- 量化：q(x) = trunc_toward_zero(x * scale)，截断方向恒为向零截断。
- 溢出饱和规则：所有定点运算结果饱和到 int64 区间
  [-(2**63), 2**63 - 1]，即 saturate(v) = min(max(v, INT64_MIN), INT64_MAX)。
- 派生值、分组聚合与阈值比较全部以 Python 整数完成，不经过浮点。
- 耗时以单调时钟毫秒整数表示（time.monotonic_ns() // 1_000_000）。
"""

from decimal import Decimal, ROUND_DOWN

from .errors import SpecError

MIN_SCALE_EXP = 0
MAX_SCALE_EXP = 6
INT64_MIN = -(2 ** 63)
INT64_MAX = 2 ** 63 - 1


def check_scale_exp(scale_exp):
    if isinstance(scale_exp, bool) or not isinstance(scale_exp, int):
        raise SpecError("scale exponent must be an integer", field="scale_exp")
    if not (MIN_SCALE_EXP <= scale_exp <= MAX_SCALE_EXP):
        raise SpecError(
            "scale exponent must be within %d..%d" % (MIN_SCALE_EXP, MAX_SCALE_EXP),
            field="scale_exp",
        )
    return scale_exp


def scale_of(scale_exp):
    check_scale_exp(scale_exp)
    return 10 ** scale_exp


def sat(value):
    """Saturate an integer to the int64 range (溢出饱和规则)."""
    if value < INT64_MIN:
        return INT64_MIN
    if value > INT64_MAX:
        return INT64_MAX
    return value


def sat_add(a, b):
    return sat(a + b)


def quantize(value, scale_exp):
    """Quantize a numeric value to a fixed-point integer.

    截断方向：向零截断（ROUND_DOWN）。结果按 int64 饱和。
    """
    scale = scale_of(scale_exp)
    if isinstance(value, bool):
        raise SpecError("boolean is not quantizable")
    try:
        dec = Decimal(str(value))
    except Exception as exc:  # noqa: BLE001
        raise SpecError("value is not quantizable: %r" % (value,)) from exc
    if not dec.is_finite():
        raise SpecError("value is not finite: %r" % (value,))
    scaled = dec * scale
    truncated = int(scaled.to_integral_value(rounding=ROUND_DOWN))
    return sat(truncated)


def tdiv(a, b):
    """Truncating integer division toward zero; b must be positive."""
    if b <= 0:
        raise ValueError("tdiv divisor must be positive")
    q = abs(a) // b
    return q if a >= 0 else -q


def derived_value(coeff_q, parent_q, scale_exp):
    """Derived fixed-point value: sat(trunc(coeff_q * parent_q / scale))."""
    return sat(tdiv(coeff_q * parent_q, scale_of(scale_exp)))


def single_value_error_ulp():
    """单值量化误差上界：向零截断 => 误差严格小于 1 ulp，上界取 1 ulp。"""
    return 1


def chain_error_bounds(item_keys_topo, derived_from, coeff_q_map, scale_exp):
    """派生链累计误差上界（单位：ulp），按拓扑序计算。

    返回 {key: (trunc_part, derived_part, chain_total)}：
    - trunc_part：本项自身量化截断贡献，恒为 1 ulp。
    - derived_part：派生放大贡献 = trunc(|coeff_q| * parent_chain_total / scale)。
    - chain_total = trunc_part + derived_part（单值链累计上界）。
    """
    scale = scale_of(scale_exp)
    bounds = {}
    for key in item_keys_topo:
        parent = derived_from.get(key) or None
        trunc_part = single_value_error_ulp()
        if parent is None:
            derived_part = 0
        else:
            parent_total = bounds[parent][2]
            derived_part = tdiv(abs(coeff_q_map[key]) * parent_total, scale)
        bounds[key] = (trunc_part, derived_part, trunc_part + derived_part)
    return bounds


def aggregation_part(sample_sizes, groups):
    """聚合合成贡献上界（ulp）：每组最多 ceil(size/groups) 个加项，
    每个加项在 tdiv 处引入不超过 1 ulp 的截断。取所有订阅节点最大值。"""
    best = 0
    for size in sample_sizes:
        part = -(-size // groups)  # ceil division
        if part > best:
            best = part
    return best
