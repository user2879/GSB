"""整数定点运算原语与显式舍入/饱和规则。

规则（在聚合摘要与统计结果中显式声明）：

* 定点表示：值 v = raw / 10**scale，raw 为有符号 64 位整数。
* 截断方向：全部为向零截断（truncation toward zero，Python // 对负数是
  向负无穷，因此这里用 int(a/b) 的向零语义并显式封装）。
* 溢出规则：int64 饱和（saturate），上溢 -> INT64_MAX，下溢 -> INT64_MIN，
  并在运算中逐元素检测，不依赖 NumPy 的静默回绕。
"""

import json

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1

# 界面延迟统计的定点缩放因子（乘数）：定点 raw = 毫秒整数 * LATENCY_SCALE。
# 即统计量保留到 1/1000 毫秒（微秒表达位）；截断方向为向零，单次量化
# 误差上界 = 1 / LATENCY_SCALE 毫秒。
LATENCY_SCALE = 1000


def trunc_div(a, b):
    """向零截断的整数除法（区别于 Python // 的向负无穷截断）。"""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def sat_int64(value):
    """把 Python 任意精度整数饱和到有符号 64 位区间。"""
    if value > INT64_MAX:
        return INT64_MAX
    if value < INT64_MIN:
        return INT64_MIN
    return int(value)


def sat_add(a, b):
    """饱和整数加法（输入输出均为 Python int）。"""
    return sat_int64(int(a) + int(b))


def sat_mul_fixed(raw, factor_raw, factor_scale):
    """定点乘法并饱和：返回 round_toward_zero(raw * factor_raw / 10**factor_scale)。"""
    product = int(raw) * int(factor_raw)
    return sat_int64(trunc_div(product, 10**factor_scale))


def quantize(value, scale):
    """把实值（以整数分子表达的场景由调用方处理）量化为定点 raw。

    本实验台所有输入本身就是整数毫秒，故 value 为 int，scale 表示额外
    放大的小数位；返回 (raw, scale)。
    """
    raw = sat_int64(int(value) * 10**scale)
    return raw, scale


def truncated_mean(values, scale):
    """定点均值：sum 饱和求和（先乘 scale 乘数）后向零截断除以 n。"""
    if not values:
        return 0, scale
    total = 0
    for v in values:
        total = sat_add(total, int(v) * scale)
    return trunc_div(total, len(values)), scale


def nearest_rank_percentile(sorted_values, pct):
    """最近秩（nearest-rank）P95：ceil(pct/100 * n)，1 基秩，确定性且无插值。"""
    if not sorted_values:
        return 0
    n = len(sorted_values)
    rank = -(-(pct * n) // 100)    # ceil(pct*n/100)，标准向上取整
    rank = max(1, min(rank, n))
    return sorted_values[rank - 1]


def latency_stats(values):
    """对整数毫秒观测序列计算均值/P95/最大值的定点表示。

    返回字典含 scale、truncation、quant_error_bound_ms 与各指标 raw。
    被记录的请求不会被剔除：values 由调用方保证包含全部观测（含超预算）。
    """
    ordered = sorted(int(v) for v in values)
    mean_raw, multiplier = truncated_mean(ordered, LATENCY_SCALE)
    p95 = nearest_rank_percentile(ordered, 95)
    maximum = ordered[-1] if ordered else 0
    return {
        "scale": multiplier,
        "truncation": "toward_zero",
        "quant_error_bound_ms": {"raw": 1, "scale": multiplier},
        "mean": {"raw": mean_raw, "scale": multiplier},
        "p95": {"raw": sat_int64(p95 * multiplier), "scale": multiplier},
        "max": {"raw": sat_int64(maximum * multiplier), "scale": multiplier},
        "count": len(ordered),
    }


def canonical_bytes(obj):
    """规范化 JSON 字节：排序键、指定分隔符、无多余空白、不转义非 ASCII。

    聚合摘要的“逐字节一致”比较即基于该编码。
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_hex(obj):
    import hashlib

    return hashlib.sha256(canonical_bytes(obj)).hexdigest()
