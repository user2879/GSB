"""确定性公共工具：规范化哈希、定点舍入、固定规则分位数、稳定排序键。"""

import hashlib
import json
import math

ROUND_DIGITS = 9


def canonical_json(obj):
    """键排序、固定分隔符的规范化 JSON 文本，供哈希与快照使用。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text):
    if isinstance(text, bytes):
        return hashlib.sha256(text).hexdigest()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(prefix, obj):
    """对任意可 JSON 化对象计算带前缀的内容指纹。"""
    return prefix + "_" + sha256_text(canonical_json(obj))[:16]


def stable_key(audio_id, segment_index, stage):
    """由段标识与阶段名派生的稳定任务排序键。

    形如 ``a001|s0003|feature``，字典序与音频、段序号、阶段顺序一致。
    """
    return "{}|s{:06d}|{}".format(audio_id, int(segment_index), stage)


def round_fixed(value, digits=ROUND_DIGITS):
    """避免浮点 -0 与平台差异的确定性四舍五入（半数向上）。"""
    if value == 0:
        return 0.0
    factor = 10 ** digits
    scaled = value * factor
    rounded = math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5)
    result = rounded / factor
    return 0.0 if result == 0 else result


def round_half_up(value):
    """定点聚合用确定性舍入：半值向远离零方向进位。"""
    if value >= 0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


def quantile(values, q, method="linear"):
    """固定规则分位数。

    ``values`` 会被复制后升序排序；并列值不影响位置计算。
    method 取值与 numpy 命名保持一致：linear / lower / higher /
    nearest / midpoint，保证跨环境逐字段一致。
    """
    if not values:
        raise ValueError("quantile requires at least one value")
    data = sorted(float(v) for v in values)
    n = len(data)
    if n == 1:
        return data[0]
    if not 0.0 <= q <= 1.0:
        raise ValueError("quantile q out of [0,1]")
    pos = q * (n - 1)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return data[lower]
    frac = pos - lower
    if method == "linear":
        return data[lower] * (1.0 - frac) + data[upper] * frac
    if method == "lower":
        return data[lower]
    if method == "higher":
        return data[upper]
    if method == "midpoint":
        return (data[lower] + data[upper]) / 2.0
    if method == "nearest":
        return data[lower] if frac < 0.5 else data[upper]
    raise ValueError("unknown quantile method: {}".format(method))


def is_finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def diff_summary(before, after):
    """两个等长数值向量的确定性差分摘要。"""
    if before is None:
        return {"status": "new", "changed_dims": list(range(len(after)))}
    changed = [i for i, (b, a) in enumerate(zip(before, after)) if b != a]
    delta = [round_fixed(a - b) for b, a in zip(before, after)]
    return {
        "status": "changed" if changed else "unchanged",
        "changed_dims": changed,
        "max_abs_delta": max((abs(d) for d in delta), default=0.0),
        "delta": delta,
    }
