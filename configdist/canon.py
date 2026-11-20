"""规范化 JSON、整数定点运算与聚合口径。

定点约定：配置数值与样本数据均为 int64  milli 单位（1000 = 1.0），
所有乘除使用向零截断的整数运算，保证跨进程、跨平台逐字节一致。
"""

import hashlib
import json

SCALE = 1000

PROFILES = ("sum", "mean", "wsum")


def canon(obj):
    """规范化 JSON 字符串（键排序、紧凑分隔符），用于逐字节比较。"""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canon_bytes(obj):
    return canon(obj).encode("utf-8")


def sha256_hex(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tdiv(a, b):
    """向零截断的整数除法（对负数同样确定）。"""
    if b == 0:
        raise ZeroDivisionError("tdiv by zero")
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def fp_mul(a, b):
    """定点乘：a、b 均为 milli 单位，结果为 milli 单位。"""
    return tdiv(a * b, SCALE)


def fp_div(a, b):
    """定点除：a、b 均为 milli 单位，结果为 milli 单位。"""
    return tdiv(a * SCALE, b)


def aggregate(profile, terms):
    """按节点声明的聚合口径对整数项序列聚合。"""
    if profile == "sum":
        return sum(terms)
    if profile == "mean":
        return tdiv(sum(terms), len(terms)) if terms else 0
    if profile == "wsum":
        return sum(t * (i + 1) for i, t in enumerate(terms))
    raise ValueError(f"未知聚合口径: {profile!r}")
