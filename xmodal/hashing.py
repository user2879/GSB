"""确定性哈希工具。

全部使用 sha256，避免 Python 内置 hash() 的随机盐值；JSON 规范化采用
排序键、无空白、固定分隔符，保证跨进程、跨机器的稳定指纹。
"""

import hashlib
import json

import numpy as np


def digest_bytes(data, prefix=""):
    h = hashlib.sha256()
    if prefix:
        h.update(prefix.encode("utf-8"))
        h.update(b"\x00")
    h.update(data)
    return h.hexdigest()


def canonical_json(obj):
    """把可 JSON 序列化对象转为规范化字节串。"""
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest_json(obj, prefix=""):
    return digest_bytes(canonical_json(obj), prefix=prefix)


def digest_text(text, prefix=""):
    return digest_bytes(text.encode("utf-8"), prefix=prefix)


def digest_array(arr, prefix=""):
    """对 NumPy 数组按 dtype 与原始字节取指纹。"""
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    if prefix:
        h.update(prefix.encode("utf-8"))
        h.update(b"\x00")
    h.update(str(a.dtype).encode("ascii"))
    h.update(b"|")
    h.update(",".join(str(d) for d in a.shape).encode("ascii"))
    h.update(b"|")
    h.update(a.tobytes())
    return h.hexdigest()


def combine(parts, prefix=""):
    """组合多个十六进制摘要或字符串，生成稳定摘要。"""
    h = hashlib.sha256()
    if prefix:
        h.update(prefix.encode("utf-8"))
        h.update(b"\x00")
    for part in parts:
        token = part if isinstance(part, str) else str(part)
        h.update(token.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def cache_key(stage, parts):
    """生成阶段缓存键，parts 为上游内容指纹、配置指纹与版本号。"""
    return combine(parts, prefix="cache:" + stage)[:32]


def stable_sort_key(sample, stage):
    """每个样本任务携带的稳定排序键（样本名 + 阶段名派生）。"""
    return digest_text("{}::{}".format(stage, sample), prefix="sortkey")[:16]


def short_hex(data, n=12):
    return digest_bytes(data)[:n]
