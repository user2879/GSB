"""确定性哈希组合规则（显式、可复现、与并行交错无关）。

设计要点
--------
1. 所有结构化对象先经 :func:`canonical_json` 稳定序列化：
   - dict 按 key 的 UTF-8 字典序排列；
   - 仅允许 str/int/float/bool/None/list/tuple/dict；
   - 分隔符固定、键顺序固定，禁止依赖 dict 遍历顺序。
2. :func:`combine` 采用「类型标签 + 长度前缀 + 内容」的显式帧格式拼接，
   避免因简单字符串拼接产生歧义；每段用 SHA-256 滚动压缩，最终返回 hex。
3. :func:`normalize_newlines` 把 CRLF/CR 统一为 LF，保证 Windows 与
   POSIX 下同一命令的标准输出参与摘要时字节一致。
这些规则只依赖输入字节与字段名，不依赖线程调度、字典哈希种子或缓存状态。
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterable

_ALLOWED = (str, int, float, bool, type(None), list, tuple, dict)


def normalize_newlines(data: str | bytes) -> str:
    """CRLF 与孤立 CR 统一为 LF，返回文本。"""
    if isinstance(data, bytes):
        data = data.decode("utf-8", errors="replace")
    # 先 CRLF -> LF，再处理残留 CR
    return data.replace("\r\n", "\n").replace("\r", "\n")


def canonical_json(obj: Any) -> str:
    """生成稳定 JSON 文本（键排序、无多余空白、保留非 ASCII）。"""
    _check_type(obj)
    return _encode(obj)


def _check_type(obj: Any) -> None:
    if not isinstance(obj, _ALLOWED):
        raise TypeError(f"不可确定性序列化的类型: {type(obj).__name__}")
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError("dict 键必须为 str 才能稳定排序")
            _check_type(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _check_type(item)


def _encode(obj: Any) -> str:
    # json.dumps(sort_keys=True, ensure_ascii=False, separators 固定)
    # 自身即满足稳定顺序；此处显式封装以便审计。
    import json
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _frame(hasher: "hashlib._Hash", tag: str, payload: bytes) -> None:
    """显式长度前缀帧：tag\\nlen\\npayload。"""
    hasher.update(tag.encode("utf-8"))
    hasher.update(b"\n")
    hasher.update(str(len(payload)).encode("ascii"))
    hasher.update(b"\n")
    hasher.update(payload)


def combine(parts: Iterable[tuple[str, Any]]) -> str:
    """按给定顺序组合若干 (tag, value)，返回 SHA-256 hex。

    value 会被确定性序列化（dict/list）或直接 UTF-8 编码（标量）。
    调用方负责以业务上稳定的顺序传入 parts；集合类输入须先排序。
    """
    hasher = hashlib.sha256()
    for tag, value in parts:
        if isinstance(value, (dict, list, tuple)):
            payload = canonical_json(value).encode("utf-8")
        elif isinstance(value, bytes):
            payload = value
        else:
            payload = canonical_json(value).encode("utf-8")
        _frame(hasher, str(tag), payload)
    return hasher.hexdigest()


def hash_file(path: str, buf_size: int = 65536) -> str:
    """对文件字节计算 SHA-256 hex（读取原始字节，不做换行转换）。"""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(buf_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def stable_digest_of_mapping(mapping: dict) -> str:
    """对一个 dict 求稳定摘要，便于快照/缓存索引整体指纹。"""
    return sha256_hex(canonical_json(mapping))
