"""内容指纹与分层缓存键工具（blake2b，确定性、平台无关）。"""

import hashlib
from typing import Any, Iterable

PREFIX_FRAGMENT = "frag"
PREFIX_WINDOW_FEATURE = "winfeat"
PREFIX_WINDOW_POSTERIOR = "winpost"
PREFIX_COMMAND = "cmd"


def digest_bytes(parts: Iterable[bytes]) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    for part in parts:
        hasher.update(len(part).to_bytes(4, "little"))
        hasher.update(part)
    return hasher.hexdigest()


def _canonical(value: Any) -> bytes:
    if isinstance(value, bool):
        return f"b:{int(value)}".encode()
    if isinstance(value, int):
        return f"i:{value}".encode()
    if isinstance(value, float):
        return f"f:{value!r}".encode()
    if isinstance(value, str):
        return f"s:{value}".encode()
    if value is None:
        return b"n:"
    if isinstance(value, (list, tuple)):
        body = b",".join(_canonical(item) for item in value)
        return b"[" + body + b"]"
    if isinstance(value, dict):
        body = b",".join(
            _canonical(k) + b"=" + _canonical(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        )
        return b"{" + body + b"}"
    raise TypeError(f"无法规范化类型: {type(value)!r}")


def content_fingerprint(value: Any) -> str:
    """对任意可 JSON 规范化的内容计算 16 位十六进制指纹。"""
    return hashlib.blake2b(_canonical(value), digest_size=16).hexdigest()


def cache_key(prefix: str, *upstream: str) -> str:
    """由上游内容指纹、参数指纹、位宽/档位派生的分层缓存键。"""
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(prefix.encode())
    for item in upstream:
        hasher.update(b"|")
        hasher.update(str(item).encode())
    return f"{prefix}:{hasher.hexdigest()}"


def stable_key_for_window(fragment_seqs: Iterable[int], window_index: int) -> tuple:
    """窗任务稳定排序键：片段序号集合（最小、最大、元组）+ 窗序号。"""
    seqs = tuple(sorted(fragment_seqs))
    return (seqs[0] if seqs else -1, seqs[-1] if seqs else -1, seqs, window_index)
