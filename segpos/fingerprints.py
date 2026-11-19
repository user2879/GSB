"""内容指纹与缓存键。

所有哈希均为确定性 SHA-256，规范化序列化保证键序无关、跨平台一致。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


def canonical_bytes(obj: Any) -> bytes:
    """键序稳定的 JSON 规范化字节。"""

    return json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def canonical_hash(obj: Any) -> str:
    return sha256_hex(canonical_bytes(obj))


def text_fingerprint(text: str) -> str:
    return sha256_hex(text)


def hit_words_fingerprint(hit_words: Iterable[str]) -> str:
    return sha256_hex(canonical_bytes(sorted(set(hit_words))))


def cache_key(text: str, hit_words: Iterable[str], prob_table_version: int) -> str:
    """句子级缓存键：文本指纹 + 命中词条集合指纹 + 概率表版本。"""

    payload = {
        "text": text_fingerprint(text),
        "hit_words": hit_words_fingerprint(hit_words),
        "prob_table_version": prob_table_version,
    }
    return sha256_hex(canonical_bytes(payload))


def stable_sort_key(sentence_id: str, stage: str) -> str:
    """由句子 ID 与阶段名派生的稳定排序键。"""

    return f"{sentence_id}\u001f{stage}"
