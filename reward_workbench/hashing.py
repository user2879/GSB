"""确定性子词切分、稳定哈希与内容指纹。

不使用任何运行时随机源（无 random、无 PYTHONHASHSEED 依赖），
全部哈希走 hashlib（SHA-256），跨平台/跨次运行逐整数一致。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .vocab import all_in_vocab


def stable_hash_int(text: str) -> int:
    """字符串 -> 64 位无符号整数（确定性）。"""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def sha256_hex(parts: list[str]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(b"\x1f")  # 字段分隔，避免拼接歧义
        h.update(part.encode("utf-8"))
    return h.hexdigest()


def tokenize(text: str, tokenizer_version: str, normalize: bool) -> list[str]:
    """子词切分。

    v1: 空白切词，可选小写归一化。
    v2: 在 v1 基础上做确定性“子词”切分——对长度 > 4 的词按固定步长切为
        前缀 + 双字子词（如 faithful -> faith, fa, ai, it, th, hf, fu, ul），
        同样可重现且只依赖词自身。
    """
    if normalize:
        text = text.lower().strip()
    base = text.split()
    if tokenizer_version == "v1":
        return base
    if tokenizer_version == "v2":
        out: list[str] = []
        for word in base:
            if len(word) <= 4:
                out.append(word)
                continue
            out.append(word[:4])
            for i in range(len(word) - 1):
                out.append(word[i:i + 2])
        return out
    raise ValueError(f"unsupported tokenizer_version {tokenizer_version!r}")


@dataclass(frozen=True)
class BucketVector:
    """桶计数向量。同时保存非零桶（稀疏）以便审计与缓存。"""

    bucket_count: int
    counts: tuple[tuple[int, int], ...]  # (bucket_id, count)，按 bucket 排序
    digest: str                          # 向量内容指纹

    def to_dict(self) -> dict:
        return {
            "bucket_count": self.bucket_count,
            "nonzero": [{"bucket": b, "count": c} for b, c in self.counts],
            "digest": self.digest,
        }

    def l1(self) -> int:
        return sum(c for _, c in self.counts)


def hash_to_bucket(token: str, bucket_count: int) -> int:
    return stable_hash_int(token) % bucket_count


def bucket_vector(text: str, tokenizer_version: str, normalize: bool,
                  bucket_count: int) -> BucketVector:
    tokens = tokenize(text, tokenizer_version, normalize)
    counts: dict[int, int] = {}
    for tok in tokens:
        bid = hash_to_bucket(tok, bucket_count)
        counts[bid] = counts.get(bid, 0) + 1
    sparse = tuple(sorted(counts.items()))
    digest = sha256_hex(
        [tokenizer_version, str(normalize), str(bucket_count)]
        + [f"{b}:{c}" for b, c in sparse]
    )
    return BucketVector(bucket_count=bucket_count, counts=sparse, digest=digest)


def text_tokens_in_vocab(text: str, normalize: bool) -> bool:
    """校验文本词项是否落在受控词表（归一化后判定）。"""
    tokens = tokenize(text, "v1", normalize)  # 词表按整词判定
    return len(tokens) > 0 and all_in_vocab(tokens)
