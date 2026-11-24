"""确定性特征向量：词项经 FNV-1a 哈希进入定长桶向量。

特征维度 feature_dim 决定向量长度，hash_buckets 决定取模桶数。
全程使用整数，保证跨机器逐字段一致；NumPy 仅承担确定性向量运算。
"""

import numpy as np

from .fingerprint import fnv1a64


def term_bucket(term: str, hash_buckets: int) -> int:
    digest = int(fnv1a64(term.encode("utf-8")), 16)
    return digest % hash_buckets


def build_feature_vector(tokens, feature_dim, hash_buckets):
    vector = np.zeros(feature_dim, dtype=np.int64)
    for token in tokens:
        bucket = term_bucket(token, hash_buckets)
        index = bucket % feature_dim
        vector[index] += 1
    return vector


def pair_feature_vector(head, tail, tokens, feature_dim, hash_buckets):
    """实体对特征：头/尾区间词项 + 区间词项距离衰减的整数向量。"""
    vector = build_feature_vector(tokens[head.start:tail.end], feature_dim, hash_buckets)
    gap = max(0, tail.start - head.end)
    gap_bucket = term_bucket(f"GAP:{gap}", hash_buckets) % feature_dim
    vector[gap_bucket] += 1
    return vector


def feature_points_for_vector(feature_dim):
    return feature_dim
