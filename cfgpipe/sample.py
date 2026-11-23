"""确定性样本生成与整数定点分组聚合（节点与平台共用同一实现）。

- 样本：numpy RandomState(seed)（MT19999 确定性序列）生成的 int64 整数。
- 聚合：第 i 个样本归入第 (i % groups) 组；每项每组的加和为
  sat_add 累加的 sat(trunc(value_q * sample / scale))，全部整数运算。
- 摘要：规范化 JSON 的 SHA-256，平台与节点据此逐字节比对。
"""

import hashlib
import json

import numpy as np

from .quantize import sat, sat_add, scale_of, tdiv

SAMPLE_LOW = -(2 ** 31)
SAMPLE_HIGH = 2 ** 31 - 1


def generate_samples(seed, size):
    rng = np.random.RandomState(seed)
    raw = rng.randint(SAMPLE_LOW, SAMPLE_HIGH, size=size, dtype=np.int64)
    return [int(x) for x in raw]


def aggregate_view(items, seed, sample_size, scale_exp, groups):
    """items: {key: {"value_q": int, ...}}；返回 {key: [group sums]}。"""
    samples = generate_samples(seed, sample_size)
    scale = scale_of(scale_exp)
    out = {}
    for key in sorted(items):
        value_q = int(items[key]["value_q"])
        group_sums = [0] * groups
        for idx, sample in enumerate(samples):
            term = sat(tdiv(value_q * sample, scale))
            g = idx % groups
            group_sums[g] = sat_add(group_sums[g], term)
        out[key] = group_sums
    return out


def view_document(items, seed, sample_size, scale_exp, groups):
    agg = aggregate_view(items, seed, sample_size, scale_exp, groups)
    return {
        "agg": agg,
        "versions": {k: int(items[k]["version"]) for k in sorted(items)},
        "scale_exp": scale_exp,
        "groups": groups,
        "seed": seed,
        "sample_size": sample_size,
    }


def canonical(doc):
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")


def view_digest(items, seed, sample_size, scale_exp, groups):
    return hashlib.sha256(
        canonical(view_document(items, seed, sample_size, scale_exp, groups))
    ).hexdigest()


def key_digests(items, seed, sample_size, scale_exp, groups):
    """逐键聚合摘要，用于 E_CONVERGE 偏差定位到具体配置键。"""
    agg = aggregate_view(items, seed, sample_size, scale_exp, groups)
    return {
        k: hashlib.sha256(canonical({"key": k, "groups": agg[k]})).hexdigest()
        for k in sorted(agg)
    }


def config_version_of(items):
    return max((int(it["version"]) for it in items.values()), default=0)
