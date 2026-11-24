"""基于内容指纹的签名分量级精确失效判定。

四级缓存键的派生口径：

* feature  : 样本内容指纹(图像/断言/区域) + 特征侧配置指纹
* signature: feature 键 + 投影种子版本 + 签名位宽（投影矩阵派生口径）
* bucket   : 逐位签名键集合 + 分桶位数
* cluster  : 全量桶归属集合 + 相似度阈值 + 签名位宽

同时维护两类引用关系：每个签名位引用的联合特征分量集合、
每个桶引用的签名位集合，供逐条复用清单给出“受影响分量编号”。
"""
from __future__ import annotations

from typing import Any

from .hashing import derive_key, fingerprint


def sample_content_fingerprint(bundle_files: dict[str, str]) -> str:
    """对样本三件套的原始字节做内容指纹（图像/断言/区域）。"""
    import hashlib

    digest = hashlib.sha256()
    for name in sorted(bundle_files):
        digest.update(name.encode("utf-8"))
        with open(bundle_files[name], "rb") as fh:
            digest.update(fh.read())
    return digest.hexdigest()


def feature_cache_key(content_fp: str, feature_config_fp: str) -> str:
    return derive_key(["feature", content_fp, feature_config_fp])


def signature_cache_key(feature_key: str, seed_version: str, signature_bits: int) -> str:
    return derive_key(["signature", feature_key, seed_version, str(signature_bits)])


def bucket_cache_key(signature_key: str, bucket_bits: int) -> str:
    return derive_key(["bucket", signature_key, str(bucket_bits)])


def cluster_cache_key(bucket_keys: dict[str, list[str]], similarity_permille: int, signature_bits: int) -> str:
    material = {
        "buckets": {sid: list(keys) for sid, keys in sorted(bucket_keys.items())},
        "similarity_permille": similarity_permille,
        "signature_bits": signature_bits,
    }
    return derive_key(["cluster", fingerprint(material)])


def affected_components_for_sample(layout, changed_modality: str) -> list[int]:
    """返回样本内容变化时受影响的联合特征分量编号。"""
    lo, hi = layout.image_component_range
    image_components = list(range(lo, hi))
    lo, hi = layout.text_component_range
    text_components = list(range(lo, hi))
    if changed_modality == "image":
        return image_components
    if changed_modality == "text":
        return text_components
    return image_components + text_components


def affected_signature_bits(component_indices: list[int], referenced_components: list[list[int]]) -> list[int]:
    """依据“签名位 -> 特征分量”引用关系反查受影响的签名位编号。"""
    component_set = set(component_indices)
    bits = [bit for bit, refs in enumerate(referenced_components) if component_set.intersection(refs)]
    return sorted(set(bits))


def affected_buckets(bit_indices: list[int], bucket_bit_sets: list[list[int]]) -> list[int]:
    """依据“桶 -> 签名位”引用关系反查受影响桶序号。"""
    bit_set = set(bit_indices)
    return sorted(
        bucket_index
        for bucket_index, refs in enumerate(bucket_bit_sets)
        if bit_set.intersection(refs)
    )


def reuse_entry(stage: str, subject: str, cache_key: str, hit: bool, reason: str,
                affected: list[int] | None = None) -> dict[str, Any]:
    return {
        "stage": stage,
        "subject": subject,
        "cache_key": cache_key,
        "status": "reused" if hit else "recomputed",
        "reason": reason,
        "affected_component_indices": list(affected or []),
    }
