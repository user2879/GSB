"""跨模态二值哈希签名生成与分桶。

投影矩阵由 ``投影种子版本 + 签名侧配置指纹 + 特征维度`` 派生的固定种子生成，
元素为定点整数，投影通过整数乘加完成。逐位符号判决采用固定决胜规则：

* 投影恰为零       -> 决胜 ``zero``，位取 ``1``
* |投影| <= 量化半步 -> 决胜 ``half_step``，位取 ``0``
* 其余：正 -> 1，负 -> 0
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fixedpoint import Quantization, fixed_matvec
from .hashing import int_to_bucket_key
from . import features as features_mod


@dataclass
class Projection:
    seed: int
    signature_bits: int
    feature_dim: int
    frac_bits: int
    matrix: list[list[int]]

    def project(self, quantized: Quantization) -> list[int]:
        return fixed_matvec(self.matrix, quantized.values)


def build_projection(seed: int, signature_bits: int, feature_dim: int, frac_bits: int) -> Projection:
    """确定性定点投影矩阵：每行是一个超平面法向量。"""
    rng = np.random.default_rng(seed)
    # 元素落在 [-2^m, 2^m) 的定点整数，等价于真实系数 [-1, 1)
    limit = 1 << frac_bits
    matrix = rng.integers(-limit, limit, size=(signature_bits, feature_dim), dtype=np.int64)
    return Projection(
        seed=seed,
        signature_bits=signature_bits,
        feature_dim=feature_dim,
        frac_bits=frac_bits,
        matrix=[[int(v) for v in row] for row in matrix],
    )


@dataclass
class SignatureResult:
    sample_id: str
    bits: str                      # MSB-first 的 0/1 字符串
    signature_int: int
    projections: list[int]
    tie_breaks: list[dict]         # 逐位决胜记录（仅记录发生决胜的位）
    bucket_keys: list[str]
    bucket_bit_sets: list[list[int]]
    referenced_components: list[list[int]]  # 每个签名位引用的联合特征分量
    quantization: Quantization

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "bits": self.bits,
            "signature_int": self.signature_int,
            "projections": list(self.projections),
            "tie_breaks": list(self.tie_breaks),
            "bucket_keys": list(self.bucket_keys),
            "bucket_bit_sets": [list(s) for s in self.bucket_bit_sets],
            "quantization": self.quantization.to_dict(),
        }


def _half_step(frac_bits: int) -> int:
    # 投影值的定点尺度为 Q(2m)，半步即 2^(2m-1)
    return 1 << (2 * frac_bits - 1)


def sign_bits(projection_values: list[int], frac_bits: int) -> tuple[str, int, list[dict]]:
    """逐位符号判决，返回位串、签名整数与决胜记录。"""
    half = _half_step(frac_bits)
    chars: list[str] = []
    tie_breaks: list[dict] = []
    for bit_index, value in enumerate(projection_values):
        if value == 0:
            bit = 1
            tie_breaks.append({"bit_index": bit_index, "rule": "zero", "projection": value, "resolved_to": bit})
        elif abs(value) <= half:
            bit = 0
            tie_breaks.append({"bit_index": bit_index, "rule": "half_step", "projection": value, "resolved_to": bit})
        else:
            bit = 1 if value > 0 else 0
        chars.append(str(bit))
    bits = "".join(chars)
    return bits, int(bits, 2), tie_breaks


def bucket_signature(bits: str, bucket_bits: int) -> tuple[list[str], list[list[int]]]:
    """按分桶位数把签名从高位切成若干子串，映射为定长十六进制桶键。

    最后一个桶若不足 ``bucket_bits`` 位则以低位零填充，桶内记录其引用的签名位集合。
    """
    total = len(bits)
    keys: list[str] = []
    bit_sets: list[list[int]] = []
    chunk_index = 0
    for start in range(0, total, bucket_bits):
        chunk = bits[start : start + bucket_bits]
        padded = chunk.ljust(bucket_bits, "0")
        value = int(padded, 2)
        keys.append(f"{chunk_index}:{int_to_bucket_key(value, bucket_bits)}")
        bit_sets.append(list(range(start, start + len(chunk))))
        chunk_index += 1
    return keys, bit_sets


def referenced_components(projection: Projection) -> list[list[int]]:
    """每个签名位引用的联合特征分量集合（投影法向量非零的列编号）。"""
    refs: list[list[int]] = []
    for row in projection.matrix:
        refs.append([idx for idx, coeff in enumerate(row) if coeff != 0])
    return refs


def hamming(sig_a: int, sig_b: int) -> int:
    return (sig_a ^ sig_b).bit_count()


def generate_signature(
    sample_id: str,
    quantized: Quantization,
    projection: Projection,
    bucket_bits: int,
) -> SignatureResult:
    """完整的签名 + 分桶流水线。"""
    values = projection.project(quantized)
    bits, signature_int, tie_breaks = sign_bits(values, projection.frac_bits)
    bucket_keys, bucket_bit_sets = bucket_signature(bits, bucket_bits)
    return SignatureResult(
        sample_id=sample_id,
        bits=bits,
        signature_int=signature_int,
        projections=values,
        tie_breaks=tie_breaks,
        bucket_keys=bucket_keys,
        bucket_bit_sets=bucket_bit_sets,
        referenced_components=referenced_components(projection),
        quantization=quantized,
    )


def layout_for(bundle, tile_size: int, image_weight: int, text_weight: int) -> features_mod.FeatureLayout:
    return features_mod.build_layout(
        bundle.image.width, bundle.image.height, tile_size, image_weight, text_weight
    )
