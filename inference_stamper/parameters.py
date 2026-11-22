"""参数矩阵与后端身份摘要的确定性构造。

显式规则（版本 IDENTITY_SCHEME_V1，写进摘要帧，规则变化即换号）：

1. 每个后端按固定层级顺序生成参数矩阵：
   - W1 形状 (hidden_dim, model_dim)
   - b1 形状 (hidden_dim,)
   - W2 形状 (model_dim, hidden_dim)
   - b2 形状 (model_dim,)
2. 各层随机种子 = mix_seed(后端种子, 身份版本号, 层序号, 模型维度, 隐藏层维度)，
   比特流 splitmix64，取值 [-1, 1) 的 float32。
3. 按量化口径量化：
   - fp32：保持 float32
   - fp16：转为 float16（IEEE 754 半精度）
   - int8：逐层 absmax 对称量化为 int8，附带 scale(float32)
   - int4：逐层 absmax 对称量化为有符号 4bit，再大端打包为字节，附带 scale
4. 逐层摘要 = SHA256(帧头 + 量化后字节 + scale 字节)。
5. 身份摘要 = SHA256(后端标识 + 身份版本号 + 维度 + 量化口径 +
   全部逐层摘要，按固定合成顺序)。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .prng import BitStream, mix_seed

IDENTITY_SCHEME_VERSION = 1
SUPPORTED_QUANTS = ("fp32", "fp16", "int8", "int4")

# 固定层级定义：(层名, 形状元组相对于 (model_dim, hidden_dim) 的说明)
LAYER_SPECS = (
    ("W1", ("hidden_dim", "model_dim")),
    ("b1", ("hidden_dim",)),
    ("W2", ("model_dim", "hidden_dim")),
    ("b2", ("model_dim",)),
)


def _frame(tag: bytes, payload: bytes) -> bytes:
    return (
        len(tag).to_bytes(2, "big")
        + tag
        + len(payload).to_bytes(8, "big")
        + payload
    )


def _layer_shape(spec: tuple[str, ...], model_dim: int,
                 hidden_dim: int) -> tuple[int, ...]:
    return tuple(model_dim if axis == "model_dim" else hidden_dim
                 for axis in spec)


def quantize(raw: np.ndarray, quant: str) -> tuple[np.ndarray, np.float32]:
    """按口径量化单层参数，返回 (量化结果, scale)。"""
    if quant == "fp32":
        return np.ascontiguousarray(raw, dtype=np.float32), np.float32(1.0)
    if quant == "fp16":
        return np.ascontiguousarray(raw.astype(np.float16)), np.float32(1.0)

    if quant in ("int8", "int4"):
        bound = int(quant[3:])  # 8 或 4
        levels = (1 << (bound - 1)) - 1  # int8=127, int4=7
        max_abs = float(np.max(np.abs(raw.astype(np.float64))))
        scale = np.float32(max_abs / levels if max_abs > 0.0 else 1.0)
        scaled = np.rint(raw.astype(np.float64) / float(scale))
        clipped = np.clip(scaled, -levels, levels).astype(np.int8)
        if quant == "int8":
            return np.ascontiguousarray(clipped), scale
        return _pack_int4(clipped), scale

    raise ValueError(f"不支持的量化口径: {quant}")


def _pack_int4(values: np.ndarray) -> np.ndarray:
    """有符号 4bit 值大端打包，奇数长度补 0 半字节。"""
    flat = values.astype(np.uint8).reshape(-1)
    nibbles = flat & 0x0F
    if nibbles.size % 2 == 1:
        nibbles = np.concatenate((nibbles, np.zeros(1, dtype=np.uint8)))
    packed = ((nibbles[0::2] << 4) | nibbles[1::2]).astype(np.uint8)
    return np.ascontiguousarray(packed)


def layer_digest(layer_name: str, layer_index: int, quant: str,
                 shape: tuple[int, ...], values: np.ndarray,
                 scale: np.float32) -> str:
    hasher = hashlib.sha256()
    hasher.update(_frame(b"layer-name", layer_name.encode("utf-8")))
    hasher.update(_frame(b"layer-index", layer_index.to_bytes(4, "big")))
    hasher.update(_frame(b"quant", quant.encode("utf-8")))
    hasher.update(_frame(b"shape",
                         np.asarray(shape, dtype=np.int64).tobytes()))
    hasher.update(_frame(b"values",
                         np.ascontiguousarray(values).tobytes()))
    hasher.update(_frame(b"scale",
                         np.asarray(scale, dtype=np.float32).tobytes()))
    return hasher.hexdigest()


@dataclass(frozen=True)
class LayerArtifact:
    name: str
    index: int
    shape: tuple[int, ...]
    quant: str
    digest: str
    scale: float
    values: np.ndarray


@dataclass(frozen=True)
class IdentityArtifact:
    backend_id: str
    model_dim: int
    hidden_dim: int
    seed: int
    quant: str
    identity_version: int
    token_spec_version: int
    scheme_version: int
    identity_digest: str
    composition_order: tuple[str, ...]
    layers: tuple[LayerArtifact, ...]


def build_identity(backend_id: str, model_dim: int, hidden_dim: int,
                   seed: int, quant: str, identity_version: int,
                   token_spec_version: int = 1) -> IdentityArtifact:
    """按显式规则生成参数矩阵并推导后端身份摘要。"""
    if quant not in SUPPORTED_QUANTS:
        raise ValueError(f"不支持的量化口径: {quant}")

    layers: list[LayerArtifact] = []
    composition_order: list[str] = []
    for layer_index, (layer_name, axis_spec) in enumerate(LAYER_SPECS):
        shape = _layer_shape(axis_spec, model_dim, hidden_dim)
        layer_seed = mix_seed(seed, identity_version, layer_index,
                              model_dim, hidden_dim)
        raw = BitStream(layer_seed).float32_array(shape)
        values, scale = quantize(raw, quant)
        digest = layer_digest(layer_name, layer_index, quant, shape,
                              values, scale)
        layers.append(LayerArtifact(
            name=layer_name,
            index=layer_index,
            shape=shape,
            quant=quant,
            digest=digest,
            scale=float(scale),
            values=values,
        ))
        composition_order.append(
            f"{layer_index}:{layer_name}:{digest[:16]}"
        )

    hasher = hashlib.sha256()
    hasher.update(_frame(b"scheme",
                         IDENTITY_SCHEME_VERSION.to_bytes(4, "big")))
    hasher.update(_frame(b"backend-id", backend_id.encode("utf-8")))
    hasher.update(_frame(b"identity-version",
                         int(identity_version).to_bytes(8, "big",
                                                        signed=True)))
    hasher.update(_frame(
        b"dims",
        np.asarray((model_dim, hidden_dim), dtype=np.int64).tobytes(),
    ))
    hasher.update(_frame(b"seed", int(seed).to_bytes(8, "big",
                                                     signed=True)))
    hasher.update(_frame(b"quant", quant.encode("utf-8")))
    for layer in layers:
        hasher.update(_frame(b"layer-digest",
                             bytes.fromhex(layer.digest)))
    identity_digest = hasher.hexdigest()

    return IdentityArtifact(
        backend_id=backend_id,
        model_dim=model_dim,
        hidden_dim=hidden_dim,
        seed=seed,
        quant=quant,
        identity_version=identity_version,
        token_spec_version=token_spec_version,
        scheme_version=IDENTITY_SCHEME_VERSION,
        identity_digest=identity_digest,
        composition_order=tuple(composition_order),
        layers=tuple(layers),
    )
