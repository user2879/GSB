"""内容指纹、配置指纹与确定性随机源。

所有派生随机量都来自显式字符串种子（配置指纹 + 维度信息），
从不读取全局 numpy/random 随机状态，因此与运行环境和调度顺序无关。
"""

from __future__ import annotations

import hashlib

import numpy as np

from .jsonio import canonical


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(text: str) -> str:
    return sha256_hex(text)[:16]


def content_fingerprint(rgb: np.ndarray) -> str:
    """由解码后 RGB 像素数组派生图像内容指纹（与文件名、修改时间无关）。"""

    if not np.isfinite(rgb).all():
        raise ValueError("像素数组包含非有限值")
    payload = rgb.astype(np.uint8).tobytes()
    digest = hashlib.sha256(b"vrep-image-v1|" + payload).hexdigest()
    return digest[:32]


def config_fingerprint(config_dict: dict) -> str:
    """完整配置指纹。"""
    return short_hash("vrep-config-v1|" + canonical(config_dict))


def sub_fingerprint(prefix: str, config_dict: dict) -> str:
    """配置子集指纹，用于精确失效判定。"""
    return short_hash(f"vrep-{prefix}-v1|" + canonical(config_dict))


def feature_config_fingerprint(config) -> str:
    """影响分块特征的配置子集：分块、特征维度、量化档位、定点位宽、分块上限。"""
    return sub_fingerprint(
        "feature-cfg",
        {
            "tile_size": config.tile_size,
            "color_dim": config.color_dim,
            "gradient_dim": config.gradient_dim,
            "quant_level": config.quant_level,
            "fixed_bits": config.fixed_bits,
            "max_tiles_per_image": config.max_tiles_per_image,
        },
    )


def embed_config_fingerprint(config) -> str:
    """影响嵌入聚合的配置子集：嵌入维度、量化档位、定点位宽。"""
    return sub_fingerprint(
        "embed-cfg",
        {
            "embed_dim": config.embed_dim,
            "quant_level": config.quant_level,
            "fixed_bits": config.fixed_bits,
        },
    )


def index_config_fingerprint(config) -> str:
    """检索索引指纹：相似度阈值 + 量化档位（决定向量解释方式）。"""
    return sub_fingerprint(
        "index-cfg",
        {
            "similarity_threshold": config.similarity_threshold,
            "quant_level": config.quant_level,
        },
    )


def deterministic_rng(seed_text: str) -> np.random.Generator:
    """由字符串种子构造独立的 numpy 生成器，绝不触碰全局随机状态。"""
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return np.random.default_rng(seed)
