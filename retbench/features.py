"""确定性视觉分块特征与文本断言特征抽取（含定点量化）。

确定性保证：
- 投影矩阵仅由 (维度, 配置指纹, 用途键) 派生，固定种子；
- 分块按行列号升序、特征按维度升序累加，不依赖线程调度；
- 量化结果记录步长/整数/残差，同输入同配置逐元素一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import RAW_STAT_DIM, REDUCED_STAT_DIM, Config
from .fingerprint import digest_many, digest_obj
from .fixedpoint import QuantSpec, quantize_array
from .rng import child_rng, root_rng


def projection_matrix(rows: int, cols: int, config_fingerprint: str,
                      purpose: str) -> np.ndarray:
    """由维度与配置指纹派生固定投影矩阵，元素在 [-1, 1) 且逐元素可复现。"""

    rng = child_rng(root_rng(20261123), f"proj:{purpose}:{config_fingerprint}:{rows}x{cols}")
    return rng.uniform(-1.0, 1.0, size=(rows, cols)).astype(np.float64)


def stable_normalize(vector: np.ndarray) -> np.ndarray:
    """减去最大值的稳定归一化（softmax 数值稳定前置），输入为浮点。"""

    vector = np.asarray(vector, dtype=np.float64)
    shifted = vector - np.max(vector)
    exponent = np.exp(shifted)
    total = exponent.sum()
    return exponent / total if total > 0 else np.ones_like(exponent) / exponent.size


def tile_raw_stats(image: np.ndarray, tile_size: int, include_edge: bool) -> np.ndarray:
    """按分块行列号升序计算每块颜色均值/梯度强度/（可选）边缘密度。"""

    height, width, _ = image.shape
    rows = height // tile_size
    cols = width // tile_size
    stats: list[np.ndarray] = []
    img = image.astype(np.float64)
    grad_y = np.zeros_like(img)
    grad_x = np.zeros_like(img)
    grad_y[1:, :, :] = np.abs(img[1:, :, :] - img[:-1, :, :])
    grad_x[:, 1:, :] = np.abs(img[:, 1:, :] - img[:, :-1, :])
    edge_map = ((grad_y.sum(axis=2) + grad_x.sum(axis=2)) > 60.0).astype(np.float64)

    for row in range(rows):
        for col in range(cols):
            y0, y1 = row * tile_size, (row + 1) * tile_size
            x0, x1 = col * tile_size, (col + 1) * tile_size
            block = img[y0:y1, x0:x1, :]
            color_mean = block.reshape(-1, 3).mean(axis=0) / 255.0
            gradient = np.concatenate([
                grad_y[y0:y1, x0:x1, :].reshape(-1, 3),
                grad_x[y0:y1, x0:x1, :].reshape(-1, 3),
            ]).mean() / 255.0
            row_stats = list(color_mean) + [float(gradient)]
            if include_edge:
                row_stats.append(float(edge_map[y0:y1, x0:x1].mean()))
            stats.append(np.asarray(row_stats, dtype=np.float64))
    return np.stack(stats, axis=0)


@dataclass
class VisualFeatures:
    sample: str
    precision: str
    grid: tuple[int, int]
    tile_keys: list[tuple[int, int]]
    raw_stats: np.ndarray
    projected: np.ndarray
    quant: dict[str, Any]
    feature_fingerprint: str
    referenced_tiles: list[tuple[int, int]]

    def q(self) -> np.ndarray:
        return self.quant["q"]


@dataclass
class TextFeatures:
    sample: str
    projected: np.ndarray
    quant: dict[str, Any]
    assertion_refs: list[dict[str, Any]]
    padded_positions: list[int]
    feature_fingerprint: str

    def q(self) -> np.ndarray:
        return self.quant["q"]


def extract_visual(sample, config: Config, config_fingerprint: str,
                   spec: QuantSpec, precision: str) -> VisualFeatures:
    include_edge = precision == "full"
    raw = tile_raw_stats(sample.image_array, config.tile_size, include_edge)
    rows, cols = config.grid
    stat_dim = RAW_STAT_DIM if precision == "full" else REDUCED_STAT_DIM
    assert raw.shape[1] == stat_dim
    matrix = projection_matrix(config.visual_dim, stat_dim,
                               config_fingerprint, f"visual-{precision}")
    projected = raw @ matrix.T
    quant = quantize_array(projected, spec)
    tile_keys = [(r, c) for r in range(rows) for c in range(cols)]
    fingerprint = digest_many(
        "visual", sample.image_fingerprint, config_fingerprint,
        precision, spec.key(), digest_obj(projected.tolist())
    )
    return VisualFeatures(
        sample=sample.name,
        precision=precision,
        grid=(rows, cols),
        tile_keys=tile_keys,
        raw_stats=raw,
        projected=projected,
        quant=quant,
        feature_fingerprint=fingerprint,
        referenced_tiles=list(tile_keys),
    )


def _text_onehot(token: str, vocab: list[str]) -> np.ndarray:
    vector = np.zeros(len(vocab), dtype=np.float64)
    if token in vocab:
        vector[vocab.index(token)] = 1.0
    return vector


def extract_text(sample, config: Config, config_fingerprint: str,
                 spec: QuantSpec, target_count: int) -> TextFeatures:
    vocab = config.vocabulary
    entity_vocab = vocab["entities"]
    attr_vocab = vocab["attributes"]
    relation_vocab = vocab["relations"]
    base_dim = len(entity_vocab) + len(attr_vocab) + len(relation_vocab) + len(entity_vocab)
    matrix = projection_matrix(config.text_dim, base_dim, config_fingerprint, "text")

    use_count = max(target_count, len(sample.assertions))
    default_rng = child_rng(root_rng(20261123),
                            f"default-assert:{config_fingerprint}:{sample.name}")
    default_vector = default_rng.uniform(-0.5, 0.5, size=(base_dim,)).astype(np.float64)

    assertion_refs: list[dict[str, Any]] = []
    rows: list[np.ndarray] = []
    for index in range(use_count):
        if index >= len(sample.assertions):
            base = default_vector.copy()
            assertion_refs.append({"index": index, "padded": True,
                                   "predicate": None, "object": None})
        else:
            assertion = sample.assertions[index]
            entity_part = _text_onehot(sample.entity, entity_vocab)
            attr_part = np.zeros(len(attr_vocab), dtype=np.float64)
            for attr in sample.attributes:
                if attr in attr_vocab:
                    attr_part[attr_vocab.index(attr)] = 1.0
            relation_part = _text_onehot(assertion["predicate"], relation_vocab)
            object_part = _text_onehot(assertion["object"], entity_vocab)
            base = np.concatenate([entity_part, attr_part, relation_part, object_part])
            assertion_refs.append({"index": index, "padded": False,
                                   "predicate": assertion["predicate"],
                                   "object": assertion["object"]})
        rows.append(base)

    base_matrix = np.stack(rows, axis=0)
    projected = base_matrix @ matrix.T
    quant = quantize_array(projected, spec)
    padded_positions = [ref["index"] for ref in assertion_refs if ref["padded"]]
    fingerprint = digest_many(
        "text", sample.text_fingerprint, config_fingerprint,
        spec.key(), target_count, digest_obj(projected.tolist())
    )
    return TextFeatures(
        sample=sample.name,
        projected=projected,
        quant=quant,
        assertion_refs=assertion_refs,
        padded_positions=padded_positions,
        feature_fingerprint=fingerprint,
    )
