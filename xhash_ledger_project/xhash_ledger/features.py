"""确定性跨模态联合特征抽取。

图像侧按 ``tile_size`` 切块，每块计算 3 通道颜色均值、平均梯度强度与边缘密度
（共 5 个统计量）；文本侧对受控词表做实体类型计数与属性名计数。
两侧分别乘以模态权重后拼接为联合特征，全程为确定性 NumPy 运算，
同一输入与配置下逐元素一致。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .validation import CONTROLLED_ATTRIBUTES, CONTROLLED_ENTITIES, SampleBundle


@dataclass
class FeatureLayout:
    """联合特征布局，供失效判定时定位分量编号。"""

    image_dim: int
    text_dim: int
    grid_x: int
    grid_y: int
    tile_size: int
    image_weight: int
    text_weight: int

    @property
    def dim(self) -> int:
        return self.image_dim + self.text_dim

    @property
    def image_component_range(self) -> tuple[int, int]:
        return 0, self.image_dim

    @property
    def text_component_range(self) -> tuple[int, int]:
        return self.image_dim, self.image_dim + self.text_dim

    def tile_ranges(self) -> list[tuple[int, int]]:
        """每个图块在联合特征向量中占据的分量区间（5 个统计量）。"""
        return [(idx * 5, idx * 5 + 5) for idx in range(self.grid_x * self.grid_y)]

    def text_entity_index(self, entity_type: str) -> int:
        return self.image_dim + CONTROLLED_ENTITIES.index(entity_type)

    def text_attribute_index(self, attribute: str) -> int:
        return self.image_dim + len(CONTROLLED_ENTITIES) + CONTROLLED_ATTRIBUTES.index(attribute)

    def to_dict(self) -> dict:
        return {
            "image_dim": self.image_dim,
            "text_dim": self.text_dim,
            "grid_x": self.grid_x,
            "grid_y": self.grid_y,
            "tile_size": self.tile_size,
            "image_weight": self.image_weight,
            "text_weight": self.text_weight,
            "dim": self.dim,
        }


def build_layout(width: int, height: int, tile_size: int, image_weight: int, text_weight: int) -> FeatureLayout:
    if width % tile_size or height % tile_size:
        raise ValueError(f"图像分辨率 {width}x{height} 必须能被分块尺寸 {tile_size} 整除")
    grid_x = width // tile_size
    grid_y = height // tile_size
    image_dim = grid_x * grid_y * 5
    text_dim = len(CONTROLLED_ENTITIES) + len(CONTROLLED_ATTRIBUTES)
    return FeatureLayout(image_dim, text_dim, grid_x, grid_y, tile_size, image_weight, text_weight)


def _image_tile_features(pixels: np.ndarray, tile_size: int) -> np.ndarray:
    """返回形状 (grid_x*grid_y, 5) 的图像块统计量（浮点）。"""
    height, width, _ = pixels.shape
    grid_x = width // tile_size
    grid_y = height // tile_size
    # 用 int16 做差分避免 uint8 回绕，结果完全确定
    base = pixels.astype(np.int16)
    grad_x = np.zeros_like(base)
    grad_y = np.zeros_like(base)
    grad_x[:, 1:] = np.abs(base[:, 1:] - base[:, :-1])
    grad_y[1:, :] = np.abs(base[1:, :] - base[:-1, :])
    grad_mag = grad_x + grad_y
    # 边缘判决：梯度强度超过固定阈值 60（0-255 尺度）
    edge_map = (grad_mag.sum(axis=2) > 60).astype(np.uint8)

    features = np.zeros((grid_x * grid_y, 5), dtype=np.float64)
    for ty in range(grid_y):
        for tx in range(grid_x):
            y0, y1 = ty * tile_size, (ty + 1) * tile_size
            x0, x1 = tx * tile_size, (tx + 1) * tile_size
            tile = base[y0:y1, x0:x1]
            grad_tile = grad_mag[y0:y1, x0:x1]
            edge_tile = edge_map[y0:y1, x0:x1]
            idx = ty * grid_x + tx
            features[idx, 0] = tile[:, :, 0].mean()
            features[idx, 1] = tile[:, :, 1].mean()
            features[idx, 2] = tile[:, :, 2].mean()
            features[idx, 3] = grad_tile.mean()
            features[idx, 4] = edge_tile.mean() * 255.0
    return features


def _text_counts(assertion: dict) -> np.ndarray:
    counts = np.zeros(len(CONTROLLED_ENTITIES) + len(CONTROLLED_ATTRIBUTES), dtype=np.float64)
    for entity in assertion["entities"]:
        counts[CONTROLLED_ENTITIES.index(entity["type"])] += 1.0
        for attribute in entity["attributes"]:
            counts[len(CONTROLLED_ENTITIES) + CONTROLLED_ATTRIBUTES.index(attribute)] += 1.0
    return counts


def extract_raw_features(bundle: SampleBundle, layout: FeatureLayout) -> np.ndarray:
    """抽取未加权的原始联合特征（图像块统计量 + 文本计数）。"""
    image_feats = _image_tile_features(bundle.image.pixels, layout.tile_size).reshape(-1)
    text_feats = _text_counts(bundle.assertion)
    return np.concatenate([image_feats, text_feats])


def compose_weighted(raw: np.ndarray, layout: FeatureLayout) -> np.ndarray:
    """按模态权重拼接联合特征向量。"""
    weighted = np.empty(layout.dim, dtype=np.float64)
    lo, hi = layout.image_component_range
    weighted[lo:hi] = raw[lo:hi] * layout.image_weight
    lo, hi = layout.text_component_range
    weighted[lo:hi] = raw[lo:hi] * layout.text_weight
    return weighted
