"""确定性分块特征抽取、定点量化与图像级嵌入聚合。

确定性保证：
- 切块顺序固定为行优先 (row, col)，序号 seq 从 0 起；
- 统计量全部使用固定顺序的 NumPy 归约，不依赖迭代器/哈希顺序；
- 投影矩阵与嵌入矩阵由 ``配置指纹 + 维度`` 派生的独立 RNG 生成，
  不读取任何全局随机状态；
- 定点量化采用半向上取整（round-half-up），逐元素记录舍入残差。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hashing import deterministic_rng, short_hash

# 投影后的数值先按固定常数归一化到有界区间，保证最窄 8 位档也不会整体截断。
FEATURE_NORMALIZER = 32.0
# 纹理判定阈值（0-255 灰度域），为固定常数。
TEXTURE_EPSILON = 8.0
# 残差与相似度统一保留的小数位数（仅用于 JSON 稳定展示，不参与判定）。
RESIDUAL_DECIMALS = 9
SIMILARITY_DECIMALS = 9


@dataclass(frozen=True)
class TileIndex:
    row: int
    col: int
    seq: int

    def sort_key(self, image: str, stage: str) -> str:
        return f"{stage}|{image}|r{self.row:04d}|c{self.col:04d}|s{self.seq:04d}"


def enumerate_tiles(height: int, width: int, tile_size: int, max_tiles: int):
    """按行列号升序列出分块；超出上限时截断。

    返回 (保留 TileIndex 列表, 被截断分块序号列表)。
    """
    rows = height // tile_size
    cols = width // tile_size
    kept: list[TileIndex] = []
    truncated: list[int] = []
    seq = 0
    for row in range(rows):
        for col in range(cols):
            if seq < max_tiles:
                kept.append(TileIndex(row=row, col=col, seq=seq))
            else:
                truncated.append(seq)
            seq += 1
    return kept, truncated


def _chunk_stats(tile: np.ndarray) -> np.ndarray:
    """固定 6 维原始统计：RGB 均值、梯度强度均值/标准差、纹理密度。"""
    pixels = tile.astype(np.float64)
    red_mean = pixels[:, :, 0].mean()
    green_mean = pixels[:, :, 1].mean()
    blue_mean = pixels[:, :, 2].mean()

    gray = (
        0.299 * pixels[:, :, 0]
        + 0.587 * pixels[:, :, 1]
        + 0.114 * pixels[:, :, 2]
    )
    grad_y, grad_x = np.gradient(gray)
    magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    grad_mean = magnitude.mean()
    grad_std = magnitude.std()

    horizontal_diff = np.abs(gray[:, 1:] - gray[:, :-1])
    texture_density = float(
        (horizontal_diff > TEXTURE_EPSILON).sum()
    ) / float(horizontal_diff.size)

    return np.array(
        [red_mean, green_mean, blue_mean, grad_mean, grad_std, texture_density],
        dtype=np.float64,
    )


def projection_matrix(feature_dim: int, feature_cfg_fp: str) -> np.ndarray:
    """由特征配置指纹派生的 (feature_dim x 6) 固定投影矩阵。"""
    rng = deterministic_rng(f"projection|{feature_cfg_fp}|d{feature_dim}")
    return rng.uniform(-0.5, 0.5, size=(feature_dim, 6))


def embedding_matrix(embed_dim: int, feature_dim: int, embed_cfg_fp: str) -> np.ndarray:
    """由嵌入配置指纹派生的定点 (embed_dim x feature_dim) 整数矩阵。"""
    rng = deterministic_rng(
        f"embedding|{embed_cfg_fp}|e{embed_dim}|f{feature_dim}"
    )
    return rng.uniform(-0.5, 0.5, size=(embed_dim, feature_dim))


def integer_bounds(quant_level: int) -> tuple[int, int]:
    """quant_level 为有符号整数总位宽：返回 (最小值, 最大值)。"""
    return -(2 ** (quant_level - 1)), 2 ** (quant_level - 1) - 1


def round_half_up(value: np.ndarray) -> np.ndarray:
    """半向上取整（含负值），与平台无关的确定性舍入。"""
    return np.floor(value + 0.5).astype(np.int64)


def quantize(values: np.ndarray, quant_level: int, fixed_bits: int):
    """浮点向量 -> 定点整数向量，返回 (整数数组, 残差数组)。"""
    scale = float(1 << fixed_bits)
    scaled = values * scale
    integers = round_half_up(scaled)
    lower, upper = integer_bounds(quant_level)
    integers = np.clip(integers, lower, upper)
    residuals = np.round(scaled - integers, RESIDUAL_DECIMALS)
    return integers.astype(np.int64), residuals.astype(np.float64)


def quantize_matrix(matrix: np.ndarray, quant_level: int, fixed_bits: int) -> np.ndarray:
    integers, _ = quantize(matrix.reshape(-1), quant_level, fixed_bits)
    return integers.reshape(matrix.shape).astype(np.int64)


def extract_chunk(
    rgb: np.ndarray,
    tile: TileIndex,
    tile_size: int,
    project: np.ndarray,
    quant_level: int,
    fixed_bits: int,
) -> dict:
    """抽取单个分块的定点特征向量与元数据。"""
    y0 = tile.row * tile_size
    x0 = tile.col * tile_size
    patch = rgb[y0 : y0 + tile_size, x0 : x0 + tile_size, :]
    stats = _chunk_stats(patch)
    projected = project @ stats / FEATURE_NORMALIZER
    if not np.isfinite(projected).all():
        raise FloatingPointError("分块投影出现非有限特征值")
    vector, residuals = quantize(projected, quant_level, fixed_bits)
    return {
        "row": tile.row,
        "col": tile.col,
        "seq": tile.seq,
        "vector": [int(value) for value in vector],
        "residual": [float(value) for value in residuals],
        "fixed_bits": fixed_bits,
        "quant_level": quant_level,
    }


def chunk_cache_key(
    content_fp: str, feature_cfg_fp: str, quant_level: int, tile: TileIndex
) -> str:
    return short_hash(
        f"tile|{content_fp}|{feature_cfg_fp}|q{quant_level}"
        f"|r{tile.row}|c{tile.col}|s{tile.seq}"
    )


def feature_bundle_key(content_fp: str, feature_cfg_fp: str, quant_level: int) -> str:
    return short_hash(f"feat|{content_fp}|{feature_cfg_fp}|q{quant_level}")


def embedding_cache_key(bundle_key: str, embed_cfg_fp: str, quant_level: int) -> str:
    return short_hash(f"emb|{bundle_key}|{embed_cfg_fp}|q{quant_level}")


def index_entry_cache_key(emb_key: str, index_cfg_fp: str) -> str:
    return short_hash(f"idxentry|{emb_key}|{index_cfg_fp}")


def chunk_weight(seq: int) -> int:
    """确定性分块权重；相等时由聚合顺序（seq 升序）决断。"""
    return 1 + (seq % 3)


def aggregate_embedding(
    seq_vector_pairs: list,
    embed_matrix_int: np.ndarray,
    fixed_bits: int,
    quant_level: int,
) -> list[int]:
    """以定点整数按 seq 升序加权聚合为图像级嵌入。

    入参为 ``(seq, vector)`` 对；权重相等时按分块序号决断
    （seq 小者先累加，整数加法结合顺序固定）。
    """
    scale = 1 << fixed_bits
    accumulator = np.zeros(embed_matrix_int.shape[0], dtype=np.int64)
    for seq, vector in sorted(seq_vector_pairs, key=lambda pair: pair[0]):
        vector = np.asarray(vector, dtype=np.int64)
        weighted = embed_matrix_int @ vector
        accumulator += np.multiply(weighted, chunk_weight(seq), dtype=np.int64)

    # 累加结果多乘了一次定点比例尺 S（矩阵元素本身带 S），按半向上取整缩回。
    rescaled = round_half_up(accumulator.astype(np.float64) / float(scale))
    lower, upper = integer_bounds(quant_level)
    rescaled = np.clip(rescaled, lower, upper)
    return [int(value) for value in rescaled]


def cosine_similarity(left: list[int], right: list[int]) -> float:
    """float64 余弦相似度，统一四舍五入到固定小数位后再参与阈值决断。"""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        similarity = 0.0
    else:
        similarity = float(np.dot(a, b) / denominator)
    return round(similarity, SIMILARITY_DECIMALS)
