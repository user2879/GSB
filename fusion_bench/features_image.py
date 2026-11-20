"""图像分块确定性特征抽取：颜色均值、梯度强度、边缘密度 + 固定投影。

全程定点整数运算；投影矩阵由 (视觉特征维度, 配置指纹) 派生的固定种子生成。
"""

import hashlib

import numpy as np

from .errors import E_SAMPLE_INVALID, PipelineError
from .fixedpoint import div_round, quantize, scale_of

_PROJ_CACHE = {}


def projection_matrix(visual_dim, cfg):
    """固定投影矩阵 5 x visual_dim，由特征维度与配置指纹派生。"""
    key = (visual_dim, cfg.fingerprint)
    if key in _PROJ_CACHE:
        return _PROJ_CACHE[key]
    seed_material = f"proj|{visual_dim}|{cfg.fingerprint}".encode()
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    mat = rng.integers(-2 * cfg.scale, 2 * cfg.scale,
                       size=(5, visual_dim), dtype=np.int64)
    mat = quantize(mat, cfg.frac_bits, cfg.quant_levels, cfg.quant_range)
    _PROJ_CACHE[key] = mat
    return mat


def block_vector(image, block_size, block_idx, cfg):
    """单个分块的 5 维定点特征：[r均值, g均值, b均值, 梯度强度, 边缘密度]。"""
    h, w = image.shape[0], image.shape[1]
    n_per_row = w // block_size
    row, col = divmod(block_idx, n_per_row)
    block = image[row * block_size:(row + 1) * block_size,
                  col * block_size:(col + 1) * block_size, :].astype(np.int64)
    n_px = block_size * block_size
    scale = scale_of(cfg.frac_bits)
    # 颜色均值归一到 [0,1] 定点，避免投影后饱和
    means = [div_round(int(block[:, :, c].sum(dtype=np.int64)) * scale,
                       n_px * 255)
             for c in range(3)]
    gray = (block[:, :, 0] * 299 + block[:, :, 1] * 587
            + block[:, :, 2] * 114) // 1000
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    grad_sum = int(gx.sum(dtype=np.int64)) + int(gy.sum(dtype=np.int64))
    grad_cnt = gx.size + gy.size
    grad_strength = div_round(grad_sum * scale, grad_cnt * 255)
    edge_cnt = int((gx > cfg.edge_threshold).sum()) + int(
        (gy > cfg.edge_threshold).sum())
    edge_density = div_round(edge_cnt * scale, grad_cnt)
    return np.array([means[0], means[1], means[2],
                     grad_strength, edge_density], dtype=np.int64)


def visual_feature_from_blocks(block_vectors, cfg):
    """由分块特征序列重建视觉特征（缓存重算路径与首次计算共用）。"""
    proj = projection_matrix(cfg.visual_dim, cfg)
    acc = np.zeros(cfg.visual_dim, dtype=np.int64)
    for vec in block_vectors:
        acc += div_round(vec @ proj, cfg.scale)
    feat = div_round(acc, len(block_vectors))
    return quantize(feat, cfg.frac_bits, cfg.quant_levels, cfg.quant_range)


def extract_visual(sample, cfg, plan):
    """抽取视觉特征，返回 (特征向量, 分块特征列表, 元数据)。"""
    blocks = [block_vector(sample.image, sample.block_size, i, cfg)
              for i in range(sample.n_blocks)]
    feat = visual_feature_from_blocks(blocks, cfg)
    feat = _apply_nonfinite_injection(feat, sample.sample_id, plan)
    meta = {
        "frac_bits": cfg.frac_bits,
        "rounding": "half_even",
        "quant_levels": cfg.quant_levels,
        "n_blocks": sample.n_blocks,
        "block_size": sample.block_size,
    }
    return feat, blocks, meta


def _apply_nonfinite_injection(feat, sample_id, plan):
    """非有限特征值注入：注入后由有限性校验以 E_SAMPLE_INVALID 拒绝。"""
    for inj in plan.for_sample(sample_id, "nonfinite_feature"):
        plan.mark_hit(inj["id"])
        corrupted = feat.astype(np.float64)
        corrupted[0] = float("inf")
        if not np.all(np.isfinite(corrupted)):
            raise PipelineError(
                E_SAMPLE_INVALID, "feature_extract",
                f"{sample_id}: non-finite feature value detected",
                injection_id=inj["id"])
    if not np.all(np.isfinite(feat.astype(np.float64))):
        raise PipelineError(E_SAMPLE_INVALID, "feature_extract",
                            f"{sample_id}: non-finite feature value detected")
    return feat
