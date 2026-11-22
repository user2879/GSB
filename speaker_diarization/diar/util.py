"""通用工具：稳定 JSON、内容指纹与定点量化原语。"""

import hashlib
import json

import numpy as np


def stable_json(obj):
    """递归按键排序的稳定 JSON 文本（分隔符固定，避免空格差异）。"""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(obj):
    """对任意可 JSON 序列化对象计算 SHA-256 内容指纹（16 进制）。"""
    payload = stable_json(obj).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def short_hash(obj):
    return content_hash(obj)[:16]


def quantize_vector(values, frac_bits):
    """把浮点数组按定点档位量化为整数。

    返回 (整数数组, 量化步长, 残差数组)，残差 = 原值 - 整数*步长。
    四舍五入由 floor(x+0.5) 实现，保证跨平台确定性。
    """
    values = np.asarray(values, dtype=np.float64)
    step = 2.0 ** (-frac_bits)
    scaled = np.floor(values / step + 0.5)
    ints = scaled.astype(np.int64)
    residual = values - ints.astype(np.float64) * step
    return ints, step, residual


def quantize_scalar(value, frac_bits):
    """标量定点化，返回 (整数, 步长, 残差)。"""
    step = 2.0 ** (-frac_bits)
    rounded = int(np.floor(value / step + 0.5))
    residual = float(value) - rounded * step
    return rounded, step, residual


def l1_distance(int_vec_a, int_vec_b):
    """定点整数向量间的 L1 距离（整数）。"""
    return int(np.sum(np.abs(np.asarray(int_vec_a, dtype=np.int64)
                             - np.asarray(int_vec_b, dtype=np.int64))))


def mean_quantized(matrix_int):
    """对整数特征矩阵按维度取均值并再次定点化。

    累加顺序固定：帧序号升序（矩阵行顺序）、维度序号升序（列顺序）。
    """
    matrix_int = np.asarray(matrix_int, dtype=np.int64)
    n = matrix_int.shape[0]
    sums = np.sum(matrix_int, axis=0)  # 行序升序累加
    means = np.floor(sums.astype(np.float64) / n + 0.5).astype(np.int64)
    return means
