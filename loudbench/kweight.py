"""ITU-R BS.1770-4 K 加权滤波与 4x 过采样真峰值。

为支持任务并行，滤波器以“冲激响应 + 分块 FFT 卷积”实现，并携带固定长度
warm-up 上下文：每个测量任务从任务起点之前的真实样本重新建立滤波状态，
丢弃 warm-up 输出，从而与连续处理在数值上一致（误差远低于门控量化精度）。
所有计算使用 float64，线程数不参与任何数学运算。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from .wavio import SAMPLE_RATE

# BS.1770-4 两级 RBJ biquad 系数（归一化角频率）
SHELF_GAIN_DB = 4.0
SHELF_FC = 1681.7632
SHELF_Q = 0.707175236
HP_FC = 38.135470876
HP_Q = 0.5003270373

# K 加权（IIR）冲激响应截断长度：24k 点≈0.5s，尾端衰减足够
KW_IR_LEN = 24_000
# 每个任务为 K 加权准备的前置 warm-up 样本
KW_WARMUP = 24_000

# 真峰值 4x 过采样 FIR（窗化 sinc）长度
TP_FIR_LEN = 65
TP_FACTOR = 4
TP_WARMUP = (TP_FIR_LEN // 2) + 2

# 5.1 声道权重：L,R,C,Lfe -> 1.0；Ls,Rs -> 1.41（sqrt(2)）
CHANNEL_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, math.sqrt(2.0), math.sqrt(2.0)])


def _rbj_high_shelf(fs: float, gain_db: float, fc: float, q: float) -> np.ndarray:
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * fc / fs
    cosw, sinw = math.cos(w0), math.sin(w0)
    alpha = sinw / (2.0 * q)
    b0 = a * ((a + 1.0) + (a - 1.0) * cosw + 2.0 * math.sqrt(a) * alpha)
    b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cosw)
    b2 = a * ((a + 1.0) + (a - 1.0) * cosw - 2.0 * math.sqrt(a) * alpha)
    a0 = (a + 1.0) - (a - 1.0) * cosw + 2.0 * math.sqrt(a) * alpha
    a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cosw)
    a2 = (a + 1.0) - (a - 1.0) * cosw - 2.0 * math.sqrt(a) * alpha
    return np.array([b0, b1, b2, a0, a1, a2], dtype=np.float64) / a0


def _rbj_high_pass(fs: float, fc: float, q: float) -> np.ndarray:
    w0 = 2.0 * math.pi * fc / fs
    cosw, sinw = math.cos(w0), math.sin(w0)
    alpha = sinw / (2.0 * q)
    b0 = (1.0 + cosw) / 2.0
    b1 = -(1.0 + cosw)
    b2 = (1.0 + cosw) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cosw
    a2 = 1.0 - alpha
    return np.array([b0, b1, b2, a0, a1, a2], dtype=np.float64) / a0


def _biquad_impulse(coef: np.ndarray, length: int) -> np.ndarray:
    b0, b1, b2, a0, a1, a2 = coef
    x = np.zeros(length, dtype=np.float64)
    y = np.zeros(length, dtype=np.float64)
    x[0] = 1.0
    for n in range(length):
        acc = b0 * x[n]
        if n >= 1:
            acc += b1 * x[n - 1] - a1 * y[n - 1]
        if n >= 2:
            acc += b2 * x[n - 2] - a2 * y[n - 2]
        y[n] = acc / a0
    return y


def kweight_impulse(fs: float = SAMPLE_RATE, length: int = KW_IR_LEN) -> np.ndarray:
    """级联高架+高通后的 K 加权冲激响应（确定性）。"""
    shelf = _rbj_high_shelf(fs, SHELF_GAIN_DB, SHELF_FC, SHELF_Q)
    hp = _rbj_high_pass(fs, HP_FC, HP_Q)
    h1 = _biquad_impulse(shelf, length)
    h2 = _biquad_impulse(hp, length)
    n = length
    size = 1
    while size < 2 * n:
        size <<= 1
    conv = np.fft.irfft(np.fft.rfft(h1, size) * np.fft.rfft(h2, size), size)
    return conv[:n].copy()


def fftconvolve_chunk(x: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    """对一维输入做与冲激响应等长延迟的因果 FFT 卷积。

    返回长度等于 len(x)，等价于对完整序列做 np.convolve(mode='full') 后
    取对应区间（长度 len(impulse)-1 的启动延迟保持在内）。
    """
    n_x = x.shape[0]
    n_h = impulse.shape[0]
    size = 1
    target = n_x + n_h - 1
    while size < target:
        size <<= 1
    full = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(impulse, size), size)
    return full[:n_x].copy()


def kweight_channels(samples: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    """对 (channels, samples) 逐声道做 K 加权，返回同形 float64。"""
    out = np.empty_like(samples, dtype=np.float64)
    for c in range(samples.shape[0]):
        out[c] = fftconvolve_chunk(samples[c], impulse)
    return out


def true_peak_fir(length: int = TP_FIR_LEN, factor: int = TP_FACTOR) -> np.ndarray:
    """4x 过采样用窗化 sinc 低通（增益 factor，截止约 fs/2）。"""
    if length % 2 == 0:
        length += 1
    half = length // 2
    n = np.arange(-half, half + 1, dtype=np.float64)
    t = n / factor
    h = np.sinc(t) * np.hanning(length)
    h *= factor / np.sum(h)
    return h.astype(np.float64)


def upsample_max(
    samples: np.ndarray,
    fir: np.ndarray,
    factor: int = TP_FACTOR,
    out_start: int = 0,
    out_end: Optional[int] = None,
) -> Tuple[float, int]:
    """对一维样本 4x 过采样，返回指定输出子窗口内的 (最大绝对值, 过采样索引)。"""
    n = samples.shape[0]
    if out_end is None:
        out_end = n
    up = np.zeros(n * factor, dtype=np.float64)
    up[::factor] = samples
    size = 1
    target = up.shape[0] + fir.shape[0] - 1
    while size < target:
        size <<= 1
    conv = np.fft.irfft(np.fft.rfft(up, size) * np.fft.rfft(fir, size), size)
    delay = fir.shape[0] // 2
    valid = conv[delay:delay + n * factor]
    lo = max(0, out_start) * factor
    hi = min(n, out_end) * factor
    window = valid[lo:hi]
    if window.size == 0:
        return 0.0, lo
    idx = int(np.argmax(np.abs(window))) + lo
    return float(np.abs(valid[idx])), idx


def to_db_tp(linear_peak: float) -> float:
    return 20.0 * math.log10(max(linear_peak, 1e-12))
