"""确定性 24bit PCM WAV 读写（仅依赖标准库与 NumPy）。

写文件时不写入任何时间戳或不可控字段，同一 float64 阵列两次写入逐字节一致。
样本约定：浮点范围 [-1.0, 1.0)，形状 (channels, samples)。
"""

from __future__ import annotations

import wave
from typing import Tuple

import numpy as np

from .errors import AudioFormatError

SAMPLE_RATE = 48_000
SAMPLE_WIDTH = 3
CHANNELS_51 = 6
INT24_MAX = 8_388_607  # 2**23 - 1


def quantize_int24(samples: np.ndarray) -> np.ndarray:
    """把 [-1,1] 浮点对称量化为 int32 容器承载的 24bit 整数。"""
    if not np.isfinite(samples).all():
        raise AudioFormatError(
            "输入包含非有限样本，无法量化为 24bit PCM",
            evidence={"non_finite": int(np.size(samples) - np.isfinite(samples).sum())},
        )
    clipped = np.clip(samples, -1.0, 1.0)
    q = np.rint(clipped * INT24_MAX).astype(np.int32)
    return q


def int24_to_bytes(q: np.ndarray) -> bytes:
    """int32 容器中的 24bit 值转小端三字节，输出长度确定。"""
    flat = q.reshape(-1).astype("<i4").view(np.uint8).reshape(-1, 4)
    out = np.empty((flat.shape[0], 3), dtype=np.uint8)
    out[:, 0] = flat[:, 0]
    out[:, 1] = flat[:, 1]
    out[:, 2] = flat[:, 2]
    return out.tobytes()


def write_wav24(path: str, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """写多声道 24bit PCM WAV。samples 形状 (channels, samples)。"""
    if samples.ndim != 2:
        raise AudioFormatError(
            "WAV 输入必须是二维 (channels, samples) 阵列",
            location={"ndim": int(samples.ndim)},
        )
    channels, n = samples.shape
    if channels <= 0 or n <= 0:
        raise AudioFormatError(
            "空的声道或样本维度",
            location={"channels": int(channels), "samples": int(n)},
        )
    q = quantize_int24(samples)
    interleaved = np.ascontiguousarray(q.T.reshape(-1))
    payload = int24_to_bytes(interleaved)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(payload)


def read_wav24(path: str) -> Tuple[np.ndarray, int, int]:
    """读 WAV，返回 (float64 (channels, samples), sample_rate, channels)。

    仅接受 6 声道 48kHz/24bit 的工作台素材，其它格式抛 E_AUDIO_FORMAT。
    """
    try:
        with wave.open(path, "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)
    except wave.Error as exc:  # pragma: no cover - 透传为业务错误
        raise AudioFormatError("WAV 容器解析失败", cause=exc)

    if channels != CHANNELS_51:
        raise AudioFormatError(
            f"仅支持 {CHANNELS_51} 声道 5.1 素材，收到 {channels} 声道",
            evidence={"channels": int(channels)},
        )
    if width != SAMPLE_WIDTH:
        raise AudioFormatError(
            f"仅支持 24bit PCM，收到 {width * 8}bit",
            evidence={"sample_width_bytes": int(width)},
        )
    if rate != SAMPLE_RATE:
        raise AudioFormatError(
            f"仅支持 {SAMPLE_RATE}Hz，收到 {rate}Hz",
            evidence={"sample_rate": int(rate)},
        )

    data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
    i32 = np.zeros((data.shape[0], 4), dtype=np.uint8)
    i32[:, :3] = data
    i32[:, 3] = np.where(data[:, 2] >= 0x80, 0xFF, 0x00)
    q = i32.view("<i4").reshape(-1).astype(np.float64) / INT24_MAX
    q = q.reshape(nframes, channels).T.copy()
    return q, rate, channels
