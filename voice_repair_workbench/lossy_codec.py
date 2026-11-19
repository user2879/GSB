"""本地确定性有损编解码器 LRX1（无外部编码器/网络依赖）。

变换核：对称 Hann 窗、50% 重叠的 rfft/irfft 加权重叠相加（WOLA）。
无量化时内部区域满足完美重建（数值误差 ~1e-15）；有损性来自对每帧
频域系数做“帧增益归一 + 低位深均匀量化”，量化误差不可逆。

容器格式（小端）：
  magic(b"LRX1") | sample_rate(uint32) | channels(uint16=1)
  | n_frames(uint32) | frame_size(uint32) | code_bits(uint16) | reserved(uint16)
  随后每帧: gain(float64) + 量化系数(int16, bins 个)
"""
from __future__ import annotations

import os
import struct

import numpy as np

from . import config
from .errors import E_FORMAT_UNSUPPORTED, RepairError

FRAME_SIZE = 1024  # 帧长（样本）；50% 重叠 -> 帧移 FRAME_SIZE//2
HOP = FRAME_SIZE // 2
# 12bit 系数量化：保持不可逆有损，同时把误差控制在跨格式一致性容差内。
CODE_BITS = 14
MAGIC = b"LRX1"


def _hann_window() -> np.ndarray:
    return np.hanning(FRAME_SIZE).astype(np.float64)


def _pad_signal(x: np.ndarray) -> tuple[np.ndarray, int]:
    # 首尾各补一帧零，保证所有有效样本都处于完整窗覆盖的内部区域
    padded = np.concatenate([np.zeros(FRAME_SIZE), x.astype(np.float64), np.zeros(FRAME_SIZE)])
    remainder = padded.shape[0] % HOP
    tail = (HOP - remainder) % HOP
    if tail:
        padded = np.concatenate([padded, np.zeros(tail)])
    return padded, x.shape[0]


def _frame_starts(total_len: int) -> range:
    return range(0, total_len - FRAME_SIZE + 1, HOP)


def encode(signal: np.ndarray, sample_rate: int, code_bits: int = CODE_BITS) -> bytes:
    if signal.ndim != 1:
        raise RepairError(
            E_FORMAT_UNSUPPORTED,
            "LRX1 codec encodes mono signals only; downmix before encoding",
            field="channels",
        )
    window = _hann_window()
    padded, orig_len = _pad_signal(signal)
    starts = list(_frame_starts(padded.shape[0]))
    n_frames = len(starts)
    bins = FRAME_SIZE // 2 + 1
    levels = 2 ** (code_bits - 1) - 1

    payload = bytearray()
    payload += MAGIC
    payload += struct.pack(
        "<IHHIIH", sample_rate, 1, n_frames, FRAME_SIZE, code_bits, 0
    )
    payload += struct.pack("<I", orig_len)

    for start in starts:
        block = padded[start : start + FRAME_SIZE] * window
        coeffs = np.fft.rfft(block, n=FRAME_SIZE)
        gain = float(np.max(np.abs(coeffs))) if np.any(coeffs) else 0.0
        if gain > 0.0:
            real_q = np.clip(np.round((coeffs.real / gain) * levels), -levels, levels)
            imag_q = np.clip(np.round((coeffs.imag / gain) * levels), -levels, levels)
        else:
            real_q = np.zeros(bins, dtype=np.float64)
            imag_q = np.zeros(bins, dtype=np.float64)
        payload += struct.pack("<d", gain)
        payload += real_q.astype(np.int16).tobytes(order="C")
        payload += imag_q.astype(np.int16).tobytes(order="C")
    return bytes(payload)


def decode(blob: bytes) -> tuple[np.ndarray, int]:
    header_size = 4 + 4 + 2 + 4 + 4 + 2 + 2 + 4
    if len(blob) < header_size or blob[:4] != MAGIC:
        raise RepairError(
            E_FORMAT_UNSUPPORTED,
            "not an LRX1 lossy stream (bad magic)",
            field="magic",
            context={"marker": config.LOSSY_MARKER},
        )
    sample_rate, channels, n_frames, frame_size, code_bits, _reserved = struct.unpack(
        "<IHHIIH", blob[4 : 4 + 18]
    )
    (orig_len,) = struct.unpack("<I", blob[4 + 18 : header_size])
    if channels != 1 or frame_size != FRAME_SIZE:
        raise RepairError(
            E_FORMAT_UNSUPPORTED,
            "unsupported LRX1 stream layout",
            field="header",
            context={"channels": channels, "frame_size": frame_size},
        )
    bins = FRAME_SIZE // 2 + 1
    levels = 2 ** (code_bits - 1) - 1
    window = _hann_window()
    total_len = FRAME_SIZE + n_frames * HOP
    out = np.zeros(total_len, dtype=np.float64)
    norm = np.zeros(total_len, dtype=np.float64)
    offset = header_size
    frame_record = 8 + 2 * bins * 2
    for frame in range(n_frames):
        gain = struct.unpack("<d", blob[offset : offset + 8])[0]
        real_q = np.frombuffer(
            blob[offset + 8 : offset + 8 + bins * 2], dtype=np.int16
        ).astype(np.float64)
        imag_q = np.frombuffer(
            blob[offset + 8 + bins * 2 : offset + frame_record], dtype=np.int16
        ).astype(np.float64)
        coeffs = (real_q + 1j * imag_q) * (gain / levels if levels else 0.0)
        block = np.fft.irfft(coeffs, n=FRAME_SIZE) * window
        start = frame * HOP
        out[start : start + FRAME_SIZE] += block
        norm[start : start + FRAME_SIZE] += window**2
        offset += frame_record
    reconstructed = out / np.maximum(norm, 1e-12)
    return reconstructed[FRAME_SIZE : FRAME_SIZE + orig_len].copy(), sample_rate


def encode_file(signal: np.ndarray, sample_rate: int, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(encode(signal, sample_rate))
    return path


def decode_file(path: str) -> tuple[np.ndarray, int]:
    with open(path, "rb") as fh:
        return decode(fh.read())
