"""Deterministic 24-bit PCM WAV input/output for 5.1 material.

Float signals live in ``[-1.0, 1.0)`` and are packed/unpacked as signed
24-bit little-endian PCM using the standard library :mod:`wave` module only,
so no third-party backend can inject non-deterministic behaviour.
"""

from __future__ import annotations

import wave

import numpy as np

from .config import BIT_DEPTH, SAMPLE_RATE, validate_audio_format
from .errors import WorkbenchError, E_AUDIO_FORMAT

_INT24_MAX = (1 << 23) - 1
_INT24_MIN = -(1 << 23)


def float_to_pcm24(signal: np.ndarray) -> bytes:
    """Pack a float ``(samples, channels)`` array into signed 24-bit bytes."""
    clipped = np.clip(signal, -1.0, 1.0)
    q = np.round(clipped * _INT24_MAX).astype(np.int32)
    # Signed two's complement 24-bit: mask off the top byte before packing.
    q = np.where(q < 0, q + (1 << 24), q).astype(np.uint32)
    flat = q.reshape(-1)
    out = np.zeros((flat.size, 3), dtype=np.uint8)
    out[:, 0] = flat & 0xFF
    out[:, 1] = (flat >> 8) & 0xFF
    out[:, 2] = (flat >> 16) & 0xFF
    return out.tobytes()


def pcm24_to_float(raw: bytes, channels: int) -> np.ndarray:
    """Inverse of :func:`float_to_pcm24`."""
    if len(raw) % 3 != 0:
        raise WorkbenchError(E_AUDIO_FORMAT, "PCM24 byte stream length is not a multiple of 3")
    packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
    values = (
        packed[:, 0].astype(np.int32)
        | (packed[:, 1].astype(np.int32) << 8)
        | (packed[:, 2].astype(np.int32) << 16)
    )
    values = np.where(values >= (1 << 23), values - (1 << 24), values)
    signal = values.astype(np.float64) / _INT24_MAX
    return signal.reshape(-1, channels)


def write_wav(path: str, signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write a multi-channel float array as a deterministic 24-bit WAV."""
    if signal.ndim != 2:
        raise WorkbenchError(
            E_AUDIO_FORMAT, f"signal must be 2-D (samples, channels), got ndim={signal.ndim}"
        )
    validate_audio_format(sample_rate, signal.shape[1], BIT_DEPTH)
    if np.isnan(signal).any() or np.isinf(signal).any():
        raise WorkbenchError(E_AUDIO_FORMAT, "signal contains NaN or Inf samples")
    pcm = float_to_pcm24(signal)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(signal.shape[1])
        wav.setsampwidth(BIT_DEPTH // 8)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def read_wav(path: str) -> tuple[np.ndarray, int, int]:
    """Return ``(signal, sample_rate, channels)`` from a 24-bit PCM WAV."""
    with wave.open(path, "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    if sample_width * 8 != BIT_DEPTH:
        raise WorkbenchError(
            E_AUDIO_FORMAT, f"sample width={sample_width * 8}-bit, expected {BIT_DEPTH}-bit"
        )
    validate_audio_format(sample_rate, channels, BIT_DEPTH)
    signal = pcm24_to_float(raw, channels)
    return signal, sample_rate, channels
