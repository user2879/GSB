"""Minimal deterministic 24-bit PCM WAV reader/writer."""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import ErrorCode, fail
from .hashing import bytes_hash, write_bytes_atomic

_PCM_24_MAX = 8_388_607
_PCM_24_MIN = -8_388_608


@dataclass(frozen=True)
class WavFormat:
    sample_rate: int
    channels: int
    bit_depth: int
    block_align: int
    total_frames: int


def encode_wav_24(
    samples: np.ndarray,
    *,
    sample_rate: int = 48_000,
    channels: int = 4,
    bit_depth: int = 24,
) -> bytes:
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2 or data.shape[1] != channels:
        fail(
            ErrorCode.AUDIO_FORMAT,
            "sample array must have exactly four channels",
            stage="audio",
            details={"shape": list(data.shape), "channels": channels},
        )
    if sample_rate != 48_000 or bit_depth != 24 or channels != 4:
        fail(
            ErrorCode.AUDIO_FORMAT,
            "only 48 kHz / 24-bit / 4-channel PCM is supported",
            stage="audio",
            details={
                "sample_rate": sample_rate,
                "bit_depth": bit_depth,
                "channels": channels,
            },
        )
    if not np.all(np.isfinite(data)):
        fail(ErrorCode.AUDIO_FORMAT, "recorded samples contain non-finite values", stage="audio")
    quantized = np.rint(np.clip(data, -1.0, 1.0 - 1.0 / 8_388_608.0) * _PCM_24_MAX).astype(np.int32)
    unsigned = (quantized & 0x00FFFFFF).astype(np.uint32).reshape(-1)
    interleaved = np.empty(unsigned.size * 3, dtype=np.uint8)
    interleaved[0::3] = (unsigned & 0xFF).astype(np.uint8)
    interleaved[1::3] = ((unsigned >> 8) & 0xFF).astype(np.uint8)
    interleaved[2::3] = ((unsigned >> 16) & 0xFF).astype(np.uint8)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(3)
        handle.setframerate(sample_rate)
        handle.writeframes(interleaved.tobytes(order="C"))
    return buffer.getvalue()


def decode_wav_24(
    data: bytes,
    *,
    expected_sample_rate: int | None = 48_000,
    expected_channels: int | None = 4,
    expected_bit_depth: int | None = 24,
) -> tuple[np.ndarray, WavFormat]:
    buffer = io.BytesIO(data)
    try:
        with wave.open(buffer, "rb") as handle:
            channels = handle.getnchannels()
            sample_rate = handle.getframerate()
            sample_width = handle.getsampwidth()
            frames = handle.getnframes()
            raw = handle.readframes(frames)
    except Exception as exc:
        fail(ErrorCode.AUDIO_FORMAT, "WAV container cannot be decoded", stage="audio", details={"error": str(exc)})

    bit_depth = sample_width * 8
    expected = {
        "sample_rate": expected_sample_rate,
        "channels": expected_channels,
        "bit_depth": expected_bit_depth,
    }
    actual = {"sample_rate": sample_rate, "channels": channels, "bit_depth": bit_depth}
    for key, value in expected.items():
        if value is not None and actual[key] != value:
            fail(
                ErrorCode.AUDIO_FORMAT,
                f"unexpected {key} in WAV",
                stage="audio",
                details={"expected": expected, "actual": actual},
            )
    if sample_width != 3 or len(raw) != frames * channels * 3:
        fail(
            ErrorCode.AUDIO_FORMAT,
            "WAV payload size or sample width is invalid",
            stage="audio",
            details={"declared_frames": frames, "payload_bytes": len(raw), "sample_width": sample_width},
        )
    encoded = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    values = encoded[:, 0] | (encoded[:, 1] << 8) | (encoded[:, 2] << 16)
    values = np.where(values >= 0x800000, values - 0x1000000, values)
    samples = values.astype(np.float64).reshape(frames, channels) / float(_PCM_24_MAX)
    return samples, WavFormat(sample_rate, channels, bit_depth, channels * sample_width, frames)


def write_wav_24(
    path: str | Path,
    samples: np.ndarray,
    *,
    sample_rate: int = 48_000,
    channels: int = 4,
    bit_depth: int = 24,
) -> tuple[str, WavFormat]:
    payload = encode_wav_24(
        samples,
        sample_rate=sample_rate,
        channels=channels,
        bit_depth=bit_depth,
    )
    write_bytes_atomic(path, payload)
    samples_check, fmt = decode_wav_24(
        payload,
        expected_sample_rate=sample_rate,
        expected_channels=channels,
        expected_bit_depth=bit_depth,
    )
    if samples_check.shape != np.asarray(samples).shape:
        fail(ErrorCode.AUDIO_FORMAT, "round-trip WAV shape mismatch", stage="audio")
    return bytes_hash(payload), fmt
