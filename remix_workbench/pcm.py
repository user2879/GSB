"""Lossless PCM conversion helpers.

The internal sample plane is signed int32, left-justified to 24 bits (Q23).
16-bit PCM therefore maps to ``value << 8`` and 24-bit PCM maps verbatim.
This keeps all integer arithmetic exact while retaining 24-bit source detail.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

Q23_MAX = 8_388_607
Q23_MIN = -8_388_608
I16_MAX = 32_767
I16_MIN = -32_768


def decode_pcm_bytes(raw: bytes, sample_width: int) -> np.ndarray:
    """Decode mono/interleaved PCM bytes to Q23 int32 integers."""
    if sample_width == 1:
        values = np.frombuffer(raw, dtype=np.uint8).astype(np.int32)
        return (values - 128) << 15
    if sample_width == 2:
        values = np.frombuffer(raw, dtype="<i2").astype(np.int32)
        return values << 8
    if sample_width == 3:
        if len(raw) % 3:
            raise ValueError("truncated 24-bit PCM payload")
        octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
        return np.where(values & 0x800000, values - 0x1000000, values).astype(np.int32)
    if sample_width == 4:
        values = np.frombuffer(raw, dtype="<i4").astype(np.int32)
        return values >> 8
    raise ValueError(f"unsupported PCM sample width: {sample_width}")


def q23_to_i16(values: np.ndarray) -> np.ndarray:
    """Round Q23 to int16, half away from zero, with deterministic saturation."""
    q = np.asarray(values, dtype=np.int64)
    magnitude = (np.abs(q) + 128) // 256
    rounded = np.where(q < 0, -magnitude, magnitude)
    return np.clip(rounded, I16_MIN, I16_MAX).astype(np.int16)


def q23_to_i24_bytes(values: np.ndarray) -> bytes:
    q = np.asarray(values, dtype=np.int32).astype(np.int64)
    q = np.clip(q, Q23_MIN, Q23_MAX)
    unsigned = q & 0xFFFFFF
    out = np.empty((q.size, 3), dtype=np.uint8)
    out[:, 0] = unsigned & 0xFF
    out[:, 1] = (unsigned >> 8) & 0xFF
    out[:, 2] = (unsigned >> 16) & 0xFF
    return out.tobytes(order="C")


def i16_to_bytes(values: np.ndarray) -> bytes:
    return np.asarray(values, dtype="<i2").tobytes(order="C")


def read_wav(path: str | Path) -> tuple[int, int, int, int, bytes]:
    """Read an uncompressed RIFF/WAVE file.

    Returns ``(sample_rate, channels, sample_width, frame_count, raw_pcm)``.
    Explicit payload-length validation catches files truncated after a valid
    RIFF header, which the stdlib ``wave`` module otherwise accepts.
    """
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frame_count = wav.getnframes()
            compression = wav.getcomptype()
            raw = wav.readframes(frame_count)
    except (wave.Error, EOFError, OSError) as exc:
        raise ValueError(f"invalid or truncated WAV {path}: {exc}") from exc
    if compression != "NONE":
        raise ValueError(f"compressed WAV is unsupported: {path}")
    if channels <= 0 or sample_rate <= 0 or sample_width not in (1, 2, 3, 4):
        raise ValueError(f"invalid WAV header values in {path}")
    expected = frame_count * channels * sample_width
    if len(raw) != expected:
        raise ValueError(
            f"truncated WAV payload in {path}: expected {expected} bytes, got {len(raw)}"
        )
    return sample_rate, channels, sample_width, frame_count, raw


def write_wav(
    path: str | Path,
    *,
    sample_rate: int,
    channels: int,
    sample_width: int,
    raw_pcm: bytes,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = (len(raw_pcm) // (channels * sample_width)) * channels * sample_width
    if expected != len(raw_pcm) or expected == 0:
        raise ValueError("PCM payload must contain a positive whole number of frames")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.setcomptype("NONE", "not compressed")
        wav.writeframes(raw_pcm)


def write_pcm_wav(
    path: str | Path,
    values: np.ndarray,
    *,
    sample_rate: int,
    sample_width: int = 2,
) -> None:
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("write_pcm_wav expects a mono sample vector")
    if sample_width == 2:
        raw = i16_to_bytes(values.astype(np.int16))
    elif sample_width == 3:
        raw = q23_to_i24_bytes(values.astype(np.int32))
    elif sample_width == 4:
        raw = (np.asarray(values, dtype=np.int32).astype(np.int64) << 8).astype(
            "<i4"
        ).tobytes()
    else:
        raise ValueError("supported output sample widths are 2, 3, and 4")
    write_wav(
        path,
        sample_rate=sample_rate,
        channels=1,
        sample_width=sample_width,
        raw_pcm=raw,
    )


def wav_info(path: str | Path) -> dict[str, int]:
    rate, channels, width, frames, _ = read_wav(path)
    return {
        "sample_rate": rate,
        "channels": channels,
        "sample_width": width,
        "frames": frames,
    }


def sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def int16_vector_sha256(values: np.ndarray) -> str:
    return sha256_bytes(np.asarray(values, dtype="<i2").tobytes(order="C"))


def int64_matrix_sha256(values: np.ndarray) -> str:
    return sha256_bytes(np.asarray(values, dtype="<i8").tobytes(order="C"))


def q23_vector_sha256(values: np.ndarray) -> str:
    return sha256_bytes(np.asarray(values, dtype="<i4").tobytes(order="C"))

