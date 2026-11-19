from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np

from .canonical import sha256_bytes
from .config import SAMPLE_RATE
from .errors import WorkbenchError


def validate_audio_bytes(data: bytes) -> tuple[np.ndarray, int]:
    try:
        with wave.open(io.BytesIO(data), "rb") as reader:
            channels = reader.getnchannels()
            sample_width = reader.getsampwidth()
            sample_rate = reader.getframerate()
            frame_count = reader.getnframes()
            compression = reader.getcomptype()
            raw = reader.readframes(frame_count)
    except (wave.Error, EOFError, OSError) as exc:
        raise WorkbenchError("E_AUDIO_FORMAT", "invalid WAV container", {"reason": str(exc)}) from exc
    if channels != 1 or sample_width != 2 or compression != "NONE":
        raise WorkbenchError(
            "E_AUDIO_FORMAT",
            "audio must be mono 16-bit PCM WAV",
            {
                "channels": channels,
                "sample_width": sample_width,
                "compression": compression,
            },
        )
    if sample_rate != SAMPLE_RATE:
        raise WorkbenchError(
            "E_AUDIO_FORMAT",
            "audio must be 44100 Hz",
            {"sample_rate": sample_rate},
        )
    if frame_count <= 0 or len(raw) != frame_count * 2:
        raise WorkbenchError("E_AUDIO_FORMAT", "WAV has no complete PCM frames")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.int16, copy=True)
    return samples, sample_rate


def read_wav_bytes(path: str | Path) -> tuple[bytes, np.ndarray]:
    data = Path(path).read_bytes()
    samples, _ = validate_audio_bytes(data)
    return data, samples


def pcm16_bytes(samples: np.ndarray) -> bytes:
    if samples.dtype != np.int16:
        raise WorkbenchError(
            "E_AUDIO_FORMAT",
            "PCM writer requires int16 samples",
            {"dtype": str(samples.dtype)},
        )
    return np.ascontiguousarray(samples, dtype="<i2").tobytes()


def wav_bytes(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    if sample_rate != SAMPLE_RATE:
        raise WorkbenchError("E_AUDIO_FORMAT", "unsupported output sample rate")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm16_bytes(samples))
    return buffer.getvalue()


def audio_content_hash(data: bytes) -> str:
    return sha256_bytes(data)
