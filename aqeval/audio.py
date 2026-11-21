"""Wave input and deterministic frame slicing."""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .errors import AudioFormatError
from .util import sha256_bytes


@dataclass(frozen=True)
class AudioData:
    file_id: str
    path: str
    samples: np.ndarray
    sample_rate: int
    fingerprint: str

    @property
    def sample_count(self) -> int:
        return int(self.samples.shape[0])


def decode_wav_bytes(data: bytes, *, expected_sample_rate: int, location: str) -> tuple[np.ndarray, int]:
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            compression = handle.getcomptype()
            frames = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError) as exc:
        raise AudioFormatError(f"cannot decode WAV: {exc}", code="E_AUDIO_FORMAT") from exc
    if compression != "NONE":
        raise AudioFormatError("compressed WAV is not supported", code="E_AUDIO_FORMAT")
    if channels != 1:
        raise AudioFormatError(f"WAV must be mono, got {channels} channels", code="E_AUDIO_FORMAT")
    if sample_width != 2:
        raise AudioFormatError(
            f"WAV must use 16-bit samples, got {sample_width * 8}-bit", code="E_AUDIO_FORMAT"
        )
    if sample_rate != expected_sample_rate:
        raise AudioFormatError(
            f"WAV sample rate must be {expected_sample_rate}, got {sample_rate}", code="E_AUDIO_FORMAT"
        )
    samples = np.frombuffer(frames, dtype="<i2").copy()
    if samples.size == 0:
        raise AudioFormatError("WAV contains no samples", code="E_AUDIO_FORMAT")
    return samples, sample_rate


def load_audio(path: str | Path, file_id: str, config: Config) -> AudioData:
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise AudioFormatError(f"cannot read audio: {exc}", code="E_AUDIO_MISSING") from exc
    samples, sample_rate = decode_wav_bytes(
        data, expected_sample_rate=config.sample_rate, location=str(path)
    )
    return AudioData(
        file_id=file_id,
        path=str(path),
        samples=samples,
        sample_rate=sample_rate,
        fingerprint=sha256_bytes(data),
    )


def frame_ranges(sample_count: int, frame_length: int, hop: int) -> list[tuple[int, int]]:
    if sample_count <= 0:
        return []
    if sample_count < frame_length:
        return [(0, sample_count)]
    return [(start, start + frame_length) for start in range(0, sample_count - frame_length + 1, hop)]

