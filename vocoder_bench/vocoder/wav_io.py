"""WAV (PCM16 mono) reader / writer using only ``wave`` and ``struct``."""
from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np


class AudioFormatError(Exception):
    def __init__(self, message: str, code: str = "E_AUDIO_FORMAT"):
        super().__init__(message)
        self.code = code


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    path = Path(path)
    with path.open("rb") as fh:
        head = fh.read(12)
        if len(head) < 12 or head[0:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise AudioFormatError(f"{path.name}: missing RIFF/WAVE container")
    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)
    except wave.Error as exc:
        raise AudioFormatError(f"{path.name}: {exc}") from exc
    if framerate != 16000:
        raise AudioFormatError(f"{path.name}: sample rate {framerate} != 16000")
    if channels != 1:
        raise AudioFormatError(f"{path.name}: {channels} channels, mono required")
    if sampwidth != 2:
        raise AudioFormatError(f"{path.name}: {sampwidth * 8} bit, 16 bit required")
    if len(raw) != nframes * channels * sampwidth:
        raise AudioFormatError(f"{path.name}: data chunk length mismatch")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    return samples, framerate


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.rint(samples), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(clipped.tobytes())


def pack_pcm16(samples: np.ndarray, rounding: str) -> np.ndarray:
    """Saturate float samples to int16 using a deterministic rounding mode."""
    if rounding == "half_up":
        rounded = np.where(samples >= 0, np.floor(samples + 0.5), np.ceil(samples - 0.5))
    elif rounding == "half_even":
        rounded = np.rint(samples)
    elif rounding == "trunc":
        rounded = np.trunc(samples)
    else:
        raise ValueError(f"unknown rounding mode: {rounding}")
    return np.clip(rounded, -32768, 32767).astype(np.int16)


def pcm16_to_float(pcm: np.ndarray) -> np.ndarray:
    return pcm.astype(np.float64)
