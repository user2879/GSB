"""Deterministic 16 kHz mono 16-bit PCM WAV input/output (stdlib wave only)."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .config import fingerprint_pcm


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if channels != 1 or sample_width != 2:
        raise ValueError(f"{path}: expected mono 16-bit PCM")
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.int64)
    return pcm, sample_rate


def write_wav(path: str | Path, pcm: np.ndarray, sample_rate: int) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(pcm, dtype=np.int64), -32768, 32767)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(clipped.astype("<i2").tobytes())
    return fingerprint_pcm(clipped.astype("<i2"))


def saturate_int16(samples: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(samples), -32768, 32767).astype("<i2")
