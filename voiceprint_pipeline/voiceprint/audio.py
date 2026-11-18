"""Strict WAV container validation and PCM decoding (16 kHz / mono / 16-bit)."""
from __future__ import annotations

import os
import wave
from dataclasses import dataclass

import numpy as np

from .errors import AudioFormatError
from .util import sha256_bytes

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit PCM


@dataclass(frozen=True)
class WaveData:
    sample_id: str
    path: str
    samples: np.ndarray  # float64 in [-1, 1)
    n_frames_pcm: int
    data_bytes: int
    data_fp: str


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AudioFormatError(message)


def read_wav(path: str, sample_id: str | None = None) -> WaveData:
    sid = sample_id if sample_id is not None else os.path.splitext(os.path.basename(path))[0]
    if not os.path.isfile(path):
        raise AudioFormatError(f"wav file not found: {path}")
    try:
        with wave.open(path, "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            n_pcm = wf.getnframes()
            comptype = wf.getcomptype()
            raw = wf.readframes(n_pcm)
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioFormatError(f"unreadable wav container: {path}: {exc}") from exc

    _check(comptype == "NONE", f"unsupported compression {comptype!r} in {path}")
    _check(frame_rate == SAMPLE_RATE, f"sample rate {frame_rate} != {SAMPLE_RATE} in {path}")
    _check(n_channels == CHANNELS, f"channels {n_channels} != {CHANNELS} in {path}")
    _check(sample_width == SAMPLE_WIDTH, f"sample width {sample_width} != 16-bit in {path}")
    expected_bytes = n_pcm * n_channels * sample_width
    _check(len(raw) == expected_bytes,
           f"data length {len(raw)} inconsistent with header {expected_bytes} in {path}")

    pcm = np.frombuffer(raw, dtype="<i2")
    samples = pcm.astype(np.float64) / 32768.0
    return WaveData(sample_id=sid, path=path, samples=samples, n_frames_pcm=n_pcm,
                    data_bytes=len(raw), data_fp=sha256_bytes(raw))


def list_wav_files(input_dir: str) -> list[str]:
    if not os.path.isdir(input_dir):
        raise AudioFormatError(f"input directory not found: {input_dir}")
    names = [name for name in os.listdir(input_dir) if name.lower().endswith(".wav")]
    return [os.path.join(input_dir, name) for name in sorted(names)]
