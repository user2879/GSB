"""WAV 音频加载与 RIFF/PCM 格式校验（标准库 wave）。"""
from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import ErrorCollector, ErrorCode

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
N_CHANNELS = 1


@dataclass
class AudioClip:
    round: int
    path: Path
    samples: np.ndarray  # float64, 归一化到 [-1, 1)
    n_samples: int
    content_hash: str


def scan_wav_files(sample_dir: str | Path) -> list[Path]:
    return sorted(Path(sample_dir).glob("*.wav"),
                  key=lambda p: p.name)


def validate_wav(path: Path, errors: ErrorCollector,
                 round_hint: int | None = None) -> bool:
    """校验 RIFF 容器头、采样率、单声道、16bit 与数据块长度一致性。"""
    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except (wave.Error, EOFError, OSError) as exc:
        errors.add(ErrorCode.E_AUDIO_FORMAT, "audio",
                   f"invalid RIFF/WAV container: {exc}",
                   round=round_hint, file=path.name)
        return False

    problems = []
    if channels != N_CHANNELS:
        problems.append(f"channels={channels}")
    if sampwidth != SAMPLE_WIDTH:
        problems.append(f"sampwidth={sampwidth}")
    if framerate != SAMPLE_RATE:
        problems.append(f"framerate={framerate}")
    expected_bytes = n_frames * channels * sampwidth
    if len(raw) != expected_bytes:
        problems.append(f"data bytes {len(raw)} != {expected_bytes}")
    if problems:
        errors.add(ErrorCode.E_AUDIO_FORMAT, "audio",
                   "; ".join(problems), round=round_hint, file=path.name)
        return False
    return True


def load_wav(path: Path, round_index: int,
             content_hash: str) -> AudioClip:
    with wave.open(str(path), "rb") as wf:
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    samples = pcm / 32768.0
    return AudioClip(round=round_index, path=path, samples=samples,
                     n_samples=samples.shape[0], content_hash=content_hash)


def write_wav(path: str | Path, samples: np.ndarray) -> None:
    """合成脚本使用：float64 [-1,1] -> 16kHz 单声道 16bit PCM WAV。"""
    pcm = np.clip(np.round(samples * 32768.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(N_CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
