from __future__ import annotations

import hashlib
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import E_AUDIO_FORMAT, E_AUDIO_SAMPLE, BusinessError, failure


@dataclass(frozen=True)
class AudioFile:
    name: str
    samples: np.ndarray
    source_path: Path | None = None

    @property
    def sample_count(self) -> int:
        return int(self.samples.size)

    @property
    def duration_ms(self) -> float:
        return self.sample_count / 16000.0 * 1000.0

    def content_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.samples.astype("<i2", copy=False).tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class AudioScan:
    files: dict[str, AudioFile]
    failures: list[dict[str, Any]]


def validate_samples(samples: Any, *, name: str = "samples") -> np.ndarray:
    array = np.asarray(samples)
    if array.ndim != 1:
        raise BusinessError(E_AUDIO_SAMPLE, f"{name} must be one-dimensional", shape=list(array.shape))
    if not np.issubdtype(array.dtype, np.integer):
        try:
            if not np.all(np.isfinite(array)):
                raise BusinessError(E_AUDIO_SAMPLE, f"{name} contains non-finite values")
        except TypeError as exc:
            raise BusinessError(E_AUDIO_SAMPLE, f"{name} is not a numeric sample array") from exc
    if array.size and (int(np.min(array)) < -32768 or int(np.max(array)) > 32767):
        raise BusinessError(E_AUDIO_SAMPLE, f"{name} contains values outside int16 range")
    return array.astype(np.int16, copy=False)


def read_wav_strict(path: str | Path) -> AudioFile:
    audio_path = Path(path)
    try:
        payload = audio_path.read_bytes()
    except OSError as exc:
        raise BusinessError(E_AUDIO_FORMAT, f"cannot read WAV: {exc}", file=audio_path.name) from exc
    if len(payload) < 44:
        raise BusinessError(E_AUDIO_FORMAT, "WAV is shorter than a minimal RIFF/PCM file", file=audio_path.name)
    if payload[0:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise BusinessError(E_AUDIO_FORMAT, "missing RIFF/WAVE container signature", file=audio_path.name)
    riff_size = struct.unpack_from("<I", payload, 4)[0]
    if riff_size + 8 != len(payload):
        raise BusinessError(
            E_AUDIO_FORMAT,
            "RIFF size does not match file length",
            file=audio_path.name,
            riff_size=riff_size,
            file_size=len(payload),
        )

    offset = 12
    fmt: tuple[int, int, int, int, int, int] | None = None
    data: bytes | None = None
    data_size = 0
    seen_chunks: list[str] = []
    while offset + 8 <= len(payload):
        chunk_id = payload[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", payload, offset + 4)[0]
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(payload):
            raise BusinessError(
                E_AUDIO_FORMAT,
                f"chunk {chunk_id!r} extends beyond end of file",
                file=audio_path.name,
                chunk=chunk_id.decode("latin-1", errors="replace"),
                chunk_size=chunk_size,
            )
        seen_chunks.append(chunk_id.decode("latin-1", errors="replace"))
        if chunk_id == b"fmt ":
            if fmt is not None:
                raise BusinessError(E_AUDIO_FORMAT, "duplicate fmt chunk", file=audio_path.name)
            if chunk_size < 16:
                raise BusinessError(E_AUDIO_FORMAT, "fmt chunk is shorter than 16 bytes", file=audio_path.name)
            fmt = struct.unpack_from("<HHIIHH", payload, chunk_start)
        elif chunk_id == b"data":
            if data is not None:
                raise BusinessError(E_AUDIO_FORMAT, "multiple data chunks are not supported", file=audio_path.name)
            data = payload[chunk_start:chunk_end]
            data_size = chunk_size
        offset = chunk_end + (chunk_size & 1)
    if offset != len(payload):
        raise BusinessError(E_AUDIO_FORMAT, "trailing bytes after final RIFF chunk", file=audio_path.name)
    if fmt is None:
        raise BusinessError(E_AUDIO_FORMAT, "missing fmt chunk", file=audio_path.name)
    if data is None:
        raise BusinessError(E_AUDIO_FORMAT, "missing data chunk", file=audio_path.name)
    audio_format, channels, sample_rate, byte_rate, block_align, bits_per_sample = fmt
    problems: list[str] = []
    if audio_format != 1:
        problems.append(f"audio format is {audio_format}, expected PCM(1)")
    if channels != 1:
        problems.append(f"channel count is {channels}, expected 1")
    if sample_rate != 16000:
        problems.append(f"sample rate is {sample_rate}, expected 16000")
    if bits_per_sample != 16:
        problems.append(f"sample width is {bits_per_sample}, expected 16")
    if block_align != 2:
        problems.append(f"block align is {block_align}, expected 2")
    if byte_rate != 32000:
        problems.append(f"byte rate is {byte_rate}, expected 32000")
    if data_size % 2:
        problems.append("data chunk length is not a multiple of 2")
    if problems:
        raise BusinessError(E_AUDIO_FORMAT, "; ".join(problems), file=audio_path.name, chunks=seen_chunks)
    samples = np.frombuffer(data, dtype="<i2").astype(np.int16, copy=True)
    validate_samples(samples, name=audio_path.name)
    return AudioFile(name=audio_path.name, samples=samples, source_path=audio_path)


def write_wav(path: str | Path, samples: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    safe = validate_samples(samples, name=target.name)
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(safe.astype("<i2", copy=False).tobytes())


def scan_wavs(directory: str | Path) -> AudioScan:
    root = Path(directory)
    files: dict[str, AudioFile] = {}
    failures: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.wav"), key=lambda item: item.name):
        try:
            audio = read_wav_strict(path)
            if audio.name in files:
                raise BusinessError(E_AUDIO_FORMAT, "duplicate WAV basename", file=audio.name)
            files[audio.name] = audio
        except BusinessError as exc:
            failures.append(failure(exc.code, exc.message, **{"file": path.name, **exc.details}))
    return AudioScan(files=files, failures=failures)
