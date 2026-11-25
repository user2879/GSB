"""RIFF/WAVE validation and deterministic PCM16 writing."""

from __future__ import annotations

import hashlib
import struct
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import AudioFormatFailure


@dataclass(frozen=True)
class AudioData:
    path: str
    samples: np.ndarray
    sample_rate: int
    channels: int
    bit_depth: int
    content_sha256: str
    byte_length: int

    @property
    def sample_count(self) -> int:
        return int(self.samples.size)


def _fail(path: Path, message: str, code: str = "E_AUDIO_FORMAT") -> None:
    raise AudioFormatFailure(str(path), message, code)


def read_pcm16_wav(path: Path, *, expected_rate: int = 16000) -> AudioData:
    """Strictly validate one WAV and decode signed little-endian PCM16.

    Every malformed byte, inconsistent chunk size, or unsupported format is
    isolated by the caller.  IEEE-float files are parsed only far enough to
    distinguish non-finite samples from an ordinary unsupported format.
    """

    try:
        raw = path.read_bytes()
    except OSError as exc:
        _fail(path, f"cannot read file: {exc}")
    if len(raw) < 12:
        _fail(path, "truncated RIFF header")
    if raw[0:4] != b"RIFF" or raw[8:12] != b"WAVE":
        _fail(path, "missing RIFF/WAVE container header")
    declared_riff = struct.unpack_from("<I", raw, 4)[0]
    riff_end = 8 + declared_riff
    if riff_end > len(raw):
        _fail(path, f"truncated RIFF payload: declared {declared_riff} bytes, file has {len(raw) - 8}")
    if riff_end < 12:
        _fail(path, "invalid RIFF size")
    if riff_end != len(raw):
        _fail(path, f"RIFF size mismatch: declared end {riff_end}, actual {len(raw)}")

    fmt: tuple[int, int, int, int, int, int] | None = None
    data_payload: bytes | None = None
    position = 12
    while position < riff_end:
        if position + 8 > riff_end:
            _fail(path, "truncated chunk header")
        chunk_id = raw[position : position + 4]
        chunk_size = struct.unpack_from("<I", raw, position + 4)[0]
        body_start = position + 8
        body_end = body_start + chunk_size
        if body_end > riff_end:
            _fail(path, f"truncated {chunk_id!r} chunk")
        body = raw[body_start:body_end]
        if chunk_id == b"fmt ":
            if chunk_size < 16:
                _fail(path, "fmt chunk shorter than 16 bytes")
            fmt = struct.unpack_from("<HHIIHH", body, 0)
        elif chunk_id == b"data":
            if data_payload is not None:
                _fail(path, "multiple data chunks are unsupported")
            data_payload = body
        position = body_end + (chunk_size & 1)
        if position > riff_end:
            _fail(path, "chunk padding exceeds RIFF payload")

    if fmt is None:
        _fail(path, "missing fmt chunk")
    if data_payload is None:
        _fail(path, "missing data chunk")
    format_tag, channels, sample_rate, byte_rate, block_align, bit_depth = fmt
    if sample_rate != expected_rate:
        _fail(path, f"sample rate must be {expected_rate}, got {sample_rate}")
    if channels != 1:
        _fail(path, f"channels must be 1, got {channels}")
    if format_tag == 3:
        if bit_depth != 32 or block_align != 4:
            _fail(path, "non-finite probe requires float32 mono samples")
        if len(data_payload) % 4:
            _fail(path, "float data length is not sample aligned")
        values = np.frombuffer(data_payload, dtype="<f4")
        if values.size and not np.isfinite(values).all():
            _fail(path, "non-finite sample value", "E_AUDIO_NONFINITE")
        _fail(path, "unsupported IEEE-float WAV; PCM16 is required")
    if format_tag != 1:
        _fail(path, f"unsupported WAV format tag {format_tag}; PCM16 is required")
    if bit_depth != 16:
        _fail(path, f"bit depth must be 16, got {bit_depth}")
    if block_align != channels * 2:
        _fail(path, f"invalid block alignment {block_align}")
    if byte_rate != sample_rate * block_align:
        _fail(path, f"invalid byte rate {byte_rate}")
    if len(data_payload) % block_align:
        _fail(path, "data chunk length is not aligned to complete samples")
    samples = np.frombuffer(data_payload, dtype="<i2").astype(np.int16, copy=True)
    payload_digest = hashlib.sha256(
        struct.pack("<III", sample_rate, channels, bit_depth) + samples.astype("<i2").tobytes()
    ).hexdigest()
    return AudioData(
        path=str(path),
        samples=samples,
        sample_rate=sample_rate,
        channels=channels,
        bit_depth=bit_depth,
        content_sha256=payload_digest,
        byte_length=len(data_payload),
    )


def write_pcm16_wav(path: Path, samples: np.ndarray | list[int], *, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(samples)
    if values.ndim != 1:
        raise ValueError("samples must be mono")
    if np.issubdtype(values.dtype, np.floating):
        if not np.isfinite(values).all():
            raise ValueError("samples must be finite")
        values = np.clip(np.rint(values), -32768, 32767).astype("<i2")
    else:
        values = np.clip(values, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(values.tobytes())


def scan_wav_files(root: Path) -> list[Path]:
    return sorted((path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".wav"), key=lambda p: p.as_posix())


def format_failure(path: Path, exc: AudioFormatFailure) -> dict:
    return exc.failure.as_dict()
