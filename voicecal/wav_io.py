"""RIFF/WAVE validation and PCM decoding with no external audio dependency."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import E_AUDIO_FORMAT
from .errors import VoicecalError


@dataclass(frozen=True)
class WavPayload:
    path: Path
    raw: bytes
    samples: np.ndarray
    sample_rate: int
    channels: int
    bits_per_sample: int
    data_bytes: int


def _fail(path: Path, reason: str, **details: object) -> VoicecalError:
    return VoicecalError(E_AUDIO_FORMAT, reason, file=str(path), **details)


def parse_wav_bytes(raw: bytes, path: Path = Path("<memory>")) -> WavPayload:
    if len(raw) < 12:
        raise _fail(path, "file too short for RIFF/WAVE header", size=len(raw))
    if raw[0:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise _fail(path, "missing RIFF/WAVE container header")
    riff_size = struct.unpack_from("<I", raw, 4)[0]
    if riff_size + 8 != len(raw):
        raise _fail(
            path,
            "RIFF chunk size does not match file length",
            riff_size=riff_size,
            file_size=len(raw),
        )

    offset = 12
    fmt: tuple[int, int, int, int, int, int] | None = None
    data: bytes | None = None
    data_offset = 0
    while offset + 8 <= len(raw):
        chunk_id = raw[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", raw, offset + 4)[0]
        payload_start = offset + 8
        payload_end = payload_start + chunk_size
        if payload_end > len(raw):
            raise _fail(path, "chunk extends beyond end of file", chunk=chunk_id.decode("latin1"))
        if chunk_id == b"fmt ":
            if chunk_size < 16:
                raise _fail(path, "fmt chunk is shorter than 16 bytes", chunk_size=chunk_size)
            fmt = struct.unpack_from("<HHIIHH", raw, payload_start)
        elif chunk_id == b"data":
            if data is not None:
                raise _fail(path, "multiple data chunks are not supported")
            data = raw[payload_start:payload_end]
            data_offset = payload_start
        offset = payload_end + (chunk_size & 1)

    if fmt is None:
        raise _fail(path, "missing fmt chunk")
    if data is None:
        raise _fail(path, "missing data chunk")
    audio_format, channels, sample_rate, byte_rate, block_align, bits = fmt
    if audio_format != 1:
        raise _fail(path, "audio format must be PCM", audio_format=audio_format)
    if channels != 1:
        raise _fail(path, "audio must be mono", channels=channels)
    if sample_rate != 16000:
        raise _fail(path, "sample rate must be 16000 Hz", sample_rate=sample_rate)
    if bits != 16:
        raise _fail(path, "bits per sample must be 16", bits_per_sample=bits)
    if block_align != 2:
        raise _fail(path, "block alignment must equal 2 bytes", block_align=block_align)
    if byte_rate != 32000:
        raise _fail(path, "byte rate must equal 32000", byte_rate=byte_rate)
    if len(data) % 2 != 0:
        raise _fail(path, "data chunk length is not a whole number of int16 samples", data_bytes=len(data))
    sample_count = len(data) // 2
    if len(data) != sample_count * 2:
        raise _fail(path, "data length and sample count disagree", data_bytes=len(data), samples=sample_count)
    if sample_count == 0:
        raise _fail(path, "audio contains no samples", data_offset=data_offset)
    samples = np.frombuffer(data, dtype="<i2").copy()
    return WavPayload(
        path=path,
        raw=raw,
        samples=samples,
        sample_rate=sample_rate,
        channels=channels,
        bits_per_sample=bits,
        data_bytes=len(data),
    )


def read_wav(path: Path, injected_bytes: bytes | None = None) -> WavPayload:
    if injected_bytes is None:
        try:
            raw = path.read_bytes()
        except FileNotFoundError as exc:
            from .constants import E_AUDIO_MISSING

            raise VoicecalError(E_AUDIO_MISSING, "audio file is missing", file=str(path)) from exc
    else:
        raw = injected_bytes
    return parse_wav_bytes(raw, path)


def scan_wav_files(root: Path) -> list[Path]:
    return sorted((path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".wav"), key=lambda p: p.relative_to(root).as_posix())
