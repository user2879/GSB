"""Strict PCM WAV reader used by the VAD pipeline."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .util import sha256_bytes


class AudioFormatError(ValueError):
    code = "E_AUDIO_FORMAT"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class AudioData:
    file: str
    content_sha256: str
    samples: np.ndarray
    sample_rate: int
    duration_ms: float

    @property
    def sample_count(self) -> int:
        return int(self.samples.size)


def read_wav(path: Path, audio_config: dict[str, Any], relative_name: str) -> AudioData:
    try:
        blob = path.read_bytes()
    except OSError as exc:
        raise AudioFormatError(f"cannot read file: {exc}") from exc
    sample_rate, samples = parse_pcm_wav(blob, audio_config)
    return AudioData(
        file=relative_name,
        content_sha256=sha256_bytes(blob),
        samples=samples,
        sample_rate=sample_rate,
        duration_ms=float(samples.size) * 1000.0 / float(sample_rate),
    )


def parse_pcm_wav(blob: bytes, audio_config: dict[str, Any]) -> tuple[int, np.ndarray]:
    if len(blob) < 12:
        raise AudioFormatError("file is shorter than a RIFF header")
    if blob[0:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise AudioFormatError("missing RIFF/WAVE container header")
    riff_size = struct.unpack_from("<I", blob, 4)[0]
    if riff_size + 8 != len(blob):
        raise AudioFormatError(
            f"RIFF chunk size mismatch: header={riff_size}, actual={len(blob) - 8}"
        )

    fmt: tuple[int, int, int, int, int, int] | None = None
    data_payload: bytes | None = None
    position = 12
    while position < len(blob):
        if position + 8 > len(blob):
            raise AudioFormatError("truncated subchunk header")
        chunk_id = blob[position : position + 4]
        chunk_size = struct.unpack_from("<I", blob, position + 4)[0]
        payload_start = position + 8
        payload_end = payload_start + chunk_size
        if payload_end > len(blob):
            raise AudioFormatError(
                f"{chunk_id!r} chunk length exceeds file boundary"
            )
        if chunk_id == b"fmt ":
            if fmt is not None:
                raise AudioFormatError("duplicate fmt chunk")
            if chunk_size < 16:
                raise AudioFormatError("fmt chunk is shorter than 16 bytes")
            fmt = struct.unpack_from("<HHIIHH", blob, payload_start)
        elif chunk_id == b"data":
            if data_payload is not None:
                raise AudioFormatError("duplicate data chunk")
            data_payload = blob[payload_start:payload_end]
        position = payload_end + (chunk_size & 1)
        if position > len(blob):
            raise AudioFormatError("missing subchunk padding byte")

    if fmt is None:
        raise AudioFormatError("missing fmt chunk")
    if data_payload is None:
        raise AudioFormatError("missing data chunk")
    if not data_payload:
        raise AudioFormatError("data chunk is empty")

    audio_format, channels, sample_rate, byte_rate, block_align, bits = fmt
    expected_rate = int(audio_config["sample_rate"])
    expected_channels = int(audio_config["channels"])
    expected_bits = int(audio_config["bits_per_sample"])
    if audio_format != 1:
        raise AudioFormatError(f"unsupported WAV format tag {audio_format}; PCM required")
    if channels != expected_channels:
        raise AudioFormatError(
            f"channel count mismatch: expected {expected_channels}, got {channels}"
        )
    if sample_rate != expected_rate:
        raise AudioFormatError(
            f"sample rate mismatch: expected {expected_rate}, got {sample_rate}"
        )
    if bits != expected_bits:
        raise AudioFormatError(
            f"bit depth mismatch: expected {expected_bits}, got {bits}"
        )
    expected_block_align = channels * (bits // 8)
    expected_byte_rate = sample_rate * expected_block_align
    if block_align != expected_block_align:
        raise AudioFormatError(
            f"block align mismatch: expected {expected_block_align}, got {block_align}"
        )
    if byte_rate != expected_byte_rate:
        raise AudioFormatError(
            f"byte rate mismatch: expected {expected_byte_rate}, got {byte_rate}"
        )
    if len(data_payload) % block_align != 0:
        raise AudioFormatError(
            "data chunk length is not a whole number of PCM samples"
        )
    samples = np.frombuffer(data_payload, dtype="<i2").astype(np.int16, copy=True)
    return sample_rate, samples
