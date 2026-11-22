from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import DiagnosticError


@dataclass(frozen=True)
class WavData:
    sample_rate: int
    channels: int
    frames: int
    pcm: np.ndarray


def _parse_chunks(data: bytes) -> tuple[dict[str, bytes], int]:
    if len(data) < 12 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("missing RIFF/WAVE header")
    riff_size = struct.unpack_from("<I", data, 4)[0]
    if riff_size != len(data) - 8:
        raise ValueError("RIFF size does not match file length")
    chunks: dict[str, bytes] = {}
    position = 12
    while position + 8 <= len(data):
        chunk_id = data[position : position + 4]
        chunk_size = struct.unpack_from("<I", data, position + 4)[0]
        data_start = position + 8
        data_end = data_start + chunk_size
        if data_end > len(data):
            raise ValueError(f"chunk {chunk_id!r} extends beyond file")
        name = chunk_id.decode("ascii", errors="replace")
        if name in chunks:
            raise ValueError(f"duplicate chunk {name!r}")
        chunks[name] = data[data_start:data_end]
        position = data_end + (chunk_size & 1)
    if position != len(data):
        raise ValueError("trailing bytes after RIFF chunks")
    return chunks, riff_size


def read_pcm16_wav(path: str | Path) -> WavData:
    wav_path = Path(path)
    try:
        data = wav_path.read_bytes()
        chunks, _ = _parse_chunks(data)
        fmt = chunks.get("fmt ")
        pcm_bytes = chunks.get("data")
        if fmt is None or pcm_bytes is None:
            raise ValueError("missing fmt or data chunk")
        if len(fmt) < 16:
            raise ValueError("short fmt chunk")
        audio_format, channels, sample_rate, byte_rate, block_align, bits = struct.unpack_from(
            "<HHIIHH", fmt, 0
        )
        if audio_format != 1:
            raise ValueError(f"unsupported audio format {audio_format}")
        if bits != 16:
            raise ValueError(f"unsupported bit depth {bits}")
        if channels < 1:
            raise ValueError("invalid channel count")
        expected_align = channels * 2
        if block_align != expected_align or byte_rate != sample_rate * expected_align:
            raise ValueError("inconsistent WAV alignment fields")
        if len(pcm_bytes) % expected_align:
            raise ValueError("PCM byte count is not frame aligned")
        pcm = np.frombuffer(pcm_bytes, dtype="<i2").copy()
        pcm = pcm.reshape((-1, channels)).T
        return WavData(
            sample_rate=int(sample_rate),
            channels=int(channels),
            frames=int(pcm.shape[1]),
            pcm=pcm,
        )
    except (OSError, ValueError, struct.error) as exc:
        raise DiagnosticError(
            code="E_TRACK_PARSE_FAILED",
            message=f"cannot read PCM16 WAV {wav_path}: {exc}",
            stage="parse",
            details={"path": str(wav_path)},
        ) from exc


def _wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    channels_pcm = np.asarray(pcm, dtype="<i2")
    if channels_pcm.ndim == 1:
        channels_pcm = channels_pcm.reshape((1, -1))
    if channels_pcm.ndim != 2:
        raise ValueError("PCM must be one- or two-dimensional")
    channels, frames = channels_pcm.shape
    interleaved = np.ascontiguousarray(channels_pcm.T).astype("<i2", copy=False)
    payload = interleaved.tobytes(order="C")
    block_align = channels * 2
    fmt = struct.pack(
        "<HHIIHH",
        1,
        channels,
        sample_rate,
        sample_rate * block_align,
        block_align,
        16,
    )
    chunks = [b"fmt " + struct.pack("<I", len(fmt)) + fmt]
    chunks.append(b"data" + struct.pack("<I", len(payload)) + payload)
    if len(payload) & 1:
        chunks.append(b"\x00")
    body = b"WAVE" + b"".join(chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def write_pcm16_wav(path: str | Path, pcm: np.ndarray, sample_rate: int) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_wav_bytes(np.asarray(pcm), sample_rate))


def write_multichannel_wav(path: str | Path, channels: np.ndarray, sample_rate: int) -> None:
    write_pcm16_wav(path, channels, sample_rate)
