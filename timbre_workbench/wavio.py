"""Strict RIFF/WAVE validation and deterministic WAV writing."""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .errors import E_AUDIO_FORMAT, WorkbenchError
from .util import normalize_relpath, sha256_bytes


@dataclass
class AudioData:
    logical_name: str
    samples: np.ndarray
    sample_rate: int
    channels: int
    bits_per_sample: int
    sha256: str
    source_path: str = ""

    @property
    def sample_count(self) -> int:
        return int(self.samples.shape[0])

    def clone(self) -> "AudioData":
        return AudioData(
            logical_name=self.logical_name,
            samples=self.samples.copy(),
            sample_rate=self.sample_rate,
            channels=self.channels,
            bits_per_sample=self.bits_per_sample,
            sha256=self.sha256,
            source_path=self.source_path,
        )


def _error(message: str, **context: Any) -> WorkbenchError:
    return WorkbenchError(E_AUDIO_FORMAT, message, context or None)


def parse_wav_bytes(
    blob: bytes,
    *,
    logical_name: str = "",
    source_path: str = "",
    required_sample_rate: int = 16000,
) -> AudioData:
    if len(blob) < 44:
        raise _error("WAV file is shorter than a canonical 44-byte header")
    if blob[0:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise _error("missing RIFF/WAVE container header")
    riff_size = struct.unpack_from("<I", blob, 4)[0]
    if riff_size != len(blob) - 8:
        raise _error(
            "RIFF size does not match file length",
            declared_riff_size=riff_size,
            actual_payload_size=len(blob) - 8,
        )

    offset = 12
    fmt: dict[str, int] | None = None
    pcm_payload: bytes | None = None
    while offset + 8 <= len(blob):
        chunk_id = blob[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", blob, offset + 4)[0]
        data_start = offset + 8
        data_end = data_start + chunk_size
        padded_end = data_end + (chunk_size & 1)
        if data_end > len(blob) or padded_end > len(blob):
            raise _error("chunk extends beyond the RIFF payload", chunk_id=chunk_id.decode("latin1"))
        chunk = blob[data_start:data_end]
        if chunk_id == b"fmt ":
            if fmt is not None:
                raise _error("duplicate fmt chunk")
            if chunk_size < 16:
                raise _error("fmt chunk is shorter than 16 bytes")
            audio_format, channels, sample_rate, byte_rate, block_align, bits = struct.unpack_from(
                "<HHIIHH", chunk, 0
            )
            fmt = {
                "audio_format": audio_format,
                "channels": channels,
                "sample_rate": sample_rate,
                "byte_rate": byte_rate,
                "block_align": block_align,
                "bits_per_sample": bits,
            }
        elif chunk_id == b"data":
            if pcm_payload is not None:
                raise _error("multiple data chunks are not supported")
            pcm_payload = chunk
        offset = padded_end

    if fmt is None:
        raise _error("missing fmt chunk")
    if pcm_payload is None:
        raise _error("missing data chunk")
    if fmt["audio_format"] != 1:
        raise _error("only PCM audio_format=1 is supported", audio_format=fmt["audio_format"])
    if fmt["channels"] != 1:
        raise _error("mono audio is required", channels=fmt["channels"])
    if fmt["sample_rate"] != required_sample_rate:
        raise _error(
            "unexpected sample rate",
            expected=required_sample_rate,
            actual=fmt["sample_rate"],
        )
    if fmt["bits_per_sample"] != 16:
        raise _error("16-bit PCM is required", bits_per_sample=fmt["bits_per_sample"])
    if fmt["block_align"] != 2 or fmt["byte_rate"] != required_sample_rate * 2:
        raise _error(
            "inconsistent block alignment or byte rate",
            block_align=fmt["block_align"],
            byte_rate=fmt["byte_rate"],
        )
    if len(pcm_payload) % fmt["block_align"] != 0:
        raise _error(
            "data chunk length is not a whole number of samples",
            data_bytes=len(pcm_payload),
            block_align=fmt["block_align"],
        )
    samples = np.frombuffer(pcm_payload, dtype="<i2").astype(np.int16, copy=True)
    return AudioData(
        logical_name=normalize_relpath(logical_name),
        samples=samples,
        sample_rate=fmt["sample_rate"],
        channels=fmt["channels"],
        bits_per_sample=fmt["bits_per_sample"],
        sha256=sha256_bytes(pcm_payload),
        source_path=source_path,
    )


def read_wav(path: Path, *, logical_name: str = "", required_sample_rate: int = 16000) -> AudioData:
    try:
        blob = path.read_bytes()
    except OSError as exc:
        raise _error("unable to read WAV file", os_error=str(exc)) from exc
    return parse_wav_bytes(
        blob,
        logical_name=logical_name or path.name,
        source_path=str(path),
        required_sample_rate=required_sample_rate,
    )


def wav_bytes(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    pcm = np.asarray(samples, dtype="<i2").tobytes(order="C")
    if len(pcm) % 2:
        raise ValueError("PCM payload must contain whole 16-bit samples")
    fmt = struct.pack(
        "<HHIIHH",
        1,
        1,
        int(sample_rate),
        int(sample_rate) * 2,
        2,
        16,
    )
    data_header = struct.pack("<4sI", b"data", len(pcm))
    chunks = struct.pack("<4sI", b"fmt ", len(fmt)) + fmt + data_header + pcm
    return struct.pack("<4sI4s", b"RIFF", len(chunks) + 4, b"WAVE") + chunks


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wav_bytes(samples, sample_rate=sample_rate))


def scan_wav_directory(directory: Path, *, required_sample_rate: int = 16000) -> tuple[list[AudioData], list[dict[str, Any]]]:
    assets: list[AudioData] = []
    failures: list[dict[str, Any]] = []
    candidates: list[tuple[str, Path]] = []
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".wav":
            relative = normalize_relpath(str(path.relative_to(directory)))
            candidates.append((relative, path))
    candidates.sort(key=lambda item: (item[0].casefold(), item[0]))
    for relative, path in candidates:
        try:
            assets.append(read_wav(path, logical_name=relative, required_sample_rate=required_sample_rate))
        except WorkbenchError as exc:
            record = exc.as_failure(file=relative, stage="audio_read", sort_key=f"{relative}|audio_read|{exc.code}")
            failures.append(record)
        except Exception as exc:  # defense in depth: one bad file must not stop the scan
            record = {
                "error_code": E_AUDIO_FORMAT,
                "message": f"unexpected WAV validation failure: {exc}",
                "file": relative,
                "stage": "audio_read",
                "sort_key": f"{relative}|audio_read|{E_AUDIO_FORMAT}",
            }
            failures.append(record)
    failures.sort(key=lambda row: row["sort_key"])
    return assets, failures


def copy_audio(asset: AudioData, logical_name: str | None = None) -> AudioData:
    copied = asset.clone()
    if logical_name is not None:
        copied.logical_name = normalize_relpath(logical_name)
    return copied

