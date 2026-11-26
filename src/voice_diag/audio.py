from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .errors import AudioFormatError
from .util import sha256_bytes


@dataclass(frozen=True)
class WavInfo:
    sample_rate: int
    channels: int
    bits_per_sample: int
    sample_count: int
    data_size: int
    pcm: np.ndarray
    content_hash: str

    @property
    def duration_samples(self) -> int:
        return self.sample_count


@dataclass(frozen=True)
class AudioFragment:
    fragment_id: str
    interval_id: str
    speaker_id: str
    path: Path
    global_start: int
    global_end: int
    arrival_index: int
    capture_start_ms: int
    arrival_time_ms: int
    wav: WavInfo

    @property
    def sample_count(self) -> int:
        return self.global_end - self.global_start


def _checked_slice(data: bytes, start: int, size: int, where: str) -> bytes:
    end = start + size
    if start < 0 or size < 0 or end > len(data):
        raise AudioFormatError("RIFF chunk extends beyond file", where=where)
    return data[start:end]


def parse_wav_bytes(
    data: bytes,
    *,
    expected_samples: int | None = None,
    where: str = "<bytes>",
    sample_rate: int = 16000,
    channels: int = 1,
    bits_per_sample: int = 16,
) -> WavInfo:
    """Strictly validate a RIFF/WAVE PCM16 container.

    Header-only inspection is not sufficient here: the declared RIFF size,
    chunk boundaries, padding, ``fmt`` consistency, and data/sample count are
    all checked before any PCM is accepted.
    """

    if len(data) < 12:
        raise AudioFormatError("file is shorter than a RIFF header", where=where)
    if data[0:4] != b"RIFF":
        raise AudioFormatError("missing RIFF signature", where=where)
    if data[8:12] != b"WAVE":
        raise AudioFormatError("missing WAVE form type", where=where)
    riff_size = struct.unpack_from("<I", data, 4)[0]
    if riff_size + 8 != len(data):
        raise AudioFormatError(
            "RIFF size does not match actual file length",
            where=where,
            details={"riff_size_plus_8": riff_size + 8, "actual": len(data)},
        )

    offset = 12
    fmt: tuple[int, int, int, int, int, int] | None = None
    pcm_bytes: bytes | None = None
    seen_fmt = False
    seen_data = False
    while offset < len(data):
        if offset + 8 > len(data):
            raise AudioFormatError("truncated RIFF chunk header", where=where)
        chunk_id = data[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
        payload_start = offset + 8
        _checked_slice(data, payload_start, chunk_size, where)
        payload = data[payload_start : payload_start + chunk_size]
        if chunk_id == b"fmt ":
            if seen_fmt:
                raise AudioFormatError("multiple fmt chunks are not accepted", where=where)
            seen_fmt = True
            if chunk_size < 16:
                raise AudioFormatError("fmt chunk is shorter than 16 bytes", where=where)
            audio_format, num_channels, rate, byte_rate, block_align, bits = struct.unpack_from(
                "<HHIIHH", payload, 0
            )
            if audio_format != 1:
                raise AudioFormatError("only integer PCM WAV is accepted", where=where, details={"audio_format": audio_format})
            if num_channels != channels:
                raise AudioFormatError(
                    "WAV channel count must be mono",
                    where=where,
                    details={"channels": num_channels},
                )
            if rate != sample_rate:
                raise AudioFormatError(
                    "WAV sample rate must be 16000",
                    where=where,
                    details={"sample_rate": rate},
                )
            if bits != bits_per_sample:
                raise AudioFormatError(
                    "WAV bit depth must be 16",
                    where=where,
                    details={"bits_per_sample": bits},
                )
            expected_block_align = num_channels * bits // 8
            if block_align != expected_block_align:
                raise AudioFormatError(
                    "WAV block alignment is inconsistent with format",
                    where=where,
                    details={"block_align": block_align, "expected": expected_block_align},
                )
            if byte_rate != rate * block_align:
                raise AudioFormatError(
                    "WAV byte rate is inconsistent with format",
                    where=where,
                    details={"byte_rate": byte_rate, "expected": rate * block_align},
                )
            fmt = (audio_format, num_channels, rate, byte_rate, block_align, bits)
        elif chunk_id == b"data":
            if seen_data:
                raise AudioFormatError("multiple data chunks are not accepted", where=where)
            seen_data = True
            pcm_bytes = payload
        # Chunks are word aligned.  A missing pad byte at EOF is malformed.
        padded_size = chunk_size + (chunk_size & 1)
        offset = payload_start + padded_size
        if offset > len(data):
            raise AudioFormatError("RIFF chunk padding extends beyond file", where=where)

    if fmt is None:
        raise AudioFormatError("WAV has no fmt chunk", where=where)
    if pcm_bytes is None:
        raise AudioFormatError("WAV has no data chunk", where=where)
    block_align = fmt[4]
    if len(pcm_bytes) % block_align != 0:
        raise AudioFormatError(
            "data chunk length is not a whole number of samples",
            where=where,
            details={"data_size": len(pcm_bytes), "block_align": block_align},
        )
    sample_count = len(pcm_bytes) // block_align
    if expected_samples is not None and sample_count != expected_samples:
        raise AudioFormatError(
            "data chunk sample count does not match the declared interval length",
            where=where,
            details={"expected_samples": expected_samples, "actual_samples": sample_count},
        )
    pcm = np.frombuffer(pcm_bytes, dtype="<i2").copy()
    if pcm.size != sample_count:
        raise AudioFormatError("PCM decode produced an inconsistent sample count", where=where)
    content_hash = sha256_bytes(pcm.tobytes(order="C"))
    return WavInfo(
        sample_rate=fmt[2],
        channels=fmt[1],
        bits_per_sample=fmt[5],
        sample_count=sample_count,
        data_size=len(pcm_bytes),
        pcm=pcm,
        content_hash=content_hash,
    )


def read_wav(
    path: str | Path,
    *,
    expected_samples: int | None = None,
    where: str | None = None,
) -> WavInfo:
    wav_path = Path(path)
    location = where or str(wav_path)
    try:
        data = wav_path.read_bytes()
    except OSError as exc:
        raise AudioFormatError(f"cannot read WAV: {exc}", where=location) from exc
    return parse_wav_bytes(data, expected_samples=expected_samples, where=location)


def write_pcm16_wav(path: str | Path, pcm: np.ndarray, sample_rate: int = 16000) -> None:
    samples = np.asarray(pcm)
    if samples.ndim != 1:
        raise ValueError("PCM input must be mono")
    if samples.dtype != np.dtype("<i2"):
        samples = np.clip(samples, -32768, 32767).astype("<i2")
    payload = samples.astype("<i2", copy=False).tobytes(order="C")
    fmt_payload = struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16)
    chunks = b"fmt " + struct.pack("<I", len(fmt_payload)) + fmt_payload
    chunks += b"data" + struct.pack("<I", len(payload)) + payload
    if len(payload) & 1:
        chunks += b"\x00"
    riff_size = 4 + len(chunks)
    data = b"RIFF" + struct.pack("<I", riff_size) + b"WAVE" + chunks
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def mix_timeline(
    fragments: Iterable[AudioFragment],
    *,
    timeline_end: int,
) -> tuple[np.ndarray, str]:
    accumulator = np.zeros(timeline_end, dtype=np.int32)
    ordered = sorted(fragments, key=lambda item: (item.arrival_index, item.fragment_id))
    for fragment in ordered:
        pcm = fragment.wav.pcm.astype(np.int32, copy=False)
        start = fragment.global_start
        end = fragment.global_end
        if start < 0 or end > timeline_end or pcm.size != end - start:
            raise AudioFormatError(
                "fragment bounds or PCM length exceed the timeline",
                where=fragment.fragment_id,
            )
        accumulator[start:end] += pcm
    mixed = np.clip(accumulator, -32768, 32767).astype("<i2")
    content_hash = sha256_bytes(mixed.tobytes(order="C"))
    return mixed, content_hash


def mix_window(
    fragments: Iterable[AudioFragment],
    *,
    start: int,
    end: int,
) -> np.ndarray:
    result = np.zeros(end - start, dtype=np.int32)
    for fragment in sorted(fragments, key=lambda item: (item.arrival_index, item.fragment_id)):
        overlap_start = max(start, fragment.global_start)
        overlap_end = min(end, fragment.global_end)
        if overlap_start >= overlap_end:
            continue
        local_start = overlap_start - fragment.global_start
        local_end = local_start + (overlap_end - overlap_start)
        out_start = overlap_start - start
        result[out_start : out_start + (overlap_end - overlap_start)] += fragment.wav.pcm[
            local_start:local_end
        ].astype(np.int32)
    return np.clip(result, -32768, 32767).astype("<i2")


def fragment_window(
    fragment: AudioFragment,
    *,
    start: int,
    end: int,
) -> tuple[np.ndarray, int, int] | None:
    overlap_start = max(start, fragment.global_start)
    overlap_end = min(end, fragment.global_end)
    if overlap_start >= overlap_end:
        return None
    local_start = overlap_start - fragment.global_start
    local_end = local_start + (overlap_end - overlap_start)
    result = np.zeros(end - start, dtype="<i2")
    out_start = overlap_start - start
    result[out_start : out_start + (overlap_end - overlap_start)] = fragment.wav.pcm[
        local_start:local_end
    ]
    return result, overlap_start, overlap_end
