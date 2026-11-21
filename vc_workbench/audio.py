"""Strict WAV loading using only the standard-library wave module."""

from __future__ import annotations

import io
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import AudioFormatError
from .util import relative_posix, sha256_bytes, sha256_value


@dataclass(slots=True, frozen=True)
class WavAudio:
    path: Path
    relative_path: str
    pcm: bytes
    samples: np.ndarray
    sample_rate: int
    channels: int
    sample_width: int
    frame_count: int
    content_fingerprint: str


@dataclass(slots=True, frozen=True)
class WavFailure:
    path: Path
    relative_path: str
    detail: str
    raw_fingerprint: str


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def validate_and_decode_wav(path: Path, relative_path: str | None = None) -> WavAudio:
    """Validate the RIFF chunk graph and decode PCM via wave.

    The validation is intentionally stricter than wave.open: truncated RIFF
    sizes, mismatched data lengths and non-PCM/16k/mono/16-bit inputs are all
    rejected with one business-facing AudioFormatError.
    """

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AudioFormatError(f"cannot read WAV: {exc}") from exc
    if len(raw) < 12:
        raise AudioFormatError("RIFF header is truncated")
    if raw[0:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise AudioFormatError("not a RIFF/WAVE container")
    riff_size = _u32(raw, 4)
    if riff_size < 4 or len(raw) != riff_size + 8:
        raise AudioFormatError(
            f"RIFF size mismatch: declared={riff_size}, actual_payload={max(0, len(raw) - 8)}"
        )

    position = 12
    fmt_chunk: bytes | None = None
    data_chunk: bytes | None = None
    while position < len(raw):
        if position + 8 > len(raw):
            raise AudioFormatError("truncated chunk header")
        chunk_id = raw[position : position + 4]
        chunk_size = _u32(raw, position + 4)
        body_start = position + 8
        body_end = body_start + chunk_size
        if body_end > len(raw):
            raise AudioFormatError(
                f"chunk {chunk_id!r} declares {chunk_size} bytes beyond file end"
            )
        body = raw[body_start:body_end]
        if chunk_id == b"fmt ":
            if fmt_chunk is not None:
                raise AudioFormatError("duplicate fmt chunk")
            fmt_chunk = body
        elif chunk_id == b"data":
            if data_chunk is not None:
                raise AudioFormatError("duplicate data chunk")
            data_chunk = body
        position = body_end + (chunk_size & 1)
        if position > len(raw):
            raise AudioFormatError("missing RIFF chunk padding byte")

    if position != len(raw):
        raise AudioFormatError("chunks do not exactly fill the RIFF container")
    if fmt_chunk is None or len(fmt_chunk) < 16:
        raise AudioFormatError("missing or truncated fmt chunk")
    if data_chunk is None:
        raise AudioFormatError("missing data chunk")

    audio_format = _u16(fmt_chunk, 0)
    channels = _u16(fmt_chunk, 2)
    sample_rate = _u32(fmt_chunk, 4)
    byte_rate = _u32(fmt_chunk, 8)
    block_align = _u16(fmt_chunk, 12)
    bits_per_sample = _u16(fmt_chunk, 14)
    if audio_format != 1:
        raise AudioFormatError(f"audio format must be PCM (1), got {audio_format}")
    if channels != 1:
        raise AudioFormatError(f"channel count must be 1, got {channels}")
    if sample_rate != 16000:
        raise AudioFormatError(f"sample rate must be 16000, got {sample_rate}")
    if bits_per_sample != 16:
        raise AudioFormatError(f"sample width must be 16 bit, got {bits_per_sample}")
    if block_align != 2 or byte_rate != 32000:
        raise AudioFormatError(
            f"inconsistent PCM layout: block_align={block_align}, byte_rate={byte_rate}"
        )
    if len(data_chunk) % 2:
        raise AudioFormatError("data chunk length is not an integral number of 16-bit samples")
    sample_count = len(data_chunk) // 2

    try:
        with wave.open(io.BytesIO(raw), "rb") as reader:
            if reader.getnchannels() != 1 or reader.getsampwidth() != 2:
                raise AudioFormatError("wave decoder disagrees with PCM fmt chunk")
            if reader.getframerate() != 16000:
                raise AudioFormatError("wave decoder sample rate is not 16000")
            if reader.getnframes() != sample_count:
                raise AudioFormatError(
                    f"data length/sample count mismatch: wave={reader.getnframes()}, chunk={sample_count}"
                )
            decoded = reader.readframes(sample_count)
    except (wave.Error, EOFError) as exc:
        raise AudioFormatError(f"wave decoder rejected container: {exc}") from exc
    if len(decoded) != len(data_chunk):
        raise AudioFormatError("wave decoder returned a different number of PCM bytes")

    samples = np.frombuffer(decoded, dtype="<i2").astype(np.int16, copy=True)
    content_fingerprint = sha256_value(
        {
            "format": "pcm_s16le_mono_16000",
            "sample_count": sample_count,
            "pcm_sha256": sha256_bytes(decoded),
        }
    )
    return WavAudio(
        path=path,
        relative_path=relative_path or path.name,
        pcm=decoded,
        samples=samples,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=2,
        frame_count=sample_count,
        content_fingerprint=content_fingerprint,
    )


def load_wav_checked(path: Path, base: Path) -> WavAudio:
    return validate_and_decode_wav(path, relative_posix(path, base))


def scan_wavs(directory: Path, base: Path) -> tuple[dict[str, WavAudio], list[WavFailure]]:
    """Scan a directory recursively in lexical relative-path order."""

    loaded: dict[str, WavAudio] = {}
    failures: list[WavFailure] = []
    if not directory.exists():
        return loaded, failures
    paths = sorted((p for p in directory.rglob("*.wav") if p.is_file()), key=lambda p: relative_posix(p, base))
    for path in paths:
        rel = relative_posix(path, base)
        try:
            loaded[rel] = validate_and_decode_wav(path, rel)
        except AudioFormatError as exc:
            try:
                raw = path.read_bytes()
                raw_hash = sha256_bytes(raw)
            except OSError:
                raw_hash = "unreadable"
            failures.append(WavFailure(path=path, relative_path=rel, detail=str(exc), raw_fingerprint=raw_hash))
    return loaded, failures


@dataclass(slots=True)
class AudioCatalog:
    base: Path
    references: dict[str, WavAudio]
    sources: dict[str, WavAudio]
    failures: list[WavFailure]
    fingerprint: str

    @classmethod
    def scan(cls, base: Path) -> "AudioCatalog":
        references, ref_failures = scan_wavs(base / "references", base)
        sources, source_failures = scan_wavs(base / "sources", base)
        failures = sorted(ref_failures + source_failures, key=lambda f: f.relative_path)
        payload = {
            "references": [
                {
                    "path": rel,
                    "content": audio.content_fingerprint,
                    "sample_count": audio.frame_count,
                }
                for rel, audio in sorted(references.items())
            ],
            "sources": [
                {
                    "path": rel,
                    "content": audio.content_fingerprint,
                    "sample_count": audio.frame_count,
                }
                for rel, audio in sorted(sources.items())
            ],
            "failures": [
                {"path": failure.relative_path, "detail": failure.detail, "raw": failure.raw_fingerprint}
                for failure in failures
            ],
        }
        return cls(
            base=base,
            references=references,
            sources=sources,
            failures=failures,
            fingerprint=sha256_value(payload),
        )

    def reference_speakers(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for rel in sorted(self.references):
            parts = Path(rel).parts
            speaker = parts[1] if len(parts) >= 3 and parts[0] == "references" else Path(rel).stem
            result.setdefault(speaker, []).append(rel)
        return {speaker: sorted(paths) for speaker, paths in sorted(result.items())}

    def failure_payloads(self) -> list[dict[str, Any]]:
        return [
            {
                "code": "E_AUDIO_FORMAT",
                "stage": "audio",
                "detail": failure.detail,
                "path": failure.relative_path,
                "exception_type": "AudioFormatError",
            }
            for failure in self.failures
        ]

    def fingerprint_for_segments(self, segments: list[Any]) -> str:
        """Fingerprint only audio visible to the current persisted segment set.

        Unreferenced files may be added for future append commands without making
        the current snapshot stale. Reference audio is always part of the state.
        """

        references = [
            {
                "path": rel,
                "content": audio.content_fingerprint,
                "sample_count": audio.frame_count,
            }
            for rel, audio in sorted(self.references.items())
        ]
        failure_by_path = {failure.relative_path: failure for failure in self.failures}
        source_entries: list[dict[str, Any]] = []
        seen: set[str] = set()
        for segment in segments:
            if isinstance(segment, dict):
                path = str(segment["path"])
                start_sample = int(segment.get("start_sample", 0))
                end_sample = segment.get("end_sample")
            else:
                path = str(segment.path)
                start_sample = int(segment.start_sample)
                end_sample = segment.end_sample
            if path in seen:
                continue
            seen.add(path)
            entry: dict[str, Any] = {
                "path": path,
                "start_sample": start_sample,
                "end_sample": end_sample,
            }
            audio = self.sources.get(path)
            if audio is not None:
                entry["content"] = audio.content_fingerprint
                entry["sample_count"] = audio.frame_count
            elif path in failure_by_path:
                failure = failure_by_path[path]
                entry["content"] = failure.raw_fingerprint
                entry["format_error"] = failure.detail
            else:
                entry["content"] = "missing"
            source_entries.append(entry)
        return sha256_value({"references": references, "sources": source_entries})
