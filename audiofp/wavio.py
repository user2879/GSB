"""Strict, dependency-free PCM WAV reader and source-synchronous writer."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np

from .errors import BusinessError
from .models import TrackAudio


def _u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def read_wav(
    path: str | Path,
    *,
    track_id: str,
    role: str,
    allowed_sample_rates: tuple[int, ...],
    declared_sample_rate: int | None = None,
    silence_threshold: float = 1e-6,
) -> TrackAudio:
    wav_path = Path(path).resolve()
    try:
        blob = wav_path.read_bytes()
    except OSError as exc:
        raise BusinessError("E_IO", "unable to read WAV", {"path": str(wav_path), "reason": str(exc)}) from exc
    if len(blob) < 12:
        raise BusinessError("E_TRUNC", "WAV is shorter than the RIFF header", {"path": str(wav_path)})
    if blob[0:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise BusinessError("E_WAV", "not a RIFF/WAVE file", {"path": str(wav_path)})
    declared_riff_size = _u32(blob, 4) + 8
    if declared_riff_size > len(blob):
        raise BusinessError(
            "E_TRUNC",
            "RIFF declared size extends past end of file",
            path=str(wav_path),
            declared_bytes=declared_riff_size,
            available_bytes=len(blob),
        )
    cursor = 12
    fmt: tuple[int, int, int, int] | None = None
    data_offset: int | None = None
    data_size: int | None = None
    while cursor + 8 <= len(blob):
        chunk_id = blob[cursor : cursor + 4]
        chunk_size = _u32(blob, cursor + 4)
        payload = cursor + 8
        if payload + chunk_size > len(blob):
            code = "E_TRUNC" if chunk_id == b"data" else "E_WAV"
            raise BusinessError(
                code,
                "WAV chunk extends past end of file",
                path=str(wav_path),
                chunk=chunk_id.decode("latin-1", errors="replace"),
                declared_bytes=chunk_size,
                available_bytes=max(0, len(blob) - payload),
            )
        if chunk_id == b"fmt ":
            if chunk_size < 16:
                raise BusinessError("E_WAV", "fmt chunk is too short", path=str(wav_path))
            audio_format = _u16(blob, payload)
            channels = _u16(blob, payload + 2)
            sample_rate = _u32(blob, payload + 4)
            block_align = _u16(blob, payload + 12)
            bits = _u16(blob, payload + 14)
            fmt = (audio_format, channels, sample_rate, bits)
            if audio_format != 1:
                raise BusinessError("E_WAV", "only PCM WAV is supported", audio_format=audio_format)
            if channels != 1:
                raise BusinessError("E_WAV", "only mono WAV is supported", channels=channels)
            if bits != 16:
                raise BusinessError("E_WAV", "only 16-bit WAV is supported", bits=bits)
            if block_align != 2:
                raise BusinessError("E_WAV", "invalid block alignment for 16-bit mono", block_align=block_align)
            if sample_rate not in allowed_sample_rates:
                raise BusinessError(
                    "E_RATE",
                    "WAV sample rate is not allowed",
                    sample_rate=sample_rate,
                    allowed_sample_rates=list(allowed_sample_rates),
                )
        elif chunk_id == b"data":
            if data_offset is not None:
                raise BusinessError("E_WAV", "multiple data chunks are not supported", path=str(wav_path))
            data_offset = payload
            data_size = chunk_size
        cursor = payload + chunk_size + (chunk_size & 1)
    if fmt is None:
        raise BusinessError("E_WAV", "missing fmt chunk", path=str(wav_path))
    if data_offset is None or data_size is None:
        raise BusinessError("E_WAV", "missing data chunk", path=str(wav_path))
    if data_size <= 0:
        raise BusinessError("E_TRUNC", "WAV data chunk is empty", path=str(wav_path))
    if data_size % 2:
        raise BusinessError("E_WAV", "16-bit data chunk has an odd byte length", path=str(wav_path))
    sample_rate = fmt[2]
    if declared_sample_rate is not None and declared_sample_rate != sample_rate:
        raise BusinessError(
            "E_RATE",
            "declared sample rate does not match WAV header",
            declared_sample_rate=declared_sample_rate,
            header_sample_rate=sample_rate,
        )
    raw_data = np.frombuffer(blob[data_offset : data_offset + data_size], dtype="<i2")
    samples = raw_data.astype(np.float64) / 32768.0
    if samples.size == 0:
        raise BusinessError("E_TRUNC", "WAV has no samples", path=str(wav_path))
    peak = float(np.max(np.abs(samples)))
    if peak <= silence_threshold:
        raise BusinessError("E_SILENT", "WAV is entirely silent", peak_amplitude=peak)
    return TrackAudio(
        track_id=track_id,
        path=wav_path,
        role=role,
        sample_rate=sample_rate,
        declared_sample_rate=declared_sample_rate,
        samples=samples,
        duration_samples=int(samples.size),
    )


def write_pcm16_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0 - 1.0 / 32768.0)
    pcm = np.rint(clipped * 32768.0).astype("<i2")
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
