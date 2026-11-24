"""Minimal deterministic RIFF/WAV reader and writer.

The parser walks RIFF chunks manually instead of relying on the wave module so
that truncation, inconsistent chunk sizes and non-finite downstream samples can
all be detected with explicit error codes.
"""
from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass

from .config import SAMPLE_RATE
from .errors import AudioFormatError


@dataclass
class WaveData:
    path: str
    samples: list          # python floats in [-1, 1]
    sample_rate: int
    channels: int
    bits_per_sample: int
    data_bytes: int

    @property
    def num_samples(self) -> int:
        return len(self.samples)


def _read_exact(handle, size: int) -> bytes:
    blob = handle.read(size)
    if len(blob) != size:
        raise AudioFormatError("truncated data chunk", {"need": size, "got": len(blob)})
    return blob


def read_wav(path: str) -> WaveData:
    if not os.path.exists(path):
        raise AudioFormatError(f"file not found: {path}")
    with open(path, "rb") as handle:
        try:
            riff = _read_exact(handle, 12)
        except AudioFormatError:
            raise AudioFormatError("file smaller than RIFF header")
        if riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise AudioFormatError("missing RIFF/WAVE container header")

        fmt = None
        data_blob = None
        declared_data_size = None
        while True:
            header = handle.read(8)
            if not header:
                break
            if len(header) < 8:
                raise AudioFormatError("truncated chunk header")
            chunk_id, chunk_size = struct.unpack("<4sI", header)
            body = handle.read(chunk_size)
            if len(body) != chunk_size:
                raise AudioFormatError("truncated chunk body",
                                       {"chunk": chunk_id.decode("ascii", "replace")})
            if chunk_size % 2 == 1:  # RIFF padding byte
                handle.read(1)
            if chunk_id == b"fmt ":
                fmt = body
            elif chunk_id == b"data":
                data_blob = body
                declared_data_size = chunk_size
                # Continue scanning to make sure no extra trailing chunks hide
                # further truncation issues.
        if fmt is None:
            raise AudioFormatError("missing fmt chunk")
        if data_blob is None:
            raise AudioFormatError("missing data chunk")
        if len(fmt) < 16:
            raise AudioFormatError("fmt chunk smaller than PCM header")

        audio_format, channels, sample_rate, _byte_rate, _block_align, \
            bits_per_sample = struct.unpack("<HHIIHH", fmt[:16])
        if audio_format != 1:
            raise AudioFormatError("only PCM (format tag 1) is supported",
                                   {"audio_format": audio_format})
        if sample_rate != SAMPLE_RATE:
            raise AudioFormatError(f"sample rate must be {SAMPLE_RATE}",
                                   {"sample_rate": sample_rate})
        if channels != 1:
            raise AudioFormatError("only mono audio is supported",
                                   {"channels": channels})
        if bits_per_sample != 16:
            raise AudioFormatError("only 16-bit PCM is supported",
                                   {"bits_per_sample": bits_per_sample})

        frame_size = channels * bits_per_sample // 8
        if declared_data_size % frame_size != 0:
            raise AudioFormatError("data chunk length is not frame aligned")
        sample_count = declared_data_size // frame_size
        if len(data_blob) != declared_data_size:
            raise AudioFormatError("data block length mismatch")

        ints = struct.unpack("<%dh" % sample_count, data_blob)
        scale = float(1 << (bits_per_sample - 1))
        samples = [value / scale for value in ints]
        for value in samples:
            if not math.isfinite(value):
                raise AudioFormatError("non-finite sample value encountered")

        return WaveData(path=path, samples=samples, sample_rate=sample_rate,
                        channels=channels, bits_per_sample=bits_per_sample,
                        data_bytes=declared_data_size)


def write_wav(path: str, samples, sample_rate: int = SAMPLE_RATE) -> None:
    data = bytearray()
    for value in samples:
        clipped = max(-1.0, min(1.0, float(value)))
        integer = int(round(clipped * 32767))
        if integer == -32768:
            integer = -32767
        data.extend(struct.pack("<h", integer))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as handle:
        byte_rate = sample_rate * 2
        handle.write(b"RIFF")
        handle.write(struct.pack("<I", 36 + len(data)))
        handle.write(b"WAVE")
        handle.write(b"fmt ")
        handle.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate,
                                 byte_rate, 2, 16))
        handle.write(b"data")
        handle.write(struct.pack("<I", len(data)))
        handle.write(bytes(data))


def truncate_wav(path: str, keep_bytes: int) -> int:
    """Fault-injection helper: cut the data block short but keep declared size."""
    with open(path, "r+b") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.truncate(keep_bytes)
    return size
