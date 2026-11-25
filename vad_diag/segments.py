"""Audio segment model and strict RIFF/WAV container validation.

A :class:`Segment` is one appendable 16 kHz / mono / 16-bit PCM fragment with
its three timing fields (capture / arrival / service-processing milliseconds).
Segments are identified by a unique, caller-assigned ``segment_id`` (we use the
stable ``S0000`` style) and carry an explicit ``capture_ms`` so out-of-order
or late arrival never changes the reconstructed *content* timeline.
"""

from __future__ import annotations

import struct
import wave
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .errors import VADFieldError, E_AUDIO_FORMAT, E_FIELD_INVALID
from .fingerprint import digest, digest_bytes

RIFF_MAGIC = b"RIFF"
WAVE_MAGIC = b"WAVE"
FMT_PCM = 1
EXPECTED_RATE = 16000
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_WIDTH = 2  # bytes -> 16 bit


@dataclass(frozen=True)
class Segment:
    segment_id: str
    capture_ms: int
    arrival_ms: int
    process_ms: int
    samples: Tuple[int, ...]          # signed 16-bit PCM, capture order
    sample_rate: int = EXPECTED_RATE
    channels: int = EXPECTED_CHANNELS
    sample_width: int = EXPECTED_SAMPLE_WIDTH
    data_byte_len: int = 0

    def __post_init__(self):
        # dataclass frozen -> use object.__setattr__ for the derived length.
        object.__setattr__(self, "data_byte_len", len(self.samples) *
                           self.sample_width * self.channels)

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def content_fingerprint(self) -> str:
        """Fingerprint over raw PCM bytes *and* the timing fields."""
        pcm = struct.pack("<%dh" % len(self.samples), *self.samples)
        return digest([
            "segment",
            self.segment_id,
            digest_bytes(pcm),
            {
                "capture_ms": self.capture_ms,
                "arrival_ms": self.arrival_ms,
                "process_ms": self.process_ms,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "sample_width": self.sample_width,
            },
        ])


def _read_riff_raw(path: str) -> Tuple[bytes, int, int, int, int]:
    """Validate the RIFF container manually and return the PCM data chunk.

    We do not rely solely on :mod:`wave` because the requirement is to verify
    the container header fields explicitly and to reject truncation.
    """
    with open(path, "rb") as handle:
        blob = handle.read()
    if len(blob) < 44:
        raise VADFieldError(
            "file too small to contain a RIFF/WAV header",
            E_AUDIO_FORMAT, stage="load")
    if blob[0:4] != RIFF_MAGIC:
        raise VADFieldError("missing RIFF magic", E_AUDIO_FORMAT,
                            stage="load")
    declared_size = struct.unpack_from("<I", blob, 4)[0]
    # declared_size counts the bytes following the 8-byte RIFF preamble.
    if declared_size + 8 != len(blob):
        raise VADFieldError(
            "RIFF chunk size does not match file length", E_AUDIO_FORMAT,
            stage="load",
            detail={"declared": declared_size + 8, "actual": len(blob)})
    if blob[8:12] != WAVE_MAGIC:
        raise VADFieldError("missing WAVE magic", E_AUDIO_FORMAT,
                            stage="load")

    offset = 12
    fmt_found = False
    data_blob: Optional[bytes] = None
    declared_data_len = 0
    channels = sample_width = rate = audio_format = 0
    while offset + 8 <= len(blob):
        chunk_id = blob[offset:offset + 4]
        chunk_size = struct.unpack_from("<I", blob, offset + 4)[0]
        body_start = offset + 8
        body_end = body_start + chunk_size
        if body_end > len(blob):
            raise VADFieldError(
                f"chunk {chunk_id!r} extends past end of file (truncated)",
                E_AUDIO_FORMAT, stage="load",
                detail={"chunk": chunk_id.decode('latin1'),
                        "declared": chunk_size})
        if chunk_id == b"fmt ":
            fmt_found = True
            if chunk_size < 16:
                raise VADFieldError(
                    "fmt chunk too small for PCM", E_AUDIO_FORMAT,
                    stage="load")
            (audio_format, channels, rate, _byte_rate, _block_align,
             bits_per_sample) = struct.unpack_from("<HHIIHH", blob,
                                                   body_start)
            sample_width = bits_per_sample // 8
        elif chunk_id == b"data":
            data_blob = blob[body_start:body_end]
            declared_data_len = chunk_size
        # Chunks are word-aligned (padded to even length).
        offset = body_end + (chunk_size & 1)

    if not fmt_found:
        raise VADFieldError("missing fmt chunk", E_AUDIO_FORMAT, stage="load")
    if data_blob is None:
        raise VADFieldError("missing data chunk", E_AUDIO_FORMAT,
                            stage="load")
    if audio_format != FMT_PCM:
        raise VADFieldError(
            f"unsupported audio format {audio_format}, PCM required",
            E_AUDIO_FORMAT, stage="load")
    if rate != EXPECTED_RATE:
        raise VADFieldError(
            f"sample rate {rate} != {EXPECTED_RATE}", E_AUDIO_FORMAT,
            stage="load", detail={"sample_rate": rate})
    if channels != EXPECTED_CHANNELS:
        raise VADFieldError(
            f"channels {channels} != {EXPECTED_CHANNELS}", E_AUDIO_FORMAT,
            stage="load", detail={"channels": channels})
    if sample_width != EXPECTED_SAMPLE_WIDTH:
        raise VADFieldError(
            f"sample width {sample_width} bytes != "
            f"{EXPECTED_SAMPLE_WIDTH} (16 bit)", E_AUDIO_FORMAT,
            stage="load", detail={"sample_width": sample_width})
    if declared_data_len != len(data_blob):
        raise VADFieldError(
            "data chunk length mismatch", E_AUDIO_FORMAT, stage="load",
            detail={"declared": declared_data_len,
                      "actual": len(data_blob)})
    return data_blob, rate, channels, sample_width, declared_data_len


def decode_pcm(data_blob: bytes, channels: int, sample_width: int
               ) -> Tuple[List[int], int]:
    """Decode interleaved PCM bytes to mono signed samples and sample count."""
    if sample_width != 2:
        raise VADFieldError("only 16-bit PCM decoding is supported",
                            E_AUDIO_FORMAT, stage="load")
    if len(data_blob) % (sample_width * channels) != 0:
        raise VADFieldError(
            "data block length is not a whole number of frames",
            E_AUDIO_FORMAT, stage="load",
            detail={"byte_len": len(data_blob)})
    frame_count = len(data_blob) // (sample_width * channels)
    interleaved = struct.unpack("<%dh" % (frame_count * channels), data_blob)
    if channels == 1:
        samples = list(interleaved)
    else:
        # Defensive: validation already forces mono, but keep the reduction.
        samples = [interleaved[i * channels] for i in range(frame_count)]
    return samples, frame_count


def parse_wav_segment(path: str, segment_id: str, capture_ms: int,
                      arrival_ms: int, process_ms: int) -> Segment:
    """Full validation path used by the loader for one segment file."""
    data_blob, rate, channels, sample_width, data_len = _read_riff_raw(path)
    samples, frame_count = decode_pcm(data_blob, channels, sample_width)
    # Cross-check data block length against sample count (explicit requirement).
    if data_len != frame_count * sample_width * channels:
        raise VADFieldError(
            "data block length inconsistent with sample count",
            E_AUDIO_FORMAT, stage="load",
            detail={"data_len": data_len, "samples": frame_count})
    _check_timing(segment_id, capture_ms, arrival_ms, process_ms)
    return Segment(segment_id, capture_ms, arrival_ms, process_ms,
                   tuple(samples), rate, channels, sample_width, data_len)


def _check_timing(segment_id: str, capture_ms: int, arrival_ms: int,
                  process_ms: int) -> None:
    for name, value in (("capture_ms", capture_ms),
                        ("arrival_ms", arrival_ms),
                        ("process_ms", process_ms)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise VADFieldError(
                f"{name} must be an integer", E_FIELD_INVALID,
                item_id=segment_id, detail={"field": name})
        if value < 0:
            raise VADFieldError(
                f"{name} must be a non-negative integer", E_FIELD_INVALID,
                item_id=segment_id, detail={"field": name, "value": value})


def build_in_memory_segment(segment_id: str, samples: List[int],
                            capture_ms: int, arrival_ms: int,
                            process_ms: int) -> Segment:
    """Construct a validated segment directly from integer samples.

    Used by the deterministic sample generator and by tests; enforces the same
    timing and sample-range rules as the file loader.
    """
    _check_timing(segment_id, capture_ms, arrival_ms, process_ms)
    for idx, sample in enumerate(samples):
        if isinstance(sample, bool) or not isinstance(sample, int):
            raise VADFieldError(
                f"sample[{idx}] must be an integer", E_FIELD_INVALID,
                item_id=segment_id)
        if not -(2 ** 15) <= sample <= 2 ** 15 - 1:
            raise VADFieldError(
                f"sample[{idx}] outside signed 16-bit range",
                E_AUDIO_FORMAT, item_id=segment_id,
                detail={"value": sample})
    return Segment(segment_id, capture_ms, arrival_ms, process_ms,
                   tuple(samples))


def write_wav(path: str, samples: List[int], rate: int = EXPECTED_RATE
              ) -> None:
    """Write a standards-compliant mono 16-bit PCM WAV file."""
    with wave.open(path, "wb") as wav:
        wav.setnchannels(EXPECTED_CHANNELS)
        wav.setsampwidth(EXPECTED_SAMPLE_WIDTH)
        wav.setframerate(rate)
        wav.writeframes(struct.pack("<%dh" % len(samples), *samples))
