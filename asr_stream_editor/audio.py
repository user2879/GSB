"""WAV loading, strict format validation and deterministic chunk planning."""

from __future__ import annotations

import hashlib
import os
import struct
import wave
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .errors import BusinessError, E_AUDIO_FORMAT

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
N_CHANNELS = 1


@dataclass(frozen=True)
class AudioClip:
    source: str
    samples: np.ndarray  # float64 mono in [-1, 1)
    pcm_bytes: bytes
    fingerprint: str

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration_ms(self) -> float:
        return self.n_samples * 1000.0 / SAMPLE_RATE


def _read_raw(path: str) -> Tuple[bytes, int, int, int]:
    """Return (pcm_bytes, sample_rate, channels, sample_width) with RIFF checks."""
    with open(path, "rb") as handle:
        blob = handle.read()
    if len(blob) < 12 or blob[0:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise BusinessError(E_AUDIO_FORMAT, f"{path}: missing RIFF/WAVE container header")
    declared_size = struct.unpack("<I", blob[4:8])[0]
    if declared_size + 8 != len(blob):
        raise BusinessError(E_AUDIO_FORMAT, f"{path}: RIFF chunk size does not match file length")
    try:
        with wave.open(path, "rb") as wav:
            params = wav.getparams()
            pcm = wav.readframes(wav.getnframes())
            n_frames = wav.getnframes()
    except wave.Error as exc:
        raise BusinessError(E_AUDIO_FORMAT, f"{path}: malformed WAVE structure: {exc}") from exc
    if params.sampwidth != SAMPLE_WIDTH:
        raise BusinessError(E_AUDIO_FORMAT, f"{path}: expected 16-bit PCM, got {params.sampwidth * 8}-bit")
    if params.nchannels != N_CHANNELS:
        raise BusinessError(E_AUDIO_FORMAT, f"{path}: expected mono, got {params.nchannels} channels")
    if params.framerate != SAMPLE_RATE:
        raise BusinessError(E_AUDIO_FORMAT, f"{path}: expected {SAMPLE_RATE} Hz, got {params.framerate}")
    expected_bytes = n_frames * params.nchannels * params.sampwidth
    if len(pcm) != expected_bytes:
        raise BusinessError(E_AUDIO_FORMAT,
                            f"{path}: data block length {len(pcm)} != samples*width {expected_bytes}")
    return pcm, params.framerate, params.nchannels, params.sampwidth


def load_wav(path: str) -> AudioClip:
    pcm, _rate, _channels, _width = _read_raw(path)
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
    fingerprint = hashlib.sha256(pcm).hexdigest()[:16]
    return AudioClip(source=os.path.basename(path), samples=samples, pcm_bytes=pcm,
                     fingerprint=fingerprint)


def load_wav_pcm(pcm: bytes, source: str) -> AudioClip:
    """Validate in-memory PCM (used when replacing one chunk's audio)."""
    if len(pcm) % SAMPLE_WIDTH != 0:
        raise BusinessError(E_AUDIO_FORMAT, f"{source}: PCM length is not a multiple of sample width")
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
    fingerprint = hashlib.sha256(pcm).hexdigest()[:16]
    return AudioClip(source=source, samples=samples, pcm_bytes=pcm, fingerprint=fingerprint)


@dataclass(frozen=True)
class FramePlan:
    frame_length: int  # samples
    frame_shift: int  # samples
    n_frames: int
    last_frame_padded: bool

    def frame_start(self, frame_index: int) -> int:
        return frame_index * self.frame_shift


def plan_frames(n_samples: int, frame_length_ms: int, frame_shift_ms: int) -> FramePlan:
    frame_length = SAMPLE_RATE * frame_length_ms // 1000
    frame_shift = SAMPLE_RATE * frame_shift_ms // 1000
    if n_samples <= frame_length:
        n_frames = 1
    else:
        n_frames = 1 + int(np.ceil((n_samples - frame_length) / frame_shift))
    last_start = (n_frames - 1) * frame_shift
    last_frame_padded = last_start + frame_length > n_samples
    return FramePlan(frame_length=frame_length, frame_shift=frame_shift,
                     n_frames=n_frames, last_frame_padded=last_frame_padded)


@dataclass(frozen=True)
class ChunkPlan:
    index: int
    start_sample: int
    end_sample: int          # exclusive
    own_start_sample: int    # exclusive-ownership region (first chunk owns from 0)
    own_end_sample: int
    hop_samples: int

    @property
    def n_samples(self) -> int:
        return self.end_sample - self.start_sample


def plan_chunks(n_samples: int, chunk_ms: int, overlap_ms: int) -> List[ChunkPlan]:
    """Sliding window: hop = chunk - overlap; last chunk ends at stream end."""
    chunk_len = SAMPLE_RATE * chunk_ms // 1000
    overlap_len = SAMPLE_RATE * overlap_ms // 1000
    hop = chunk_len - overlap_len
    plans: List[ChunkPlan] = []
    if n_samples == 0:
        return plans
    start = 0
    index = 0
    while start < n_samples:
        end = min(start + chunk_len, n_samples)
        own_start = 0 if index == 0 else start
        if end == n_samples:
            own_end = n_samples
        else:
            own_end = start + hop
        plans.append(ChunkPlan(index=index, start_sample=start, end_sample=end,
                               own_start_sample=own_start, own_end_sample=own_end,
                               hop_samples=hop))
        if end == n_samples:
            break
        start += hop
        index += 1
    return plans


def frame_to_ms(frame_index: int, frame_shift_ms: int) -> int:
    return frame_index * frame_shift_ms


def replace_chunk_samples(samples: np.ndarray, chunk: ChunkPlan,
                          replacement: np.ndarray) -> np.ndarray:
    """Replace only the chunk's exclusive ownership region.

    Length must match the ownership region so frame/time geometry stays fixed.
    """
    region_len = chunk.own_end_sample - chunk.own_start_sample
    if replacement.shape[0] != region_len:
        raise BusinessError(
            E_AUDIO_FORMAT,
            f"replacement length {replacement.shape[0]} != ownership region {region_len}")
    out = samples.copy()
    out[chunk.own_start_sample:chunk.own_end_sample] = replacement.astype(np.float64)
    return out
