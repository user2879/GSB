"""Deterministic multi-speaker PCM synthesizer.

A local numpy random generator with a fixed seed drives every random decision,
so repeated runs produce byte-identical WAV files without touching global RNG
state. Each speaker is a glottal pulse train filtered by two formant resonators
plus an aspiration noise component.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .audioio import write_wav
from .config import SAMPLE_RATE

# (fundamental f0, formant-1 Hz, formant-2 Hz) per speaker profile.
SPEAKER_PROFILES = {
    0: (120.0, 700.0, 1220.0),
    1: (185.0, 500.0, 1600.0),
    2: (95.0, 350.0, 900.0),
}


@dataclass
class SynthSpec:
    duration_s: float = 6.0
    speaker_count: int = 3
    seed: int = 20240601
    pause_ms_range: Tuple[int, int] = (120, 240)
    overlap_ms_range: Tuple[int, int] = (80, 160)
    utter_ms_range: Tuple[int, int] = (500, 900)


@dataclass
class SynthTruth:
    path: str
    segments: List[Dict]
    overlaps: List[Dict] = field(default_factory=list)
    pcm_fingerprint: str = ""


def _resonator(samples: np.ndarray, freq: float, bw: float, sr: int) -> np.ndarray:
    """Second-order biquad band resonator applied with lfilter semantics."""
    r = np.exp(-np.pi * bw / sr)
    theta = 2.0 * np.pi * freq / sr
    a1 = -2.0 * r * np.cos(theta)
    a2 = r * r
    b0 = (1.0 - r * r) * 0.5
    y = np.zeros_like(samples)
    y1 = y2 = 0.0
    for i, x0 in enumerate(samples):
        y0 = b0 * x0 - a1 * y1 - a2 * y2
        y[i] = y0
        y2, y1 = y1, y0
    return y


def _glottal_train(length: int, f0: float, sr: int, rng: np.random.Generator,
                   phase0: float) -> np.ndarray:
    train = np.zeros(length, dtype=np.float64)
    period = max(1, int(round(sr / f0)))
    start = int(phase0 * period)
    for center in range(start, length, period):
        # Two-sample glottal pulse keeps harmonics rich but cheap.
        if 0 <= center < length:
            train[center] += 1.0
        if center + 1 < length:
            train[center + 1] += 0.35
    jitter = rng.normal(0.0, 0.004, length)
    return train + jitter


def _render_voice(length: int, profile, rng: np.random.Generator,
                  sr: int, phase0: float) -> np.ndarray:
    f0, formant1, formant2 = profile
    source = _glottal_train(length, f0, sr, rng, phase0)
    voiced = _resonator(source, formant1, 90.0, sr)
    voiced = _resonator(voiced, formant2, 120.0, sr)
    aspiration = rng.normal(0.0, 0.05, length)
    aspiration = _resonator(aspiration, formant2, 300.0, sr)
    mix = voiced * 4.0 + aspiration
    # Slow amplitude envelope avoids hard clicks at segment edges.
    ramp = max(1, sr // 200)
    env = np.ones(length)
    env[:ramp] = np.linspace(0.0, 1.0, ramp)
    env[-ramp:] = np.linspace(1.0, 0.0, ramp)
    return mix * env


def synthesize(spec: SynthSpec, path: str) -> SynthTruth:
    rng = np.random.default_rng(spec.seed)
    total = int(spec.duration_s * SAMPLE_RATE)
    mix = np.zeros(total, dtype=np.float64)
    truth: List[Dict] = []
    overlaps: List[Dict] = []
    cursor = SAMPLE_RATE // 20  # 50 ms lead-in
    speaker_ids = list(range(spec.speaker_count))
    last_speaker = -1
    segment_index = 0

    while cursor < total - SAMPLE_RATE // 10:
        choices = [s for s in speaker_ids if s != last_speaker]
        speaker = int(rng.choice(choices))
        last_speaker = speaker
        utter_ms = int(rng.integers(*spec.utter_ms_range))
        length = min(int(utter_ms * SAMPLE_RATE / 1000), total - cursor)
        if length <= SAMPLE_RATE // 20:
            break
        start = cursor
        end = cursor + length
        phase0 = float(rng.random())
        voice = _render_voice(length, SPEAKER_PROFILES[speaker], rng,
                              SAMPLE_RATE, phase0)
        mix[start:end] += voice
        truth.append({
            "index": segment_index,
            "speaker": speaker,
            "start_sample": int(start),
            "end_sample": int(end),
            "start_ms": round(start * 1000.0 / SAMPLE_RATE, 3),
            "end_ms": round(end * 1000.0 / SAMPLE_RATE, 3),
            "f0": SPEAKER_PROFILES[speaker][0],
        })
        segment_index += 1

        gap_ms = int(rng.integers(*spec.pause_ms_range))
        gap = int(gap_ms * SAMPLE_RATE / 1000)
        if rng.random() < 0.4 and end + SAMPLE_RATE // 10 < total:
            overlap_ms = int(rng.integers(*spec.overlap_ms_range))
            overlap_len = int(overlap_ms * SAMPLE_RATE / 1000)
            next_choices = [s for s in speaker_ids if s != speaker]
            other = int(rng.choice(next_choices))
            ov_start = max(0, end - overlap_len)
            ov_end = min(total, end + max(1, overlap_len // 2))
            ov_voice = _render_voice(ov_end - ov_start,
                                     SPEAKER_PROFILES[other], rng,
                                     SAMPLE_RATE, float(rng.random()))
            mix[ov_start:ov_end] += 0.8 * ov_voice
            overlaps.append({
                "start_sample": int(ov_start),
                "end_sample": int(ov_end),
                "speakers": [speaker, other],
                "start_ms": round(ov_start * 1000.0 / SAMPLE_RATE, 3),
                "end_ms": round(ov_end * 1000.0 / SAMPLE_RATE, 3),
            })
            cursor = end - gap // 3
        else:
            cursor = end + gap

    peak = float(np.max(np.abs(mix)))
    if peak > 0:
        mix = mix / peak * 0.9
    write_wav(path, mix.tolist())
    import hashlib
    fingerprint = hashlib.sha256(np.asarray(mix, dtype=np.float64).tobytes()).hexdigest()
    return SynthTruth(path=path, segments=truth, overlaps=overlaps,
                      pcm_fingerprint=fingerprint)


def write_silence(path: str, duration_s: float = 3.0) -> SynthTruth:
    total = int(duration_s * SAMPLE_RATE)
    write_wav(path, [0.0] * total)
    return SynthTruth(path=path, segments=[], overlaps=[])
