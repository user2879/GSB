"""Deterministic 5.1 programme-material synthesizer.

A fixed-seed :class:`numpy.random.Generator` drives speech-like, music-like
and transient-like segments. Each segment is calibrated to a declared
BS.1770-4 integrated loudness target *before* cross-fading, then the segments
are concatenated with linear cross-fades. Re-running the same configuration
produces byte-identical WAV and label files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .audio_io import write_wav
from .config import (
    CHANNEL_COUNT,
    CHANNEL_LABELS,
    SAMPLE_RATE,
    ms_to_samples,
)
from .bs1770 import channel_gains_db, k_weighting_coeffs
from scipy import signal as sps

CROSSFADE_MS = 10


@dataclass(frozen=True)
class SegmentSpec:
    kind: str  # "speech" | "music" | "transient"
    duration_s: float
    target_lkfs: float
    target_lra_lu: float


DEFAULT_PROGRAMME: tuple[SegmentSpec, ...] = (
    SegmentSpec("speech", 3.0, -24.0, 6.0),
    SegmentSpec("music", 4.0, -20.0, 11.0),
    SegmentSpec("transient", 2.0, -26.0, 18.0),
    SegmentSpec("speech", 3.0, -23.0, 7.0),
    SegmentSpec("music", 3.0, -18.0, 10.0),
    SegmentSpec("transient", 3.0, -27.0, 20.0),
)


def _weighted_mean_square(signal: np.ndarray) -> float:
    b, a = k_weighting_coeffs(SAMPLE_RATE)
    weighted_signal = sps.lfilter(b, a, signal, axis=0)
    gains = channel_gains_db()
    linear = 10.0 ** (gains / 10.0)
    linear[np.isneginf(gains)] = 0.0
    return float(np.mean(weighted_signal**2, axis=0) @ linear)


def _calibrate_to_target(signal: np.ndarray, target_lkfs: float) -> np.ndarray:
    """Apply a single global gain so the weighted level hits the target."""
    ms = _weighted_mean_square(signal)
    current_lkfs = -0.691 + 10.0 * np.log10(ms)
    gain_db = target_lkfs - current_lkfs
    return signal * (10.0 ** (gain_db / 20.0))


def _speech_segment(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SAMPLE_RATE
    # Band-passed noise shaped by a syllable-rate amplitude envelope.
    base = rng.standard_normal(n)
    syllable = 0.45 + 0.55 * (0.5 + 0.5 * np.sin(2 * np.pi * 3.2 * t))
    breathing = 0.45 + 0.55 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.6 * t + 1.1))
    shaped = base * syllable * breathing
    mono = np.clip(shaped, -1.0, 1.0)
    return _pan_to_surround(mono, rng)


def _music_segment(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SAMPLE_RATE
    f0 = 220.0 + float(rng.integers(-20, 21))
    lfo = 0.6 + 0.4 * np.sin(2 * np.pi * 0.25 * t)
    mono = (
        0.55 * np.sin(2 * np.pi * f0 * t)
        + 0.3 * np.sin(2 * np.pi * f0 * 1.5 * t + 0.7)
        + 0.15 * np.sin(2 * np.pi * f0 * 2.01 * t + 1.3)
    ) * lfo
    return _pan_to_surround(np.clip(mono, -1.0, 1.0), rng)


def _transient_segment(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SAMPLE_RATE
    mono = np.zeros(n)
    # Damped bursts separated by quiet gaps -> large loudness range.
    period = ms_to_samples(700)
    decay_tau = ms_to_samples(28)
    tone = rng.uniform(700.0, 1800.0)
    for start in range(0, n, period):
        length = min(period, n - start)
        local_t = np.arange(length)
        burst = np.exp(-local_t / decay_tau) * np.sin(
            2 * np.pi * tone * local_t / SAMPLE_RATE
        )
        mono[start : start + length] += burst
    floor = 0.012 * rng.standard_normal(n)
    return _pan_to_surround(np.clip(mono + floor, -1.0, 1.0), rng)


def _pan_to_surround(mono: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = mono.shape[0]
    signal = np.zeros((n, CHANNEL_COUNT))
    # Stable per-segment random panning coefficients, LFE carries only content.
    coeffs = rng.uniform(0.45, 0.9, size=5)
    signal[:, 0] = mono * coeffs[0]
    signal[:, 1] = mono * coeffs[1]
    signal[:, 2] = mono * coeffs[2]
    signal[:, 4] = mono * coeffs[3]
    signal[:, 5] = mono * coeffs[4]
    signal[:, 3] = mono * 0.10  # LFE
    return signal


def _build_segment(spec: SegmentSpec, rng: np.random.Generator) -> np.ndarray:
    n = ms_to_samples(spec.duration_s * 1000.0)
    if spec.kind == "speech":
        raw = _speech_segment(n, rng)
    elif spec.kind == "music":
        raw = _music_segment(n, rng)
    elif spec.kind == "transient":
        raw = _transient_segment(n, rng)
    else:
        raise ValueError(f"unknown segment kind: {spec.kind!r}")
    return _calibrate_to_target(raw, spec.target_lkfs)


def _apply_crossfades(segments: list[np.ndarray]) -> np.ndarray:
    fade = ms_to_samples(CROSSFADE_MS)
    ramp_up = np.linspace(0.0, 1.0, fade)[:, None]
    ramp_down = np.linspace(1.0, 0.0, fade)[:, None]
    output = segments[0].copy()
    boundaries: list[int] = []
    for nxt in segments[1:]:
        boundary = output.shape[0]
        boundaries.append(boundary)
        head = output[-fade:] * ramp_down + nxt[:fade] * ramp_up
        output[-fade:] = head
        output = np.concatenate([output, nxt[fade:]], axis=0)
    return output


def generate_programme(
    seed: int,
    programme: tuple[SegmentSpec, ...] = DEFAULT_PROGRAMME,
) -> tuple[np.ndarray, list[dict]]:
    """Return the calibrated multi-channel signal and ground-truth labels."""
    rng = np.random.default_rng(seed)
    segments: list[np.ndarray] = []
    labels: list[dict] = []
    cursor = 0
    fade_samples = ms_to_samples(CROSSFADE_MS)
    for index, spec in enumerate(programme):
        seg = _build_segment(spec, rng)
        segments.append(seg)
        start_sample = cursor
        end_sample = cursor + seg.shape[0]
        labels.append(
            {
                "segment_id": index,
                "kind": spec.kind,
                "start_sample": int(start_sample),
                "end_sample": int(end_sample),
                "start_s": round(start_sample / SAMPLE_RATE, 6),
                "end_s": round(end_sample / SAMPLE_RATE, 6),
                "target_integrated_lkfs": spec.target_lkfs,
                "target_lra_lu": spec.target_lra_lu,
                "crossfade_samples": fade_samples,
            }
        )
        cursor = end_sample
    signal = _apply_crossfades(segments)
    # Each cross-fade overlaps two segments by ``fade_samples``; shift every
    # boundary after the first join and clamp the final end to the waveform.
    for index in range(1, len(labels)):
        shift = fade_samples * index
        labels[index]["start_sample"] -= shift
        labels[index]["end_sample"] -= shift
        labels[index]["start_s"] = round(labels[index]["start_sample"] / SAMPLE_RATE, 6)
        labels[index]["end_s"] = round(labels[index]["end_sample"] / SAMPLE_RATE, 6)
    labels[-1]["end_sample"] = int(signal.shape[0])
    labels[-1]["end_s"] = round(signal.shape[0] / SAMPLE_RATE, 6)
    return signal.astype(np.float64), labels


def render_material(
    wav_path: str,
    label_path: str,
    seed: int,
    programme: tuple[SegmentSpec, ...] = DEFAULT_PROGRAMME,
) -> dict:
    """Render WAV + JSON label manifest and return the generation manifest."""
    signal, labels = generate_programme(seed, programme)
    write_wav(wav_path, signal)
    manifest = {
        "sample_rate": SAMPLE_RATE,
        "bit_depth": 24,
        "channels": CHANNEL_COUNT,
        "channel_order": list(CHANNEL_LABELS),
        "seed": seed,
        "crossfade_ms": CROSSFADE_MS,
        "total_samples": int(signal.shape[0]),
        "total_duration_s": round(signal.shape[0] / SAMPLE_RATE, 6),
        "segments": labels,
    }
    with open(label_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    return manifest
