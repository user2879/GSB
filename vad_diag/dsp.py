"""Deterministic digital signal-processing primitives.

All spectral math uses NumPy with explicit, fixed-length FFTs and deterministic
windowing; the *accumulated* features that enter the cache are quantized to
fixed-point integers elsewhere, so floating-point ordering never leaks into the
reproducible ledger artifacts.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

# A small, fixed Mel filter bank.  Edges are in Hz; filters are triangular.
MEL_LOW_HZ = 0.0
MEL_HIGH_HZ = 4000.0
NUM_MEL_BINS = 6


def frame_intervals(total_samples: int, frame_length: int,
                    frame_shift: int) -> List[Tuple[int, int]]:
    """Return global ``[start, end)`` frame intervals.

    Frames are anchored at sample zero with hop ``frame_shift``.  A trailing
    partial frame shorter than ``frame_length`` is dropped deterministically,
    and the decision is recorded in the stage metadata.
    """
    intervals: List[Tuple[int, int]] = []
    if total_samples < frame_length:
        return intervals
    start = 0
    while start + frame_length <= total_samples:
        intervals.append((start, start + frame_length))
        start += frame_shift
    return intervals


def window_intervals(total_samples: int, window_length: int,
                     window_shift: int) -> List[Tuple[int, int]]:
    return frame_intervals(total_samples, window_length, window_shift)


def _hann(length: int) -> np.ndarray:
    """Periodic Hann window (NumPy-independent, fully deterministic)."""
    if length <= 1:
        return np.ones(length, dtype=np.float64)
    idx = np.arange(length, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * math.pi * idx / length)


def frame_energy(samples: Sequence[int]) -> float:
    """Mean squared amplitude, normalized to [0, 1] for 16-bit PCM."""
    if len(samples) == 0:
        return 0.0
    arr = np.asarray(samples, dtype=np.float64) / 32768.0
    return float(np.mean(arr * arr))


def spectral_flatness(samples: Sequence[int]) -> float:
    """Wiener entropy in [0, 1]; tonality-like measure used by the VAD gate."""
    n = len(samples)
    if n == 0:
        return 0.0
    arr = np.asarray(samples, dtype=np.float64) * _hann(n)
    spectrum = np.fft.rfft(arr, n=n)
    power = (spectrum.real ** 2 + spectrum.imag ** 2)
    power = power[1:]  # drop DC
    power = np.maximum(power, 1e-12)
    geometric = math.exp(float(np.mean(np.log(power))))
    arithmetic = float(np.mean(power))
    if arithmetic <= 0.0:
        return 0.0
    return min(1.0, geometric / arithmetic)


def _mel_filters(fft_bins: int, sample_rate: int) -> np.ndarray:
    """Build a deterministic triangular Mel filter-bank matrix."""
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_points = np.linspace(hz_to_mel(MEL_LOW_HZ),
                             hz_to_mel(min(MEL_HIGH_HZ, sample_rate / 2)),
                             NUM_MEL_BINS + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((fft_bins + 1) * hz_points / sample_rate).astype(int)
    filters = np.zeros((NUM_MEL_BINS, fft_bins), dtype=np.float64)
    for m in range(1, NUM_MEL_BINS + 1):
        left, center, right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(left, min(center, fft_bins)):
            if center > left:
                filters[m - 1, k] = (k - left) / max(1, (center - left))
        for k in range(center, min(right, fft_bins)):
            if right > center:
                filters[m - 1, k] = (right - k) / max(1, (right - center))
    return filters


_MEL_CACHE = {}


def mel_energy(samples: Sequence[int], sample_rate: int) -> List[float]:
    """Log-Mel energies for one frame (length = :data:`NUM_MEL_BINS`)."""
    n = len(samples)
    if n == 0:
        return [0.0] * NUM_MEL_BINS
    arr = np.asarray(samples, dtype=np.float64) * _hann(n)
    spectrum = np.fft.rfft(arr, n=n)
    power = (spectrum.real ** 2 + spectrum.imag ** 2) / n
    fft_bins = power.shape[0]
    key = (fft_bins, sample_rate)
    if key not in _MEL_CACHE:
        _MEL_CACHE[key] = _mel_filters(fft_bins, sample_rate)
    filters = _MEL_CACHE[key]
    filtered = filters @ power
    filtered = np.maximum(filtered, 1e-10)
    return [float(math.log(v)) for v in filtered]


def zero_crossing_rate(samples: Sequence[int]) -> float:
    if len(samples) < 2:
        return 0.0
    arr = np.asarray(samples, dtype=np.int64)
    crossings = np.sum(np.abs(np.diff(np.sign(arr))) > 0)
    return float(crossings) / float(len(arr) - 1)


def spectral_centroid(samples: Sequence[int], sample_rate: int) -> float:
    n = len(samples)
    if n == 0:
        return 0.0
    arr = np.asarray(samples, dtype=np.float64) * _hann(n)
    magnitude = np.abs(np.fft.rfft(arr, n=n))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    total = float(np.sum(magnitude))
    if total <= 0.0:
        return 0.0
    return float(np.sum(freqs * magnitude) / total) / float(sample_rate / 2)


# Feature dimension layout used by the window stage.
FEATURE_DIM = NUM_MEL_BINS + 2  # 6 Mel + ZCR + centroid


def raw_frame_feature(samples: Sequence[int], sample_rate: int) -> List[float]:
    """Full real-valued per-frame feature vector in a fixed dimension order.

    Every component is normalized into a bounded range so the fixed-point Qn
    accumulators never saturate trivially and quantization-level changes move
    the integer and de-quantized values meaningfully.  Dimension order is part
    of the deterministic contract: 6 normalized log-Mel energies, then ZCR,
    then normalized spectral centroid.
    """
    mel = mel_energy(samples, sample_rate)
    # Observed log-Mel range for 16-bit PCM is roughly [8, 24]; map it to a
    # bounded [0, 1] interval so fixed-point accumulation stays in range.
    mel_norm = [min(1.0, max(0.0, (m - 8.0) / 16.0)) for m in mel]
    return mel_norm + [
        zero_crossing_rate(samples),
        spectral_centroid(samples, sample_rate),
    ]
