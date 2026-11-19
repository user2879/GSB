"""Deterministic DSP primitives shared by generation and analysis."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import fft as sp_fft
from scipy import signal


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    if seconds < 0.0:
        raise ValueError("duration must be non-negative")
    return int(round(seconds * sample_rate))


def exponential_sweep(
    *,
    sample_rate: int,
    duration_s: float,
    low_hz: float,
    high_hz: float,
    amplitude: float,
    fade_s: float,
) -> np.ndarray:
    """Generate a deterministic exponential sine sweep with raised-cosine edges."""

    count = seconds_to_samples(duration_s, sample_rate)
    if count <= 1:
        raise ValueError("sweep must contain at least two samples")
    t = np.arange(count, dtype=np.float64) / float(sample_rate)
    duration = count / float(sample_rate)
    log_ratio = math.log(high_hz / low_hz)
    phase = 2.0 * math.pi * low_hz * duration / log_ratio * (
        np.exp(t / duration * log_ratio) - 1.0
    )
    sweep = np.sin(phase)
    fade_count = min(seconds_to_samples(fade_s, sample_rate), count // 2)
    if fade_count > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade_count) / fade_count)
        sweep[:fade_count] *= ramp
        sweep[-fade_count:] *= ramp[::-1]
    peak = float(np.max(np.abs(sweep)))
    if peak == 0.0:
        raise ValueError("generated sweep is empty")
    return np.asarray(sweep * (amplitude / peak), dtype=np.float64)


def sync_pulse(
    *,
    sample_rate: int,
    amplitude: float,
    center_hz: float,
    sigma_samples: float,
    radius_samples: int,
) -> tuple[np.ndarray, int]:
    """Return a short, zero-mean-ish pulse and the index of its nominal centre."""

    offsets = np.arange(-radius_samples, radius_samples + 1, dtype=np.float64)
    envelope = np.exp(-0.5 * np.square(offsets / sigma_samples))
    carrier = np.cos(2.0 * math.pi * center_hz * offsets / sample_rate)
    pulse = amplitude * envelope * carrier
    pulse -= float(np.mean(pulse)) * envelope * 0.25
    return pulse.astype(np.float64), radius_samples


def apply_fractional_delay(signal_in: np.ndarray, delay_samples: float, sample_rate: int) -> np.ndarray:
    """Band-limited fractional delay using an FFT phase ramp and zero padding."""

    x = np.asarray(signal_in, dtype=np.float64)
    count = x.size
    if count == 0:
        return x.copy()
    pad = max(64, int(math.ceil(abs(delay_samples))) + 64)
    nfft = sp_fft.next_fast_len(count + 2 * pad, real=True)
    buffered = np.zeros(nfft, dtype=np.float64)
    buffered[pad : pad + count] = x
    spectrum = np.fft.rfft(buffered)
    frequencies = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    spectrum *= np.exp(-2.0j * math.pi * frequencies * delay_samples / sample_rate)
    delayed = np.fft.irfft(spectrum, n=nfft)
    return np.asarray(delayed[pad : pad + count], dtype=np.float64)


def band_limit(
    signal_in: np.ndarray,
    *,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band limiting used by the simulated room path."""

    nyquist = sample_rate / 2.0
    low = max(low_hz, 1.0e-4)
    high = min(high_hz, nyquist * 0.9999)
    if not (0.0 < low < high < nyquist):
        raise ValueError("invalid band limits for simulation")
    sos = signal.butter(order, [low, high], btype="bandpass", fs=sample_rate, output="sos")
    return np.asarray(signal.sosfiltfilt(sos, np.asarray(signal_in, dtype=np.float64)), dtype=np.float64)


def tukey_window(count: int, alpha: float) -> np.ndarray:
    if count <= 0:
        raise ValueError("window length must be positive")
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return signal.windows.tukey(count, alpha=alpha, sym=True).astype(np.float64)


def smooth_band_mask(
    frequencies_hz: np.ndarray,
    low_hz: float,
    high_hz: float,
    taper_hz: float,
) -> np.ndarray:
    f = np.asarray(frequencies_hz, dtype=np.float64)
    taper = max(taper_hz, 1.0e-9)
    mask = np.zeros_like(f)
    mask[(f >= low_hz) & (f <= high_hz)] = 1.0
    left = (f >= low_hz - taper) & (f < low_hz)
    right = (f > high_hz) & (f <= high_hz + taper)
    mask[left] = 0.5 + 0.5 * np.cos(np.pi * (low_hz - f[left]) / taper)
    mask[right] = 0.5 + 0.5 * np.cos(np.pi * (f[right] - high_hz) / taper)
    return np.clip(mask, 0.0, 1.0)


def parabolic_peak(values: np.ndarray) -> tuple[int, float, float]:
    """Return peak index, sub-sample offset and local curvature."""

    y = np.asarray(values, dtype=np.float64)
    if y.size < 3:
        raise ValueError("at least three samples are required for parabolic interpolation")
    peak = int(np.argmax(y))
    if peak == 0 or peak == y.size - 1:
        return peak, 0.0, 0.0
    left = float(y[peak - 1])
    center = float(y[peak])
    right = float(y[peak + 1])
    curvature = left - 2.0 * center + right
    if abs(curvature) < 1.0e-30:
        return peak, 0.0, curvature
    delta = 0.5 * (left - right) / curvature
    delta = float(np.clip(delta, -0.5, 0.5))
    return peak, delta, curvature


def quartic_peak_offset(values: np.ndarray, peak: int) -> float:
    """Estimate a local peak offset with a five-point quartic fit."""

    y = np.asarray(values, dtype=np.float64)
    if peak < 2 or peak > y.size - 3:
        return 0.0
    positions = np.arange(-2, 3, dtype=np.float64)
    samples = y[peak - 2 : peak + 3]
    coefficients = np.polyfit(positions, samples, 4)
    derivative = np.polyder(coefficients)
    roots = np.roots(derivative)
    real_roots = [float(root.real) for root in roots if abs(root.imag) < 1.0e-8 and abs(root.real) <= 1.0]
    if not real_roots:
        return 0.0
    return min(real_roots, key=lambda value: abs(value))


def robust_sigma(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=np.float64).ravel()
    if y.size == 0:
        return 0.0
    median = float(np.median(y))
    mad = float(np.median(np.abs(y - median)))
    if mad == 0.0:
        return float(np.std(y))
    return float(1.482602218505602 * mad)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(weights))
    if total <= 0.0:
        return float(np.mean(values))
    return float(np.sum(values * weights) / total)


def phase_degrees(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    ratio = np.asarray(numerator) / np.asarray(denominator)
    return np.degrees(np.angle(ratio))


def db_ratio(numerator: np.ndarray, denominator: np.ndarray, floor: float = 1.0e-300) -> np.ndarray:
    num = np.maximum(np.abs(numerator), floor)
    den = np.maximum(np.abs(denominator), floor)
    return 20.0 * np.log10(num / den)


def next_odd(value: int) -> int:
    value = int(value)
    return value if value % 2 == 1 else value + 1


def overlap_fraction(band: tuple[float, float], other: tuple[float, float]) -> float:
    lo = max(band[0], other[0])
    hi = min(band[1], other[1])
    if hi <= lo:
        return 0.0
    width = max(other[1] - other[0], 1.0e-12)
    return float(np.clip((hi - lo) / width, 0.0, 1.0))


def rms(values: Sequence[float] | np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(data))))
