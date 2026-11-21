"""Deterministic frame features: mel energy, cepstrum and F0 candidates."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .config import WorkbenchConfig
from .util import q_from_float


def frame_count(sample_count: int, frame_length: int, frame_shift: int) -> int:
    if sample_count <= 0:
        return 0
    if sample_count <= frame_length:
        return 1
    return 1 + int(math.ceil((sample_count - frame_length) / frame_shift))


def frame_ranges(sample_count: int, cfg: WorkbenchConfig) -> list[tuple[int, int, bool]]:
    """Return (start_sample, valid_samples, padded) in ascending frame order."""

    length = cfg.frame_length_samples
    shift = cfg.frame_shift_samples
    count = frame_count(sample_count, length, shift)
    ranges: list[tuple[int, int, bool]] = []
    for index in range(count):
        start = index * shift
        valid = max(0, min(length, sample_count - start))
        ranges.append((start, valid, valid < length))
    return ranges


@lru_cache(maxsize=16)
def _hann_window(length: int) -> np.ndarray:
    n = np.arange(length, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / length)


@lru_cache(maxsize=16)
def _mel_filterbank(cfg: WorkbenchConfig) -> np.ndarray:
    """Triangular filters over FFT bins; rows are mel bands in ascending order."""

    fft_bins = cfg.frame_length_samples // 2 + 1
    frequencies = np.arange(fft_bins, dtype=np.float64) * cfg.sample_rate / cfg.frame_length_samples

    def hz_to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_low = hz_to_mel(0.0)
    mel_high = hz_to_mel(cfg.mel_max_hz)
    points = np.linspace(mel_low, mel_high, cfg.mel_bands + 2, dtype=np.float64)
    hz_points = np.array([mel_to_hz(float(point)) for point in points], dtype=np.float64)
    bank = np.zeros((cfg.mel_bands, fft_bins), dtype=np.float64)
    for band in range(cfg.mel_bands):
        left, center, right = hz_points[band], hz_points[band + 1], hz_points[band + 2]
        for bin_index in range(fft_bins):
            frequency = float(frequencies[bin_index])
            if left < frequency < right:
                if frequency <= center:
                    bank[band, bin_index] = (frequency - left) / max(center - left, 1e-12)
                else:
                    bank[band, bin_index] = (right - frequency) / max(right - center, 1e-12)
        if not np.any(bank[band]):
            nearest = int(np.argmin(np.abs(frequencies - center)))
            bank[band, nearest] = 1.0
    return bank


@lru_cache(maxsize=16)
def _dct_matrix(order: int, mel_bands: int) -> np.ndarray:
    matrix = np.zeros((order + 1, mel_bands), dtype=np.float64)
    for coefficient in range(order + 1):
        factor = math.sqrt(1.0 / mel_bands) if coefficient == 0 else math.sqrt(2.0 / mel_bands)
        for band in range(mel_bands):
            matrix[coefficient, band] = factor * math.cos(
                math.pi * coefficient * (band + 0.5) / mel_bands
            )
    return matrix


@lru_cache(maxsize=16)
def _fft_size(frame_length: int) -> int:
    size = 1
    while size < frame_length:
        size *= 2
    return max(size, 256)


def _energy_quantities(frame: np.ndarray, valid_samples: int, cfg: WorkbenchConfig) -> tuple[int, int]:
    valid = frame[:valid_samples] if valid_samples > 0 else frame[:1]
    rms = float(np.sqrt(np.mean(np.square(valid), dtype=np.float64))) if valid.size else 0.0
    db = 20.0 * math.log10(max(rms, 1e-12))
    return q_from_float(rms, cfg.scale), q_from_float(db, cfg.scale)


def _f0_candidate(frame: np.ndarray, energy_db_q: int, cfg: WorkbenchConfig) -> tuple[int, int]:
    if energy_db_q < cfg.energy_threshold_q:
        return -1, 0
    x = frame - float(np.mean(frame))
    if float(np.max(np.abs(x), initial=0.0)) < 1e-7:
        return -1, 0
    windowed = x * _hann_window(frame.size)
    nfft = _fft_size(frame.size * 2)
    spectrum = np.fft.rfft(windowed, n=nfft)
    autocorrelation = np.fft.irfft(np.abs(spectrum) ** 2, n=nfft)[: frame.size]
    zero_lag = float(autocorrelation[0])
    if zero_lag <= 1e-12:
        return -1, 0
    min_lag = max(1, int(math.floor(cfg.sample_rate / cfg.f0_max_hz)))
    max_lag = min(frame.size - 2, int(math.ceil(cfg.sample_rate / cfg.f0_min_hz)))
    if max_lag <= min_lag:
        return -1, 0
    search = autocorrelation[min_lag : max_lag + 1]
    peak_offset = int(np.argmax(search))
    peak_lag = min_lag + peak_offset
    peak_value = float(search[peak_offset]) / zero_lag
    if peak_value < 0.18:
        return -1, 0
    lag = float(peak_lag)
    if 0 < peak_offset < len(search) - 1:
        left = float(search[peak_offset - 1])
        center = float(search[peak_offset])
        right = float(search[peak_offset + 1])
        denominator = left - 2.0 * center + right
        if abs(denominator) > 1e-12:
            lag += 0.5 * (left - right) / denominator
    lag = max(float(min_lag), min(float(max_lag), lag))
    f0_hz = cfg.sample_rate / lag
    if not (cfg.f0_min_hz <= f0_hz <= cfg.f0_max_hz):
        return -1, 0
    position = (f0_hz - cfg.f0_min_hz) / (cfg.f0_max_hz - cfg.f0_min_hz)
    unit = int(round(position * (cfg.f0_quantization_levels - 1)))
    unit = max(0, min(cfg.f0_quantization_levels - 1, unit))
    return unit, q_from_float(f0_hz, cfg.scale)


def analyze_frame(
    frame: np.ndarray,
    cfg: WorkbenchConfig,
    *,
    frame_index: int,
    start_sample: int,
    valid_samples: int,
    padded: bool,
) -> dict[str, object]:
    energy_q, energy_db_q = _energy_quantities(frame, valid_samples, cfg)
    pre_emphasized = np.empty_like(frame)
    if frame.size:
        pre_emphasized[0] = frame[0]
        pre_emphasized[1:] = frame[1:] - 0.97 * frame[:-1]
    windowed = pre_emphasized * _hann_window(frame.size)
    nfft = _fft_size(frame.size)
    spectrum = np.fft.rfft(windowed, n=nfft)
    power = (np.abs(spectrum) ** 2) / float(nfft)

    bank = _mel_filterbank(cfg)
    # The band loop is deliberately ascending; only this reduction defines mel energy.
    mel_energy = np.zeros(cfg.mel_bands, dtype=np.float64)
    for band in range(cfg.mel_bands):
        mel_energy[band] = float(np.sum(power[: bank.shape[1]] * bank[band], dtype=np.float64))
    log_mel_db = 10.0 * np.log10(np.maximum(mel_energy, 1e-12))
    cepstrum = _dct_matrix(cfg.cepstrum_order, cfg.mel_bands) @ log_mel_db
    f0_unit, f0_hz_q = _f0_candidate(frame, energy_db_q, cfg)
    return {
        "frame_index": int(frame_index),
        "start_sample": int(start_sample),
        "valid_samples": int(valid_samples),
        "padded": bool(padded),
        "energy_q": int(energy_q),
        "energy_db_q": int(energy_db_q),
        "mel_energy_q": [q_from_float(float(value), cfg.scale) for value in log_mel_db],
        "cepstrum_q": [q_from_float(float(value), cfg.scale) for value in cepstrum],
        "f0_unit": int(f0_unit),
        "source_f0_hz_q": int(f0_hz_q),
    }


def quantile_from_sorted(values: list[int], probability: float, method: str) -> int:
    """Exact integer interpolation over sorted integer values."""

    from fractions import Fraction
    from .util import q_div

    if not values:
        return 0
    if len(values) == 1:
        return int(values[0])
    p = Fraction(str(probability))
    position = p * (len(values) - 1)
    lower = int(position)
    upper = min(len(values) - 1, lower + 1)
    fraction = position - lower
    lo, hi = int(values[lower]), int(values[upper])
    if method == "lower":
        selected = lo
    elif method == "higher":
        selected = hi
    elif method == "midpoint":
        return q_div(lo + hi, 2)
    elif method == "nearest":
        selected = lo if fraction < Fraction(1, 2) else hi
    else:
        return int(lo + q_div((hi - lo) * fraction.numerator * 2, fraction.denominator * 2))
    return int(selected)


def fixed_stats(values: list[int], quantiles: tuple[float, ...], method: str) -> dict[str, object]:
    if not values:
        return {
            "count": 0,
            "mean_q": 0,
            "variance_q": 0,
            "quantiles_q": [0 for _ in quantiles],
        }
    n = len(values)
    total = 0
    for value in values:
        total += int(value)
    mean_q = int(round(total / n))
    variance_numerator = 0
    for value in values:
        variance_numerator += (int(value) * n - total) ** 2
    variance_denominator = n * n * n
    variance_q = (variance_numerator + variance_denominator // 2) // variance_denominator
    ordered = sorted(int(value) for value in values)
    return {
        "count": n,
        "mean_q": mean_q,
        "variance_q": int(variance_q),
        "quantiles_q": [quantile_from_sorted(ordered, q, method) for q in quantiles],
    }


def vector_stats(
    vectors: list[list[int]], quantiles: tuple[float, ...], method: str
) -> dict[str, object]:
    dimensions = len(vectors[0]) if vectors else 0
    means: list[int] = []
    variances: list[int] = []
    quantile_vectors: list[list[int]] = []
    count = len(vectors)
    for dimension in range(dimensions):
        values = [int(vector[dimension]) for vector in vectors]
        total = 0
        for value in values:
            total += value
        mean_q = int(round(total / count)) if count else 0
        variance_numerator = 0
        if count:
            for value in values:
                variance_numerator += (value * count - total) ** 2
            variance_denominator = count * count * count
            variance_q = (variance_numerator + variance_denominator // 2) // variance_denominator
        else:
            variance_q = 0
        ordered = sorted(values)
        means.append(mean_q)
        variances.append(int(variance_q))
        quantile_vectors.append([quantile_from_sorted(ordered, q, method) for q in quantiles])
    return {
        "count": count,
        "mean_q": means,
        "variance_q": variances,
        "quantiles_q": quantile_vectors,
    }
