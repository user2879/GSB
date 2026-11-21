"""Windowing, harmonic frame synthesis, overlap-add and waveform metrics."""
from __future__ import annotations

import math

import numpy as np

TWO_PI = 2.0 * math.pi
ENERGY_SCALE = 32767.0 * 0.85


def make_window(window_type: str, length: int) -> np.ndarray:
    idx = np.arange(length, dtype=np.float64)
    if window_type == "rect":
        return np.ones(length, dtype=np.float64)
    if length == 1:
        return np.ones(length, dtype=np.float64)
    phase = TWO_PI * idx / (length - 1)
    if window_type == "hann":
        return 0.5 - 0.5 * np.cos(phase)
    if window_type == "hamming":
        return 0.54 - 0.46 * np.cos(phase)
    raise ValueError(f"unknown window type: {window_type}")


def synthesize_frame(amplitudes: np.ndarray, phases_rad: np.ndarray,
                         f0: float, gain: float, length: int,
                         sample_rate: int) -> np.ndarray:
    """Sinusoidal frame: phase at frame origin, oscillator runs within frame."""
    frame = np.zeros(length, dtype=np.float64)
    time_idx = np.arange(length, dtype=np.float64)
    for harmonic in range(amplitudes.shape[0]):
        freq = (harmonic + 1) * float(f0)
        frame += float(amplitudes[harmonic]) * np.sin(
            phases_rad[harmonic] + TWO_PI * freq * time_idx / sample_rate
        )
    return frame * float(gain)


def overlap_add(frames: list[np.ndarray], hop: int, window_type: str,
                normalization_rule: str) -> np.ndarray:
    if not frames:
        return np.zeros(0, dtype=np.float64)
    length = frames[0].shape[0]
    total = length + hop * (len(frames) - 1)
    output = np.zeros(total, dtype=np.float64)
    weight = np.zeros(total, dtype=np.float64)
    window = make_window(window_type, length)
    for idx, frame in enumerate(frames):
        start = idx * hop
        output[start:start + length] += frame * window
        weight[start:start + length] += window
    if normalization_rule == "cola":
        valid = weight > 1e-6
        output[valid] /= weight[valid]
        # Edge samples whose window weight is negligible: hold the nearest
        # interior sample instead of dividing by a near-zero weight.
        output[~valid] = 0.0
    return output


def apply_energy_scale(wave_float: np.ndarray, rounding: str) -> np.ndarray:
    from .wav_io import pack_pcm16

    return pack_pcm16(wave_float * ENERGY_SCALE, rounding)


def normalized_cross_correlation(reference: np.ndarray, candidate: np.ndarray,
                                 search_ms: float, sample_rate: int) -> tuple[int, float]:
    """Best lag aligning candidate to reference; ties resolve to smaller |lag|."""
    max_lag = int(round(search_ms * sample_rate / 1000.0))
    n = min(reference.shape[0], candidate.shape[0])
    ref = reference[:n].astype(np.float64)
    cand = candidate[:n].astype(np.float64)
    ref_n = ref - ref.mean()
    cand_n = cand - cand.mean()
    ref_energy = math.sqrt(float(np.dot(ref_n, ref_n)))
    cand_energy = math.sqrt(float(np.dot(cand_n, cand_n)))
    if ref_energy == 0.0 or cand_energy == 0.0:
        return 0, 0.0
    best_lag, best_corr = 0, -2.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a = ref_n[-lag:]
            b = cand_n[:n + lag]
        elif lag > 0:
            a = ref_n[:n - lag]
            b = cand_n[lag:]
        else:
            a, b = ref_n, cand_n
        corr = float(np.dot(a, b)) / (
            ref_energy * cand_energy * math.sqrt(len(a) / n)
        )
        if corr > best_corr + 1e-12 or (
            abs(corr - best_corr) <= 1e-12 and abs(lag) < abs(best_lag)
        ):
            best_corr, best_lag = corr, lag
    return best_lag, best_corr


def spectral_distance(reference: np.ndarray, candidate: np.ndarray) -> float:
    n = min(reference.shape[0], candidate.shape[0])
    ref = reference[:n].astype(np.float64)
    cand = candidate[:n].astype(np.float64)
    ref_spec = np.abs(np.fft.rfft(ref))
    cand_spec = np.abs(np.fft.rfft(cand))
    ref_log = np.log1p(ref_spec)
    cand_log = np.log1p(cand_spec)
    denom = math.sqrt(float(np.dot(ref_log, ref_log))) + 1e-12
    return float(np.linalg.norm(ref_log - cand_log) / denom)


def waveform_rmse(reference: np.ndarray, candidate: np.ndarray) -> float:
    n = min(reference.shape[0], candidate.shape[0])
    ref = reference[:n].astype(np.float64) / 32768.0
    cand = candidate[:n].astype(np.float64) / 32768.0
    return float(math.sqrt(np.mean((ref - cand) ** 2)))


def frame_summaries(frames: list[np.ndarray]) -> list[dict]:
    summaries = []
    for idx, frame in enumerate(frames):
        peak = float(np.max(np.abs(frame))) if frame.size else 0.0
        rms = float(math.sqrt(np.mean(frame ** 2))) if frame.size else 0.0
        summaries.append({
            "frame_index": idx,
            "peak_float": round(peak, 6),
            "rms_float": round(rms, 6),
            "samples": int(frame.shape[0]),
        })
    return summaries


def shift_wave(wave: np.ndarray, lag: int) -> np.ndarray:
    if lag == 0:
        return wave
    result = np.zeros_like(wave)
    if lag > 0:
        result[lag:] = wave[:-lag] if lag < len(wave) else 0
    else:
        result[:lag] = wave[-lag:]
    return result
