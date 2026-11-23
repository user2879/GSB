"""Deterministic segment-level acoustic feature accumulation."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from .segment import Frame, iter_frames
from .util import stable_float


def _percentile(values: list[float], q: float, interpolation: str) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(q)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    if interpolation == "lower":
        return ordered[lower]
    if interpolation == "higher":
        return ordered[upper]
    if interpolation == "midpoint":
        return 0.5 * (ordered[lower] + ordered[upper])
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _mean(values: Iterable[float]) -> float:
    total = 0.0
    count = 0
    for value in values:
        total += float(value)
        count += 1
    return total / count if count else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    squared = 0.0
    for value in values:
        delta = float(value) - mean
        squared += delta * delta
    return math.sqrt(squared / len(values))


def _frame_indices_for_segment(
    samples: np.ndarray,
    sample_rate: int,
    segmentation: dict[str, Any],
    start_sample: int,
    end_sample: int,
) -> list[Frame]:
    frames = iter_frames(samples, sample_rate, segmentation)
    selected = [frame for frame in frames if frame.start_sample >= start_sample and frame.end_sample <= end_sample]
    if selected:
        return selected
    if end_sample <= start_sample:
        return []
    frame_length = max(1, int(round(sample_rate * float(segmentation["frame_length_ms"]) / 1000.0)))
    clipped_end = min(len(samples), max(end_sample, start_sample + 1))
    block = samples[start_sample:clipped_end].astype(np.float64) / 32768.0
    if block.size:
        rms = float(np.sqrt(np.mean(np.square(block), dtype=np.float64)))
        db = max(float(segmentation.get("energy_floor_dbfs", -70.0)), 20.0 * math.log10(max(rms, 1e-12)))
    else:
        db = float(segmentation.get("energy_floor_dbfs", -70.0))
    hop = max(1, int(round(sample_rate * float(segmentation["frame_shift_ms"]) / 1000.0)))
    synthetic_index = start_sample // hop
    return [Frame(index=synthetic_index, start_sample=start_sample, end_sample=clipped_end, rms_dbfs=stable_float(db))]


def _estimate_f0(frame: np.ndarray, sample_rate: int, feature_config: dict[str, Any]) -> float | None:
    if frame.size < 16:
        return None
    x = frame.astype(np.float64)
    x -= float(np.mean(x))
    energy = float(np.dot(x, x))
    if energy <= 1e-12:
        return None
    autocorrelation = np.correlate(x, x, mode="full")[len(x) - 1 :]
    if autocorrelation[0] <= 0:
        return None
    autocorrelation = autocorrelation / autocorrelation[0]
    min_lag = max(2, int(math.floor(sample_rate / float(feature_config["f0_max_hz"]))))
    max_lag = min(len(x) - 2, int(math.ceil(sample_rate / float(feature_config["f0_min_hz"]))))
    if max_lag <= min_lag:
        return None
    search = autocorrelation[min_lag : max_lag + 1]
    peak_offset = int(np.argmax(search))
    peak_lag = min_lag + peak_offset
    peak_value = float(autocorrelation[peak_lag])
    if peak_value < float(feature_config.get("voicing_threshold", 0.28)):
        return None
    lag = float(peak_lag)
    if 0 < peak_lag < len(autocorrelation) - 1:
        left = float(autocorrelation[peak_lag - 1])
        center = float(autocorrelation[peak_lag])
        right = float(autocorrelation[peak_lag + 1])
        denominator = left - 2.0 * center + right
        if abs(denominator) > 1e-12:
            offset = 0.5 * (left - right) / denominator
            lag += max(-0.5, min(0.5, offset))
    f0 = sample_rate / max(lag, 1.0)
    if not float(feature_config["f0_min_hz"]) <= f0 <= float(feature_config["f0_max_hz"]):
        return None
    return stable_float(f0)


def _levinson_durbin(autocorrelation: np.ndarray, order: int) -> np.ndarray:
    error = float(autocorrelation[0])
    coefficients = np.zeros(order + 1, dtype=np.float64)
    coefficients[0] = 1.0
    if error <= 1e-12:
        return coefficients
    for i in range(1, order + 1):
        if i - 1 > 0:
            acc = float(autocorrelation[i]) + float(np.dot(coefficients[1:i], autocorrelation[i - 1 : 0 : -1]))
        else:
            acc = float(autocorrelation[i])
        reflection = -acc / error
        if not math.isfinite(reflection) or abs(reflection) >= 1.0:
            return coefficients
        previous = coefficients.copy()
        for j in range(1, i):
            coefficients[j] = previous[j] + reflection * previous[i - j]
        coefficients[i] = reflection
        error *= 1.0 - reflection * reflection
        if error <= 1e-12:
            break
    return coefficients


def _lpc_formants(frame: np.ndarray, sample_rate: int, feature_config: dict[str, Any]) -> list[float]:
    formant_count = int(feature_config.get("formant_count", 3))
    if frame.size < 64:
        return [0.0] * formant_count
    x = frame.astype(np.float64)
    x -= float(np.mean(x))
    preemphasis = float(feature_config.get("lpc_preemphasis", 0.0))
    if frame.size > 1 and preemphasis:
        preemphasized = np.empty_like(x)
        preemphasized[0] = x[0]
        preemphasized[1:] = x[1:] - preemphasis * x[:-1]
        x = preemphasized
    window = np.hanning(len(x))
    x = x * window
    order = max(int(feature_config.get("lpc_order", 18)), 2 * formant_count + 2)
    order = min(order, len(x) - 2)
    autocorrelation = np.array(
        [float(np.dot(x[: len(x) - lag], x[lag:])) for lag in range(order + 1)],
        dtype=np.float64,
    )
    if autocorrelation[0] <= 1e-12:
        return [0.0] * formant_count
    coefficients = _levinson_durbin(autocorrelation, order)
    roots = np.roots(coefficients)
    candidates: list[tuple[float, float]] = []
    for root in roots:
        if abs(root.imag) < 1e-9 or root.real <= 0 or abs(root) >= 1.0:
            continue
        frequency = abs(math.atan2(root.imag, root.real)) * sample_rate / (2.0 * math.pi)
        if not float(feature_config.get("formant_min_hz", 250.0)) <= frequency <= float(
            feature_config.get("formant_max_hz", 3800.0)
        ):
            continue
        bandwidth = -0.5 * (sample_rate / (2.0 * math.pi)) * math.log(max(abs(root), 1e-12))
        if bandwidth > 650.0:
            continue
        candidates.append((frequency, bandwidth))
    candidates.sort(key=lambda item: (item[0], item[1]))
    selected: list[float] = []
    for frequency, _bandwidth in candidates:
        if any(abs(frequency - prior) < 90.0 for prior in selected):
            continue
        selected.append(stable_float(frequency))
        if len(selected) == formant_count:
            break
    while len(selected) < formant_count:
        selected.append(0.0)
    return selected


def _syllable_rate(frame_db: list[float], sample_rate: int, segmentation: dict[str, Any], feature_config: dict[str, Any]) -> float:
    if len(frame_db) < 6:
        return 0.0
    hop_seconds = float(segmentation["frame_shift_ms"]) / 1000.0
    envelope = np.array([10.0 ** (value / 20.0) for value in frame_db], dtype=np.float64)
    envelope -= float(np.mean(envelope))
    if float(np.dot(envelope, envelope)) <= 1e-12:
        return 0.0
    nfft = 1
    while nfft < len(envelope) * 4:
        nfft <<= 1
    spectrum = np.abs(np.fft.rfft(envelope, n=nfft))
    low = max(1, int(math.ceil(float(feature_config.get("syllable_min_hz", 1.5)) * nfft * hop_seconds)))
    high = min(len(spectrum) - 1, int(math.floor(float(feature_config.get("syllable_max_hz", 8.0)) * nfft * hop_seconds)))
    if high < low:
        return 0.0
    index = low + int(np.argmax(spectrum[low : high + 1]))
    return stable_float(index / (nfft * hop_seconds))


def extract_features(
    samples: np.ndarray,
    sample_rate: int,
    start_sample: int,
    end_sample: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    segmentation = config["segmentation"]
    feature_config = config["features"]
    frames = _frame_indices_for_segment(
        samples,
        sample_rate,
        segmentation,
        start_sample,
        end_sample,
    )
    valid_threshold = float(feature_config.get("valid_frame_dbfs", -48.0))
    valid_frames = [frame for frame in frames if frame.rms_dbfs >= valid_threshold]
    effective = valid_frames if valid_frames else []
    frame_indices = [frame.index for frame in effective]
    frame_db = [frame.rms_dbfs for frame in effective]

    f0_values: list[float] = []
    formant_tracks = [[] for _ in range(int(feature_config.get("formant_count", 3)))]
    for frame in effective:
        block = samples[frame.start_sample : frame.end_sample].astype(np.float64) / 32768.0
        f0 = _estimate_f0(block, sample_rate, feature_config)
        if f0 is not None:
            f0_values.append(f0)
        formants = _lpc_formants(block, sample_rate, feature_config)
        for index, frequency in enumerate(formants[: len(formant_tracks)]):
            if frequency > 0:
                formant_tracks[index].append(frequency)

    quantiles = [float(value) for value in feature_config.get("quantiles", [])]
    interpolation = str(feature_config.get("quantile_interpolation", "linear"))
    q_values: dict[str, float] = {}
    for q in quantiles:
        label = f"p{int(round(q * 100)):02d}"
        q_values[label] = stable_float(_percentile(f0_values, q, interpolation))

    features: dict[str, Any] = {
        "valid_frame_count": len(effective),
        "frame_indices": frame_indices,
        "accumulation_order": "frame_index_ascending",
        "f0_mean_hz": stable_float(_mean(f0_values)),
        "f0_std_hz": stable_float(_std(f0_values)),
        "f0_min_hz": stable_float(min(f0_values) if f0_values else 0.0),
        "f0_max_hz": stable_float(max(f0_values) if f0_values else 0.0),
        "f0_voiced_frame_count": len(f0_values),
        "rms_dbfs_mean": stable_float(_mean(frame_db)),
        "voiced_ratio": stable_float(len(f0_values) / len(effective) if effective else 0.0),
        "speech_rate_sps": _syllable_rate(frame_db, sample_rate, segmentation, feature_config),
        "formant_track_counts": [len(track) for track in formant_tracks],
    }
    features.update(q_values)
    for index, track in enumerate(formant_tracks, start=1):
        mean_value = _mean(track)
        features[f"f{index}_mean_hz"] = stable_float(mean_value)
        features[f"f{index}_median_hz"] = stable_float(
            _percentile(track, 0.5, interpolation)
        )
    return features
