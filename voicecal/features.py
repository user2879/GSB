"""Deterministic frame slicing, acoustic features, and segment aggregation."""

from __future__ import annotations

import math
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from .util import stable_sort_tasks

ALGORITHM_METADATA = {
    "frame_window": "hann",
    "pitch_estimator": "normalized_autocorrelation",
    "pitch_candidates": "top_three_local_peaks",
    "pitch_interpolation": "none",
    "energy_estimator": "frame_rms_dbfs",
    "speech_rate_estimator": "energy_onset_peaks_per_second",
    "rhythm_estimator": "onset_interval_coefficient_of_variation",
    "frame_rhythm_features": "energy_delta_db_activity_onset",
    "aggregation": "explicit_index_order_fsum",
}


@lru_cache(maxsize=16)
def _hann_cached(length: int) -> tuple[float, ...]:
    if length <= 1:
        return tuple(1.0 for _ in range(length))
    indices = np.arange(length, dtype=np.float64)
    return tuple((0.5 - 0.5 * np.cos((2.0 * math.pi * indices) / (length - 1))).tolist())


def _hann(length: int) -> np.ndarray:
    return np.asarray(_hann_cached(length), dtype=np.float64)


def _rms_dbfs(frame: np.ndarray) -> float:
    if frame.size == 0:
        return -160.0
    centered = frame.astype(np.float64) / 32768.0
    mean_square = float(np.dot(centered, centered)) / centered.size
    return 20.0 * math.log10(math.sqrt(max(mean_square, 1e-16)))


def _pitch_candidates(
    frame: np.ndarray,
    sample_rate: int,
    min_hz: float,
    max_hz: float,
    correlation_threshold: float,
) -> tuple[list[float], float | None, float]:
    if frame.size < 3:
        return [], None, 0.0
    centered = frame.astype(np.float64) - float(np.mean(frame.astype(np.float64)))
    min_lag = max(2, int(math.floor(sample_rate / max_hz)))
    max_lag = min(frame.size - 2, int(math.ceil(sample_rate / min_hz)))
    if max_lag <= min_lag:
        return [], None, 0.0
    length = centered.size
    prefix = np.concatenate(([0.0], np.cumsum(centered * centered, dtype=np.float64)))
    autocorrelation = np.correlate(centered, centered, mode="full")[length - 1 :]
    lags = np.arange(min_lag, max_lag + 1)
    left_energy = prefix[length - lags]
    right_energy = prefix[length] - prefix[lags]
    denominators = np.sqrt(np.maximum(left_energy * right_energy, 1e-30))
    correlations_array = autocorrelation[lags] / denominators
    correlations = {lag: float(correlations_array[index]) for index, lag in enumerate(lags.tolist())}
    candidates: list[tuple[int, float]] = []
    for lag in range(min_lag + 1, max_lag):
        correlation = correlations[lag]
        if correlation >= correlation_threshold and correlation >= correlations[lag - 1] and correlation >= correlations[lag + 1]:
            candidates.append((lag, correlation))
    candidates.sort(key=lambda item: (-item[1], item[0]))
    selected = candidates[:3]
    hz_values = [round(sample_rate / lag, 6) for lag, _ in selected]
    best_corr = max((corr for _, corr in candidates), default=0.0)
    best_hz = hz_values[0] if hz_values else None
    return hz_values, best_hz, best_corr


def extract_frame(
    file_key: str,
    frame_index: int,
    start_sample: int,
    original_count: int,
    frame_samples: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    frame_len = int(frame_samples.size)
    padded = original_count < frame_len
    window = _hann(frame_len)
    windowed = frame_samples.astype(np.float64) * window
    rms = _rms_dbfs(frame_samples)
    energy = 10.0 ** (rms / 20.0)
    zero_crossings = 0
    if frame_samples.size > 1:
        signs = np.signbit(frame_samples.astype(np.float64))
        zero_crossings = int(np.count_nonzero(signs[1:] != signs[:-1]))
    zcr = zero_crossings / max(frame_samples.size - 1, 1)
    candidates, f0_hz, correlation = _pitch_candidates(
        windowed,
        int(config["sample_rate"]),
        float(config["pitch_min_hz"]),
        float(config["pitch_max_hz"]),
        float(config["voiced_correlation"]),
    )
    voiced = rms > float(config["silence_floor_dbfs"]) and correlation >= float(config["voiced_correlation"])
    if not voiced:
        f0_hz = None
    return {
        "sort_key": f"{file_key}|{frame_index:012d}|extract.frame",
        "frame_index": frame_index,
        "start_sample": start_sample,
        "original_sample_count": original_count,
        "effective_sample_count": frame_len,
        "padded": padded,
        "rms_dbfs": round(rms, 6),
        "energy": round(energy, 10),
        "zero_crossing_rate": round(zcr, 6),
        "f0_candidates_hz": candidates,
        "f0_hz": round(f0_hz, 6) if f0_hz is not None else None,
        "autocorrelation": round(correlation, 6),
        "voiced": voiced,
    }


@dataclass
class WorkStats:
    frame_tasks: int = 0
    aggregation_tasks: int = 0
    initial_evidence_tasks: int = 0
    observed_threads: int = 0
    active_workers: int = 0
    max_active_workers: int = 0
    active_initial_workers: int = 0
    max_active_initial_workers: int = 0
    lock: Any = field(default_factory=lambda: __import__("threading").Lock())

    def mark_thread(self) -> None:
        import threading

        with self.lock:
            self.observed_threads = max(self.observed_threads, threading.active_count())

    def enter_worker(self) -> None:
        with self.lock:
            self.active_workers += 1
            self.max_active_workers = max(self.max_active_workers, self.active_workers)

    def leave_worker(self) -> None:
        with self.lock:
            self.active_workers = max(0, self.active_workers - 1)

    def enter_initial(self) -> None:
        with self.lock:
            self.initial_evidence_tasks += 1
            self.active_initial_workers += 1
            self.max_active_initial_workers = max(self.max_active_initial_workers, self.active_initial_workers)

    def leave_initial(self) -> None:
        with self.lock:
            self.active_initial_workers = max(0, self.active_initial_workers - 1)


def slice_frames(samples: np.ndarray, frame_length: int, frame_shift: int) -> list[tuple[int, int, np.ndarray]]:
    frames: list[tuple[int, int, np.ndarray]] = []
    total = int(samples.size)
    start = 0
    while start < total:
        raw = samples[start : start + frame_length]
        if raw.size < frame_length:
            raw = np.pad(raw, (0, frame_length - raw.size), mode="constant")
        frames.append((start, int(samples[start : start + frame_length].size), raw))
        start += frame_shift
    return frames


def extract_audio_frames(
    samples: np.ndarray,
    file_key: str,
    frame_config: dict[str, Any],
    max_workers: int,
    stats: WorkStats,
) -> list[dict[str, Any]]:
    frames = slice_frames(samples, int(frame_config["frame_length_samples"]), int(frame_config["frame_shift_samples"]))
    tasks: list[tuple[str, Callable[[], dict[str, Any]]]] = []
    for frame_index, (start, original_count, raw) in enumerate(frames):
        sort_key = f"{file_key}|{frame_index:012d}|extract.frame"

        def run_task(index: int = frame_index, begin: int = start, count: int = original_count, data: np.ndarray = raw) -> dict[str, Any]:
            stats.mark_thread()
            stats.enter_worker()
            try:
                return extract_frame(file_key, index, begin, count, data, frame_config)
            finally:
                stats.leave_worker()

        tasks.append((sort_key, run_task))
    with stats.lock:
        stats.frame_tasks += len(tasks)
    if max_workers <= 1 or len(tasks) <= 1:
        results = [task() for _, task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="voicecal-frame") as pool:
            futures = [(sort_key, pool.submit(task)) for sort_key, task in tasks]
            results = [future.result() for _, future in futures]
    ordered = stable_sort_tasks(results)
    activity_threshold = float(frame_config["silence_floor_dbfs"]) + float(frame_config["onset_margin_db"])
    previous_rms: float | None = None
    previous_active = False
    for frame in ordered:
        current_rms = float(frame["rms_dbfs"])
        active = current_rms >= activity_threshold
        frame["speech_rhythm"] = {
            "energy_delta_db": round(current_rms - previous_rms, 6) if previous_rms is not None else 0.0,
            "activity": active,
            "onset": bool(active and not previous_active and previous_rms is not None),
            "activity_threshold_dbfs": round(activity_threshold, 6),
        }
        previous_rms = current_rms
        previous_active = active
    return ordered


def _ordered_mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return math.fsum(materialized) / len(materialized) if materialized else 0.0


def _ordered_std(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    mean = math.fsum(materialized) / len(materialized)
    variance = math.fsum((value - mean) * (value - mean) for value in materialized) / len(materialized)
    return math.sqrt(max(variance, 0.0))


def aggregate_features(frames: list[dict[str, Any]], sample_count: int, sample_rate: int, frame_config: dict[str, Any]) -> dict[str, Any]:
    ordered = sorted(frames, key=lambda frame: frame["frame_index"])
    voiced = [frame for frame in ordered if frame["voiced"] and frame["f0_hz"] is not None]
    f0_values = [float(frame["f0_hz"]) for frame in voiced]
    rms_values = [float(frame["rms_dbfs"]) for frame in ordered]
    duration_s = sample_count / float(sample_rate)
    duration_ms = duration_s * 1000.0

    threshold = max(
        float(frame_config["silence_floor_dbfs"]) + float(frame_config["onset_margin_db"]),
        min(rms_values) + 0.65 * (max(rms_values) - min(rms_values)) if rms_values else 0.0,
    )
    onset_indices: list[int] = []
    min_gap_frames = max(1, int(round(float(frame_config["min_onset_gap_ms"]) / float(frame_config["frame_shift_ms"]))))
    for index, frame in enumerate(ordered):
        previous = ordered[index - 1] if index else None
        if frame["rms_dbfs"] >= threshold and (previous is None or previous["rms_dbfs"] < threshold):
            if not onset_indices or index - onset_indices[-1] >= min_gap_frames:
                onset_indices.append(index)
    intervals = [
        (onset_indices[index] - onset_indices[index - 1]) * float(frame_config["frame_shift_ms"]) / 1000.0
        for index in range(1, len(onset_indices))
    ]
    mean_interval = _ordered_mean(intervals)
    rhythm_variability = _ordered_std(intervals) / mean_interval if mean_interval > 0 else 0.0
    speech_rate = len(onset_indices) / duration_s if duration_s > 0 else 0.0
    return {
        "frames_used": len(ordered),
        "padded_frame_count": sum(1 for frame in ordered if frame["padded"]),
        "voiced_frame_count": len(voiced),
        "duration_ms": round(duration_ms, 6),
        "f0_mean_hz": round(_ordered_mean(f0_values), 6),
        "f0_std_hz": round(_ordered_std(f0_values), 6),
        "energy_mean_dbfs": round(_ordered_mean(rms_values), 6),
        "energy_min_dbfs": round(min(rms_values), 6) if rms_values else -160.0,
        "energy_max_dbfs": round(max(rms_values), 6) if rms_values else -160.0,
        "energy_dynamic_db": round(max(rms_values) - min(rms_values), 6) if rms_values else 0.0,
        "voiced_ratio": round(len(voiced) / len(ordered), 6) if ordered else 0.0,
        "speech_rate_sps": round(speech_rate, 6),
        "rhythm_variability": round(rhythm_variability, 6),
        "onset_count": len(onset_indices),
        "onset_threshold_dbfs": round(threshold, 6),
        "feature_subset": [
            "f0_mean_hz",
            "f0_std_hz",
            "energy_mean_dbfs",
            "energy_dynamic_db",
            "speech_rate_sps",
            "rhythm_variability",
        ],
    }


def aggregate_audio(
    frames: list[dict[str, Any]],
    sample_count: int,
    file_key: str,
    frame_config: dict[str, Any],
    stats: WorkStats,
) -> dict[str, Any]:
    stats.mark_thread()
    stats.aggregation_tasks += 1
    return aggregate_features(frames, sample_count, int(frame_config["sample_rate"]), frame_config)
