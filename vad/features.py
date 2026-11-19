"""Block-wise deterministic frame feature extraction."""

from __future__ import annotations

from typing import Any, Iterable, Iterator

import numpy as np

from .util import round_float, round_significant


def frame_geometry(config: dict[str, Any], sample_rate: int) -> tuple[int, int]:
    frame_samples = int(round(float(config["frame_ms"]) * sample_rate / 1000.0))
    hop_samples = int(round(float(config["hop_ms"]) * sample_rate / 1000.0))
    if frame_samples < 2 or hop_samples < 1:
        raise ValueError("configured frame/hop produces invalid sample geometry")
    return frame_samples, hop_samples


def frame_count(sample_count: int, hop_samples: int) -> int:
    if sample_count <= 0:
        return 0
    return int((sample_count + hop_samples - 1) // hop_samples)


def frame_starts(sample_count: int, hop_samples: int) -> np.ndarray:
    if sample_count <= 0:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, sample_count, hop_samples, dtype=np.int64)


def iter_frame_blocks(
    samples: np.ndarray,
    starts: np.ndarray,
    frame_samples: int,
    block_size: int = 1024,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (start_index, padded int16 frames, valid sample counts)."""

    total = int(starts.size)
    for offset in range(0, total, block_size):
        block_starts = starts[offset : offset + block_size]
        result = np.zeros((block_starts.size, frame_samples), dtype=np.int16)
        valid = np.empty(block_starts.size, dtype=np.int64)
        for row, start in enumerate(block_starts.tolist()):
            count = min(frame_samples, samples.size - int(start))
            valid[row] = count
            result[row, :count] = samples[int(start) : int(start) + count]
        yield offset, result, valid


def zcr_from_frames(frames: np.ndarray) -> np.ndarray:
    if frames.shape[1] < 2:
        return np.zeros(frames.shape[0], dtype=np.float64)
    left = frames[:, :-1]
    right = frames[:, 1:]
    crossings = ((left > 0) & (right < 0)) | ((left < 0) & (right > 0))
    return np.mean(crossings, axis=1, dtype=np.float64)


def flatness_from_frames(
    frames: np.ndarray, window: np.ndarray, epsilon: float
) -> np.ndarray:
    windowed = frames.astype(np.float64, copy=False) * window[np.newaxis, :]
    spectrum = np.abs(np.fft.rfft(windowed, axis=1))[:, 1:]
    arithmetic = np.mean(spectrum, axis=1, dtype=np.float64) + epsilon
    geometric = np.exp(np.mean(np.log(spectrum + epsilon), axis=1, dtype=np.float64))
    result = geometric / arithmetic
    result = np.clip(result, 0.0, 1.0)
    result[arithmetic <= epsilon * 1.000001] = 0.0
    return result


def extract_features(
    samples: np.ndarray,
    sample_rate: int,
    config: dict[str, Any],
    precision: int,
    *,
    block_size: int = 1024,
) -> dict[str, Any]:
    frame_samples, hop_samples = frame_geometry(config, sample_rate)
    starts = frame_starts(int(samples.size), hop_samples)
    window_name = str(config.get("fft_window", "hann")).lower()
    if window_name == "hann":
        window = np.hanning(frame_samples).astype(np.float64)
    elif window_name in {"rect", "rectangular"}:
        window = np.ones(frame_samples, dtype=np.float64)
    else:
        raise ValueError(f"unsupported fft_window: {window_name}")
    energy_floor = float(config.get("energy_floor", 1e-9))
    flatness_epsilon = float(config.get("flatness_epsilon", 1e-12))
    hop_ms = float(hop_samples) * 1000.0 / float(sample_rate)
    sample_to_ms = 1000.0 / float(sample_rate)
    frame_ms = float(frame_samples) * 1000.0 / float(sample_rate)
    frames_out: list[dict[str, Any]] = []
    for offset, frames, valid in iter_frame_blocks(
        samples, starts, frame_samples, block_size=block_size
    ):
        as_float = frames.astype(np.float64, copy=False)
        rms = np.sqrt(np.mean(as_float * as_float, axis=1, dtype=np.float64))
        rms_log = np.log(np.maximum(rms, energy_floor))
        zcr = zcr_from_frames(frames)
        flatness = flatness_from_frames(frames, window, flatness_epsilon)
        rms_log = round_significant(rms_log, precision)
        zcr = round_significant(zcr, precision)
        flatness = round_significant(flatness, precision)
        for row, start in enumerate(starts[offset : offset + frames.shape[0]].tolist()):
            index = offset + row
            frames_out.append(
                {
                    "frame_index": index,
                    "start_ms": round(float(start) * sample_to_ms, 9),
                    "frame_ms": round(frame_ms, 9),
                    "is_last_frame": bool(int(start) + frame_samples > samples.size),
                    "valid_samples": int(valid[row]),
                    "rms_log": float(rms_log[row]),
                    "zcr": float(zcr[row]),
                    "flatness": float(flatness[row]),
                }
            )
    return {
        "frame_count": len(frames_out),
        "duration_ms": round(float(samples.size) * 1000.0 / sample_rate, 9),
        "frame_samples": frame_samples,
        "hop_samples": hop_samples,
        "frame_ms": round(frame_ms, 9),
        "hop_ms": round(hop_ms, 9),
        "significant_digits": precision,
        "frames": frames_out,
    }


def feature_statistics(features: dict[str, Any], digits: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    frames: Iterable[dict[str, Any]] = features["frames"]
    for key in ("rms_log", "zcr", "flatness"):
        values = np.asarray([float(frame[key]) for frame in frames], dtype=np.float64)
        if values.size:
            result[key] = {
                "mean": round_float(float(np.mean(values)), digits),
                "std": round_float(float(np.std(values)), digits),
                "min": round_float(float(np.min(values)), digits),
                "max": round_float(float(np.max(values)), digits),
            }
        else:
            result[key] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return result


def frame_rms_values(samples: np.ndarray, config: dict[str, Any], sample_rate: int) -> np.ndarray:
    """Lightweight energy-only scan used before strategy selection."""

    frame_samples, hop_samples = frame_geometry(config, sample_rate)
    starts = frame_starts(int(samples.size), hop_samples)
    energy_floor = float(config.get("energy_floor", 1e-9))
    output = np.empty(starts.size, dtype=np.float64)
    for offset, frames, _valid in iter_frame_blocks(samples, starts, frame_samples):
        as_float = frames.astype(np.float64, copy=False)
        rms = np.sqrt(np.mean(as_float * as_float, axis=1, dtype=np.float64))
        output[offset : offset + frames.shape[0]] = np.log(
            np.maximum(rms, energy_floor)
        )
    return output
