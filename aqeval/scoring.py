"""Frame features, deterministic dimension bands and fixed-point aggregation."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Iterable

import numpy as np

from .config import DIMENSIONS, Config
from .levels import LevelMap
from .models import FrameSummary
from .util import MICRO_SCALE, micro_to_permille, round_half_up_fraction, sha256_bytes


SCORER_VERSION = "dimension-bands-v1"
LEVEL_TARGET_DBFS = -23.0

LEVEL_BANDS = (
    (3.0, 1000),
    (6.0, 850),
    (10.0, 650),
    (18.0, 350),
    (math.inf, 100),
)
NOISE_BANDS = (
    (-60.0, 1000),
    (-50.0, 900),
    (-40.0, 700),
    (-30.0, 450),
    (-20.0, 200),
    (math.inf, 50),
)
CLIPPING_BANDS = (
    (0.0, 1000),
    (0.001, 900),
    (0.005, 700),
    (0.02, 400),
    (0.05, 150),
    (math.inf, 0),
)
CONTINUITY_BANDS = (
    (0.01, 1000),
    (0.02, 900),
    (0.05, 700),
    (0.10, 450),
    (0.25, 200),
    (math.inf, 0),
)


def _max_zero_run(samples: np.ndarray) -> int:
    nonzero = np.flatnonzero(samples != 0)
    if nonzero.size == 0:
        return int(samples.size)
    longest = int(nonzero[0])
    if nonzero.size > 1:
        longest = max(longest, int(np.max(np.diff(nonzero) - 1)))
    longest = max(longest, int(samples.size - 1 - nonzero[-1]))
    return max(0, longest)


def compute_frame_features(
    file_id: str,
    frame_index: int,
    samples: np.ndarray,
    sample_rate: int,
    clipping_amplitude_threshold: float,
) -> dict[str, Any]:
    if samples.size == 0:
        raise ValueError("frame must not be empty")
    float_samples = samples.astype(np.float64) / 32768.0
    mean_square = float(np.mean(float_samples * float_samples))
    rms = math.sqrt(max(0.0, mean_square))
    rms_db = 20.0 * math.log10(max(rms, 1e-12))
    peak = float(np.max(np.abs(float_samples)))
    if samples.size <= 1:
        zcr = 0.0
    else:
        signs = np.signbit(float_samples)
        zcr = float(np.count_nonzero(signs[1:] != signs[:-1])) / float(samples.size - 1)
    zero_run = _max_zero_run(samples)
    clip_level = int(round(clipping_amplitude_threshold * 32768.0))
    clipped_samples = int(np.count_nonzero(np.abs(samples.astype(np.int32)) >= clip_level))
    return {
        "file_id": file_id,
        "frame_index": frame_index,
        "rms_db": rms_db,
        "rms_linear": rms,
        "zcr": zcr,
        "peak": peak,
        "max_zero_run": zero_run,
        "zero_run_ratio": zero_run / float(samples.size),
        "clipped_samples": clipped_samples,
        "clipped_fraction": clipped_samples / float(samples.size),
        "frame_fingerprint": sha256_bytes(samples.tobytes()),
        "task_key": frame_task_key(file_id, frame_index),
    }


def frame_task_key(file_id: str, frame_index: int) -> str:
    return f"{file_id}|frame|{frame_index:06d}"


def summarize_frames(
    file_id: str,
    frames: list[dict[str, Any]],
    config: Config,
) -> dict[str, Any]:
    ordered = sorted(frames, key=lambda item: item["frame_index"])
    if not ordered:
        raise ValueError("cannot summarize an empty frame set")
    rms_values = [float(item["rms_db"]) for item in ordered]
    active = [value for value in rms_values if value > config.silence_threshold_dbfs]
    level_metric = float(np.median(active)) if active else config.silence_threshold_dbfs
    sorted_db = sorted(rms_values)
    low_count = max(1, int(math.ceil(len(sorted_db) * config.noise_floor_percentile / 100.0)))
    cutoff = sorted_db[low_count - 1]
    low_frames = [item for item in ordered if float(item["rms_db"]) <= cutoff]
    low_power = sum(float(item["rms_linear"]) ** 2 for item in low_frames) / len(low_frames)
    noise_floor_db = 10.0 * math.log10(max(low_power, 1e-24))
    frame_length = config.frame_length_samples
    clipped_total = sum(int(item["clipped_samples"]) for item in ordered)
    clipping_fraction = clipped_total / float(max(1, len(ordered) * frame_length))
    max_zero_run = max(int(item["max_zero_run"]) for item in ordered)
    zero_run_threshold = int(config.consecutive_zero_samples)
    effective_zero_run = max_zero_run if max_zero_run >= zero_run_threshold else 0
    max_frame_length = max(1, frame_length)
    continuity_metric = effective_zero_run / float(max_frame_length)
    dropout_frame_count = sum(
        1 for item in ordered if int(item["max_zero_run"]) >= zero_run_threshold
    )
    zcr_mean = sum(float(item["zcr"]) for item in ordered) / len(ordered)
    peak_max = max(float(item["peak"]) for item in ordered)
    return {
        "file_id": file_id,
        "frame_count": len(ordered),
        "frame_indexes": [int(item["frame_index"]) for item in ordered],
        "level_metric_dbfs": level_metric,
        "level_deviation_db": abs(level_metric - LEVEL_TARGET_DBFS),
        "noise_floor_dbfs": noise_floor_db,
        "noise_floor_frame_count": len(low_frames),
        "noise_floor_cutoff_dbfs": cutoff,
        "clipping_fraction": clipping_fraction,
        "max_zero_run": max_zero_run,
        "zero_run_threshold": zero_run_threshold,
        "effective_zero_run": effective_zero_run,
        "dropout_frame_count": dropout_frame_count,
        "dropout_frame_fraction": dropout_frame_count / float(len(ordered)),
        "continuity_metric": continuity_metric,
        "zcr_mean": zcr_mean,
        "peak_max": peak_max,
        "active_frame_count": len(active),
        "rms_median_dbfs": float(np.median(rms_values)),
        "frames": ordered,
    }


def _band_index_and_score(metric: float, bands: tuple[tuple[float, int], ...]) -> tuple[int, int]:
    for index, (upper, score) in enumerate(bands):
        if metric <= upper:
            return index, score
    return len(bands) - 1, bands[-1][1]


def _boundary_if_exact(
    metric_name: str,
    metric: float,
    bands: tuple[tuple[float, int], ...],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for index, (upper, score) in enumerate(bands):
        if math.isfinite(upper) and metric == upper:
            events.append(
                {
                    "kind": "raw_band_threshold",
                    "metric": metric_name,
                    "value": metric,
                    "threshold": upper,
                    "upper_score": score,
                    "next_score": bands[index + 1][1] if index + 1 < len(bands) else score,
                    "rule": "metric <= threshold uses the upper score bucket",
                }
            )
    return events


def _quantize_score_micro(raw_score: int, levels: tuple[int, ...]) -> tuple[int, bool, int]:
    candidates = [(abs(raw_score - level * 1000), level) for level in levels]
    _, selected = min(candidates)
    return selected * 1000, raw_score == selected * 1000, selected


def score_dimensions(
    file_id: str,
    summary: dict[str, Any],
    config: Config,
) -> dict[str, Any]:
    metrics = {
        "level": float(summary["level_deviation_db"]),
        "noise": float(summary["noise_floor_dbfs"]),
        "clipping": float(summary["clipping_fraction"]),
        "continuity": float(summary["continuity_metric"]),
    }
    bands = {
        "level": LEVEL_BANDS,
        "noise": NOISE_BANDS,
        "clipping": CLIPPING_BANDS,
        "continuity": CONTINUITY_BANDS,
    }
    raw_scores: dict[str, int] = {}
    dimensions_micro: dict[str, int] = {}
    boundaries: list[dict[str, Any]] = []
    quantized_levels: dict[str, int] = {}
    for name in DIMENSIONS:
        _, raw = _band_index_and_score(metrics[name], bands[name])
        raw_scores[name] = raw
        events = _boundary_if_exact(name, metrics[name], bands[name])
        for event in events:
            boundaries.append({"file_id": file_id, "dimension": name, **event})
        quantized_micro, exact, selected_level = _quantize_score_micro(raw * 1000, config.quantization_levels)
        dimensions_micro[name] = quantized_micro
        quantized_levels[name] = selected_level
        if exact:
            boundaries.append(
                {
                    "file_id": file_id,
                    "dimension": name,
                    "kind": "quantization_threshold",
                    "value": raw,
                    "threshold": selected_level,
                    "rule": "raw score equals a configured quantization tier",
                }
            )
    combined = combine_fixed_total(dimensions_micro, config)
    boundaries.sort(
        key=lambda item: (
            item["file_id"],
            DIMENSIONS.index(item["dimension"]),
            item["kind"],
            float(item.get("threshold", 0.0)),
        )
    )
    return {
        "file_id": file_id,
        "dimensions_micro": dimensions_micro,
        "dimensions_permille": {name: level for name, level in quantized_levels.items()},
        "raw_scores_permille": raw_scores,
        "metrics": metrics,
        "weighted_contributions_micro": combined["weighted_contributions_micro"],
        "total_micro": combined["total_micro"],
        "total_permille": combined["total_permille"],
        "boundaries": boundaries,
        "scorer_version": SCORER_VERSION,
    }


def combine_fixed_total(
    dimensions_micro: dict[str, int],
    config: Config,
) -> dict[str, Any]:
    weights = config.fraction_weights
    weight_sum = sum(weights.values(), Fraction(0))
    weighted = sum(
        (Fraction(dimensions_micro[name]) * weights[name] for name in DIMENSIONS),
        Fraction(0),
    ) / weight_sum
    total_micro = round_half_up_fraction(weighted)
    total_micro = max(0, min(MICRO_SCALE, total_micro))
    return {
        "total_micro": total_micro,
        "total_permille": micro_to_permille(total_micro),
        "weighted_contributions_micro": {
            name: round_half_up_fraction(
                Fraction(dimensions_micro[name]) * weights[name] / weight_sum
            )
            for name in DIMENSIONS
        },
    }


def grade_total(total_permille: int, level_map: LevelMap) -> tuple[str, str, bool]:
    entry = level_map.level_for_permille(total_permille)
    at_threshold = total_permille == entry.min_total_permille
    return entry.code, entry.name, at_threshold


def grade_total_micro(total_micro: int, level_map: LevelMap) -> tuple[str, str, bool]:
    entry = level_map.level_for_micro(total_micro)
    at_threshold = total_micro == entry.min_total_permille * 1000
    return entry.code, entry.name, at_threshold


def frame_summaries_from_features(frames: Iterable[dict[str, Any]]) -> list[FrameSummary]:
    return [
        FrameSummary(
            frame_index=int(item["frame_index"]),
            rms_db=float(item["rms_db"]),
            zcr=float(item["zcr"]),
            peak=float(item["peak"]),
            max_zero_run=int(item["max_zero_run"]),
            zero_run_ratio=float(item["zero_run_ratio"]),
            clipped_samples=int(item["clipped_samples"]),
            frame_fingerprint=str(item["frame_fingerprint"]),
        )
        for item in sorted(frames, key=lambda value: value["frame_index"])
    ]


def frame_set_digest(frames: Iterable[dict[str, Any]]) -> str:
    ordered = sorted(frames, key=lambda item: item["frame_index"])
    return sha256_bytes(
        ("\n".join(str(item["frame_fingerprint"]) for item in ordered)).encode("ascii")
    )
