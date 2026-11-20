"""Baseline-derived, shared calibration for heterogeneous score types."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from .util import sha256_json

CALIBRATION_METHOD = "baseline_empirical_quantile_affine_v1"


def _softmax(row: Iterable[float]) -> list[float]:
    values = np.asarray(list(row), dtype=np.float64)
    maximum = float(np.max(values))
    exponentials = np.exp(values - maximum)
    total = float(np.sum(exponentials))
    if not math.isfinite(total) or total <= 0.0:
        return [0.0] * values.size
    return [float(value) for value in exponentials / total]


def _normalize_row(row: Iterable[float]) -> list[float]:
    values = np.asarray(list(row), dtype=np.float64)
    if values.size == 0:
        return []
    total = float(np.sum(values))
    if not math.isfinite(total) or total <= 0.0:
        return [0.0] * values.size
    return [float(value / total) for value in values]


def to_raw_probabilities(
    row: list[float],
    score_type: str,
) -> list[float]:
    """Convert logits with natural exponent; L1-normalize probability rows."""

    if score_type == "logit":
        return _softmax(row)
    if score_type == "probability":
        return _normalize_row(row)
    raise ValueError(f"unsupported score_type: {score_type}")


def derive_calibration(
    baseline_alignments: list[dict[str, Any]],
    config: dict[str, Any],
    baseline_align_keys: dict[str, str],
) -> tuple[dict[str, Any], str]:
    """Derive all calibration parameters from baseline shards and config only."""

    classes = list(config["classes"])
    quantile_low, quantile_high = (float(value) for value in config["calibration_quantiles"])
    collected: list[list[float]] = [[] for _ in classes]

    for packet in baseline_alignments:
        score_type = packet["source_score_type"]
        for frame in packet["frames"]:
            if frame.get("unmapped"):
                continue
            raw = frame["mapped_scores"]
            if score_type == "probability" and not any(abs(float(value)) > 0.0 for value in raw):
                continue
            normalized = to_raw_probabilities(raw, score_type)
            if not any(value > 0.0 for value in normalized):
                continue
            for class_index, value in enumerate(normalized):
                collected[class_index].append(float(value))

    per_class: dict[str, dict[str, Any]] = {}
    for class_index, class_name in enumerate(classes):
        values = np.asarray(collected[class_index], dtype=np.float64)
        if values.size:
            low = float(np.quantile(values, quantile_low, method="linear"))
            high = float(np.quantile(values, quantile_high, method="linear"))
        else:
            low, high = 0.0, 1.0
        if not math.isfinite(low) or not math.isfinite(high):
            low, high = 0.0, 1.0
        if high <= low:
            low = max(0.0, min(low, 1.0))
            high = min(1.0, max(high, low + 1e-12))
        per_class[class_name] = {
            "low": low,
            "high": high,
            "count": int(values.size),
        }

    params = {
        "method": CALIBRATION_METHOD,
        "logit_conversion": "natural_softmax",
        "probability_conversion": "row_l1_normalization",
        "quantile_low": quantile_low,
        "quantile_high": quantile_high,
        "classes": classes,
        "per_class": per_class,
        "tolerance": float(config["calibration_tolerance"]),
        "baseline_packet_ids": sorted(baseline_align_keys),
    }
    fingerprint = sha256_json(
        {
            "kind": "calibration",
            "params": params,
            "baseline_align_keys": {
                packet_id: baseline_align_keys[packet_id]
                for packet_id in sorted(baseline_align_keys)
            },
        }
    )
    return params, fingerprint


def calibrate_packet(
    alignment: dict[str, Any],
    params: dict[str, Any],
    calibration_fingerprint: str,
) -> dict[str, Any]:
    """Apply the one shared calibration map to every packet frame."""

    classes = params["classes"]
    low = np.asarray([params["per_class"][name]["low"] for name in classes], dtype=np.float64)
    high = np.asarray([params["per_class"][name]["high"] for name in classes], dtype=np.float64)
    widths = np.maximum(high - low, 1e-15)
    tolerance = float(params["tolerance"])
    score_type = alignment["source_score_type"]

    output_frames: list[dict[str, Any]] = []
    sum_deviations: list[float] = []
    conversion_residuals: list[float] = []
    for frame in alignment["frames"]:
        mapped_scores = list(frame["mapped_scores"])
        unmapped = bool(frame.get("unmapped"))
        empty_probability = score_type == "probability" and not any(
            abs(float(value)) > 0.0 for value in mapped_scores
        )
        if unmapped or empty_probability:
            pre_calibration = [0.0] * len(classes)
            calibrated = [0.0] * len(classes)
        else:
            pre_calibration = to_raw_probabilities(mapped_scores, score_type)
            clipped = np.clip(np.asarray(pre_calibration, dtype=np.float64), low, high)
            scaled = np.clip((clipped - low) / widths, 0.0, 1.0)
            total = float(np.sum(scaled))
            if total <= 1e-15 or not math.isfinite(total):
                calibrated = list(pre_calibration)
            else:
                calibrated = [float(value / total) for value in scaled]
        probability_sum = float(sum(calibrated))
        sum_deviation = abs(probability_sum - 1.0)
        conversion_residual = float(
            np.mean(np.abs(np.asarray(calibrated) - np.asarray(pre_calibration)))
        )
        sum_deviations.append(sum_deviation)
        conversion_residuals.append(conversion_residual)
        output_frames.append(
            {
                **{
                    key: frame[key]
                    for key in (
                        "global_frame_index",
                        "global_time_ms",
                        "original_frame_indices",
                        "original_frame_index_anchor",
                        "interpolation_fraction",
                        "resampled",
                        "edge_hold",
                        "unmapped",
                        "alignment_residual",
                        "mapping_residual",
                    )
                },
                "source_score_type": score_type,
                "pre_calibration_probability": pre_calibration,
                "calibrated_probability": calibrated,
                "probability_sum": probability_sum,
                "sum_deviation": sum_deviation,
                "conversion_residual": conversion_residual,
                "residual_violation": sum_deviation > tolerance,
            }
        )

    violation_frames = [
        frame["global_frame_index"]
        for frame in output_frames
        if frame["residual_violation"]
    ]
    return {
        **{key: alignment[key] for key in (
            "packet_id",
            "source_file",
            "source_score_type",
            "content_sha256",
            "audio_file",
            "start_ms",
            "end_ms",
            "frame_shift_ms",
            "frame_grid_version",
            "class_codebook_version",
            "detector_version",
            "mapping_entries",
            "mapping_by_name",
            "unmapped_frame_count",
            "unmapped_frames",
            "unmapped_row_indices",
            "original_frame_count",
            "target_frame_count",
            "resampled",
            "alignment_method",
            "uncovered_intervals_ms",
            "edge_hold_frame_count",
            "mean_alignment_residual",
            "max_alignment_residual",
            "injection_ids",
        )},
        "calibration_fingerprint": calibration_fingerprint,
        "calibration_method": params["method"],
        "frames": output_frames,
        "frame_count": len(output_frames),
        "residual_violation_count": len(violation_frames),
        "residual_violation_frames": violation_frames,
        "max_sum_deviation": max(sum_deviations, default=0.0),
        "mean_conversion_residual": (
            float(sum(conversion_residuals) / len(conversion_residuals))
            if conversion_residuals
            else 0.0
        ),
    }
