from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from .errors import SyncError


def _segments(count: int, size: int) -> list[tuple[int, int]]:
    if count == 0:
        return []
    size = max(3, int(size))
    result: list[tuple[int, int]] = []
    start = 0
    while start < count:
        stop = min(count, start + size)
        if count - stop in (1, 2):
            stop = count
        result.append((start, stop))
        start = stop
    return result


def _linear_fit(x: np.ndarray, y: np.ndarray, confidence: float) -> dict[str, float]:
    n = int(x.size)
    if n < 2 or float(np.ptp(x)) <= 0.0:
        raise SyncError("E_PULSE_MISSING", "insufficient distinct pulse anchors for linear fit", {"anchors": n})
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    centered_x = x - x_mean
    denominator = float(np.dot(centered_x, centered_x))
    slope = float(np.dot(centered_x, y - y_mean) / denominator)
    intercept = y_mean
    predicted = intercept + slope * centered_x
    residual = y - predicted
    residual_rmse = float(np.sqrt(np.mean(np.square(residual))))
    residual_max = float(np.max(np.abs(residual)))
    if n > 2:
        residual_sigma = float(np.sqrt(np.sum(np.square(residual)) / (n - 2)))
        t_critical = float(student_t.ppf(0.5 + confidence / 2.0, n - 2))
        slope_se = residual_sigma / math.sqrt(denominator)
        center_se = residual_sigma / math.sqrt(n)
    else:
        residual_sigma = residual_rmse
        t_critical = float(student_t.ppf(0.5 + confidence / 2.0, 1))
        slope_se = residual_sigma / math.sqrt(denominator)
        center_se = residual_sigma / math.sqrt(n)
    return {
        "n": float(n),
        "x_mean": x_mean,
        "offset_at_center": intercept,
        "slope": slope,
        "slope_ppm": slope * 1.0e6,
        "slope_ppm_ci95_low": (slope - t_critical * slope_se) * 1.0e6,
        "slope_ppm_ci95_high": (slope + t_critical * slope_se) * 1.0e6,
        "offset_center_ci95_samples": t_critical * center_se,
        "residual_rmse_samples": residual_rmse,
        "residual_max_samples": residual_max,
        "residual_sigma_samples": residual_sigma,
    }


class DriftCurve:
    def __init__(self, anchors_x: np.ndarray, anchors_y: np.ndarray) -> None:
        order = np.argsort(anchors_x)
        self.x = np.asarray(anchors_x, dtype=np.float64)[order]
        self.y = np.asarray(anchors_y, dtype=np.float64)[order]
        if self.x.size == 0:
            raise SyncError("E_PULSE_MISSING", "cannot construct empty drift curve")
        if self.x.size > 1:
            keep = np.concatenate(([True], np.diff(self.x) > 0.0))
            self.x = self.x[keep]
            self.y = self.y[keep]

    def __call__(self, samples: np.ndarray | float) -> np.ndarray:
        values = np.asarray(samples, dtype=np.float64)
        if self.x.size == 1:
            return np.full_like(values, self.y[0], dtype=np.float64)
        return np.interp(values, self.x, self.y)

    def to_json(self) -> dict[str, Any]:
        return {"x_samples": self.x.tolist(), "offset_samples": self.y.tolist(), "interpolation": "linear"}


def _smooth_anchor_predictions(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    count = x.size
    result = np.empty(count, dtype=np.float64)
    half = max(1, window // 2)
    for index in range(count):
        start = max(0, index - half)
        stop = min(count, index + half + 1)
        if stop - start < 4:
            result[index] = y[index]
            continue
        fit = _linear_fit(x[start:stop], y[start:stop], 0.95)
        result[index] = fit["offset_at_center"] + fit["slope"] * (x[index] - fit["x_mean"])
    return result


def _build_curve(
    reference_samples: np.ndarray,
    offsets: np.ndarray,
    fixed_delay_samples: float,
    selected_strategy: str,
    config: dict[str, Any],
    duration_samples: int,
) -> tuple[DriftCurve, list[int]]:
    if selected_strategy == "full_exact":
        predicted = _smooth_anchor_predictions(reference_samples, offsets, int(config["fit"]["segment_pulses"]))
        anchors_x = reference_samples
        anchors_y = predicted
        anchor_indices = list(range(reference_samples.size))
    elif selected_strategy == "block_accumulation":
        block_pulses = max(3, int(config["strategy"]["block_pulses"]))
        anchors_x_list: list[float] = [0.0]
        anchors_y_list: list[float] = [fixed_delay_samples]
        anchor_indices = []
        for start in range(0, reference_samples.size, block_pulses):
            stop = min(reference_samples.size, start + block_pulses)
            if stop - start < 3:
                start = max(0, stop - 3)
            fit = _linear_fit(reference_samples[start:stop], offsets[start:stop], float(config["fit"]["confidence"]))
            center = float(reference_samples[start:stop].mean())
            anchors_x_list.append(center)
            anchors_y_list.append(fit["offset_at_center"])
            anchor_indices.append(start + (stop - start) // 2)
        anchors_x_list.append(float(reference_samples[-1]))
        last_fit = _linear_fit(reference_samples[max(0, reference_samples.size - block_pulses):], offsets[max(0, offsets.size - block_pulses):], float(config["fit"]["confidence"]))
        anchors_y_list.append(last_fit["offset_at_center"] + last_fit["slope"] * (float(reference_samples[-1]) - last_fit["x_mean"]))
        anchors_x = np.asarray(anchors_x_list, dtype=np.float64)
        anchors_y = np.asarray(anchors_y_list, dtype=np.float64)
    elif selected_strategy == "sparse_anchor_interpolation":
        stride = max(1, int(config["strategy"]["sparse_stride"]))
        indices = list(range(0, reference_samples.size, stride))
        if indices[-1] != reference_samples.size - 1:
            indices.append(reference_samples.size - 1)
        anchor_indices = indices
        anchors_x = np.concatenate(([0.0], reference_samples[np.asarray(indices)]))
        anchors_y = np.concatenate(([fixed_delay_samples], offsets[np.asarray(indices)]))
    else:
        raise SyncError("E_STRATEGY_BUDGET", "unsupported strategy for drift curve", {"strategy": selected_strategy})
    return DriftCurve(anchors_x, anchors_y), anchor_indices


def estimate_drift(
    paired: dict[str, Any],
    manifest: dict[str, Any],
    config: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    sample_rate = int(config["sample_rate_hz"])
    reference = np.asarray(paired["truth_reference_samples"], dtype=np.float64)
    tolerance_ppm = float(config["drift_tolerance_ppm"])
    confidence = float(config["fit"]["confidence"])
    segment_size = int(config["fit"]["segment_pulses"])
    selected = str(decision["selected"])
    result_channels: list[dict[str, Any]] = []
    curves: dict[int, DriftCurve] = {}
    all_segments: list[dict[str, Any]] = []
    for channel_data in manifest["channels"]:
        channel = int(channel_data["index"])
        pair = paired["channels"][channel]
        observed = np.asarray(pair["paired_detected"], dtype=np.float64)
        offsets = observed - reference
        if channel == int(config["reference_channel"]):
            curve = DriftCurve(np.asarray([0.0, float(manifest["frames"])]), np.asarray([0.0, 0.0]))
            curves[channel] = curve
            result_channels.append(
                {
                    "channel": channel,
                    "role": "reference",
                    "fixed_delay_samples": 0.0,
                    "fixed_delay_ci95_samples": 0.0,
                    "cumulative_drift_end_samples": 0.0,
                    "max_abs_rate_ppm": 0.0,
                    "segments": [],
                    "curve": curve.to_json(),
                    "detector_bias_samples": float(paired["reference_detector_bias_samples"]),
                    "extra_detected_count": int(pair["extra_detected_count"]),
                }
            )
            continue
        segments: list[dict[str, Any]] = []
        global_predicted = np.zeros(reference.size, dtype=np.float64)
        for start, stop in _segments(reference.size, segment_size):
            fit = _linear_fit(reference[start:stop], offsets[start:stop], confidence)
            predicted = fit["offset_at_center"] + fit["slope"] * (reference[start:stop] - fit["x_mean"])
            global_predicted[start:stop] = predicted
            segment_start = float(reference[start])
            segment_stop = float(reference[stop - 1])
            rate_ppm = fit["slope_ppm"]
            if abs(rate_ppm) > tolerance_ppm:
                raise SyncError(
                    "E_DRIFT_EXCEED",
                    "estimated drift rate exceeds declared tolerance",
                    {
                        "channel": channel,
                        "segment": len(segments),
                        "reference_start_sample": segment_start,
                        "reference_stop_sample": segment_stop,
                        "estimated_rate_ppm": rate_ppm,
                        "tolerance_ppm": tolerance_ppm,
                        "slope_ppm_ci95": [fit["slope_ppm_ci95_low"], fit["slope_ppm_ci95_high"]],
                        "residual_rmse_samples": fit["residual_rmse_samples"],
                    },
                )
            segment = {
                "segment": len(segments),
                "reference_start_sample": segment_start,
                "reference_stop_sample": segment_stop,
                "reference_start_s": segment_start / sample_rate,
                "reference_stop_s": segment_stop / sample_rate,
                "pulse_count": int(stop - start),
                "rate_ppm": rate_ppm,
                "rate_ppm_ci95_low": fit["slope_ppm_ci95_low"],
                "rate_ppm_ci95_high": fit["slope_ppm_ci95_high"],
                "offset_at_center_samples": fit["offset_at_center"],
                "offset_center_ci95_samples": fit["offset_center_ci95_samples"],
                "residual_rmse_samples": fit["residual_rmse_samples"],
                "residual_max_samples": fit["residual_max_samples"],
                "cumulative_drift_start_samples": fit["offset_at_center"] + fit["slope"] * (segment_start - fit["x_mean"]),
                "cumulative_drift_stop_samples": fit["offset_at_center"] + fit["slope"] * (segment_stop - fit["x_mean"]),
            }
            segments.append(segment)
            all_segments.append({"channel": channel, **segment})
        first_fit = _linear_fit(reference, offsets, confidence)
        early_stop = min(reference.size, max(4, segment_size))
        early_fit = _linear_fit(reference[:early_stop], offsets[:early_stop], confidence)
        fixed_delay = early_fit["offset_at_center"] + early_fit["slope"] * (0.0 - early_fit["x_mean"])
        fixed_delay_ci = early_fit["offset_center_ci95_samples"] + abs(early_fit["slope"]) * abs(early_fit["x_mean"])
        curve, anchor_indices = _build_curve(reference, offsets, fixed_delay, selected, config, int(manifest["frames"]))
        curves[channel] = curve
        cumulative_end = float(curve(float(manifest["frames"])) - fixed_delay)
        max_abs_rate = max(abs(float(segment["rate_ppm"])) for segment in segments)
        result_channels.append(
            {
                "channel": channel,
                "role": "slave",
                "fixed_delay_samples": -fixed_delay,
                "fixed_delay_observed_offset_samples": fixed_delay,
                "fixed_delay_ci95_samples": fixed_delay_ci,
                "global_linear_rate_ppm": first_fit["slope_ppm"],
                "cumulative_drift_end_samples": cumulative_end,
                "max_abs_rate_ppm": max_abs_rate,
                "segments": segments,
                "curve": curve.to_json(),
                "curve_anchor_indices": anchor_indices,
                "fit_residual_rmse_samples": float(np.sqrt(np.mean(np.square(offsets - global_predicted)))),
                "detector_bias_samples": float(paired["reference_detector_bias_samples"]),
                "extra_detected_count": int(pair["extra_detected_count"]),
            }
        )
    return {
        "schema_version": 1,
        "selected_strategy": selected,
        "sample_rate_hz": sample_rate,
        "tolerance_ppm": tolerance_ppm,
        "confidence": confidence,
        "channels": result_channels,
        "curves_by_channel": curves,
        "segments": all_segments,
    }


def drift_estimate_report(estimate: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    channels = []
    for channel in estimate["channels"]:
        channels.append({key: value for key, value in channel.items() if key != "curve_anchor_indices"})
    return {
        "schema_version": 1,
        "selected_strategy": estimate["selected_strategy"],
        "decision": decision,
        "sample_rate_hz": estimate["sample_rate_hz"],
        "tolerance_ppm": estimate["tolerance_ppm"],
        "confidence": estimate["confidence"],
        "channels": channels,
    }
