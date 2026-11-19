from __future__ import annotations

import math
from typing import Any

import numpy as np

from .errors import require


def _truth_offset_map(manifest: dict[str, Any], channel: int) -> np.ndarray:
    reference = np.asarray(manifest["reference_pulse_samples"], dtype=np.float64)
    observed = np.asarray(manifest["channels"][channel]["expected_observed_pulse_samples"], dtype=np.float64)
    return observed - reference


def verify_alignment(
    original_pairs: dict[str, Any],
    aligned_pairs: dict[str, Any],
    manifest: dict[str, Any],
    estimate: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    channels: list[dict[str, Any]] = []
    for channel_estimate in estimate["channels"]:
        channel = int(channel_estimate["channel"])
        reference = np.asarray(manifest["reference_pulse_samples"], dtype=np.float64)
        before = np.asarray(original_pairs["channels"][channel]["paired_detected"], dtype=np.float64) - reference
        after = np.asarray(aligned_pairs["channels"][channel]["paired_detected"], dtype=np.float64) - reference
        require(before.size == after.size == reference.size, "E_PULSE_MISSING", "pulse counts differ during verification", channel=channel)
        if channel == int(config["reference_channel"]):
            truth_delay = 0.0
            truth_offset_at_zero = 0.0
            truth_offsets = np.zeros_like(reference)
            fixed_error = 0.0
            curve = estimate["curves_by_channel"][channel]
            estimated_delay = 0.0
            observed_offset_at_zero = 0.0
        else:
            channel_manifest = manifest["channels"][channel]
            truth_delay = float(channel_manifest["clock"]["delay_samples"])
            truth_offset_at_zero = -truth_delay
            truth_offsets = _truth_offset_map(manifest, channel)
            detector_bias = float(original_pairs["reference_detector_bias_samples"])
            observed_offset_at_zero = float(channel_estimate["fixed_delay_observed_offset_samples"])
            estimated_delay = -observed_offset_at_zero + detector_bias
            fixed_error = float(estimated_delay - truth_delay)
            curve = estimate["curves_by_channel"][channel]
        drift_error = (curve(reference) - observed_offset_at_zero) - (truth_offsets - truth_offset_at_zero)
        predicted_post = fixed_error + drift_error
        after_measurement_component = after - predicted_post
        details = []
        for index in range(reference.size):
            details.append(
                {
                    "pulse_index": index,
                    "reference_sample": float(reference[index]),
                    "reference_time_s": float(reference[index]) / int(config["sample_rate_hz"]),
                    "detected_before_sample": float(before[index] + reference[index]),
                    "residual_before_samples": float(before[index]),
                    "detected_aligned_sample": float(after[index] + reference[index]),
                    "residual_after_samples": float(after[index]),
                    "truth_fixed_delay_samples": truth_delay,
                    "estimated_fixed_delay_samples": estimated_delay,
                    "fixed_delay_error_samples": fixed_error,
                    "truth_cumulative_drift_samples": float(truth_offsets[index] - truth_offset_at_zero),
                    "estimated_cumulative_drift_samples": float(curve(reference[index]) - observed_offset_at_zero),
                    "accumulated_drift_error_samples": float(drift_error[index]),
                    "predicted_post_compensation_error_samples": float(predicted_post[index]),
                    "measurement_component_after_samples": float(after_measurement_component[index]),
                }
            )
        before_rmse = float(np.sqrt(np.mean(np.square(before))))
        after_rmse = float(np.sqrt(np.mean(np.square(after))))
        if channel == int(config["reference_channel"]):
            status = "REFERENCE"
        elif after_rmse < before_rmse:
            status = "IMPROVED"
        elif after_rmse > before_rmse:
            status = "DEGRADED"
        else:
            status = "UNCHANGED"
        channels.append(
            {
                "channel": channel,
                "role": channel_estimate["role"],
                "pulse_count": int(reference.size),
                "before_rmse_samples": before_rmse,
                "after_rmse_samples": after_rmse,
                "before_max_abs_samples": float(np.max(np.abs(before))),
                "after_max_abs_samples": float(np.max(np.abs(after))),
                "improvement_factor": float(before_rmse / after_rmse) if after_rmse > 0.0 else math.inf,
                "status": status,
                "truth_fixed_delay_samples": truth_delay,
                "estimated_fixed_delay_samples": estimated_delay,
                "fixed_delay_error_samples": fixed_error,
                "accumulated_drift_error_rmse_samples": float(np.sqrt(np.mean(np.square(drift_error)))),
                "predicted_post_error_rmse_samples": float(np.sqrt(np.mean(np.square(predicted_post)))),
                "measurement_component_rmse_samples": float(np.sqrt(np.mean(np.square(after_measurement_component)))),
                "pulses": details,
            }
        )
    slave_channels = [channel for channel in channels if channel["role"] == "slave"]
    summary = {
        "slave_channel_count": len(slave_channels),
        "before_rmse_samples_mean": float(np.mean([channel["before_rmse_samples"] for channel in slave_channels])) if slave_channels else 0.0,
        "after_rmse_samples_mean": float(np.mean([channel["after_rmse_samples"] for channel in slave_channels])) if slave_channels else 0.0,
        "after_rmse_samples_max": float(np.max([channel["after_rmse_samples"] for channel in slave_channels])) if slave_channels else 0.0,
        "fixed_delay_error_rmse_samples": float(np.sqrt(np.mean([channel["fixed_delay_error_samples"] ** 2 for channel in slave_channels]))) if slave_channels else 0.0,
        "accumulated_drift_error_rmse_samples": float(np.sqrt(np.mean([channel["accumulated_drift_error_rmse_samples"] ** 2 for channel in slave_channels]))) if slave_channels else 0.0,
        "all_channels_improved": all(channel["status"] == "IMPROVED" for channel in slave_channels),
        "status": "IMPROVED" if all(channel["status"] == "IMPROVED" for channel in slave_channels) else "DEGRADED_OR_NO_IMPROVEMENT",
        "accuracy_target_samples": float(config["accuracy_target_samples"]),
        "target_met": all(channel["after_rmse_samples"] <= float(config["accuracy_target_samples"]) for channel in slave_channels),
    }
    return {
        "schema_version": 1,
        "summary": summary,
        "channels": channels,
    }
