from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .errors import SyncError, require
from .wavio import decode_pcm24, validate_pcm24_wav


def _tone_basis(times: np.ndarray, tones_hz: list[float]) -> np.ndarray:
    columns = [np.ones(times.size, dtype=np.float64)]
    for frequency in tones_hz:
        angle = 2.0 * np.pi * frequency * times
        columns.append(np.sin(angle))
        columns.append(np.cos(angle))
    return np.column_stack(columns)


def _precise_pulse_center(
    values: np.ndarray,
    approximate_index: int,
    sample_rate_hz: int,
    sigma_samples: float,
    tones_hz: list[float],
    threshold: float,
    gate_radius: int | None,
    context: dict[str, Any],
) -> float:
    if gate_radius is None:
        gate_start = 0
        gate_stop = values.size
    else:
        gate_start = max(0, approximate_index - gate_radius)
        gate_stop = min(values.size, approximate_index + gate_radius + 1)
    coarse = gate_start + int(np.argmax(values[gate_start:gate_stop]))
    if float(values[coarse]) < threshold:
        raise SyncError(
            "E_PULSE_MISSING",
            "pulse transient is below the detection threshold",
            {
                **context,
                "approximate_sample": float(approximate_index),
                "window_peak": float(values[coarse]),
                "threshold": threshold,
            },
        )
    radius = int(np.ceil(7.0 * sigma_samples))
    start = max(0, coarse - radius)
    stop = min(values.size, coarse + radius + 1)
    local = values[start:stop].astype(np.float64, copy=False)
    indices = np.arange(start, stop, dtype=np.float64)
    times = indices / float(sample_rate_hz)
    distance = np.abs(indices - coarse)
    centered = (distance >= 3.2 * sigma_samples) & (distance <= 7.0 * sigma_samples)
    if int(centered.sum()) < len(tones_hz) * 2 + 4:
        raise SyncError("E_PULSE_MISSING", "pulse window has too few baseline samples", context)
    basis = _tone_basis(times, tones_hz)
    coefficients, _, _, _ = np.linalg.lstsq(basis[centered], local[centered], rcond=None)
    residual = local - basis @ coefficients
    centroid_mask = np.abs(indices - coarse) <= 4.0 * sigma_samples
    energy = np.square(residual[centroid_mask])
    total = float(energy.sum())
    if total <= 0.0:
        raise SyncError("E_PULSE_MISSING", "pulse residual energy is zero", context)
    center = float(np.dot(energy, indices[centroid_mask]) / total)
    return center


def detect_window_center(
    values: np.ndarray,
    approximate_index: int,
    config: dict[str, Any],
    context: dict[str, Any],
) -> float:
    sample_rate = int(config["sample_rate_hz"])
    sigma = max(1.0, float(config["signal"]["pulse_sigma_s"]) * sample_rate)
    tones = [float(value) for value in config["signal"]["tones_hz"]]
    return _precise_pulse_center(
        values,
        approximate_index,
        sample_rate,
        sigma,
        tones,
        float(config["detector"]["peak_threshold"]),
        None,
        context,
    )


def detect_truth_gated_pulses(
    wav_path: Path,
    manifest: dict[str, Any],
    config: dict[str, Any],
    frames: int,
) -> dict[int, np.ndarray]:
    import wave

    channel_count = int(config["channel_count"])
    sample_rate = int(config["sample_rate_hz"])
    validate_pcm24_wav(wav_path, channel_count, sample_rate, frames)
    detector = config["detector"]
    tolerance = int(round(float(detector["pair_tolerance_s"]) * sample_rate))
    threshold = float(detector["peak_threshold"])
    tracking_gate = max(1, int(round(float(detector.get("tracking_gate_s", 0.15)) * sample_rate)))
    sigma = max(1.0, float(config["signal"]["pulse_sigma_s"]) * sample_rate)
    tones = [float(value) for value in config["signal"]["tones_hz"]]
    truth = [int(value) for value in manifest["reference_pulse_samples"]]
    positions: dict[int, list[float]] = {channel: [] for channel in range(channel_count)}
    last_offset: list[float | None] = [None for _ in range(channel_count)]
    previous_offset: list[float | None] = [None for _ in range(channel_count)]
    with wave.open(str(wav_path), "rb") as wav:
        for pulse_index, reference_sample in enumerate(truth):
            start = max(0, reference_sample - tolerance)
            stop = min(frames, reference_sample + tolerance + 1)
            wav.setpos(start)
            samples = decode_pcm24(wav.readframes(stop - start), channel_count)
            approximate = reference_sample - start
            for channel in range(channel_count):
                if last_offset[channel] is None:
                    predicted_offset = 0.0
                    gate_radius = tolerance
                elif previous_offset[channel] is None:
                    predicted_offset = float(last_offset[channel])
                    gate_radius = tracking_gate
                else:
                    predicted_offset = float(last_offset[channel]) + (float(last_offset[channel]) - float(previous_offset[channel]))
                    gate_radius = tracking_gate
                center = _precise_pulse_center(
                    samples[:, channel],
                    approximate + int(round(predicted_offset)),
                    sample_rate,
                    sigma,
                    tones,
                    threshold,
                    gate_radius,
                    {
                        "channel": channel,
                        "pulse_index": pulse_index,
                        "reference_sample": reference_sample,
                    },
                )
                absolute_center = start + center
                positions[channel].append(absolute_center)
                previous_offset[channel] = last_offset[channel]
                last_offset[channel] = absolute_center - reference_sample
    return {channel: np.asarray(values, dtype=np.float64) for channel, values in positions.items()}


def pair_pulses(
    detected: dict[int, np.ndarray],
    truth_reference_samples: list[int],
    detector: dict[str, Any],
    sample_rate_hz: int,
    reference_channel: int = 0,
) -> dict[str, Any]:
    truth = np.asarray(truth_reference_samples, dtype=np.float64)
    require(truth.size > 0, "E_PULSE_MISSING", "manifest contains no pulse truth")
    tolerance = float(detector["pair_tolerance_s"]) * sample_rate_hz
    channels: dict[int, dict[str, Any]] = {}
    for channel, positions in sorted(detected.items()):
        positions = np.asarray(positions, dtype=np.float64)
        if positions.size == 0:
            raise SyncError(
                "E_PULSE_MISSING",
                "no pulses detected in channel",
                {"channel": channel, "truth_pulses": int(truth.size), "detected_pulses": 0},
            )
        right = np.searchsorted(positions, truth)
        left = np.clip(right - 1, 0, positions.size - 1)
        right = np.clip(right, 0, positions.size - 1)
        use_left = np.abs(positions[left] - truth) <= np.abs(positions[right] - truth)
        nearest_index = np.where(use_left, left, right)
        nearest = positions[nearest_index]
        error = nearest - truth
        valid = np.abs(error) <= tolerance
        if not bool(valid.all()):
            missing_indices = np.flatnonzero(~valid)
            details = []
            for index in missing_indices[:32]:
                details.append(
                    {
                        "truth_index": int(index),
                        "truth_sample": float(truth[index]),
                        "nearest_detected_sample": float(nearest[index]),
                        "distance_samples": float(error[index]),
                    }
                )
            raise SyncError(
                "E_PULSE_MISSING",
                "one or more reference pulses could not be paired within tolerance",
                {
                    "channel": channel,
                    "missing_count": int(missing_indices.size),
                    "truth_count": int(truth.size),
                    "detected_count": int(positions.size),
                    "tolerance_samples": tolerance,
                    "missing": details,
                },
            )
        used = nearest_index[valid]
        if np.unique(used).size != used.size:
            raise SyncError(
                "E_PULSE_MISSING",
                "detected pulses are duplicated across reference truth positions",
                {"channel": channel, "truth_count": int(truth.size), "detected_count": int(positions.size)},
            )
        used_mask = np.zeros(positions.size, dtype=bool)
        used_mask[used] = True
        channels[channel] = {
            "detected_positions": positions.tolist(),
            "paired_detected": nearest.tolist(),
            "reference_positions": truth.tolist(),
            "offset_samples": error.tolist(),
            "extra_detected_count": int((~used_mask).sum()),
        }
    require(reference_channel in channels, "E_PULSE_MISSING", "reference channel was not detected", channel=reference_channel)
    reference_bias = float(np.median(np.asarray(channels[reference_channel]["offset_samples"], dtype=np.float64)))
    for payload in channels.values():
        payload["detector_bias_samples"] = reference_bias
    return {
        "truth_reference_samples": truth.tolist(),
        "reference_detector_bias_samples": reference_bias,
        "channels": channels,
    }


def detect_and_pair(
    wav_path: Path,
    manifest: dict[str, Any],
    config: dict[str, Any],
    frames: int | None = None,
) -> dict[str, Any]:
    expected_frames = int(manifest["frames"]) if frames is None else int(frames)
    detected = detect_truth_gated_pulses(
        wav_path,
        manifest,
        config,
        expected_frames,
    )
    return pair_pulses(
        detected,
        manifest["reference_pulse_samples"],
        config["detector"],
        int(config["sample_rate_hz"]),
        int(config["reference_channel"]),
    )
