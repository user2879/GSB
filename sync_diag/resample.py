from __future__ import annotations

import hashlib
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np

from .errors import SyncError, require
from .estimate import DriftCurve
from .wavio import decode_pcm24, encode_pcm24, validate_pcm24_wav, wav_header_bytes, write_wav_header


def _phase_table(half_width: int, beta: float, phase_bins: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    taps = np.arange(-half_width + 1, half_width + 1, dtype=np.float64)
    phases = np.arange(phase_bins, dtype=np.float64) / float(phase_bins)
    delta = taps[None, :] - phases[:, None]
    window = np.kaiser(taps.size, beta)
    weights = np.sinc(delta) * window[None, :]
    weights /= weights.sum(axis=1, keepdims=True)
    return taps, weights


def _coordinates(
    curves: dict[int, DriftCurve],
    channels: list[int],
    start: int,
    stop: int,
    phase_break_samples: float = 0.0,
    phase_break_at: int | None = None,
) -> dict[int, np.ndarray]:
    indices = np.arange(start, stop, dtype=np.float64)
    result: dict[int, np.ndarray] = {}
    for channel in channels:
        values = indices + curves[channel](indices)
        if phase_break_samples != 0.0 and phase_break_at is not None and start >= phase_break_at:
            values = values + phase_break_samples
        result[channel] = values
    return result


def validate_phase_continuity(
    curves: dict[int, DriftCurve],
    channels: list[int],
    frames: int,
    block_frames: int,
    tolerance_samples: float,
) -> dict[str, Any]:
    maximum_error = 0.0
    boundaries = []
    for boundary in range(block_frames, frames, block_frames):
        for channel in channels:
            left = float(boundary - 2 + curves[channel](boundary - 2))
            middle = float(boundary - 1 + curves[channel](boundary - 1))
            right = float(boundary + curves[channel](boundary))
            previous_step = middle - left
            boundary_step = right - middle
            error = abs(boundary_step - previous_step)
            maximum_error = max(maximum_error, error)
            if error > tolerance_samples:
                boundaries.append(
                    {
                        "channel": channel,
                        "boundary_sample": boundary,
                        "coordinate_step_samples": boundary_step,
                        "previous_coordinate_step_samples": previous_step,
                        "deviation_samples": error,
                    }
                )
    if boundaries:
        raise SyncError(
            "E_RESAMPLE_PHASE",
            "interpolation phase is discontinuous at block boundary",
            {
                "tolerance_samples": tolerance_samples,
                "max_deviation_samples": maximum_error,
                "boundaries": boundaries[:32],
            },
        )
    return {
        "validated_boundaries": int(max(0, math.ceil((frames - block_frames) / block_frames))),
        "max_coordinate_step_deviation_samples": maximum_error,
        "tolerance_samples": tolerance_samples,
    }


def _render_interval(
    source: wave.Wave_read,
    source_start: int,
    source_stop: int,
    channels: list[int],
    coordinates: dict[int, np.ndarray],
    output_start: int,
    output_stop: int,
    config: dict[str, Any],
    taps: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    channel_count = int(config["channel_count"])
    source.setpos(source_start)
    raw = source.readframes(source_stop - source_start)
    samples = decode_pcm24(raw, channel_count).astype(np.float64, copy=False)
    output = np.empty((output_stop - output_start, channel_count), dtype=np.float64)
    reference_channel = int(config["reference_channel"])
    reference_indices = np.arange(output_start, output_stop, dtype=np.int64) - source_start
    output[:, reference_channel] = samples[reference_indices, reference_channel]
    phase_bins = weights.shape[0]
    for channel in channels:
        if channel == reference_channel:
            continue
        coordinate = coordinates[channel]
        base = np.floor(coordinate).astype(np.int64)
        fraction = coordinate - base
        phase_index = np.rint(fraction * phase_bins).astype(np.int64) % phase_bins
        index = base[:, None] + taps[None, :].astype(np.int64) - source_start
        index = np.clip(index, 0, samples.shape[0] - 1)
        values = samples[index, channel]
        output[:, channel] = np.sum(values * weights[phase_index], axis=1)
    return output


def _source_interval(
    coordinates: dict[int, np.ndarray],
    half_width: int,
    capture_frames: int,
) -> tuple[int, int]:
    minimum = min(float(values.min()) for values in coordinates.values())
    maximum = max(float(values.max()) for values in coordinates.values())
    source_start = max(0, int(math.floor(minimum)) - half_width + 1)
    source_stop = min(capture_frames, int(math.ceil(maximum)) + half_width + 2)
    require(source_stop > source_start, "E_RESAMPLE_PHASE", "empty source block", source_start=source_start, source_stop=source_stop)
    return source_start, source_stop


def compensate_wav(
    source_path: Path,
    output_path: Path | None,
    manifest: dict[str, Any],
    config: dict[str, Any],
    curves: dict[int, DriftCurve],
    block_frames: int | None = None,
    phase_break_samples: float = 0.0,
    phase_break_at: int | None = None,
) -> dict[str, Any]:
    sample_rate = int(config["sample_rate_hz"])
    channel_count = int(config["channel_count"])
    capture_frames = int(manifest["frames"])
    output_frames = int(manifest.get("analysis_frames", capture_frames))
    validate_pcm24_wav(source_path, channel_count, sample_rate, capture_frames)
    settings = config["resample"]
    if block_frames is None:
        block_frames = int(round(float(settings["block_s"]) * sample_rate))
    block_frames = max(64, int(block_frames))
    half_width = max(2, int(settings["sinc_half_width"]))
    taps, weights = _phase_table(half_width, float(settings["kaiser_beta"]))
    phase_bins = weights.shape[0]
    channels = list(range(channel_count))
    continuity = validate_phase_continuity(
        curves,
        channels,
        output_frames,
        block_frames,
        float(settings["phase_tolerance_samples"]),
    )
    maximum_input_coordinate = max(float(curves[channel](output_frames - 1)) + output_frames - 1 for channel in channels)
    require(
        maximum_input_coordinate + half_width + 2 <= capture_frames,
        "E_RESAMPLE_PHASE",
        "capture guard is too short for the final interpolation phase",
        max_input_coordinate=maximum_input_coordinate,
        capture_frames=capture_frames,
        half_width=half_width,
    )
    with wave.open(str(source_path), "rb") as source:
        target = write_wav_header(output_path, channel_count, sample_rate, output_frames) if output_path is not None else None
        digest = hashlib.sha256(wav_header_bytes(channel_count, sample_rate, output_frames))
        previous_coordinates: dict[int, np.ndarray] | None = None
        try:
            for start in range(0, output_frames, block_frames):
                stop = min(output_frames, start + block_frames)
                coordinates = _coordinates(
                    curves,
                    channels,
                    start,
                    stop,
                    phase_break_samples=phase_break_samples,
                    phase_break_at=phase_break_at,
                )
                if previous_coordinates is not None:
                    for channel in channels:
                        before_step = float(previous_coordinates[channel][-1] - previous_coordinates[channel][-2])
                        boundary_step = float(coordinates[channel][0] - previous_coordinates[channel][-1])
                        if abs(boundary_step - before_step) > float(settings["phase_tolerance_samples"]):
                            raise SyncError(
                                "E_RESAMPLE_PHASE",
                                "interpolation phase jump detected between processing blocks",
                                {
                                    "channel": channel,
                                    "block_start": start,
                                    "previous_step_samples": before_step,
                                    "boundary_step_samples": boundary_step,
                                    "deviation_samples": abs(boundary_step - before_step),
                                    "tolerance_samples": float(settings["phase_tolerance_samples"]),
                                },
                            )
                source_start, source_stop = _source_interval(coordinates, half_width, capture_frames)
                output = _render_interval(
                    source,
                    source_start,
                    source_stop,
                    channels,
                    coordinates,
                    start,
                    stop,
                    config,
                    taps,
                    weights,
                )
                encoded = encode_pcm24(output)
                if target is not None:
                    target.writeframes(encoded)
                digest.update(encoded)
                previous_coordinates = coordinates
        finally:
            if target is not None:
                target.close()
    return {
        "output_frames": output_frames,
        "capture_frames": capture_frames,
        "block_frames": block_frames,
        "sinc_half_width": half_width,
        "taps": int(taps.size),
        "phase_bins": phase_bins,
        "continuity": continuity,
        "output_sha256": digest.hexdigest(),
        "retained": output_path is not None,
    }


def detect_aligned_pulse_positions(
    source_path: Path,
    manifest: dict[str, Any],
    config: dict[str, Any],
    curves: dict[int, DriftCurve],
    pulse_radius: int = 256,
) -> dict[int, np.ndarray]:
    from .detect import detect_window_center

    sample_rate = int(config["sample_rate_hz"])
    channel_count = int(config["channel_count"])
    capture_frames = int(manifest["frames"])
    output_frames = int(manifest.get("analysis_frames", capture_frames))
    validate_pcm24_wav(source_path, channel_count, sample_rate, capture_frames)
    settings = config["resample"]
    half_width = max(2, int(settings["sinc_half_width"]))
    taps, weights = _phase_table(half_width, float(settings["kaiser_beta"]))
    channels = list(range(channel_count))
    truth = [int(value) for value in manifest["reference_pulse_samples"]]
    positions: dict[int, list[float]] = {channel: [] for channel in channels}
    with wave.open(str(source_path), "rb") as source:
        for pulse_index, reference_sample in enumerate(truth):
            start = max(0, reference_sample - pulse_radius)
            stop = min(output_frames, reference_sample + pulse_radius + 1)
            coordinates = _coordinates(curves, channels, start, stop)
            source_start, source_stop = _source_interval(coordinates, half_width, capture_frames)
            output = _render_interval(
                source,
                source_start,
                source_stop,
                channels,
                coordinates,
                start,
                stop,
                config,
                taps,
                weights,
            )
            approximate = reference_sample - start
            for channel in channels:
                center = detect_window_center(
                    output[:, channel],
                    approximate,
                    config,
                    {"channel": channel, "pulse_index": pulse_index, "reference_sample": reference_sample, "alignment_window": True},
                )
                positions[channel].append(start + center)
    return {channel: np.asarray(values, dtype=np.float64) for channel, values in positions.items()}
