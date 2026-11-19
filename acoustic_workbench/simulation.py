"""Deterministic source generation and local room-path simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .audio import decode_wav_24, encode_wav_24
from .dsp import (
    apply_fractional_delay,
    band_limit,
    exponential_sweep,
    parabolic_peak,
    robust_sigma,
    seconds_to_samples,
    sync_pulse,
)
from .errors import ErrorCode, WorkbenchError, fail
from .hashing import bytes_hash, content_hash
from .models import ChannelConfig, MeasurementConfig, MeasurementRun


@dataclass(frozen=True)
class MeasurementLayout:
    n_sweep: int
    n_lead_in: int
    n_guard_pre: int
    n_guard_post: int
    n_segment: int
    n_frames: int
    sweep_starts: tuple[int, ...]
    segment_starts: tuple[int, ...]


def measurement_layout(config: MeasurementConfig) -> MeasurementLayout:
    n_sweep = seconds_to_samples(config.sweep_duration_s, config.sample_rate)
    lead = seconds_to_samples(config.lead_in_s, config.sample_rate)
    guard_pre = seconds_to_samples(config.guard_pre_s, config.sample_rate)
    guard_post = seconds_to_samples(config.guard_post_s, config.sample_rate)
    segment = guard_pre + n_sweep + guard_post
    starts = tuple(lead + index * segment for index in range(config.repeats))
    sweep_starts = tuple(start + guard_pre for start in starts)
    return MeasurementLayout(
        n_sweep=n_sweep,
        n_lead_in=lead,
        n_guard_pre=guard_pre,
        n_guard_post=guard_post,
        n_segment=segment,
        n_frames=lead + config.repeats * segment,
        sweep_starts=sweep_starts,
        segment_starts=starts,
    )


def _build_source(config: MeasurementConfig, layout: MeasurementLayout) -> np.ndarray:
    source = np.zeros(layout.n_frames, dtype=np.float64)
    sweep = exponential_sweep(
        sample_rate=config.sample_rate,
        duration_s=config.sweep_duration_s,
        low_hz=config.sweep_low_hz,
        high_hz=config.sweep_high_hz,
        amplitude=config.drive_amplitude,
        fade_s=config.fade_s,
    )
    pulse, centre = sync_pulse(
        sample_rate=config.sample_rate,
        amplitude=config.sync_pulse_amplitude,
        center_hz=config.sync_pulse_center_hz,
        sigma_samples=config.sync_pulse_sigma_samples,
        radius_samples=config.sync_pulse_radius_samples,
    )
    for start, sweep_start in zip(layout.segment_starts, layout.sweep_starts):
        source[sweep_start : sweep_start + sweep.size] += sweep
        pulse_start = start - centre
        pulse_end = pulse_start + pulse.size
        if pulse_start < 0:
            fail(ErrorCode.SYNC_MARK, "sync pulse precedes the recording", stage="simulation")
        source[pulse_start:pulse_end] += pulse
    return source


def _noise_for_channel(
    config: MeasurementConfig,
    channel_seed: int,
    count: int,
    sigma: float,
) -> np.ndarray:
    if sigma <= 0.0:
        return np.zeros(count, dtype=np.float64)
    seed = int((config.master_seed ^ channel_seed) & 0xFFFFFFFFFFFFFFFF)
    generator = np.random.Generator(np.random.PCG64(seed))
    return generator.standard_normal(count, dtype=np.float64) * sigma


def propagate_channel(
    source: np.ndarray,
    channel: ChannelConfig,
    config: MeasurementConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one channel through delay, gain, band limit and fixed-seed noise."""

    path = apply_fractional_delay(np.asarray(source, dtype=np.float64), channel.delay_samples, config.sample_rate)
    path *= channel.gain
    path = band_limit(
        path,
        sample_rate=config.sample_rate,
        low_hz=channel.band_low_hz,
        high_hz=channel.band_high_hz,
    )
    noise = _noise_for_channel(config, channel.noise_seed, path.size, channel.noise_rms)
    return path + noise, noise


def _sync_markers(config: MeasurementConfig, layout: MeasurementLayout) -> tuple[dict[str, Any], ...]:
    marker_rows: list[dict[str, Any]] = []
    for repeat_index, source_sample in enumerate(layout.segment_starts):
        truths: list[dict[str, Any]] = []
        for channel_index, channel in enumerate(config.channels):
            truth = float(source_sample) + float(channel.delay_samples)
            truths.append(
                {
                    "channel_index": channel_index,
                    "channel": channel.name,
                    "truth_sample_fractional": truth,
                    "truth_sample_nearest": int(round(truth)),
                    "configured_delay_samples": float(channel.delay_samples),
                }
            )
        marker_rows.append(
            {
                "repeat_index": repeat_index,
                "source_sample": int(source_sample),
                "sweep_start_sample": int(layout.sweep_starts[repeat_index]),
                "channels": truths,
            }
        )
    return tuple(marker_rows)


def estimate_noise_sigma(recorded: np.ndarray, config: MeasurementConfig) -> np.ndarray:
    """Recover the per-channel quiet-segment noise estimate from recorded samples."""

    layout = measurement_layout(config)
    values = np.zeros(len(config.channels), dtype=np.float64)
    probe_end = max(32, layout.n_lead_in - 64)
    for channel_index, channel in enumerate(config.channels):
        quiet = np.asarray(recorded[16:probe_end, channel_index], dtype=np.float64)
        if quiet.size > 8:
            values[channel_index] = robust_sigma(np.diff(quiet)) / np.sqrt(2.0)
        else:
            values[channel_index] = float(channel.noise_rms)
    return values


def generate_measurement(config: MeasurementConfig) -> MeasurementRun:
    """Generate the deterministic source, simulated recording and sync truth."""

    layout = measurement_layout(config)
    source = _build_source(config, layout)
    recorded = np.zeros((layout.n_frames, len(config.channels)), dtype=np.float64)
    for channel_index, channel in enumerate(config.channels):
        recorded[:, channel_index], _ = propagate_channel(source, channel, config)

    peak = float(np.max(np.abs(recorded)))
    if peak >= 1.0:
        fail(
            ErrorCode.AUDIO_FORMAT,
            "simulated recording would clip 24-bit PCM",
            stage="simulation",
            details={"peak": peak},
        )
    wav_bytes = encode_wav_24(
        recorded,
        sample_rate=config.sample_rate,
        channels=len(config.channels),
        bit_depth=config.bit_depth,
    )
    decoded, _ = decode_wav_24(
        wav_bytes,
        expected_sample_rate=config.sample_rate,
        expected_channels=len(config.channels),
        expected_bit_depth=config.bit_depth,
    )
    recorded = decoded
    noise_sigma = estimate_noise_sigma(recorded, config)
    sync_markers = _sync_markers(config, layout)
    source_hash = content_hash({"source": source})
    recorded_hash = content_hash({"recorded": recorded})
    wav_hash = bytes_hash(wav_bytes)
    return MeasurementRun(
        config=config,
        source=source,
        recorded=recorded,
        wav_bytes=wav_bytes,
        sync_markers=sync_markers,
        sweep_starts=layout.sweep_starts,
        source_hash=source_hash,
        recorded_hash=recorded_hash,
        wav_hash=wav_hash,
        noise_sigma_estimate=noise_sigma,
    )


def validate_sync_marks(
    run: MeasurementRun,
    *,
    threshold_relative: float = 0.015,
    search_radius_samples: int = 16,
    tolerance_samples: float = 1.5,
) -> tuple[dict[str, Any], ...]:
    """Verify that every declared source marker produced a detectable pulse."""

    config = run.config
    detected_rows: list[dict[str, Any]] = []
    overall_peak = float(np.max(np.abs(run.recorded)))
    if overall_peak <= 0.0:
        fail(ErrorCode.SYNC_MARK, "recording is silent; sync pulse cannot be verified", stage="sync")
    threshold = overall_peak * threshold_relative
    for marker in run.sync_markers:
        repeat_index = int(marker["repeat_index"])
        channels: list[dict[str, Any]] = []
        for truth_row in marker["channels"]:
            channel_index = int(truth_row["channel_index"])
            expected = int(truth_row["truth_sample_nearest"])
            lo = max(0, expected - search_radius_samples)
            hi = min(run.recorded.shape[0], expected + search_radius_samples + 1)
            window = np.abs(run.recorded[lo:hi, channel_index])
            if window.size == 0:
                fail(ErrorCode.SYNC_MARK, "sync search window is empty", stage="sync")
            local_peak = int(np.argmax(window)) + lo
            level = float(np.max(window))
            if level < threshold:
                fail(
                    ErrorCode.SYNC_MARK,
                    "declared sync pulse is missing or below threshold",
                    stage="sync",
                    details={
                        "repeat": repeat_index,
                        "channel": channel_index,
                        "expected_sample": expected,
                        "level": level,
                        "threshold": threshold,
                    },
                )
            search_lo = max(0, local_peak - 1)
            search_hi = min(run.recorded.shape[0], local_peak + 2)
            local_values = np.abs(run.recorded[search_lo:search_hi, channel_index])
            local_index, delta, _ = parabolic_peak(local_values)
            detected = float(search_lo + local_index + delta)
            error = detected - float(truth_row["truth_sample_fractional"])
            if abs(error) > tolerance_samples:
                fail(
                    ErrorCode.SYNC_MARK,
                    "detected sync pulse disagrees with truth marker",
                    stage="sync",
                    details={
                        "repeat": repeat_index,
                        "channel": channel_index,
                        "truth": truth_row["truth_sample_fractional"],
                        "detected": detected,
                        "tolerance": tolerance_samples,
                    },
                )
            channels.append(
                {
                    "channel_index": channel_index,
                    "channel": truth_row["channel"],
                    "truth_sample_fractional": float(truth_row["truth_sample_fractional"]),
                    "detected_sample_fractional": detected,
                    "error_samples": error,
                    "level": level,
                    "status": "OK",
                }
            )
        detected_rows.append(
            {
                "repeat_index": repeat_index,
                "source_sample": int(marker["source_sample"]),
                "sweep_start_sample": int(marker["sweep_start_sample"]),
                "channels": channels,
            }
        )
    return tuple(detected_rows)


def manifest_dict(run: MeasurementRun, layout: MeasurementLayout) -> dict[str, Any]:
    config = run.config
    return {
        "schema": "acoustic-workbench.measurement-manifest/1.0",
        "audio": {
            "format": "PCM",
            "sample_rate": config.sample_rate,
            "bit_depth": config.bit_depth,
            "channels": len(config.channels),
            "frames": layout.n_frames,
            "duration_s": layout.n_frames / float(config.sample_rate),
        },
        "sweep": {
            "kind": "exponential-sine",
            "low_hz": config.sweep_low_hz,
            "high_hz": config.sweep_high_hz,
            "duration_s": config.sweep_duration_s,
            "repeats": config.repeats,
            "sweep_starts": list(layout.sweep_starts),
        },
        "channels": [
            {
                "index": index,
                "name": channel.name,
                "delay_samples": channel.delay_samples,
                "gain": channel.gain,
                "band_low_hz": channel.band_low_hz,
                "band_high_hz": channel.band_high_hz,
                "noise_seed": channel.noise_seed,
                "noise_rms": channel.noise_rms,
            }
            for index, channel in enumerate(config.channels)
        ],
        "sync_marker_truth": list(run.sync_markers),
        "hashes": {
            "config": content_hash(config),
            "source": run.source_hash,
            "recorded_samples": run.recorded_hash,
            "wav": run.wav_hash,
        },
    }
