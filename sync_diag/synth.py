from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates, spline_filter

from .config import config_content_hash
from .hashing import sha256_file, write_json
from .manifest import write_manifest
from .signal import ChannelClock, TestSignal
from .wavio import encode_pcm24, write_wav_header


def _resolve_manifest(config: dict[str, Any], frames: int) -> tuple[dict[str, Any], list[ChannelClock], TestSignal]:
    signal = TestSignal(config)
    clocks = [ChannelClock.from_config(channel, int(config["sample_rate_hz"])) for channel in config["channels"]]
    reference_pulses = signal.reference_pulse_samples(frames)
    truth = signal.truth_at_reference_samples(np.asarray(reference_pulses, dtype=np.float64), clocks)
    channels: list[dict[str, Any]] = []
    for channel, clock, observed in zip(config["channels"], clocks, truth):
        channels.append(
            {
                "index": int(channel["index"]),
                "role": channel["role"],
                "clock": {
                    "base_ppm": clock.base_ppm,
                    "temp_ppm_per_c": clock.temp_ppm_per_c,
                    "temperature_c": {
                        "mean": clock.temp_mean_c,
                        "amplitude_c": clock.temp_amplitude_c,
                        "period_s": clock.temp_period_s,
                        "phase_rad": clock.temp_phase_rad,
                    },
                    "delay_samples": clock.delay_samples,
                },
                "expected_observed_pulse_samples": [round(float(value), 9) for value in observed],
            }
        )
    manifest = {
        "schema_version": 1,
        "kind": "multichannel_sync_truth",
        "config_sha256": config_content_hash(config),
        "seed": int(config["seed"]),
        "sample_rate_hz": int(config["sample_rate_hz"]),
        "bit_depth": int(config["bit_depth"]),
        "channel_count": int(config["channel_count"]),
        "reference_channel": int(config["reference_channel"]),
        "frames": frames,
        "analysis_frames": frames,
        "duration_s": frames / float(config["sample_rate_hz"]),
        "signal_signature_sha256": signal.signature,
        "pulse_period_samples": signal.pulse_period_samples,
        "pulse_offset_samples": signal.pulse_offset_samples,
        "reference_pulse_samples": reference_pulses,
        "channels": channels,
    }
    return manifest, clocks, signal


def synthesize_dataset(config: dict[str, Any], out_dir: Path, chunk_frames: int = 131072) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_frames = int(round(float(config["duration_s"]) * int(config["sample_rate_hz"])))
    manifest, clocks, signal = _resolve_manifest(config, analysis_frames)
    last_samples = np.asarray([analysis_frames - 1], dtype=np.float64)
    max_source_tail = max(float(clock.source_coordinate(last_samples)[0]) for clock in clocks)
    guard_frames = max(0, int(np.ceil(max_source_tail - (analysis_frames - 1)))) + 64
    frames = analysis_frames + guard_frames
    manifest["frames"] = frames
    manifest["capture_guard_frames"] = guard_frames
    manifest["duration_s"] = analysis_frames / float(config["sample_rate_hz"])
    wav_path = out_dir / "capture.wav"
    wav = write_wav_header(wav_path, int(config["channel_count"]), int(config["sample_rate_hz"]), frames)
    try:
        for start in range(0, frames, chunk_frames):
            count = min(chunk_frames, frames - start)
            sample_numbers = np.arange(start, start + count, dtype=np.float64)
            coordinates = [clock.source_coordinate(sample_numbers) for clock in clocks]
            source_start = int(np.floor(min(float(values.min()) for values in coordinates))) - 4
            source_stop = int(np.ceil(max(float(values.max()) for values in coordinates))) + 5
            base = signal.sample_indices(source_start, source_stop - source_start)
            filtered = spline_filter(base, order=3, mode="nearest")
            output = np.empty((count, int(config["channel_count"])), dtype=np.float64)
            output[:, int(config["reference_channel"])] = base[(start - source_start) : (start - source_start + count)]
            for clock in clocks:
                if clock.index == int(config["reference_channel"]):
                    continue
                positions = coordinates[clock.index] - float(source_start)
                output[:, clock.index] = map_coordinates(filtered, [positions], order=3, mode="nearest", prefilter=False)
            wav.writeframes(encode_pcm24(output))
    finally:
        wav.close()
    wav_sha256 = sha256_file(wav_path)
    manifest["wav_sha256"] = wav_sha256
    manifest_path = out_dir / "manifest.json"
    manifest_sha256 = write_manifest(manifest_path, manifest)
    resolved = dict(config)
    write_json(out_dir / "config.resolved.json", resolved)
    return {
        "wav_path": str(wav_path),
        "manifest_path": str(manifest_path),
        "wav_sha256": wav_sha256,
        "manifest_sha256": manifest_sha256,
        "frames": frames,
        "analysis_frames": analysis_frames,
        "capture_guard_frames": guard_frames,
        "pulse_count": len(manifest["reference_pulse_samples"]),
    }
