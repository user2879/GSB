from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from .errors import SyncError, require
from .hashing import sha256_json


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "seed": 20261119,
    "sample_rate_hz": 48000,
    "bit_depth": 24,
    "channel_count": 8,
    "duration_s": 30.0,
    "reference_channel": 0,
    "signal": {
        "tones_hz": [137.0, 443.0, 1297.0, 2381.0],
        "tone_amplitudes": [0.045, 0.040, 0.035, 0.030],
        "pulse_period_s": 0.75,
        "pulse_offset_s": 0.25,
        "pulse_sigma_s": 0.00035,
        "pulse_amplitude": 0.75,
    },
    "channels": [],
    "drift_tolerance_ppm": 50.0,
    "accuracy_target_samples": 0.75,
    "detector": {
        "peak_threshold": 0.45,
        "min_distance_s": 0.25,
        "pair_tolerance_s": 0.22,
        "tracking_gate_s": 0.15,
        "block_s": 5.0,
        "overlap_s": 0.5,
    },
    "strategy": {
        "override": "auto",
        "memory_budget_mb": 128.0,
        "time_budget_s": 180.0,
        "full_max_duration_s": 120.0,
        "full_max_pulses": 1200,
        "block_pulses": 6,
        "sparse_stride": 4,
        "min_memory_mb": 8.0,
    },
    "resample": {
        "engine": "windowed_sinc",
        "sinc_half_width": 8,
        "kaiser_beta": 8.6,
        "block_s": 2.0,
        "comparison_block_s": [0.25, 2.0],
        "phase_tolerance_samples": 1.0e-3,
        "consistency_tolerance_samples": 0.05,
        "quantize_wav": True,
    },
    "fit": {
        "segment_pulses": 8,
        "confidence": 0.95,
    },
    "pipeline": {
        "auto_drop_aligned_after_s": 120.0,
    },
}


def _default_channels(seed: int, count: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    channels: list[dict[str, Any]] = []
    for index in range(count):
        if index == 0:
            channels.append(
                {
                    "index": 0,
                    "role": "reference",
                    "base_ppm": 0.0,
                    "temp_ppm_per_c": 0.0,
                    "temperature_c": {"mean": 25.0, "amplitude_c": 0.0, "period_s": 600.0, "phase_rad": 0.0},
                    "delay_samples": 0.0,
                }
            )
            continue
        channels.append(
            {
                "index": index,
                "role": "slave",
                "base_ppm": float(rng.uniform(-8.0, 8.0)),
                "temp_ppm_per_c": float(rng.uniform(0.4, 1.8)),
                "temperature_c": {
                    "mean": 25.0,
                    "amplitude_c": float(rng.uniform(3.0, 7.0)),
                    "period_s": float(rng.uniform(240.0, 720.0)),
                    "phase_rad": float(rng.uniform(-3.14, 3.14)),
                },
                "delay_samples": float(120.0 + 37.0 * index + rng.uniform(-8.0, 8.0)),
            }
        )
    return channels


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def validate_config(config: dict[str, Any]) -> None:
    require(config.get("channel_count") == 8, "E_CH_FORMAT", "demo requires exactly 8 channels", actual=config.get("channel_count"))
    require(config.get("bit_depth") == 24, "E_CH_FORMAT", "demo requires PCM24", actual=config.get("bit_depth"))
    require(config.get("sample_rate_hz") == 48000, "E_CH_FORMAT", "demo requires 48 kHz", actual=config.get("sample_rate_hz"))
    require(float(config.get("duration_s", 0)) > 0, "E_CONFIG", "duration_s must be positive")
    channels = config.get("channels", [])
    require(len(channels) == 8, "E_CH_FORMAT", "channels array must contain 8 entries", actual=len(channels))
    require([item.get("index") for item in channels] == list(range(8)), "E_CH_FORMAT", "channel indices must be 0..7")
    require(config.get("reference_channel") == 0, "E_CONFIG", "reference channel must be 0")
    require(channels[0].get("role") == "reference", "E_CONFIG", "channel 0 role must be reference")
    require(all(item.get("role") == "slave" for item in channels[1:]), "E_CONFIG", "channels 1..7 must be slaves")
    signal = config["signal"]
    require(len(signal["tones_hz"]) == len(signal["tone_amplitudes"]), "E_CONFIG", "tone arrays differ in length")
    require(float(signal["pulse_period_s"]) > 0, "E_CONFIG", "pulse period must be positive")
    require(0.0 <= float(signal["pulse_offset_s"]) < float(signal["pulse_period_s"]), "E_CONFIG", "pulse offset outside period")
    for channel in channels[1:]:
        require(float(channel["delay_samples"]) >= 0.0, "E_CONFIG", "delay_samples must be nonnegative", channel=channel["index"])
    require(float(config["drift_tolerance_ppm"]) > 0, "E_CONFIG", "drift tolerance must be positive")
    require(float(config["accuracy_target_samples"]) > 0, "E_CONFIG", "accuracy target must be positive")


def load_config(path: Path | None, duration_s: float | None = None) -> dict[str, Any]:
    source: dict[str, Any] = {}
    if path is not None:
        try:
            source = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SyncError("E_CONFIG", "cannot load config", {"path": str(path), "reason": str(exc)}) from exc
    merged = _deep_merge(DEFAULT_CONFIG, source)
    if not merged.get("channels"):
        merged["channels"] = _default_channels(int(merged["seed"]), int(merged["channel_count"]))
    if not merged["signal"].get("tone_phase_rad"):
        rng = np.random.default_rng(int(merged["seed"]) ^ 0x5EED)
        merged["signal"]["tone_phase_rad"] = [float(value) for value in rng.uniform(0.0, 2.0 * np.pi, len(merged["signal"]["tones_hz"]))]
    if duration_s is not None:
        merged["duration_s"] = float(duration_s)
    validate_config(merged)
    content = copy.deepcopy(merged)
    content.pop("config_sha256", None)
    merged["config_sha256"] = sha256_json(content)
    return merged


def config_content_hash(config: dict[str, Any]) -> str:
    content = copy.deepcopy(config)
    content.pop("config_sha256", None)
    return sha256_json(content)
