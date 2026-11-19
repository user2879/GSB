"""Configuration and manifest loading with strict validation."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .util import read_json, sha256_json


class ConfigError(ValueError):
    """Raised for invalid configuration or sample-manifest data."""


PROFILE_REQUIRED_FIELDS = {
    "id": str,
    "target_frames": (int, float),
    "target_density": (int, float),
    "target_samples": (int, float),
    "feature_precision": int,
    "noise_window_multiplier": (int, float),
    "threshold_estimation": str,
}


@dataclass(frozen=True)
class VadConfig:
    raw: dict[str, Any]
    path: Path
    fingerprint: str

    @property
    def audio(self) -> dict[str, Any]:
        return self.raw["audio"]

    @property
    def profiles(self) -> list[dict[str, Any]]:
        return self.raw["scale_profiles"]

    def subset_fingerprint(self, keys: tuple[str, ...]) -> str:
        subset = {key: self.raw.get(key) for key in keys}
        return sha256_json(subset)

    def stage_fingerprint(self, stage: str) -> str:
        materials: dict[str, Any] = {
            "stage": stage,
            "audio": self.raw.get("audio"),
            "frame_ms": self.raw.get("frame_ms"),
            "hop_ms": self.raw.get("hop_ms"),
        }
        if stage == "policy":
            materials.update(
                {
                    "high_threshold": self.raw.get("high_threshold"),
                    "scale_profiles": self.raw.get("scale_profiles"),
                    "scale_profile_override": self.raw.get("scale_profile_override"),
                    "scale_tie_break": self.raw.get("scale_tie_break"),
                }
            )
        elif stage == "features":
            materials.update(
                {
                    "fft_window": self.raw.get("fft_window"),
                    "energy_floor": self.raw.get("energy_floor"),
                    "flatness_epsilon": self.raw.get("flatness_epsilon"),
                }
            )
        elif stage == "state":
            materials.update(
                {
                    "high_threshold": self.raw.get("high_threshold"),
                    "zcr_threshold": self.raw.get("zcr_threshold"),
                    "flatness_threshold": self.raw.get("flatness_threshold"),
                    "hysteresis": self.raw.get("hysteresis"),
                    "low_margin": self.raw.get("low_margin"),
                    "confirm_frames": self.raw.get("confirm_frames"),
                    "release_frames": self.raw.get("release_frames"),
                    "noise_window_frames": self.raw.get("noise_window_frames"),
                    "equal_boundary_policy": self.raw.get("equal_boundary_policy"),
                }
            )
        elif stage == "endpoints":
            materials.update(
                {
                    "min_speech_ms": self.raw.get("min_speech_ms"),
                    "min_silence_ms": self.raw.get("min_silence_ms"),
                }
            )
        else:
            raise ValueError(f"unknown stage: {stage}")
        return sha256_json(materials)


@dataclass(frozen=True)
class SampleManifest:
    raw: dict[str, Any]
    path: Path
    fingerprint: str
    samples_by_file: dict[str, dict[str, Any]]
    noise_reference_samples: tuple[str, ...]


def load_config(path: str | Path) -> VadConfig:
    config_path = Path(path).resolve()
    try:
        raw = read_json(config_path)
    except FileNotFoundError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"config file not found: {config_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read config: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("config root must be an object")
    _validate_config(raw)
    return VadConfig(
        raw=copy.deepcopy(raw), path=config_path, fingerprint=sha256_json(raw)
    )


def load_manifest(path: str | Path) -> SampleManifest:
    manifest_path = Path(path).resolve()
    try:
        raw = read_json(manifest_path)
    except FileNotFoundError as exc:
        raise ConfigError(f"sample manifest not found: {manifest_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read sample manifest: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("sample manifest root must be an object")
    samples = raw.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ConfigError("sample manifest must contain a non-empty samples array")
    seen: set[str] = set()
    by_file: dict[str, dict[str, Any]] = {}
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ConfigError(f"manifest samples[{index}] must be an object")
        name = sample.get("file")
        if not isinstance(name, str) or not name:
            raise ConfigError(f"manifest samples[{index}].file must be non-empty")
        normalized = name.replace("\\", "/")
        if normalized in seen:
            raise ConfigError(f"duplicate manifest sample file: {normalized}")
        seen.add(normalized)
        expected = sample.get("expected_segments")
        if (
            not isinstance(expected, list)
            or len(expected) != 2
            or any(not isinstance(item, int) or item < 0 for item in expected)
            or expected[0] > expected[1]
        ):
            raise ConfigError(
                f"manifest samples[{index}].expected_segments must be [min, max]"
            )
        if not isinstance(sample.get("scale_profile"), str):
            raise ConfigError(
                f"manifest samples[{index}].scale_profile must be a string"
            )
        shards = sample.get("shard_count", 1)
        if not isinstance(shards, int) or shards < 1:
            raise ConfigError(
                f"manifest samples[{index}].shard_count must be a positive integer"
            )
        by_file[normalized] = sample
    noise_refs = raw.get("noise_reference_samples", [])
    if not isinstance(noise_refs, list) or any(
        not isinstance(item, str) for item in noise_refs
    ):
        raise ConfigError("noise_reference_samples must be an array of strings")
    return SampleManifest(
        raw=copy.deepcopy(raw),
        path=manifest_path,
        fingerprint=sha256_json(raw),
        samples_by_file=by_file,
        noise_reference_samples=tuple(
            item.replace("\\", "/") for item in noise_refs
        ),
    )


def _validate_config(raw: dict[str, Any]) -> None:
    required_numbers = (
        "frame_ms",
        "hop_ms",
        "high_threshold",
        "zcr_threshold",
        "flatness_threshold",
        "hysteresis",
        "low_margin",
        "min_speech_ms",
        "min_silence_ms",
        "noise_window_frames",
        "concurrency",
        "snapshot_interval",
    )
    for key in required_numbers:
        value = raw.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ConfigError(f"{key} must be numeric")
    if raw["frame_ms"] <= 0 or raw["hop_ms"] <= 0:
        raise ConfigError("frame_ms and hop_ms must be positive")
    if raw["hop_ms"] > raw["frame_ms"]:
        raise ConfigError("hop_ms must not exceed frame_ms")
    if int(raw["concurrency"]) < 1 or int(raw["concurrency"]) != raw["concurrency"]:
        raise ConfigError("concurrency must be a positive integer")
    if (
        int(raw["snapshot_interval"]) < 1
        or int(raw["snapshot_interval"]) != raw["snapshot_interval"]
    ):
        raise ConfigError("snapshot_interval must be a positive integer")
    for key in ("confirm_frames", "release_frames"):
        if not isinstance(raw.get(key), int) or raw[key] < 1:
            raise ConfigError(f"{key} must be a positive integer")
    if raw["confirm_frames"] < 2:
        raise ConfigError("confirm_frames must be at least 2")
    if not isinstance(raw.get("noise_window_frames"), int) or raw[
        "noise_window_frames"
    ] < 1:
        raise ConfigError("noise_window_frames must be a positive integer")
    if raw.get("equal_boundary_policy") != "silence_first":
        raise ConfigError("equal_boundary_policy must be 'silence_first'")
    if not 0.0 <= float(raw["zcr_threshold"]) <= 1.0:
        raise ConfigError("zcr_threshold must be within [0, 1]")
    if not 0.0 <= float(raw["flatness_threshold"]) <= 1.0:
        raise ConfigError("flatness_threshold must be within [0, 1]")
    if raw["min_speech_ms"] < 0 or raw["min_silence_ms"] < 0:
        raise ConfigError("minimum speech/silence durations must be non-negative")
    if raw["hysteresis"] < 0 or raw["low_margin"] < 0:
        raise ConfigError("hysteresis and low_margin must be non-negative")
    audio = raw.get("audio")
    if not isinstance(audio, dict):
        raise ConfigError("audio must be an object")
    expected_audio = {
        "sample_rate": 16000,
        "channels": 1,
        "bits_per_sample": 16,
        "audio_format": "pcm",
    }
    for key, expected in expected_audio.items():
        if audio.get(key) != expected:
            raise ConfigError(f"audio.{key} must be {expected!r}")
    profiles = raw.get("scale_profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ConfigError("scale_profiles must be a non-empty array")
    identifiers: set[str] = set()
    for index, profile in enumerate(profiles):
        if not isinstance(profile, dict):
            raise ConfigError(f"scale_profiles[{index}] must be an object")
        for field, expected_type in PROFILE_REQUIRED_FIELDS.items():
            if field not in profile or not isinstance(profile[field], expected_type):
                raise ConfigError(
                    f"scale_profiles[{index}].{field} has invalid type"
                )
            if isinstance(profile[field], bool):
                raise ConfigError(
                    f"scale_profiles[{index}].{field} has invalid type"
                )
        if profile["id"] in identifiers:
            raise ConfigError(f"duplicate scale profile id: {profile['id']}")
        if not profile["id"]:
            raise ConfigError("scale profile id must be non-empty")
        identifiers.add(profile["id"])
        if not 1 <= profile["feature_precision"] <= 15:
            raise ConfigError("profile feature_precision must be within [1, 15]")
        for numeric_field in (
            "target_frames",
            "target_density",
            "target_samples",
            "noise_window_multiplier",
        ):
            if not math.isfinite(float(profile[numeric_field])):
                raise ConfigError(f"profile {numeric_field} must be finite")
        if profile["noise_window_multiplier"] <= 0:
            raise ConfigError("profile noise_window_multiplier must be positive")
        if profile["target_frames"] <= 0 or profile["target_samples"] <= 0:
            raise ConfigError("profile target_frames/target_samples must be positive")
        if profile["target_density"] < 0:
            raise ConfigError("profile target_density must be non-negative")
        if profile["threshold_estimation"] not in {"quantile", "fixed_ratio"}:
            raise ConfigError(
                "profile threshold_estimation must be 'quantile' or 'fixed_ratio'"
            )
        if "low_quantile" in profile and not 0.0 < float(profile["low_quantile"]) < 1.0:
            raise ConfigError("profile low_quantile must be within (0, 1)")
    override = raw.get("scale_profile_override")
    if override is not None and override not in identifiers:
        raise ConfigError("scale_profile_override must name a configured profile")
    tolerance = raw.get("noise_feature_tolerance")
    if not isinstance(tolerance, dict):
        raise ConfigError("noise_feature_tolerance must be an object")
