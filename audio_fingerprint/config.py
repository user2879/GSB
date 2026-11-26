from __future__ import annotations

from copy import deepcopy
from typing import Any

from .biz import (
    INVALID_BUCKET_CAPACITY,
    INVALID_BUDGET,
    INVALID_CONFIG,
    INVALID_PEAK_THRESHOLD,
    INVALID_TIME_GEAR,
    issue_dict,
)
from .canonical import fingerprint


TIME_GEAR_STEPS = {
    "tq_fine": 1,
    "tq_standard": 2,
    "tq_coarse": 4,
}

SPECTRUM_GEAR_SHIFTS = {
    "sq_fine": 0,
    "sq_standard": 3,
    "sq_coarse": 5,
}

DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "internal_sample_rate": 11025,
    "frame_size": 256,
    "hop_size": 128,
    "window": "hann",
    "band_count": 8,
    "band_min_hz": 180,
    "band_max_hz": 5000,
    "fixed_fraction_bits": 12,
    "spectrum_quant_gear": "sq_standard",
    "peak_threshold": 900,
    "peak_band_start": 0,
    "peak_band_end": 8,
    "peak_neighborhood_radius": 1,
    "max_peaks_per_frame": 4,
    "pair_lookahead_frames": 8,
    "max_targets_per_anchor": 3,
    "time_quant_gear": "tq_standard",
    "bucket_capacity": 8,
    "candidate_limit": 5,
    "vote_fraction_bits": 8,
    "max_concurrency": 2,
    "batch_sample_budget": 200000,
    "clip_sample_quota": 100000,
    "frame_threshold_overrides": {},
}


def is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _fallback(key: str) -> Any:
    return deepcopy(DEFAULT_CONFIG[key])


def parse_config(data: dict[str, Any] | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate a config object. Invalid fields fall back and emit issues."""

    raw = {} if data is None else data
    config = deepcopy(DEFAULT_CONFIG)
    issues: list[dict[str, Any]] = []

    if not isinstance(raw, dict):
        issues.append(issue_dict(INVALID_CONFIG, "config", "configuration must be an object"))
        return config, issues

    for key, value in raw.items():
        if key not in DEFAULT_CONFIG:
            if key not in {"time_quant_step", "spectrum_quant_shift"}:
                issues.append(issue_dict(INVALID_CONFIG, f"config.{key}", "unknown configuration key"))

    overrides = raw.get("frame_threshold_overrides", {})
    if not isinstance(overrides, dict):
        issues.append(issue_dict(INVALID_CONFIG, "config.frame_threshold_overrides", "must be an object"))
        config["frame_threshold_overrides"] = {}
    else:
        clean_overrides: dict[str, dict[str, int | None]] = {}
        for clip_id, frame_map in overrides.items():
            if not isinstance(clip_id, str) or not isinstance(frame_map, dict):
                issues.append(issue_dict(INVALID_CONFIG, "config.frame_threshold_overrides", "invalid override entry"))
                continue
            clean_map: dict[str, int | None] = {}
            for frame_key, threshold in frame_map.items():
                key_text = str(frame_key)
                try:
                    frame_index = int(key_text)
                except ValueError:
                    frame_index = -1
                if frame_index < 0 or (threshold is not None and not is_positive_int(threshold)):
                    issues.append(
                        issue_dict(
                            INVALID_PEAK_THRESHOLD,
                            f"config.frame_threshold_overrides.{clip_id}.{key_text}",
                            "frame override must be a positive integer or null",
                        )
                    )
                    continue
                clean_map[str(frame_index)] = None if threshold is None else int(threshold)
            clean_overrides[clip_id] = clean_map
        config["frame_threshold_overrides"] = clean_overrides

    positive_fields = {
        "internal_sample_rate": INVALID_CONFIG,
        "frame_size": INVALID_CONFIG,
        "hop_size": INVALID_CONFIG,
        "band_count": INVALID_CONFIG,
        "band_min_hz": INVALID_CONFIG,
        "band_max_hz": INVALID_CONFIG,
        "fixed_fraction_bits": INVALID_CONFIG,
        "peak_threshold": INVALID_PEAK_THRESHOLD,
        "peak_band_end": INVALID_CONFIG,
        "peak_neighborhood_radius": INVALID_CONFIG,
        "max_peaks_per_frame": INVALID_CONFIG,
        "pair_lookahead_frames": INVALID_CONFIG,
        "max_targets_per_anchor": INVALID_CONFIG,
        "bucket_capacity": INVALID_BUCKET_CAPACITY,
        "candidate_limit": INVALID_CONFIG,
        "vote_fraction_bits": INVALID_CONFIG,
        "max_concurrency": INVALID_CONFIG,
        "batch_sample_budget": INVALID_BUDGET,
        "clip_sample_quota": INVALID_BUDGET,
    }
    for key, code in positive_fields.items():
        if key not in raw:
            continue
        value = raw[key]
        numeric_ok = is_positive_int(value)
        if key in {"band_min_hz", "band_max_hz"} and isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric_ok = value > 0
        if not numeric_ok:
            issues.append(
                issue_dict(
                    code,
                    f"config.{key}",
                    f"{key} must be a positive integer",
                    value=value,
                )
            )
            continue
        config[key] = value

    if "peak_band_start" in raw:
        value = raw["peak_band_start"]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            issues.append(
                issue_dict(
                    INVALID_CONFIG,
                    "config.peak_band_start",
                    "peak_band_start must be a non-negative integer",
                    value=value,
                )
            )
        else:
            config["peak_band_start"] = value

    if "window" in raw:
        if raw["window"] not in {"hann", "ones"}:
            issues.append(issue_dict(INVALID_CONFIG, "config.window", "unsupported window", value=raw["window"]))
        else:
            config["window"] = raw["window"]

    gear = raw.get("time_quant_gear", config["time_quant_gear"])
    if gear not in TIME_GEAR_STEPS:
        issues.append(
            issue_dict(
                INVALID_TIME_GEAR,
                "config.time_quant_gear",
                "time quantization gear is not allowed",
                value=gear,
                allowed=sorted(TIME_GEAR_STEPS),
            )
        )
    else:
        config["time_quant_gear"] = gear

    spectrum_gear = raw.get("spectrum_quant_gear", config["spectrum_quant_gear"])
    if spectrum_gear not in SPECTRUM_GEAR_SHIFTS:
        issues.append(
            issue_dict(
                INVALID_CONFIG,
                "config.spectrum_quant_gear",
                "spectrum quantization gear is not allowed",
                value=spectrum_gear,
                allowed=sorted(SPECTRUM_GEAR_SHIFTS),
            )
        )
    else:
        config["spectrum_quant_gear"] = spectrum_gear

    if config["peak_band_start"] < 0 or config["peak_band_end"] > config["band_count"]:
        issues.append(
            issue_dict(
                INVALID_CONFIG,
                "config.peak_band_range",
                "peak band range is outside band_count",
                start=config["peak_band_start"],
                end=config["peak_band_end"],
                band_count=config["band_count"],
            )
        )
        config["peak_band_start"] = DEFAULT_CONFIG["peak_band_start"]
        config["peak_band_end"] = min(DEFAULT_CONFIG["peak_band_end"], config["band_count"])

    if config["peak_band_start"] >= config["peak_band_end"]:
        issues.append(
            issue_dict(INVALID_CONFIG, "config.peak_band_range", "peak band range is empty")
        )
        config["peak_band_start"] = 0
        config["peak_band_end"] = config["band_count"]

    if config["hop_size"] > config["frame_size"]:
        issues.append(issue_dict(INVALID_CONFIG, "config.hop_size", "hop_size must be <= frame_size"))
        config["hop_size"] = config["frame_size"] // 2

    config["time_quant_step"] = TIME_GEAR_STEPS[config["time_quant_gear"]]
    config["spectrum_quant_shift"] = SPECTRUM_GEAR_SHIFTS[config["spectrum_quant_gear"]]
    return config, issues


def full_config_fingerprint(config: dict[str, Any]) -> str:
    excluded = {"time_quant_step", "spectrum_quant_shift"}
    return fingerprint({k: v for k, v in sorted(config.items()) if k not in excluded}, "cfg_all_")


def spectral_fingerprint(config: dict[str, Any]) -> str:
    keys = [
        "internal_sample_rate",
        "frame_size",
        "hop_size",
        "window",
        "band_count",
        "band_min_hz",
        "band_max_hz",
        "fixed_fraction_bits",
        "spectrum_quant_gear",
        "spectrum_quant_shift",
    ]
    return fingerprint({k: config[k] for k in keys}, "spec_cfg_")


def peak_rule_fingerprint(config: dict[str, Any], frame_override: int | None = None) -> str:
    keys = [
        "peak_threshold",
        "peak_band_start",
        "peak_band_end",
        "peak_neighborhood_radius",
        "max_peaks_per_frame",
    ]
    payload = {k: config[k] for k in keys}
    payload["frame_threshold_override"] = frame_override
    return fingerprint(payload, "peak_rule_")


def pair_fingerprint_config(config: dict[str, Any]) -> str:
    keys = ["pair_lookahead_frames", "max_targets_per_anchor", "time_quant_gear", "time_quant_step"]
    return fingerprint({k: config[k] for k in keys}, "pair_cfg_")


def bucket_config_fingerprint(config: dict[str, Any]) -> str:
    return fingerprint({"bucket_capacity": config["bucket_capacity"]}, "bucket_cfg_")


def vote_config_fingerprint(config: dict[str, Any]) -> str:
    keys = ["candidate_limit", "vote_fraction_bits"]
    return fingerprint({k: config[k] for k in keys}, "vote_cfg_")
