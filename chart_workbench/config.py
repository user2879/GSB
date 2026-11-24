"""Calibration configuration loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .util import sha256_json


REQUIRED_NUMERIC = {
    "tick_min_length": int,
    "line_width_tolerance": int,
    "axis_detection_threshold": float,
    "broken_axis_detection_threshold": float,
    "log_axis_detection_threshold": float,
    "label_occlusion_tolerance": float,
    "fixed_decimal_places": int,
    "quantization_step": float,
    "thread_pool_concurrency": int,
    "tick_comparison_budget": int,
    "page_data_point_quota": int,
    "snapshot_commit_interval_pages": int,
}


class ConfigError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def load_config(path: Path, *, output_override: Path | None = None) -> Dict[str, Any]:
    path = Path(path).resolve()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot load configuration {path}: {exc}") from exc
    _require(isinstance(raw, dict), "configuration root must be an object")
    _require(raw.get("schema_version", 1) == 1, "unsupported configuration schema_version")
    _require(isinstance(raw.get("sample_dir"), str) and raw["sample_dir"], "sample_dir must be a non-empty string")
    _require(isinstance(raw.get("output_dir"), str) and raw["output_dir"], "output_dir must be a non-empty string")
    for key, expected_type in REQUIRED_NUMERIC.items():
        _require(key in raw, f"configuration is missing {key}")
        value = raw[key]
        if expected_type is int:
            _require(isinstance(value, int) and not isinstance(value, bool), f"{key} must be an integer")
        else:
            _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{key} must be numeric")
    _require(raw.get("inverse_mode") in {"ols", "endpoints"}, "inverse_mode must be 'ols' or 'endpoints'")
    _require(raw["tick_min_length"] >= 1, "tick_min_length must be at least 1")
    _require(raw["line_width_tolerance"] >= 0, "line_width_tolerance must be non-negative")
    _require(0.0 < raw["axis_detection_threshold"] <= 1.0, "axis_detection_threshold must be in (0, 1]")
    _require(raw["broken_axis_detection_threshold"] >= 0, "broken_axis_detection_threshold must be non-negative")
    _require(raw["log_axis_detection_threshold"] >= 0, "log_axis_detection_threshold must be non-negative")
    _require(0.0 <= raw["label_occlusion_tolerance"] <= 1.0, "label_occlusion_tolerance must be in [0, 1]")
    _require(0 <= raw["fixed_decimal_places"] <= 9, "fixed_decimal_places must be in [0, 9]")
    _require(raw["quantization_step"] >= 0, "quantization_step must be non-negative")
    _require(raw["thread_pool_concurrency"] >= 1, "thread_pool_concurrency must be at least 1")
    _require(raw["tick_comparison_budget"] >= 1, "tick_comparison_budget must be at least 1")
    _require(raw["page_data_point_quota"] >= 0, "page_data_point_quota must be non-negative")
    _require(raw["snapshot_commit_interval_pages"] >= 1, "snapshot_commit_interval_pages must be at least 1")

    config = dict(raw)
    config["_config_path"] = str(path)
    config["_sample_dir"] = str((path.parent / raw["sample_dir"]).resolve())
    if output_override is not None:
        config["_output_dir"] = str(Path(output_override).resolve())
    else:
        config["_output_dir"] = str((path.parent / raw["output_dir"]).resolve())
    config["_config_hash"] = sha256_json(_hash_payload(config))
    return config


def _hash_payload(config: Dict[str, Any]) -> Dict[str, Any]:
    execution_only = {"output_dir", "thread_pool_concurrency", "snapshot_commit_interval_pages"}
    return {
        key: value
        for key, value in config.items()
        if not key.startswith("_") and key not in execution_only
    }


def semantic_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the path-free subset serialized into reports and diagnostics."""

    return _hash_payload(config)
