"""Configuration loading, validation and analysis-parameter projection."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .errors import ConfigError
from .fingerprint import fingerprint


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "image_validation": {
        "width": 720,
        "height": 960,
        "allowed_modes": ["L", "RGB", "RGBA"],
    },
    "binarization_threshold": 180,
    "block_min_size": {"width": 12, "height": 8},
    "projection_window": {"horizontal": 6, "vertical": 5},
    "column_gap_threshold": 24,
    "column_candidate_max_width_ratio": 0.55,
    "column_span_ratio": 0.62,
    "column_span_min_overlap_ratio": 0.15,
    "column_min_overlap_ratio": 0.25,
    "role_detection": {
        "title_body_height_ratio": {"min": 1.5, "max": 3.0},
        "image_min_ink_density": 0.46,
        "image_max_projection_periodicity": 0.18,
        "header_footer_margin_ratio": 0.075,
        "role_priority": ["image", "header_footer", "title", "body"],
    },
    "reading_order_rule": "column_first",
    "region_band_gap": 8,
    "thread_pool_workers": 4,
    "pages_per_batch": 3,
    "snapshot_interval_pages": 2,
}


ANALYSIS_STAGE_KEYS = {
    "image": ["image_validation"],
    "detect": ["binarization_threshold", "block_min_size", "projection_window"],
    "role": ["role_detection"],
    "columns": [
        "column_gap_threshold",
        "column_candidate_max_width_ratio",
        "column_span_ratio",
        "column_span_min_overlap_ratio",
        "column_min_overlap_ratio",
    ],
    "order": ["reading_order_rule", "region_band_gap"],
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _require_int(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{name} must be an integer >= {minimum}")
    return value


def _require_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number")
    return float(value)


def normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be an object")
    config = _deep_merge(DEFAULT_CONFIG, raw)

    threshold = _require_int(config["binarization_threshold"], "binarization_threshold")
    if threshold < 0 or threshold > 255:
        raise ConfigError("binarization_threshold must be between 0 and 255")

    minimum = config["block_min_size"]
    _require_int(minimum.get("width"), "block_min_size.width", 1)
    _require_int(minimum.get("height"), "block_min_size.height", 1)

    window = config["projection_window"]
    _require_int(window.get("horizontal"), "projection_window.horizontal", 0)
    _require_int(window.get("vertical"), "projection_window.vertical", 0)

    _require_int(config["column_gap_threshold"], "column_gap_threshold", 1)
    for key in (
        "column_candidate_max_width_ratio",
        "column_span_ratio",
        "column_span_min_overlap_ratio",
        "column_min_overlap_ratio",
    ):
        value = _require_number(config[key], key)
        if not 0 <= value <= 1:
            raise ConfigError(f"{key} must be in [0, 1]")

    role = config["role_detection"]
    ratio = role["title_body_height_ratio"]
    minimum_ratio = _require_number(ratio.get("min"), "title_body_height_ratio.min")
    maximum_ratio = _require_number(ratio.get("max"), "title_body_height_ratio.max")
    if minimum_ratio < 0 or maximum_ratio < minimum_ratio:
        raise ConfigError("title_body_height_ratio must satisfy 0 <= min <= max")
    image_density = _require_number(role["image_min_ink_density"], "image_min_ink_density")
    if not 0 <= image_density <= 1:
        raise ConfigError("image_min_ink_density must be in [0, 1]")
    periodicity = _require_number(
        role["image_max_projection_periodicity"],
        "image_max_projection_periodicity",
    )
    if periodicity < 0:
        raise ConfigError("image_max_projection_periodicity must be >= 0")
    margin_ratio = _require_number(
        role["header_footer_margin_ratio"], "header_footer_margin_ratio"
    )
    if not 0 <= margin_ratio < 0.5:
        raise ConfigError("header_footer_margin_ratio must be in [0, 0.5)")
    priority = role["role_priority"]
    allowed_roles = {"image", "title", "body", "header_footer"}
    if (
        not isinstance(priority, list)
        or set(priority) != allowed_roles
        or len(priority) != len(allowed_roles)
    ):
        raise ConfigError(
            "role_priority must be a permutation of "
            "['image', 'title', 'header_footer', 'body']"
        )

    if config["reading_order_rule"] not in {"column_first", "region_first"}:
        raise ConfigError("reading_order_rule must be column_first or region_first")
    _require_int(config["region_band_gap"], "region_band_gap", 0)
    workers = _require_int(config["thread_pool_workers"], "thread_pool_workers", 1)
    if workers > 64:
        raise ConfigError("thread_pool_workers must be <= 64")
    _require_int(config["pages_per_batch"], "pages_per_batch", 1)
    _require_int(config["snapshot_interval_pages"], "snapshot_interval_pages", 1)

    image_validation = config["image_validation"]
    _require_int(image_validation.get("width"), "image_validation.width", 1)
    _require_int(image_validation.get("height"), "image_validation.height", 1)
    modes = image_validation.get("allowed_modes")
    if not isinstance(modes, list) or not modes or not all(
        isinstance(mode, str) for mode in modes
    ):
        raise ConfigError("image_validation.allowed_modes must be a non-empty string list")

    return config


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return normalize_config({})
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return normalize_config(raw)


def config_hash(config: dict[str, Any]) -> str:
    return fingerprint(config)


def stage_parameters(config: dict[str, Any], stage: str) -> dict[str, Any]:
    try:
        keys = ANALYSIS_STAGE_KEYS[stage]
    except KeyError as exc:  # pragma: no cover - programming error
        raise ConfigError(f"unknown stage: {stage}") from exc
    return {key: copy.deepcopy(config[key]) for key in keys}

