"""Detection configuration loading and validation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .errors import ConfigurationError
from .util import load_structured, sha256_json


REQUIRED_BANDS = ["red", "green", "blue", "elevation"]


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigurationError(f"missing configuration key {context}.{key}")
    return mapping[key]


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer", {"value": value})
    return value


def validate_config(config: dict[str, Any], tile_sizes: list[int] | None = None) -> dict[str, Any]:
    window = _require(config, "window", "config")
    quantization = _require(config, "quantization", "config")
    detector = _require(config, "detector", "config")
    fusion = _require(config, "fusion", "config")
    scheduler = _require(config, "scheduler", "config")

    window_size = _positive_int(_require(window, "size", "window"), "window.size")
    step = _positive_int(_require(window, "step", "window"), "window.step")
    overlap = _positive_int(
        _require(window, "context_overlap", "window"), "window.context_overlap"
    )
    if window_size - 2 * overlap <= 0:
        raise ConfigurationError(
            "window size must exceed twice the context overlap",
            {"window_size": window_size, "context_overlap": overlap},
        )
    if step > window_size:
        raise ConfigurationError("window.step must not exceed window.size")
    if tile_sizes:
        for tile_size in sorted(set(tile_sizes)):
            if tile_size % window_size != 0:
                raise ConfigurationError(
                    "window.size must divide every tile edge",
                    {"window_size": window_size, "tile_size": tile_size},
                )

    fraction_bits = _positive_int(
        _require(quantization, "fraction_bits", "quantization"),
        "quantization.fraction_bits",
    )
    gear = _positive_int(_require(quantization, "gear", "quantization"), "quantization.gear")
    allowed_gears = _require(quantization, "allowed_gears", "quantization")
    allowed_bits = _require(quantization, "allowed_fraction_bits", "quantization")
    if not isinstance(allowed_gears, list) or gear not in allowed_gears:
        raise ConfigurationError(
            "quantization.gear is outside the configured set",
            {"gear": gear, "allowed_gears": allowed_gears},
        )
    if not isinstance(allowed_bits, list) or fraction_bits not in allowed_bits:
        raise ConfigurationError(
            "quantization.fraction_bits is outside the configured set",
            {"fraction_bits": fraction_bits, "allowed_fraction_bits": allowed_bits},
        )
    dimension_scales = _require(quantization, "dimension_scales_milli", "quantization")
    for dimension in (
        "vegetation_index",
        "normalized_water_index",
        "texture_gradient",
        "elevation_slope",
    ):
        _positive_int(dimension_scales.get(dimension), f"dimension_scales_milli.{dimension}")

    threshold = _require(detector, "threshold_permille", "detector")
    if isinstance(threshold, bool) or not isinstance(threshold, int) or not 0 <= threshold <= 1000:
        raise ConfigurationError("detector.threshold_permille must be in [0, 1000]")
    bandwidth = _positive_int(
        _require(detector, "kde_bandwidth_milli", "detector"), "detector.kde_bandwidth_milli"
    )
    distance_weight = _require(detector, "distance_weight_milli", "detector")
    density_weight = _require(detector, "density_weight_milli", "detector")
    if distance_weight + density_weight != 1000:
        raise ConfigurationError("detector distance and density weights must sum to 1000")
    _positive_int(_require(detector, "final_score_step_micro", "detector"), "final_score_step_micro")

    _require(fusion, "strategy_version", "fusion")
    overlap_width = _positive_int(_require(fusion, "overlap_width", "fusion"), "fusion.overlap_width")
    _positive_int(
        _require(fusion, "min_component_pixels", "fusion"), "fusion.min_component_pixels"
    )
    if overlap_width > window_size:
        raise ConfigurationError("fusion.overlap_width must not exceed window.size")

    max_workers = _positive_int(_require(scheduler, "max_workers", "scheduler"), "scheduler.max_workers")
    _positive_int(
        _require(scheduler, "global_window_budget", "scheduler"),
        "scheduler.global_window_budget",
    )
    _positive_int(
        _require(scheduler, "tile_window_quota", "scheduler"), "scheduler.tile_window_quota"
    )
    _positive_int(
        _require(scheduler, "fusion_recompute_limit", "scheduler"),
        "scheduler.fusion_recompute_limit",
    )

    bands = config.get("tile", {}).get("bands", REQUIRED_BANDS)
    if bands != REQUIRED_BANDS:
        raise ConfigurationError(
            "tile.bands must be red, green, blue, elevation in that order", {"bands": bands}
        )
    versions = config.get("tile", {}).get("band_set_versions")
    if not isinstance(versions, list) or not versions:
        raise ConfigurationError("tile.band_set_versions must be a non-empty list")
    if not isinstance(config.get("tile", {}).get("size"), int):
        raise ConfigurationError("tile.size must be an integer")
    if config.get("tile", {}).get("size") % window_size != 0:
        raise ConfigurationError("window.size must divide tile.size")

    validated = copy.deepcopy(config)
    validated["_derived"] = {
        "window_size": window_size,
        "step": step,
        "context_overlap": overlap,
        "core_size": window_size - 2 * overlap,
        "max_workers": min(max_workers, max(1, os_cpu_count())),
        "requested_max_workers": max_workers,
        "bandwidth_micro": bandwidth * 1000,
        "threshold_micro": threshold * 1000,
    }
    return validated


def os_cpu_count() -> int:
    import os

    return os.cpu_count() or 1


def load_config(path: Path, tile_sizes: list[int] | None = None) -> dict[str, Any]:
    raw = load_structured(path)
    if not isinstance(raw, dict):
        raise ConfigurationError("configuration root must be an object")
    return validate_config(raw, tile_sizes)


def config_hash(config: dict[str, Any]) -> str:
    payload = copy.deepcopy(config)
    payload.pop("_derived", None)
    return sha256_json(payload)


def window_parameter_key(config: dict[str, Any]) -> str:
    window = config["window"]
    return sha256_json(
        {
            "window_size": window["size"],
            "step": window["step"],
            "context_overlap": window["context_overlap"],
            "feature_version": "features-v1",
        }
    )


def quantization_key(config: dict[str, Any]) -> str:
    return sha256_json(
        {
            "fraction_bits": config["quantization"]["fraction_bits"],
            "gear": config["quantization"]["gear"],
            "dimension_scales_milli": config["quantization"]["dimension_scales_milli"],
            "quantization_version": "fixed-quant-v1",
        }
    )

