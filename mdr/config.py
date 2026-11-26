from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .utils import fingerprint, read_json


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "seed": 400,
    "width": 96,
    "height": 72,
    "mode": "RGB",
    "frac_bits": 10,
    "allowed_frac_bits": [8, 10, 11, 12, 16],
    "quant_level": "Q1",
    "allowed_quant_levels": ["Q0", "Q1", "Q2"],
    "quant_shifts": {"Q0": 0, "Q1": 4, "Q2": 7},
    "feature_patch_radius": 2,
    "feature_score_threshold": 1400,
    "feature_nms_distance": 4,
    "descriptor_max_distance": 240,
    "min_parallax": 4,
    "voxel_size": 16,
    "residual_threshold": 1800,
    "depth_radius": 5,
    "depth_background_z": 220,
    "occupancy_min_tri": 1,
    "occupancy_min_depth": 6,
    "max_subdivision_depth": 2,
    "score_weights": {"residual": 1, "depth": 1},
    "diagnostic_tolerance_q": 32,
    "voxel_bounds": {
        "min_x": -24,
        "max_x": 40,
        "min_y": -16,
        "max_y": 16,
        "min_z": 40,
        "max_z": 200,
    },
    "budgets": {
        "max_workers": 4,
        "global_point_budget": 256,
        "view_point_quota": 96,
        "max_subdivisions_per_batch": 64,
    },
    "tie_rules": {
        "rounding": "half_step_ties_toward_zero",
        "depth_conflict": "minimum_depth_then_view_order_then_pixel_lex",
        "stable_order": "view_index_asc_then_x_asc_then_y_asc",
    },
}


class ConfigValidationError(ValueError):
    def __init__(self, errors: list[dict[str, str]]):
        super().__init__("; ".join(item["reason"] for item in errors))
        self.errors = errors


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def merge_defaults(raw: dict[str, Any] | None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if not raw:
        return config
    for key, value in raw.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    return config


def validate_config(config: dict[str, Any]) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []

    def require_positive_int(key: str, value: Any) -> None:
        if not _is_int(value) or value <= 0:
            errors.append({"ref": f"config.{key}", "reason": f"{key} must be a positive integer"})

    for key in (
        "width",
        "height",
        "frac_bits",
        "feature_patch_radius",
        "feature_score_threshold",
        "feature_nms_distance",
        "descriptor_max_distance",
        "min_parallax",
        "voxel_size",
        "residual_threshold",
        "depth_radius",
        "depth_background_z",
        "occupancy_min_tri",
        "occupancy_min_depth",
        "max_subdivision_depth",
        "diagnostic_tolerance_q",
    ):
        require_positive_int(key, config.get(key))

    if config.get("mode") not in {"RGB", "RGBA", "L"}:
        errors.append({"ref": "config.mode", "reason": "mode must be RGB, RGBA, or L"})

    allowed_bits = config.get("allowed_frac_bits", [])
    if not isinstance(allowed_bits, list) or config.get("frac_bits") not in allowed_bits:
        errors.append({"ref": "config.frac_bits", "reason": "frac_bits is not in allowed enum"})

    allowed_levels = config.get("allowed_quant_levels", [])
    level = config.get("quant_level")
    if not isinstance(allowed_levels, list) or level not in allowed_levels:
        errors.append({"ref": "config.quant_level", "reason": "quant_level is not in allowed enum"})

    shifts = config.get("quant_shifts", {})
    if not isinstance(shifts, dict) or level not in shifts or not _is_int(shifts[level]) or shifts[level] < 0:
        errors.append({"ref": "config.quant_shifts", "reason": "missing non-negative shift for quant_level"})

    bounds = config.get("voxel_bounds", {})
    for axis in ("x", "y", "z"):
        low = bounds.get(f"min_{axis}")
        high = bounds.get(f"max_{axis}")
        if not _is_int(low) or not _is_int(high) or low >= high:
            errors.append({"ref": f"config.voxel_bounds.{axis}", "reason": "bounds must be ordered integers"})

    budgets = config.get("budgets", {})
    for key in ("max_workers", "global_point_budget", "view_point_quota", "max_subdivisions_per_batch"):
        if not isinstance(budgets, dict) or not _is_int(budgets.get(key)) or budgets[key] <= 0:
            errors.append({"ref": f"config.budgets.{key}", "reason": "budget must be a positive integer"})

    weights = config.get("score_weights", {})
    for key in ("residual", "depth"):
        if not isinstance(weights, dict) or not _is_int(weights.get(key)) or weights[key] < 0:
            errors.append({"ref": f"config.score_weights.{key}", "reason": "weight must be a non-negative integer"})

    return errors


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    raw = read_json(path) if path else {}
    if not isinstance(raw, dict):
        raise ConfigValidationError([{"ref": str(path), "reason": "configuration root must be an object"}])
    config = merge_defaults(raw)
    errors = validate_config(config)
    if errors:
        raise ConfigValidationError(errors)
    return config


def config_fingerprint(config: dict[str, Any]) -> str:
    return fingerprint(config)


def quantization(config: dict[str, Any]) -> tuple[int, int, int]:
    bits = int(config["frac_bits"])
    level = str(config["quant_level"])
    shift = int(config["quant_shifts"][level])
    return bits, 1 << bits, shift
