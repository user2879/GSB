"""Solver configuration loading and validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .errors import ConfigError
from .hashing import sha256_json


ENGINE_VERSION = "pose3d-engine/1.0.0"
SNAPSHOT_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
SUPPORTED_ROUNDING = {"round_half_even"}


@dataclass(frozen=True)
class SolverConfig:
    """Validated runtime configuration.

    The ``*_fingerprint`` methods split the configuration into the smallest
    useful stage boundaries.  For example, changing only the outlier threshold
    invalidates rejection/finalization but permits triangulation reuse.
    """

    image_dir: str = "views"
    camera_dir: str = "cameras"
    observations_file: str = "observations.jsonl"
    version_file: str = "input_version.json"

    min_views: int = 3
    min_disparity_deg: float = 1.0
    legal_min_disparity_max_deg: float = 90.0
    degenerate_relative_tolerance: float = 1e-10
    degenerate_policy: str = "fail"

    max_rounds: int = 3
    residual_threshold_px: float = 2.0
    legal_residual_threshold_max_px: float = 1_000_000.0
    min_inliers: int = 3

    quantization_levels: tuple[int, int] = (16, 17)
    rounding: str = "round_half_even"
    error_tolerance: float = 1e-3

    global_residual_budget: int = 100_000
    per_keypoint_quota: int = 5_000

    max_workers: int = 4
    snapshot_commit_interval: int = 1
    report_precision_digits: int = 15
    raw: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def primary_bits(self) -> int:
        return self.quantization_levels[0]

    @property
    def high_bits(self) -> int:
        return self.quantization_levels[1]

    @property
    def inputs_fingerprint(self) -> str:
        return sha256_json(
            {
                "image_dir": self.image_dir,
                "camera_dir": self.camera_dir,
                "observations_file": self.observations_file,
                "version_file": self.version_file,
                "min_views": self.min_views,
            }
        )

    @property
    def triangulation_fingerprint(self) -> str:
        return sha256_json(
            {
                "min_views": self.min_views,
                "min_disparity_deg": self.min_disparity_deg,
                "legal_min_disparity_max_deg": self.legal_min_disparity_max_deg,
                "degenerate_relative_tolerance": self.degenerate_relative_tolerance,
                "degenerate_policy": self.degenerate_policy,
            }
        )

    @property
    def rejection_fingerprint(self) -> str:
        return sha256_json(
            {
                "max_rounds": self.max_rounds,
                "residual_threshold_px": self.residual_threshold_px,
                "min_inliers": self.min_inliers,
            }
        )

    @property
    def quantization_fingerprint(self) -> str:
        return sha256_json(
            {
                "levels": list(self.quantization_levels),
                "rounding": self.rounding,
                "tolerance": self.error_tolerance,
            }
        )

    @property
    def budget_fingerprint(self) -> str:
        return sha256_json(
            {
                "global_residual_budget": self.global_residual_budget,
                "per_keypoint_quota": self.per_keypoint_quota,
            }
        )

    @property
    def scheduler_fingerprint(self) -> str:
        return sha256_json(
            {
                "max_workers": self.max_workers,
                "snapshot_commit_interval": self.snapshot_commit_interval,
            }
        )

    def replace_for_point(self, **changes: Any) -> "SolverConfig":
        """Return a point-local copy without mutating the base configuration."""

        raw = dict(self.raw)
        return replace(self, raw=raw, **changes)

    def describe(self) -> dict[str, Any]:
        return {
            "engine_version": ENGINE_VERSION,
            "image_dir": self.image_dir,
            "camera_dir": self.camera_dir,
            "observations_file": self.observations_file,
            "version_file": self.version_file,
            "min_views": self.min_views,
            "min_disparity_deg": self.min_disparity_deg,
            "legal_min_disparity_max_deg": self.legal_min_disparity_max_deg,
            "degenerate_relative_tolerance": self.degenerate_relative_tolerance,
            "degenerate_policy": self.degenerate_policy,
            "max_rounds": self.max_rounds,
            "residual_threshold_px": self.residual_threshold_px,
            "min_inliers": self.min_inliers,
            "quantization_levels": list(self.quantization_levels),
            "rounding": self.rounding,
            "error_tolerance": self.error_tolerance,
            "global_residual_budget": self.global_residual_budget,
            "per_keypoint_quota": self.per_keypoint_quota,
            "max_workers": self.max_workers,
            "snapshot_commit_interval": self.snapshot_commit_interval,
        }

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.describe())


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a JSON object")
    return value


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer")
    return value


def _as_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be a number")
    out = float(value)
    if not math.isfinite(out):
        raise ConfigError(f"{label} must be finite")
    return out


def parse_config(data: dict[str, Any]) -> SolverConfig:
    """Validate a decoded solver configuration object."""

    root = _require_mapping(data, "config")
    inputs = _require_mapping(root.get("inputs", {}), "inputs")
    triangulation = _require_mapping(root.get("triangulation", {}), "triangulation")
    rejection = _require_mapping(root.get("outlier_rejection", {}), "outlier_rejection")
    quantization = _require_mapping(root.get("quantization", {}), "quantization")
    budget = _require_mapping(root.get("budget", {}), "budget")
    scheduler = _require_mapping(root.get("scheduler", {}), "scheduler")

    min_views = _as_int(triangulation.get("min_views", 3), "triangulation.min_views")
    min_disparity = _as_float(
        triangulation.get("min_disparity_deg", 1.0), "triangulation.min_disparity_deg"
    )
    legal_max_disparity = _as_float(
        triangulation.get("legal_min_disparity_max_deg", 90.0),
        "triangulation.legal_min_disparity_max_deg",
    )
    degenerate_tolerance = _as_float(
        triangulation.get("degenerate_relative_tolerance", 1e-10),
        "triangulation.degenerate_relative_tolerance",
    )
    degenerate_policy = str(triangulation.get("degenerate_policy", "fail"))

    max_rounds = _as_int(rejection.get("max_rounds", 3), "outlier_rejection.max_rounds")
    threshold = _as_float(
        rejection.get("threshold_px", 2.0), "outlier_rejection.threshold_px"
    )
    legal_threshold_max = _as_float(
        rejection.get("legal_threshold_max_px", 1_000_000.0),
        "outlier_rejection.legal_threshold_max_px",
    )
    min_inliers = _as_int(rejection.get("min_inliers", 3), "outlier_rejection.min_inliers")

    levels_raw = quantization.get("levels", [16, 17])
    if not isinstance(levels_raw, list) or len(levels_raw) != 2:
        raise ConfigError("quantization.levels must contain exactly two integer widths")
    levels = tuple(_as_int(v, f"quantization.levels[{i}]") for i, v in enumerate(levels_raw))
    rounding = str(quantization.get("rounding", "round_half_even"))
    error_tolerance = _as_float(
        quantization.get("error_tolerance", 1e-3), "quantization.error_tolerance"
    )

    global_budget = _as_int(
        budget.get("global_residual_budget", 100_000),
        "budget.global_residual_budget",
    )
    per_point_quota = _as_int(
        budget.get("per_keypoint_quota", 5_000), "budget.per_keypoint_quota"
    )
    budget_unit = str(budget.get("unit", "reprojection_evaluation"))
    if budget_unit != "reprojection_evaluation":
        raise ConfigError(
            "budget.unit must be 'reprojection_evaluation'; the sample ledger "
            "counts each active observation residual evaluation"
        )

    max_workers = _as_int(scheduler.get("max_workers", 4), "scheduler.max_workers")
    commit_interval = _as_int(
        scheduler.get("snapshot_commit_interval", 1),
        "scheduler.snapshot_commit_interval",
    )
    report_precision = _as_int(
        scheduler.get("report_precision_digits", 15),
        "scheduler.report_precision_digits",
    )

    if min_views < 2:
        raise ConfigError("triangulation.min_views must be at least 2")
    if min_disparity < 0:
        raise ConfigError("triangulation.min_disparity_deg must be non-negative")
    if legal_max_disparity <= 0:
        raise ConfigError("triangulation.legal_min_disparity_max_deg must be positive")
    if min_disparity > legal_max_disparity:
        raise ConfigError(
            "triangulation.min_disparity_deg exceeds legal_min_disparity_max_deg"
        )
    if degenerate_tolerance <= 0:
        raise ConfigError("triangulation.degenerate_relative_tolerance must be positive")
    if degenerate_policy != "fail":
        raise ConfigError("triangulation.degenerate_policy currently supports 'fail'")
    if max_rounds < 0:
        raise ConfigError("outlier_rejection.max_rounds must be non-negative")
    if threshold <= 0:
        raise ConfigError("outlier_rejection.threshold_px must be greater than zero")
    if threshold > legal_threshold_max:
        raise ConfigError(
            "outlier_rejection.threshold_px exceeds legal_threshold_max_px"
        )
    if min_inliers < 2:
        raise ConfigError("outlier_rejection.min_inliers must be at least 2")
    if levels[0] < 0 or levels[1] != levels[0] + 1:
        raise ConfigError(
            "quantization.levels must be [primary_bits, primary_bits + 1]"
        )
    if rounding not in SUPPORTED_ROUNDING:
        raise ConfigError(f"unsupported quantization.rounding: {rounding}")
    if error_tolerance <= 0:
        raise ConfigError("quantization.error_tolerance must be positive")
    if global_budget < 0:
        raise ConfigError("budget.global_residual_budget must be non-negative")
    if per_point_quota <= 0:
        raise ConfigError("budget.per_keypoint_quota must be positive")
    if max_workers < 1:
        raise ConfigError("scheduler.max_workers must be at least 1")
    if commit_interval < 1:
        raise ConfigError("scheduler.snapshot_commit_interval must be at least 1")
    if report_precision < 6:
        raise ConfigError("scheduler.report_precision_digits must be at least 6")

    cfg = SolverConfig(
        image_dir=str(inputs.get("image_dir", "views")),
        camera_dir=str(inputs.get("camera_dir", "cameras")),
        observations_file=str(inputs.get("observations_file", "observations.jsonl")),
        version_file=str(inputs.get("version_file", "input_version.json")),
        min_views=min_views,
        min_disparity_deg=min_disparity,
        legal_min_disparity_max_deg=legal_max_disparity,
        degenerate_relative_tolerance=degenerate_tolerance,
        degenerate_policy=degenerate_policy,
        max_rounds=max_rounds,
        residual_threshold_px=threshold,
        legal_residual_threshold_max_px=legal_threshold_max,
        min_inliers=min_inliers,
        quantization_levels=levels,
        rounding=rounding,
        error_tolerance=error_tolerance,
        global_residual_budget=global_budget,
        per_keypoint_quota=per_point_quota,
        max_workers=max_workers,
        snapshot_commit_interval=commit_interval,
        report_precision_digits=report_precision,
        raw=dict(root),
    )
    return cfg


def load_config(path: str | Path) -> SolverConfig:
    config_path = Path(path)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read solver config {config_path}: {exc}") from exc
    return parse_config(data)
