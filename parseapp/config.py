"""Configuration loading, validation and deterministic fingerprints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when the parse configuration is invalid."""

    code = "E_CONFIG_INVALID"


_ALIASES = {
    "thread_pool_size": "thread_pool_concurrency",
    "thread_pool_count": "thread_pool_concurrency",
    "max_workers": "thread_pool_concurrency",
    "concurrency": "thread_pool_concurrency",
    "batch_detection_points": "batch_detection_point_budget",
    "batch_detection_budget": "batch_detection_point_budget",
    "batch_detection_points_budget": "batch_detection_point_budget",
    "detection_point_budget": "batch_detection_point_budget",
    "batch_point_budget": "batch_detection_point_budget",
    "chart_point_quota": "page_chart_point_quota",
    "per_page_chart_point_quota": "page_chart_point_quota",
    "chart_points_per_page": "page_chart_point_quota",
    "snapshot_interval": "snapshot_commit_interval",
    "snapshot_every": "snapshot_commit_interval",
    "normalization_mode": "coordinate_normalization",
    "chart_coordinate_normalization": "coordinate_normalization",
    "image_width": "expected_image_width",
    "image_height": "expected_image_height",
    "pixel_mode": "expected_pixel_mode",
}

_DEFAULTS: dict[str, Any] = {
    "expected_image_width": None,
    "expected_image_height": None,
    "expected_pixel_mode": None,
    "binary_threshold": 160,
    "min_line_length": 40,
    "line_merge_tolerance": 2,
    "cell_min_area": 100,
    "chart_decision_threshold": 0.55,
    "coordinate_normalization": "minmax",
    "thread_pool_concurrency": 4,
    "batch_detection_point_budget": 5000,
    "page_chart_point_quota": 20,
    "snapshot_commit_interval": 2,
}

_VALID_MODES = {"L8", "LA8", "RGB8", "RGBA8", "P8"}


@dataclass(frozen=True)
class ParseConfig:
    values: Mapping[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def as_dict(self) -> dict[str, Any]:
        return dict(self.values)

    def canonical_json(self) -> str:
        return json.dumps(
            self.values, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )

    @property
    def config_hash(self) -> str:
        return _sha256_text(self.canonical_json())

    def stage_params(self, stage: str) -> dict[str, Any]:
        if stage == "line":
            keys = (
                "binary_threshold",
                "min_line_length",
                "line_merge_tolerance",
            )
        elif stage == "grid":
            keys = ("cell_min_area",)
        elif stage == "chart":
            keys = (
                "binary_threshold",
                "chart_decision_threshold",
                "coordinate_normalization",
                "page_chart_point_quota",
            )
        else:
            raise ValueError(f"unknown stage: {stage}")
        return {key: self.values[key] for key in keys}

    def stage_param_hash(self, stage: str) -> str:
        payload = json.dumps(
            self.stage_params(stage),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return _sha256_text(payload)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load_config(path: str | Path | Mapping[str, Any]) -> ParseConfig:
    if isinstance(path, Mapping):
        raw = dict(path)
    else:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"cannot read config: {exc}") from exc
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"config is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a JSON object")

    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        canonical = _ALIASES.get(key, key)
        if canonical in normalized and normalized[canonical] != value:
            raise ConfigError(f"conflicting aliases for {canonical}")
        normalized[canonical] = value

    values = dict(_DEFAULTS)
    values.update(normalized)
    normalization = values["coordinate_normalization"]
    if normalization in {"min_max", "axis_minmax"}:
        values["coordinate_normalization"] = "minmax"
    elif normalization in {"pixel", "pixel_ratio", "axis_relative"}:
        values["coordinate_normalization"] = "pixel_relative"
    mode = values["expected_pixel_mode"]
    mode_aliases = {
        "L": "L8",
        "LA": "LA8",
        "RGB": "RGB8",
        "RGBA": "RGBA8",
        "P": "P8",
    }
    if isinstance(mode, str):
        values["expected_pixel_mode"] = mode_aliases.get(mode, mode)
    _validate_values(values)
    return ParseConfig(values={key: values[key] for key in sorted(values)})


def _validate_values(values: dict[str, Any]) -> None:
    threshold = values["binary_threshold"]
    if not _is_int(threshold) or not 0 <= threshold <= 255:
        raise ConfigError("binary_threshold must be an integer in [0, 255]")

    if not _is_int(values["min_line_length"]) or values["min_line_length"] < 2:
        raise ConfigError("min_line_length must be an integer >= 2")
    if not _is_int(values["line_merge_tolerance"]) or values["line_merge_tolerance"] < 0:
        raise ConfigError("line_merge_tolerance must be an integer >= 0")
    if not _is_int(values["cell_min_area"]) or values["cell_min_area"] < 0:
        raise ConfigError("cell_min_area must be an integer >= 0")

    chart_threshold = values["chart_decision_threshold"]
    if (
        isinstance(chart_threshold, bool)
        or not isinstance(chart_threshold, (int, float))
        or not 0 <= float(chart_threshold) <= 1
    ):
        raise ConfigError("chart_decision_threshold must be a number in [0, 1]")
    values["chart_decision_threshold"] = float(chart_threshold)

    if values["coordinate_normalization"] not in {"minmax", "pixel_relative"}:
        raise ConfigError(
            "coordinate_normalization must be 'minmax' or 'pixel_relative'"
        )

    concurrency = values["thread_pool_concurrency"]
    if not _is_int(concurrency) or not 1 <= concurrency <= 256:
        raise ConfigError("thread_pool_concurrency must be an integer in [1, 256]")
    if (
        not _is_int(values["batch_detection_point_budget"])
        or values["batch_detection_point_budget"] < 0
    ):
        raise ConfigError("batch_detection_point_budget must be an integer >= 0")
    if (
        not _is_int(values["page_chart_point_quota"])
        or values["page_chart_point_quota"] < 0
    ):
        raise ConfigError("page_chart_point_quota must be an integer >= 0")
    if not _is_int(values["snapshot_commit_interval"]) or values[
        "snapshot_commit_interval"
    ] < 1:
        raise ConfigError("snapshot_commit_interval must be an integer >= 1")

    for key in ("expected_image_width", "expected_image_height"):
        value = values[key]
        if value is not None and (not _is_int(value) or value <= 0):
            raise ConfigError(f"{key} must be a positive integer or null")

    mode = values["expected_pixel_mode"]
    if mode is not None and mode not in _VALID_MODES:
        raise ConfigError(
            f"expected_pixel_mode must be one of {sorted(_VALID_MODES)} or null"
        )
