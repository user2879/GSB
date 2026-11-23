"""YAML configuration loading and strict validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .stable import public_failure, sha256_json, sort_failures


ALLOWED_QUANTIZATION_LEVELS = (1, 2, 4, 8, 16, 32)
ALLOWED_PRECISION_LEVELS = (4, 6, 8, 10, 12, 16)
ALLOWED_BLOCK_SIZES = (4, 8, 16)


@dataclass(frozen=True)
class MotionModel:
    linear_gain: float
    damping: float
    search_radius: int


@dataclass(frozen=True)
class NoiseLayer:
    amplitude: float
    seed_salt: int


@dataclass(frozen=True)
class RenderConfig:
    frame_width: int
    frame_height: int
    prediction_steps: int
    history_frames: int
    motion_model: MotionModel
    noise_layer: NoiseLayer
    fixed_point_bits: int
    quantization_levels: tuple[int, ...]
    precision_levels: tuple[int, ...]
    scale_tiers: tuple[int, ...]
    latency_budget_ms: int
    thread_pool_workers: int
    snapshot_commit_interval_batches: int
    config_hash: str

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "prediction_steps": self.prediction_steps,
            "history_frames": self.history_frames,
            "motion_model": {
                "linear_gain": self.motion_model.linear_gain,
                "damping": self.motion_model.damping,
                "search_radius": self.motion_model.search_radius,
            },
            "noise_layer": {
                "amplitude": self.noise_layer.amplitude,
                "seed_salt": self.noise_layer.seed_salt,
            },
            "fixed_point_bits": self.fixed_point_bits,
            "quantization_levels": list(self.quantization_levels),
            "precision_levels": list(self.precision_levels),
            "scale_tiers": list(self.scale_tiers),
            "latency_budget_ms": self.latency_budget_ms,
            "thread_pool_workers": self.thread_pool_workers,
            "snapshot_commit_interval_batches": self.snapshot_commit_interval_batches,
        }


@dataclass
class ConfigLoadResult:
    config: RenderConfig | None
    failures: list[dict[str, Any]]
    source: dict[str, Any] | None

    @property
    def ok(self) -> bool:
        return self.config is not None and not self.failures


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _config_failure(field: str, message: str) -> dict[str, Any]:
    return public_failure("E_CONFIG_INVALID", "__config__", message, field=field)


def _read_mapping(value: Any, field: str, failures: list[dict[str, Any]]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    failures.append(_config_failure(field, f"{field} must be a mapping"))
    return {}


def _positive_int(
    mapping: dict[str, Any],
    field: str,
    failures: list[dict[str, Any]],
    *,
    default: int | None = None,
) -> int | None:
    if field not in mapping:
        if default is not None:
            return default
        failures.append(_config_failure(field, f"{field} is required"))
        return None
    value = mapping[field]
    if not _is_int(value) or value <= 0:
        failures.append(_config_failure(field, f"{field} must be a positive integer"))
        return None
    return value


def _finite_number(
    mapping: dict[str, Any],
    field: str,
    failures: list[dict[str, Any]],
    *,
    default: float | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if field not in mapping:
        if default is not None:
            value = default
        else:
            failures.append(_config_failure(field, f"{field} is required"))
            return None
    else:
        value = mapping[field]
    if not _is_number(value):
        failures.append(_config_failure(field, f"{field} must be a finite number"))
        return None
    value = float(value)
    if minimum is not None and value < minimum:
        failures.append(_config_failure(field, f"{field} must be >= {minimum}"))
        return None
    if maximum is not None and value > maximum:
        failures.append(_config_failure(field, f"{field} must be <= {maximum}"))
        return None
    return value


def _integer_enum_list(
    mapping: dict[str, Any],
    field: str,
    allowed: tuple[int, ...],
    failures: list[dict[str, Any]],
) -> tuple[int, ...] | None:
    value = mapping.get(field)
    if not isinstance(value, list) or not value:
        failures.append(_config_failure(field, f"{field} must be a non-empty list"))
        return None
    result: list[int] = []
    for index, item in enumerate(value):
        if not _is_int(item) or item not in allowed:
            failures.append(
                _config_failure(
                    f"{field}[{index}]",
                    f"{field}[{index}]={item!r} is not in allowed enum {list(allowed)}",
                )
            )
            continue
        result.append(int(item))
    return tuple(result) if len(result) == len(value) else None


def _extract_dimensions(source: dict[str, Any]) -> tuple[Any, Any]:
    width = source.get("frame_width")
    height = source.get("frame_height")
    frame_size = source.get("frame_size")
    if isinstance(frame_size, dict):
        width = width if width is not None else frame_size.get("width")
        height = height if height is not None else frame_size.get("height")
    elif isinstance(frame_size, (list, tuple)) and len(frame_size) == 2:
        width = width if width is not None else frame_size[0]
        height = height if height is not None else frame_size[1]
    return width, height


def parse_config(source: Any) -> ConfigLoadResult:
    failures: list[dict[str, Any]] = []
    if not isinstance(source, dict):
        return ConfigLoadResult(
            config=None,
            failures=[_config_failure("$", "configuration root must be a mapping")],
            source=None,
        )

    width_value, height_value = _extract_dimensions(source)
    width = None
    height = None
    if not _is_int(width_value) or width_value <= 0:
        failures.append(_config_failure("frame_width", "frame_width must be a positive integer"))
    else:
        width = int(width_value)
    if not _is_int(height_value) or height_value <= 0:
        failures.append(_config_failure("frame_height", "frame_height must be a positive integer"))
    else:
        height = int(height_value)

    prediction_steps = _positive_int(source, "prediction_steps", failures)
    history_frames = _positive_int(source, "history_frames", failures, default=4)
    fixed_point_bits = _positive_int(source, "fixed_point_bits", failures)
    latency_budget_ms = _positive_int(source, "latency_budget_ms", failures)
    workers = _positive_int(source, "thread_pool_workers", failures)
    commit_interval = _positive_int(source, "snapshot_commit_interval_batches", failures)

    motion_source = _read_mapping(source.get("motion_model"), "motion_model", failures)
    linear_gain = _finite_number(
        motion_source, "linear_gain", failures, minimum=0.0
    )
    damping = _finite_number(
        motion_source, "damping", failures, minimum=0.0, maximum=1.0
    )
    search_radius = _positive_int(motion_source, "search_radius", failures)

    noise_source = _read_mapping(source.get("noise_layer"), "noise_layer", failures)
    amplitude = _finite_number(noise_source, "amplitude", failures, minimum=0.0)
    seed_salt_value = noise_source.get("seed_salt", 0)
    seed_salt = None
    if not _is_int(seed_salt_value):
        failures.append(_config_failure("noise_layer.seed_salt", "seed_salt must be an integer"))
    else:
        seed_salt = int(seed_salt_value)

    quantization = _integer_enum_list(
        source, "quantization_levels", ALLOWED_QUANTIZATION_LEVELS, failures
    )
    precision = _integer_enum_list(source, "precision_levels", ALLOWED_PRECISION_LEVELS, failures)

    tiers_value = source.get("scale_tiers")
    tiers: tuple[int, ...] | None = None
    if not isinstance(tiers_value, list) or not tiers_value:
        failures.append(_config_failure("scale_tiers", "scale_tiers must be a non-empty list"))
    else:
        parsed_tiers: list[int] = []
        valid = True
        for index, item in enumerate(tiers_value):
            if not _is_int(item) or item <= 0:
                failures.append(
                    _config_failure(
                        f"scale_tiers[{index}]", "each scale tier must be a positive integer"
                    )
                )
                valid = False
            else:
                parsed_tiers.append(int(item))
        if valid and any(b <= a for a, b in zip(parsed_tiers, parsed_tiers[1:])):
            failures.append(_config_failure("scale_tiers", "scale_tiers must be strictly increasing"))
            valid = False
        if valid:
            tiers = tuple(parsed_tiers)

    if precision is not None and fixed_point_bits is not None:
        if min(precision) > fixed_point_bits:
            failures.append(
                _config_failure(
                    "precision_levels",
                    "at least one precision level must be <= fixed_point_bits",
                )
            )

    failures = sort_failures(failures)
    if failures:
        return ConfigLoadResult(config=None, failures=failures, source=dict(source))

    assert width is not None and height is not None
    assert prediction_steps is not None and history_frames is not None
    assert fixed_point_bits is not None and latency_budget_ms is not None
    assert workers is not None and commit_interval is not None
    assert linear_gain is not None and damping is not None and search_radius is not None
    assert amplitude is not None and seed_salt is not None
    assert quantization is not None and precision is not None and tiers is not None

    canonical_source = {
        "frame_width": width,
        "frame_height": height,
        "prediction_steps": prediction_steps,
        "history_frames": history_frames,
        "motion_model": {
            "linear_gain": linear_gain,
            "damping": damping,
            "search_radius": search_radius,
        },
        "noise_layer": {"amplitude": amplitude, "seed_salt": seed_salt},
        "fixed_point_bits": fixed_point_bits,
        "quantization_levels": list(quantization),
        "precision_levels": list(precision),
        "scale_tiers": list(tiers),
        "latency_budget_ms": latency_budget_ms,
        "thread_pool_workers": workers,
        "snapshot_commit_interval_batches": commit_interval,
    }
    config = RenderConfig(
        frame_width=width,
        frame_height=height,
        prediction_steps=prediction_steps,
        history_frames=history_frames,
        motion_model=MotionModel(linear_gain, damping, search_radius),
        noise_layer=NoiseLayer(amplitude, seed_salt),
        fixed_point_bits=fixed_point_bits,
        quantization_levels=tuple(quantization),
        precision_levels=tuple(precision),
        scale_tiers=tuple(tiers),
        latency_budget_ms=latency_budget_ms,
        thread_pool_workers=workers,
        snapshot_commit_interval_batches=commit_interval,
        config_hash=sha256_json(canonical_source),
    )
    return ConfigLoadResult(config=config, failures=[], source=dict(source))


def load_config(path: str | Path) -> ConfigLoadResult:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        return ConfigLoadResult(
            config=None,
            failures=[_config_failure("$", f"cannot read config: {exc}")],
            source=None,
        )
    try:
        source = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return ConfigLoadResult(
            config=None,
            failures=[_config_failure("$", f"invalid YAML: {exc}")],
            source=None,
        )
    return parse_config(source)
