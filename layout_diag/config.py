"""Configuration loading, validation, and effective tier expansion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import (
    DEFAULT_STRATEGIES,
    FIXED_BITS_ALLOWED,
    FIXED_PIXEL_MODE,
    FIXED_RESOLUTION,
    QUANT_STEPS_ALLOWED,
)
from .models import FailureLog
from .utils import read_json, stable_hash


@dataclass(frozen=True)
class WindowSpec:
    width: int
    height: int
    step_x: int
    step_y: int

    def as_dict(self) -> dict[str, int]:
        return {
            "width": self.width,
            "height": self.height,
            "step_x": self.step_x,
            "step_y": self.step_y,
        }


@dataclass(frozen=True)
class CCSpec:
    min_area: int
    max_aspect: float
    merge_gap: int
    split_ratio: float
    ink_threshold: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_area": self.min_area,
            "max_aspect": self.max_aspect,
            "merge_gap": self.merge_gap,
            "split_ratio": self.split_ratio,
            "ink_threshold": self.ink_threshold,
        }


@dataclass(frozen=True)
class PrecisionSpec:
    fixed_bits: int
    quant_step: int

    def as_dict(self) -> dict[str, int]:
        return {"fixed_bits": self.fixed_bits, "quant_step": self.quant_step}


@dataclass(frozen=True)
class StrategySpec:
    name: str
    window: WindowSpec
    merge_gap: int
    precision: PrecisionSpec
    capacity_points: int
    capacity_columns: int
    capacity_chars_per_column: int
    single_image_cost: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "window": self.window.as_dict(),
            "merge_gap": self.merge_gap,
            "precision": self.precision.as_dict(),
            "capacity_points": self.capacity_points,
            "capacity_columns": self.capacity_columns,
            "capacity_chars_per_column": self.capacity_chars_per_column,
            "single_image_cost": self.single_image_cost,
        }


@dataclass(frozen=True)
class Config:
    path: str
    raw: dict[str, Any]
    resolution: tuple[int, int]
    pixel_mode: str
    tier: str
    window: WindowSpec
    cc: CCSpec
    precision: PrecisionSpec
    max_workers: int
    point_budget: int
    keypoint_version: str
    strategies: dict[str, StrategySpec]

    def content_hash(self) -> str:
        return stable_hash(self.raw)


@dataclass(frozen=True)
class EffectiveConfig:
    tier: str
    tier_explicit: bool
    resolution: tuple[int, int]
    pixel_mode: str
    window: WindowSpec
    cc: CCSpec
    precision: PrecisionSpec
    max_workers: int
    point_budget: int
    keypoint_version: str
    strategy: StrategySpec

    def params_hash(self) -> str:
        return stable_hash(
            {
                "tier": self.tier,
                "resolution": self.resolution,
                "pixel_mode": self.pixel_mode,
                "window": self.window.as_dict(),
                "cc": self.cc.as_dict(),
                "precision": self.precision.as_dict(),
                "keypoint_version": self.keypoint_version,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "tier_explicit": self.tier_explicit,
            "resolution": {"width": self.resolution[0], "height": self.resolution[1]},
            "pixel_mode": self.pixel_mode,
            "window": self.window.as_dict(),
            "cc": self.cc.as_dict(),
            "precision": self.precision.as_dict(),
            "max_workers": self.max_workers,
            "point_budget": self.point_budget,
            "keypoint_version": self.keypoint_version,
        }


def _int(raw: Any, default: int) -> int:
    return raw if isinstance(raw, int) and not isinstance(raw, bool) else default


def _number(raw: Any, default: float) -> float:
    return float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else default


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _read_window(raw: Any) -> WindowSpec:
    if not isinstance(raw, dict):
        raw = {}
    return WindowSpec(
        width=_int(raw.get("width"), 64),
        height=_int(raw.get("height"), 64),
        step_x=_int(raw.get("step_x"), 32),
        step_y=_int(raw.get("step_y"), 32),
    )


def load_config(path: str | Path, failures: FailureLog | None = None) -> Config:
    """Load a config and append all detectable validation failures."""

    if failures is None:
        failures = FailureLog()
    path = Path(path)
    try:
        raw = read_json(path)
        if not isinstance(raw, dict):
            raise ValueError("root is not an object")
    except Exception as exc:  # pragma: no cover - exact OS error is platform-specific
        failures.add("E_CONFIG_VALUE", "config", str(path), f"cannot load config: {exc}")
        raw = {}

    resolution_raw = raw.get("resolution", {})
    if "resolution" in raw and not isinstance(resolution_raw, dict):
        failures.add("E_CONFIG_VALUE", "config", "resolution", "resolution must be an object")
        resolution_raw = {}
    for key in ("width", "height"):
        if isinstance(resolution_raw, dict) and key in resolution_raw and not _is_int(resolution_raw.get(key)):
            failures.add("E_CONFIG_VALUE", "config", f"resolution.{key}", "resolution values must be integers")
    width = _int(resolution_raw.get("width") if isinstance(resolution_raw, dict) else None, FIXED_RESOLUTION[0])
    height = _int(resolution_raw.get("height") if isinstance(resolution_raw, dict) else None, FIXED_RESOLUTION[1])
    if (width, height) != FIXED_RESOLUTION:
        failures.add(
            "E_CONFIG_VALUE",
            "config",
            "resolution",
            "resolution must equal the fixed synthesis resolution",
            expected=list(FIXED_RESOLUTION),
            actual=[width, height],
        )

    pixel_mode = raw.get("pixel_mode", FIXED_PIXEL_MODE)
    if pixel_mode != FIXED_PIXEL_MODE:
        failures.add(
            "E_CONFIG_VALUE",
            "config",
            "pixel_mode",
            "pixel mode must be L",
            expected=FIXED_PIXEL_MODE,
            actual=pixel_mode,
        )

    tier = raw.get("tier", "auto")
    if not isinstance(tier, str):
        failures.add("E_TIER", "config", "tier", "tier must be a string", actual=tier)
        tier = "auto"

    raw_window = raw.get("window")
    if "window" in raw and not isinstance(raw_window, dict):
        failures.add("E_WINDOW_ALIGN", "config", "window", "window must be an object")
        raw_window = {}
    if isinstance(raw_window, dict):
        for key in ("width", "height", "step_x", "step_y"):
            if key in raw_window and not _is_int(raw_window.get(key)):
                failures.add("E_WINDOW_ALIGN", "config", f"window.{key}", "window values must be integers")
    window = _read_window(raw_window)
    for key, value in (
        ("width", window.width),
        ("height", window.height),
        ("step_x", window.step_x),
        ("step_y", window.step_y),
    ):
        if value <= 0:
            failures.add("E_WINDOW_ALIGN", "config", f"window.{key}", "value must be positive", actual=value)
    if window.width > 0 and width % window.width != 0:
        failures.add(
            "E_WINDOW_ALIGN",
            "config",
            "window.width",
            "window width must divide image width",
            window=window.width,
            image=width,
        )
    if window.height > 0 and height % window.height != 0:
        failures.add(
            "E_WINDOW_ALIGN",
            "config",
            "window.height",
            "window height must divide image height",
            window=window.height,
            image=height,
        )
    if window.step_x > window.width or window.step_y > window.height:
        failures.add("E_WINDOW_ALIGN", "config", "window.step", "step cannot exceed window size")

    cc_raw = raw.get("cc", {})
    if "cc" in raw and not isinstance(cc_raw, dict):
        failures.add("E_CONFIG_VALUE", "config", "cc", "cc must be an object")
        cc_raw = {}
    if isinstance(cc_raw, dict):
        for key in ("min_area", "merge_gap", "ink_threshold"):
            if key in cc_raw and not _is_int(cc_raw.get(key)):
                failures.add("E_CONFIG_VALUE", "config", f"cc.{key}", "value must be an integer")
        for key in ("max_aspect", "split_ratio"):
            if key in cc_raw and not _is_number(cc_raw.get(key)):
                failures.add("E_CONFIG_VALUE", "config", f"cc.{key}", "value must be numeric")
    min_area = _int(cc_raw.get("min_area"), 10)
    max_aspect = _number(cc_raw.get("max_aspect"), 3.2)
    merge_gap = _int(cc_raw.get("merge_gap"), 2)
    split_ratio = _number(cc_raw.get("split_ratio"), 1.9)
    ink_threshold = _int(cc_raw.get("ink_threshold"), 192)
    if min_area <= 0:
        failures.add("E_CONFIG_VALUE", "config", "cc.min_area", "area lower bound must be positive", actual=min_area)
    if max_aspect <= 0:
        failures.add("E_CONFIG_VALUE", "config", "cc.max_aspect", "aspect upper bound must be positive", actual=max_aspect)
    if merge_gap < 0:
        failures.add("E_CONFIG_VALUE", "config", "cc.merge_gap", "merge gap cannot be negative", actual=merge_gap)
    if split_ratio <= 1:
        failures.add("E_CONFIG_VALUE", "config", "cc.split_ratio", "split ratio must exceed one", actual=split_ratio)
    if not 0 < ink_threshold < 256:
        failures.add("E_CONFIG_VALUE", "config", "cc.ink_threshold", "ink threshold must be in 1..255", actual=ink_threshold)

    precision_raw = raw.get("precision", {})
    if "precision" in raw and not isinstance(precision_raw, dict):
        failures.add("E_QUANT_ENUM", "config", "precision", "precision must be an object")
        precision_raw = {}
    if isinstance(precision_raw, dict):
        for key in ("fixed_bits", "quant_step"):
            if key in precision_raw and not _is_int(precision_raw.get(key)):
                failures.add("E_QUANT_ENUM", "config", f"precision.{key}", "quantization values must be integers")
    fixed_bits = _int(precision_raw.get("fixed_bits"), 8)
    quant_step = _int(precision_raw.get("quant_step"), 2)
    if fixed_bits not in FIXED_BITS_ALLOWED:
        failures.add(
            "E_QUANT_ENUM",
            "config",
            "precision.fixed_bits",
            "fixed point fractional width is outside the allowed enumeration",
            allowed=list(FIXED_BITS_ALLOWED),
            actual=fixed_bits,
        )
    if quant_step not in QUANT_STEPS_ALLOWED:
        failures.add(
            "E_QUANT_ENUM",
            "config",
            "precision.quant_step",
            "quantization step is outside the allowed enumeration",
            allowed=list(QUANT_STEPS_ALLOWED),
            actual=quant_step,
        )

    if "max_workers" in raw and not _is_int(raw.get("max_workers")):
        failures.add("E_CONFIG_VALUE", "config", "max_workers", "max_workers must be an integer")
    if "point_budget" in raw and not _is_int(raw.get("point_budget")):
        failures.add("E_CONFIG_VALUE", "config", "point_budget", "point_budget must be an integer")
    max_workers = _int(raw.get("max_workers"), 4)
    point_budget = _int(raw.get("point_budget"), 96)
    if max_workers <= 0:
        failures.add("E_CONFIG_VALUE", "config", "max_workers", "max_workers must be positive", actual=max_workers)
    if point_budget < 0:
        failures.add("E_CONFIG_VALUE", "config", "point_budget", "point_budget cannot be negative", actual=point_budget)

    keypoint_version = raw.get("keypoint_version", "v1")
    if not isinstance(keypoint_version, str) or not keypoint_version:
        failures.add("E_KEYPOINT_VERSION", "config", "keypoint_version", "active keypoint version must be a non-empty string")
        keypoint_version = "v1"

    strategies = _load_strategies(raw.get("strategies"), failures)
    if tier not in ("auto", "custom", *strategies.keys()):
        failures.add(
            "E_TIER",
            "config",
            "tier",
            "unknown scale tier",
            allowed=["auto", "custom", *sorted(strategies.keys())],
            actual=tier,
        )

    return Config(
        path=str(path),
        raw=raw,
        resolution=(width, height),
        pixel_mode=pixel_mode,
        tier=tier,
        window=window,
        cc=CCSpec(min_area, max_aspect, merge_gap, split_ratio, ink_threshold),
        precision=PrecisionSpec(fixed_bits, quant_step),
        max_workers=max_workers,
        point_budget=point_budget,
        keypoint_version=keypoint_version,
        strategies=strategies,
    )


def _load_strategies(raw: Any, failures: FailureLog) -> dict[str, StrategySpec]:
    source = raw if isinstance(raw, dict) and raw else DEFAULT_STRATEGIES
    strategies: dict[str, StrategySpec] = {}
    for name in ("small", "medium", "large"):
        entry = source.get(name, DEFAULT_STRATEGIES[name]) if isinstance(source, dict) else DEFAULT_STRATEGIES[name]
        if not isinstance(entry, dict):
            failures.add("E_TIER", "config", f"strategies.{name}", "strategy must be an object")
            entry = DEFAULT_STRATEGIES[name]
        window = _read_window(entry.get("window"))
        cc_raw = entry.get("cc", {}) if isinstance(entry.get("cc"), dict) else {}
        precision_raw = entry.get("precision", {}) if isinstance(entry.get("precision"), dict) else {}
        merge_gap = _int(cc_raw.get("merge_gap"), DEFAULT_STRATEGIES[name]["cc"]["merge_gap"])
        fixed_bits = _int(precision_raw.get("fixed_bits"), DEFAULT_STRATEGIES[name]["precision"]["fixed_bits"])
        quant_step = _int(precision_raw.get("quant_step"), DEFAULT_STRATEGIES[name]["precision"]["quant_step"])
        capacities = {
            key: _int(entry.get(key), DEFAULT_STRATEGIES[name][key])
            for key in ("capacity_points", "capacity_columns", "capacity_chars_per_column")
        }
        cost = _number(entry.get("single_image_cost"), DEFAULT_STRATEGIES[name]["single_image_cost"])

        _validate_strategy_window(name, window, failures)
        if merge_gap < 0:
            failures.add("E_CONFIG_VALUE", "config", f"strategies.{name}.cc.merge_gap", "merge gap cannot be negative")
        if fixed_bits not in FIXED_BITS_ALLOWED:
            failures.add(
                "E_QUANT_ENUM",
                "config",
                f"strategies.{name}.precision.fixed_bits",
                "fixed point fractional width is outside the allowed enumeration",
                allowed=list(FIXED_BITS_ALLOWED),
                actual=fixed_bits,
            )
        if quant_step not in QUANT_STEPS_ALLOWED:
            failures.add(
                "E_QUANT_ENUM",
                "config",
                f"strategies.{name}.precision.quant_step",
                "quantization step is outside the allowed enumeration",
                allowed=list(QUANT_STEPS_ALLOWED),
                actual=quant_step,
            )
        if any(value <= 0 for value in capacities.values()):
            failures.add("E_TIER", "config", f"strategies.{name}", "strategy capacities must be positive")
        if cost <= 0:
            failures.add("E_TIER", "config", f"strategies.{name}.single_image_cost", "cost must be positive")
        strategies[name] = StrategySpec(
            name=name,
            window=window,
            merge_gap=merge_gap,
            precision=PrecisionSpec(fixed_bits, quant_step),
            capacity_points=capacities["capacity_points"],
            capacity_columns=capacities["capacity_columns"],
            capacity_chars_per_column=capacities["capacity_chars_per_column"],
            single_image_cost=cost,
        )
    return strategies


def _validate_strategy_window(name: str, window: WindowSpec, failures: FailureLog) -> None:
    width, height = FIXED_RESOLUTION
    for key, value in (
        ("width", window.width),
        ("height", window.height),
        ("step_x", window.step_x),
        ("step_y", window.step_y),
    ):
        if value <= 0:
            failures.add("E_WINDOW_ALIGN", "config", f"strategies.{name}.window.{key}", "value must be positive")
    if window.width > 0 and width % window.width != 0:
        failures.add(
            "E_WINDOW_ALIGN",
            "config",
            f"strategies.{name}.window.width",
            "strategy window width must divide image width",
            window=window.width,
            image=width,
        )
    if window.height > 0 and height % window.height != 0:
        failures.add(
            "E_WINDOW_ALIGN",
            "config",
            f"strategies.{name}.window.height",
            "strategy window height must divide image height",
            window=window.height,
            image=height,
        )


def effective_config(config: Config, tier: str) -> EffectiveConfig:
    strategy = config.strategies[tier]
    cc = CCSpec(
        min_area=config.cc.min_area,
        max_aspect=config.cc.max_aspect,
        merge_gap=strategy.merge_gap,
        split_ratio=config.cc.split_ratio,
        ink_threshold=config.cc.ink_threshold,
    )
    return EffectiveConfig(
        tier=tier,
        tier_explicit=config.tier != "auto",
        resolution=config.resolution,
        pixel_mode=config.pixel_mode,
        window=strategy.window,
        cc=cc,
        precision=strategy.precision,
        max_workers=config.max_workers,
        point_budget=config.point_budget,
        keypoint_version=config.keypoint_version,
        strategy=strategy,
    )


def custom_effective_config(config: Config) -> EffectiveConfig:
    synthetic = StrategySpec(
        name="custom",
        window=config.window,
        merge_gap=config.cc.merge_gap,
        precision=config.precision,
        capacity_points=10_000_000,
        capacity_columns=32,
        capacity_chars_per_column=128,
        single_image_cost=0.5,
    )
    return EffectiveConfig(
        tier="custom",
        tier_explicit=True,
        resolution=config.resolution,
        pixel_mode=config.pixel_mode,
        window=config.window,
        cc=config.cc,
        precision=config.precision,
        max_workers=config.max_workers,
        point_budget=config.point_budget,
        keypoint_version=config.keypoint_version,
        strategy=synthetic,
    )
