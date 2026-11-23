"""Configuration loading and validation.

The YAML config describes windows, index weights, fixed-point representation,
the three budget constraints, runtime parallelism and snapshot locations.
Validation failures raise :class:`PipelineFatal` for genuinely malformed config
files; per-tile semantic faults are handled later by the loader with business
codes.  Config validation itself maps each bad field to a stable business code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

from .errors import Code, PipelineFatal
from .fixedpoint import ALLOWED_FRAC_BITS, ALLOWED_GEARS, Gear

REQUIRED_BANDSETS = ("BS_V1", "BS_V2")
REQUIRED_WEIGHTS = ("vegetation", "water", "texture", "slope")


@dataclass
class WindowCfg:
    size: int
    stride: int
    context_overlap: int


@dataclass
class Weights:
    vegetation: float
    water: float
    texture: float
    slope: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "vegetation": self.vegetation,
            "water": self.water,
            "texture": self.texture,
            "slope": self.slope,
        }


@dataclass
class FixedCfg:
    fractional_bits: int
    gear: Gear

    @property
    def gear_value(self) -> str:
        return self.gear.value


@dataclass
class Config:
    raw: Dict[str, Any]
    window: WindowCfg
    weights: Weights
    fixed: FixedCfg
    pixel_budget: int
    tile_quota: int
    fusion_batch_limit: int
    max_workers: int
    min_component_pixels: int
    allowed_bandsets: tuple
    fusion_strategy_version: str
    manifest_path: str
    snapshot_dir: str
    report_dir: str
    sample_root: str

    def fingerprint_inputs(self) -> Dict[str, Any]:
        """Stable subset used for config hashing / snapshot staleness."""
        return {
            "window": (self.window.size, self.window.stride, self.window.context_overlap),
            "weights": tuple(round(self.weights.as_dict()[k], 9) for k in REQUIRED_WEIGHTS),
            "fixed": (self.fixed.fractional_bits, self.fixed.gear.value),
            "pixel_budget": self.pixel_budget,
            "tile_quota": self.tile_quota,
            "fusion_batch_limit": self.fusion_batch_limit,
            "min_component_pixels": self.min_component_pixels,
            "allowed_bandsets": tuple(sorted(self.allowed_bandsets)),
            "fusion_strategy_version": self.fusion_strategy_version,
        }


def _require(mapping: Dict[str, Any], key: str, code: str) -> Any:
    if key not in mapping:
        raise PipelineFatal(code, f"missing config key: {key}")
    return mapping[key]


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or isinstance(value, float) and not value.is_integer():
        raise PipelineFatal(Code.E_BUDGET_INVALID, f"{name} must be a positive integer, got {value!r}")
    ivalue = int(value)
    if ivalue <= 0:
        raise PipelineFatal(Code.E_BUDGET_INVALID, f"{name} must be a positive integer, got {ivalue}")
    return ivalue


def load_config(path: str | Path) -> Config:
    p = Path(path)
    if not p.exists():
        raise PipelineFatal(Code.E_FILE_NOT_FOUND, f"config not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PipelineFatal(Code.E_CONFIG_PARSE, f"invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise PipelineFatal(Code.E_CONFIG_PARSE, "config root must be a mapping")

    win_raw = _require(raw, "window", Code.E_CONFIG_KEY_MISSING)
    window = WindowCfg(
        size=_positive_int(win_raw, "window.size") if False else _win_int(win_raw, "size"),
        stride=_win_int(win_raw, "stride"),
        context_overlap=_win_int(win_raw, "context_overlap"),
    )
    if window.stride <= 0 or window.context_overlap < 0:
        raise PipelineFatal(Code.E_BUDGET_INVALID, "window.stride>0 and context_overlap>=0 required")

    w_raw = _require(raw, "weights", Code.E_CONFIG_KEY_MISSING)
    weight_values = {}
    for name in REQUIRED_WEIGHTS:
        if name not in w_raw:
            raise PipelineFatal(Code.E_CONFIG_KEY_MISSING, f"missing weight: {name}")
        value = w_raw[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PipelineFatal(Code.E_WEIGHT_INVALID, f"weight {name} must be numeric, got {value!r}")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise PipelineFatal(Code.E_WEIGHT_INVALID, f"weight {name} must be non-negative finite, got {value!r}")
        weight_values[name] = float(value)
    if not any(v > 0 for v in weight_values.values()):
        raise PipelineFatal(Code.E_WEIGHT_INVALID, "at least one index weight must be positive")
    weights = Weights(**weight_values)

    fp_raw = _require(raw, "fixed_point", Code.E_CONFIG_KEY_MISSING)
    frac = _require(fp_raw, "fractional_bits", Code.E_FIXED_ENUM)
    if frac not in ALLOWED_FRAC_BITS:
        raise PipelineFatal(Code.E_FIXED_ENUM, f"fractional_bits must be one of {ALLOWED_FRAC_BITS}, got {frac}")
    gear_value = _require(fp_raw, "quantization_gear", Code.E_FIXED_ENUM)
    if gear_value not in ALLOWED_GEARS:
        raise PipelineFatal(Code.E_FIXED_ENUM, f"quantization_gear must be one of {ALLOWED_GEARS}, got {gear_value}")
    fixed = FixedCfg(fractional_bits=frac, gear=Gear(gear_value))

    pixel_budget = _positive_int(_require(raw, "global_pixel_budget", Code.E_BUDGET_INVALID), "global_pixel_budget")
    tile_quota = _positive_int(_require(raw, "tile_pixel_quota", Code.E_BUDGET_INVALID), "tile_pixel_quota")
    fusion_batch = _positive_int(_require(raw, "fusion_batch_limit", Code.E_BUDGET_INVALID), "fusion_batch_limit")
    min_component = _positive_int(raw.get("min_component_pixels", 1), "min_component_pixels")

    max_workers = raw.get("max_workers", 1)
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise PipelineFatal(Code.E_BUDGET_INVALID, "max_workers must be a positive integer")

    bandsets = raw.get("allowed_bandsets", list(REQUIRED_BANDSETS))
    if not isinstance(bandsets, list) or not bandsets:
        raise PipelineFatal(Code.E_BANDSET_UNKNOWN, "allowed_bandsets must be a non-empty list")

    fusion_version = raw.get("fusion_strategy_version", "FV1")
    if fusion_version not in ("FV1", "FV2"):
        raise PipelineFatal(Code.E_FIXED_ENUM, f"unknown fusion_strategy_version {fusion_version}")

    for key in ("manifest_path", "snapshot_dir", "report_dir", "sample_root"):
        _require(raw, key, Code.E_CONFIG_KEY_MISSING)

    return Config(
        raw=raw,
        window=window,
        weights=weights,
        fixed=fixed,
        pixel_budget=pixel_budget,
        tile_quota=tile_quota,
        fusion_batch_limit=fusion_batch,
        max_workers=max_workers,
        min_component_pixels=min_component,
        allowed_bandsets=tuple(bandsets),
        fusion_strategy_version=fusion_version,
        manifest_path=raw["manifest_path"],
        snapshot_dir=raw["snapshot_dir"],
        report_dir=raw["report_dir"],
        sample_root=raw["sample_root"],
    )


def _win_int(mapping: Dict[str, Any], key: str) -> int:
    if key not in mapping:
        raise PipelineFatal(Code.E_CONFIG_KEY_MISSING, f"missing window.{key}")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PipelineFatal(Code.E_BUDGET_INVALID, f"window.{key} must be positive integer, got {value!r}")
    return int(value)
