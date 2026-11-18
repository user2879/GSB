"""Parsing and validation of generation config and material descriptions."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ConfigInvalid

ALLOWED_ROUNDING = ("truncate", "half_up", "half_even")
ALLOWED_BLEND = ("weighted", "additive", "multiply")
ALLOWED_PIXEL_MODES = ("RGB", "RGBA", "L")
ALLOWED_BITS = range(0, 17)
CHANNELS = ("diffuse", "roughness", "metalness", "normal")


@dataclass(frozen=True)
class LayerSpec:
    index: int
    source_texture: str
    frequency: int
    amplitude: float
    seed: int
    fingerprint: int


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    base: float
    scale: float
    bias: float


@dataclass(frozen=True)
class MaterialSpec:
    name: str
    layers: tuple[str, ...]
    weights: tuple[float, ...]
    channels: dict[str, ChannelSpec]
    blend_mode: str


@dataclass(frozen=True)
class GenerationConfig:
    name: str
    resolution_tiers: tuple[int, ...]
    active_tier: int
    noise_layers: tuple[LayerSpec, ...]
    blend_mode: str
    fractional_bits: int
    quantize: bool
    rounding: str
    workers: int
    point_budget: int
    snapshot_interval: int
    tolerances: dict[str, float]
    bin_width: int
    raw_json: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def width(self) -> int:
        return self.active_tier

    @property
    def height(self) -> int:
        return self.active_tier


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise ConfigInvalid(f"missing json file: {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigInvalid(f"invalid json in {path.name}: {exc.msg}") from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigInvalid(message)


def _is_positive_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and value > 0


def parse_layer_spec(raw: dict[str, Any], index: int) -> LayerSpec:
    _require(isinstance(raw, dict), f"noise_layers[{index}] must be an object")
    source = raw.get("source_texture")
    _require(isinstance(source, str) and source, f"noise_layers[{index}].source_texture must be a non-empty string")
    frequency = raw.get("frequency")
    amplitude = raw.get("amplitude")
    _require(isinstance(frequency, int) and not isinstance(frequency, bool) and frequency >= 1,
             f"noise_layers[{index}].frequency must be a positive integer")
    _require(_is_positive_finite(amplitude), f"noise_layers[{index}].amplitude must be a positive finite number")
    seed = raw.get("seed", 0)
    _require(isinstance(seed, int) and not isinstance(seed, bool), f"noise_layers[{index}].seed must be an integer")
    fp = raw.get("fingerprint")
    if fp is None:
        fp = index + 1
    _require(isinstance(fp, int) and not isinstance(fp, bool) and fp >= 0,
             f"noise_layers[{index}].fingerprint must be a non-negative integer")
    return LayerSpec(index=index, source_texture=source, frequency=frequency,
                     amplitude=float(amplitude), seed=seed, fingerprint=fp)


def parse_channel_spec(name: str, raw: dict[str, Any]) -> ChannelSpec:
    _require(isinstance(raw, dict), f"channel {name} must be an object")
    out = {}
    for key in ("base", "scale", "bias"):
        value = raw.get(key)
        _require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)),
                 f"channel {name}.{key} must be a finite number")
        out[key] = float(value)
    return ChannelSpec(name=name, base=out["base"], scale=out["scale"], bias=out["bias"])


def parse_material(raw: dict[str, Any], known_layers: set[str]) -> MaterialSpec:
    _require(isinstance(raw, dict), "each material must be an object")
    name = raw.get("name")
    _require(isinstance(name, str) and name, "material.name must be a non-empty string")
    layers = raw.get("layers")
    weights = raw.get("weights")
    _require(isinstance(layers, list) and layers, f"material {name}.layers must be a non-empty list")
    _require(isinstance(weights, list) and len(weights) == len(layers),
             f"material {name}.weights length must match layers")
    for item in layers:
        _require(isinstance(item, str) and item, f"material {name} layer names must be non-empty strings")
        _require(item in known_layers, f"material {name} references unknown layer '{item}'")
    for item in weights:
        _require(isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item)) and item >= 0,
                 f"material {name} weights must be finite non-negative numbers")
    _require(sum(float(w) for w in weights) > 0, f"material {name} weights must have a positive sum")
    channels_raw = raw.get("channels")
    _require(isinstance(channels_raw, dict), f"material {name}.channels must be an object")
    missing = [c for c in CHANNELS if c not in channels_raw]
    _require(not missing, f"material {name} missing channels: {', '.join(missing)}")
    extra = [c for c in channels_raw if c not in CHANNELS]
    _require(not extra, f"material {name} has unknown channels: {', '.join(extra)}")
    channels = {name_: parse_channel_spec(name_, channels_raw[name_]) for name_ in CHANNELS}
    blend_mode = raw.get("blend_mode", "weighted")
    _require(blend_mode in ALLOWED_BLEND, f"material {name} blend_mode must be one of {ALLOWED_BLEND}")
    return MaterialSpec(name=name, layers=tuple(layers), weights=tuple(float(w) for w in weights),
                        channels=channels, blend_mode=blend_mode)


def load_generation_config(path: str | Path) -> GenerationConfig:
    raw = _read_json(Path(path))
    _require(isinstance(raw, dict), "generation config root must be an object")
    name = raw.get("name", "run")
    _require(isinstance(name, str), "config.name must be a string")

    tiers = raw.get("resolution_tiers")
    _require(isinstance(tiers, list) and len(tiers) >= 1, "resolution_tiers must be a non-empty list")
    for tier in tiers:
        _require(isinstance(tier, int) and not isinstance(tier, bool) and tier > 0,
                 "resolution_tiers must contain positive integers")
    _require(all(tiers[i] < tiers[i + 1] for i in range(len(tiers) - 1)),
             "resolution_tiers must be strictly monotonically increasing")
    active_tier = raw.get("active_tier", tiers[-1])
    _require(active_tier in tiers, "active_tier must be one of resolution_tiers")

    layers_raw = raw.get("noise_layers")
    _require(isinstance(layers_raw, list) and len(layers_raw) >= 1, "noise_layers must be a non-empty list")
    layer_specs = tuple(parse_layer_spec(item, i) for i, item in enumerate(layers_raw))
    layer_ids = {f"layer{spec.index:02d}" for spec in layer_specs}

    blend_mode = raw.get("blend_mode", "weighted")
    _require(blend_mode in ALLOWED_BLEND, f"blend_mode must be one of {ALLOWED_BLEND}")

    bits = raw.get("fractional_bits", 8)
    _require(isinstance(bits, int) and not isinstance(bits, bool) and bits in ALLOWED_BITS,
             f"fractional_bits must be an integer within {min(ALLOWED_BITS)}..{max(ALLOWED_BITS)}")

    quantize = raw.get("quantize", True)
    _require(isinstance(quantize, bool), "quantize must be a boolean")

    rounding = raw.get("rounding", "half_even")
    _require(rounding in ALLOWED_ROUNDING, f"rounding must be one of {ALLOWED_ROUNDING}")

    workers = raw.get("workers", 1)
    _require(isinstance(workers, int) and not isinstance(workers, bool) and workers >= 1,
             "workers must be a positive integer")
    point_budget = raw.get("point_budget", 100000)
    _require(isinstance(point_budget, int) and not isinstance(point_budget, bool) and point_budget >= 1,
             "point_budget must be a positive integer")
    snapshot_interval = raw.get("snapshot_interval", 1)
    _require(isinstance(snapshot_interval, int) and not isinstance(snapshot_interval, bool) and snapshot_interval >= 1,
             "snapshot_interval must be a positive integer")

    tolerances = raw.get("tolerances", {})
    _require(isinstance(tolerances, dict), "tolerances must be an object")
    default_tol = {"noise_overlay": 4.0, "quantize": 2.0, "material_blend": 4.0, "channel_compose": 2.0}
    for key, value in tolerances.items():
        _require(key in default_tol, f"unknown tolerance stage: {key}")
        _require(isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0,
                 f"tolerance {key} must be a non-negative number")
    default_tol.update({k: float(v) for k, v in tolerances.items()})

    bin_width = raw.get("bin_width", 16)
    _require(isinstance(bin_width, int) and not isinstance(bin_width, bool) and bin_width >= 1,
             "bin_width must be a positive integer")

    return GenerationConfig(
        name=name, resolution_tiers=tuple(tiers), active_tier=active_tier,
        noise_layers=layer_specs, blend_mode=blend_mode, fractional_bits=bits,
        quantize=quantize, rounding=rounding, workers=workers, point_budget=point_budget,
        snapshot_interval=snapshot_interval, tolerances=default_tol, bin_width=bin_width, raw_json=raw,
    )


def load_materials(path: str | Path, config: GenerationConfig) -> list[MaterialSpec]:
    raw = _read_json(Path(path))
    _require(isinstance(raw, list) and raw, "materials root must be a non-empty list")
    layer_ids = {f"layer{spec.index:02d}" for spec in config.noise_layers}
    materials = [parse_material(item, layer_ids) for item in raw]
    names = [m.name for m in materials]
    _require(len(set(names)) == len(names), "material names must be unique")
    return materials
