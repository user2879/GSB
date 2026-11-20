"""Typed, validated in-memory models for generation config and materials."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import E_CONFIG_INVALID, E_LAYER_INVALID, E_WEIGHT_INVALID, Failure
from .fixedpoint import ROUND_MODES

BLEND_MODES = ("weighted_sum", "normalized_weight")
CHANNELS = ("diffuse", "roughness", "metallic", "normal")
MIN_FRAC_BITS, MAX_FRAC_BITS = 1, 20
MIN_LEVELS, MAX_LEVELS = 2, 1 << 16


def _is_pos_finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x)) and float(x) > 0


@dataclass(frozen=True)
class NoiseLayer:
    index: int
    fingerprint: int
    frequency: float
    amplitude: float
    base_texture: str


@dataclass(frozen=True)
class FixedPointConfig:
    frac_bits: int
    unit_levels: int
    signed_levels: int
    round_mode: str

    @property
    def high_tier_frac_bits(self) -> int:
        return min(MAX_FRAC_BITS, self.frac_bits + 1)


@dataclass(frozen=True)
class GenerationConfig:
    resolution_tiers: tuple[int, ...]
    layers: tuple[NoiseLayer, ...]
    blend_mode: str
    fixed_point: FixedPointConfig
    tolerance: float
    threads: int
    point_budget: int
    snapshot_interval: int
    config_hash: str
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    def with_overrides(self, **overrides: Any) -> "GenerationConfig":
        raw = json.loads(json.dumps(self.raw))
        fp_raw = raw["fixed_point"]
        if "blend_mode" in overrides:
            raw["blend_mode"] = overrides["blend_mode"]
        if "frac_bits" in overrides or "unit_levels" in overrides or "round_mode" in overrides:
            fp_raw.update({k: v for k, v in overrides.items() if k in fp_raw})
        if "layers" in overrides:
            raw["layers"] = overrides["layers"]
        return parse_generation_config(raw)


def parse_generation_config(data: dict[str, Any]) -> GenerationConfig:
    failures: list[Failure] = []
    tiers = data.get("resolution_tiers")
    if not isinstance(tiers, list) or not tiers or any(
        not isinstance(t, int) or isinstance(t, bool) or t <= 0 for t in tiers
    ) or any(tiers[i] >= tiers[i + 1] for i in range(len(tiers) - 1)):
        raise ValueError(
            f"{E_CONFIG_INVALID}: resolution_tiers must be a strictly increasing list of positive ints"
        )

    fp_raw = data.get("fixed_point", {})
    frac_bits = fp_raw.get("frac_bits")
    unit_levels = fp_raw.get("unit_levels")
    signed_levels = fp_raw.get("signed_levels")
    round_mode = fp_raw.get("round_mode", "half_even")
    for name, val in (("frac_bits", frac_bits), ("unit_levels", unit_levels), ("signed_levels", signed_levels)):
        if not isinstance(val, int) or isinstance(val, bool):
            raise ValueError(f"{E_CONFIG_INVALID}: fixed_point.{name} must be an int")
    if not (MIN_FRAC_BITS <= frac_bits <= MAX_FRAC_BITS):
        raise ValueError(f"{E_CONFIG_INVALID}: frac_bits must be within [{MIN_FRAC_BITS},{MAX_FRAC_BITS}]")
    for name, val in (("unit_levels", unit_levels), ("signed_levels", signed_levels)):
        if not (MIN_LEVELS <= val <= MAX_LEVELS):
            raise ValueError(f"{E_CONFIG_INVALID}: {name} must be within [{MIN_LEVELS},{MAX_LEVELS}]")
    if round_mode not in ROUND_MODES:
        raise ValueError(f"{E_CONFIG_INVALID}: round_mode must be one of {ROUND_MODES}")

    blend_mode = data.get("blend_mode", "normalized_weight")
    if blend_mode not in BLEND_MODES:
        raise ValueError(f"{E_CONFIG_INVALID}: blend_mode must be one of {BLEND_MODES}")

    raw_layers = data.get("layers", [])
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ValueError(f"{E_CONFIG_INVALID}: at least one noise layer is required")
    layers: list[NoiseLayer] = []
    for i, layer in enumerate(raw_layers):
        freq = layer.get("frequency")
        amp = layer.get("amplitude")
        if not _is_pos_finite(freq) or not _is_pos_finite(amp):
            failures.append(
                Failure(E_LAYER_INVALID, f"layer[{i}]", f"frequency/amplitude must be positive finite numbers")
            )
            continue
        fp = layer.get("fingerprint")
        if not isinstance(fp, int) or isinstance(fp, bool):
            failures.append(Failure(E_LAYER_INVALID, f"layer[{i}]", "fingerprint must be an int"))
            continue
        base_texture = layer.get("base_texture", "")
        if not isinstance(base_texture, str) or not base_texture:
            failures.append(Failure(E_LAYER_INVALID, f"layer[{i}]", "base_texture must name a texture id"))
            continue
        layers.append(NoiseLayer(i, int(fp), float(freq), float(amp), base_texture))
    if failures:
        raise _ConfigParseError(failures)

    threads = data.get("threads", 1)
    point_budget = data.get("point_budget", 0)
    snapshot_interval = data.get("snapshot_interval", 1)
    if not isinstance(threads, int) or threads < 1:
        raise ValueError(f"{E_CONFIG_INVALID}: threads must be a positive int")
    if not isinstance(point_budget, int) or point_budget < 0:
        raise ValueError(f"{E_CONFIG_INVALID}: point_budget must be a non-negative int (0 = unlimited)")
    if not isinstance(snapshot_interval, int) or snapshot_interval < 1:
        raise ValueError(f"{E_CONFIG_INVALID}: snapshot_interval must be a positive int")
    tolerance = float(data.get("tolerance", 1e-3))
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError(f"{E_CONFIG_INVALID}: tolerance must be a positive finite number")

    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    import hashlib

    config_hash = hashlib.blake2b(canonical, digest_size=16).hexdigest()
    return GenerationConfig(
        tuple(tiers),
        tuple(layers),
        blend_mode,
        FixedPointConfig(frac_bits, unit_levels, signed_levels, round_mode),
        tolerance,
        threads,
        point_budget,
        snapshot_interval,
        config_hash,
        data,
    )


class _ConfigParseError(Exception):
    def __init__(self, failures: list[Failure]):
        super().__init__("; ".join(f"{f.code}:{f.target}" for f in failures))
        self.failures = failures


def load_generation_config(path: str | Path) -> tuple[GenerationConfig | None, list[Failure]]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [Failure(E_CONFIG_INVALID, str(path), f"cannot parse config: {exc}")]
    try:
        return parse_generation_config(data), []
    except _ConfigParseError as exc:
        return None, exc.failures
    except ValueError as exc:
        return None, [Failure(E_CONFIG_INVALID, str(path), str(exc))]


@dataclass(frozen=True)
class Material:
    material_id: str
    # channel -> {layer_index -> weight}; missing layer means weight 0
    layer_weights: dict[str, dict[int, float]]

    def weight_sum(self, channel: str) -> float:
        return sum(self.layer_weights.get(channel, {}).values())


def parse_materials(data: dict[str, Any]) -> tuple[list[Material], list[Failure]]:
    failures: list[Failure] = []
    materials: list[Material] = []
    raw_materials = data.get("materials", [])
    if not isinstance(raw_materials, list) or not raw_materials:
        return [], [Failure(E_CONFIG_INVALID, "materials", "materials list is empty or malformed")]
    for entry in raw_materials:
        mid = str(entry.get("material_id", ""))
        if not mid:
            failures.append(Failure(E_WEIGHT_INVALID, "<unknown>", "material_id is required"))
            continue
        weights_raw = entry.get("layer_weights", {})
        weights: dict[str, dict[int, float]] = {}
        bad = False
        for channel in CHANNELS:
            chan_raw = weights_raw.get(channel, {})
            if not isinstance(chan_raw, dict):
                failures.append(Failure(E_WEIGHT_INVALID, f"{mid}.{channel}", "weights must be an object"))
                bad = True
                continue
            chan_map: dict[int, float] = {}
            for key, val in chan_raw.items():
                try:
                    layer_idx = int(key)
                except (TypeError, ValueError):
                    failures.append(Failure(E_WEIGHT_INVALID, f"{mid}.{channel}.{key}", "layer key must be int"))
                    bad = True
                    continue
                if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(float(val)):
                    failures.append(Failure(E_WEIGHT_INVALID, f"{mid}.{channel}.{key}", "weight must be finite"))
                    bad = True
                    continue
                if float(val) != 0.0:
                    chan_map[layer_idx] = float(val)
            weights[channel] = chan_map
        if bad:
            continue
        for channel in CHANNELS:
            if sum(weights[channel].values()) <= 0:
                failures.append(
                    Failure(E_WEIGHT_INVALID, f"{mid}.{channel}", "sum of channel layer weights must be positive")
                )
                bad = True
        if not bad:
            materials.append(Material(mid, weights))
    return materials, failures


def load_materials(path: str | Path) -> tuple[list[Material], list[Failure]]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], [Failure(E_CONFIG_INVALID, str(path), f"cannot parse materials: {exc}")]
    return parse_materials(data)
