"""Typed in-memory representations of the deterministic local inputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

KNOWN_ROUNDING = ("half_even", "half_up", "half_down", "trunc", "floor")
KNOWN_SATURATION = ("clip", "wrap")
KNOWN_SCHEME = ("sym", "asym")
# Fixed integer range per quantization level. Switching the level genuinely
# changes the representable integer interval and therefore the integers.
LEVEL_RANGES: dict[str, tuple[int, int]] = {
    "int4": (-8, 7),
    "uint4": (0, 15),
    "int8": (-128, 127),
    "uint8": (0, 255),
    "int16": (-32768, 32767),
    "uint16": (0, 65535),
}


@dataclass(frozen=True)
class WeightLayer:
    name: str
    shape: tuple[int, ...]
    weights: np.ndarray
    content_version: int
    group_hint: int

    @property
    def size(self) -> int:
        return int(self.weights.size)

    @property
    def input_dim(self) -> int:
        return int(self.shape[-1])


@dataclass(frozen=True)
class CalibrationSet:
    inputs: np.ndarray
    references: np.ndarray
    sample_ids: tuple[str, ...]
    tier_sizes: dict[str, int]

    @property
    def sample_count(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def dim(self) -> int:
        return int(self.inputs.shape[1])

    def sample_refs(self, tier: str) -> tuple[str, ...]:
        limit = self.tier_sizes[tier]
        return tuple(self.sample_ids[: min(limit, self.sample_count)])


@dataclass(frozen=True)
class DistillConfig:
    temperature: float
    steps: int
    soft_weight: float
    learning_rate: float
    tolerance: float


@dataclass(frozen=True)
class BudgetConfig:
    global_cal_points: int
    batch_quant_points: int
    per_layer_quotas: dict[str, int]


@dataclass(frozen=True)
class LayerQuantConfig:
    name: str
    bits: int
    level: str
    group_size: int
    q_lo: int
    q_hi: int
    frac_bits: int
    scheme: str
    rounding: str
    saturation: str

    def to_material(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bits": self.bits,
            "level": self.level,
            "group_size": self.group_size,
            "q_lo": self.q_lo,
            "q_hi": self.q_hi,
            "frac_bits": self.frac_bits,
            "scheme": self.scheme,
            "rounding": self.rounding,
            "saturation": self.saturation,
        }


@dataclass(frozen=True)
class QuantConfig:
    defaults: dict[str, Any]
    layers: dict[str, dict[str, Any]]
    distill: DistillConfig
    budgets: BudgetConfig
    workers: int
    tiers: dict[str, int]
    default_tier: str
    layer_tiers: dict[str, str]
    receipts_dir: str

    def resolve_layer(self, layer_name: str) -> LayerQuantConfig:
        merged = dict(self.defaults)
        merged.update(self.layers.get(layer_name, {}))
        return LayerQuantConfig(
            name=layer_name,
            bits=int(merged["bits"]),
            level=str(merged["level"]),
            group_size=int(merged["group_size"]),
            q_lo=int(merged["q_lo"]),
            q_hi=int(merged["q_hi"]),
            frac_bits=int(merged["frac_bits"]),
            scheme=str(merged["scheme"]),
            rounding=str(merged["rounding"]),
            saturation=str(merged["saturation"]),
        )

    def tier_for(self, layer_name: str) -> str:
        return self.layer_tiers.get(layer_name, self.default_tier)


def level_range(level: str) -> tuple[int, int]:
    if level not in LEVEL_RANGES:
        raise ValueError(f"unknown quantization level: {level}")
    return LEVEL_RANGES[level]
