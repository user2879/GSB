"""In-memory models used by validation, rendering and snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .fixed import CHANNEL_ORDER


def _canonical_dict(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_canonical_dict(item) for item in value]
    if isinstance(value, list):
        return [_canonical_dict(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _canonical_dict(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class LayerSpec:
    index: int
    frequency_milli: int
    amplitude_milli: int
    seed: int
    valid: bool = True

    def parameter_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "frequency_milli": self.frequency_milli,
            "amplitude_milli": self.amplitude_milli,
            "seed": self.seed,
            "valid": self.valid,
        }

    @property
    def frequency(self) -> float:
        return self.frequency_milli / 1000.0

    @property
    def amplitude(self) -> float:
        return self.amplitude_milli / 1000.0


@dataclass(frozen=True)
class FixedConfig:
    fraction_bits: int
    quant_step: int
    tie_rule: str
    guard_bits: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "fraction_bits": self.fraction_bits,
            "quant_step": self.quant_step,
            "tie_rule": self.tie_rule,
            "guard_bits": self.guard_bits,
        }


@dataclass(frozen=True)
class BudgetConfig:
    layer_points: int
    batch_points: int

    def to_dict(self) -> dict[str, int]:
        return {
            "layer_points": self.layer_points,
            "batch_points": self.batch_points,
        }


@dataclass
class RenderConfig:
    name: str
    texture_path: str
    resolution_tiers: tuple[int, ...]
    resolution: int
    seed: int
    algorithm: str
    layers: tuple[LayerSpec, ...]
    fixed: FixedConfig
    budgets: BudgetConfig
    blend_mode: str
    preview_budget_ms: int
    thread_pool_size: int
    commit_interval_commands: int
    error_tolerance_quant_steps: int
    source_path: str
    source_hash: str
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.resolution

    @property
    def height(self) -> int:
        return self.resolution

    @property
    def point_count(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "texture_path": self.texture_path,
            "resolution_tiers": list(self.resolution_tiers),
            "resolution": self.resolution,
            "seed": self.seed,
            "algorithm": self.algorithm,
            "layers": [layer.parameter_dict() for layer in self.layers],
            "fixed": self.fixed.to_dict(),
            "budgets": self.budgets.to_dict(),
            "blend_mode": self.blend_mode,
            "preview_budget_ms": self.preview_budget_ms,
            "thread_pool_size": self.thread_pool_size,
            "commit_interval_commands": self.commit_interval_commands,
            "error_tolerance_quant_steps": self.error_tolerance_quant_steps,
            "source_path": self.source_path,
            "source_hash": self.source_hash,
            "raw": _canonical_dict(self.raw),
        }

    def clone(self, **changes: Any) -> "RenderConfig":
        values = {
            "name": self.name,
            "texture_path": self.texture_path,
            "resolution_tiers": self.resolution_tiers,
            "resolution": self.resolution,
            "seed": self.seed,
            "algorithm": self.algorithm,
            "layers": self.layers,
            "fixed": self.fixed,
            "budgets": self.budgets,
            "blend_mode": self.blend_mode,
            "preview_budget_ms": self.preview_budget_ms,
            "thread_pool_size": self.thread_pool_size,
            "commit_interval_commands": self.commit_interval_commands,
            "error_tolerance_quant_steps": self.error_tolerance_quant_steps,
            "source_path": self.source_path,
            "source_hash": self.source_hash,
            "raw": self.raw,
        }
        values.update(changes)
        return RenderConfig(**values)


@dataclass(frozen=True)
class LayerWeight:
    index: int
    weight_q: int
    raw_weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "weight_q": self.weight_q,
            "raw_weight": self.raw_weight,
        }


@dataclass
class ChannelSpec:
    name: str
    base_weight_q: int
    layer_weights: tuple[LayerWeight, ...]
    valid: bool = True

    def parameter_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_weight_q": self.base_weight_q,
            "layer_weights": [item.to_dict() for item in self.layer_weights],
            "valid": self.valid,
        }

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return tuple(item.index for item in self.layer_weights)

    def clone(self, **changes: Any) -> "ChannelSpec":
        values = {
            "name": self.name,
            "base_weight_q": self.base_weight_q,
            "layer_weights": self.layer_weights,
            "valid": self.valid,
        }
        values.update(changes)
        return ChannelSpec(**values)


@dataclass
class MaterialConfig:
    name: str
    channels: dict[str, ChannelSpec]
    tie_rule: str
    equal_weight_rule: str
    source_path: str
    source_hash: str
    raw: dict[str, Any] = field(default_factory=dict)

    def ordered_channels(self) -> tuple[ChannelSpec, ...]:
        return tuple(self.channels[name] for name in CHANNEL_ORDER if name in self.channels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "channels": {
                name: channel.parameter_dict() for name, channel in self.channels.items()
            },
            "tie_rule": self.tie_rule,
            "equal_weight_rule": self.equal_weight_rule,
            "source_path": self.source_path,
            "source_hash": self.source_hash,
            "raw": _canonical_dict(self.raw),
        }

    def clone(self, **changes: Any) -> "MaterialConfig":
        values = {
            "name": self.name,
            "channels": self.channels,
            "tie_rule": self.tie_rule,
            "equal_weight_rule": self.equal_weight_rule,
            "source_path": self.source_path,
            "source_hash": self.source_hash,
            "raw": self.raw,
        }
        values.update(changes)
        return MaterialConfig(**values)


@dataclass
class TextureData:
    path: str
    width: int
    height: int
    mode: str
    png_color_type: int
    png_bit_depth: int
    rgb: np.ndarray
    content_key: str
    file_sha256: str

    def summary(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
            "png_color_type": self.png_color_type,
            "png_bit_depth": self.png_bit_depth,
            "content_key": self.content_key,
            "file_sha256": self.file_sha256,
        }


@dataclass
class LayerArtifact:
    index: int
    key: str
    texture_key: str
    parameter_key: str
    frac_bits: int
    quant_step: int
    raw: np.ndarray
    quantized: np.ndarray
    residual: np.ndarray
    decisions: np.ndarray
    complete: bool = True


@dataclass
class LayerPartial:
    index: int
    key: str
    texture_key: str
    parameter_key: str
    frac_bits: int
    quant_step: int
    raw: np.ndarray
    quantized: np.ndarray
    residual: np.ndarray
    decisions: np.ndarray
    points_done: int


@dataclass
class ChannelArtifact:
    name: str
    key: str
    layer_keys: tuple[str, ...]
    layer_indices: tuple[int, ...]
    array: np.ndarray
    frac_bits: int
    quant_step: int
    tie_count: int
    rounding_count: int
    parameter_key: str = ""
    blend_mode: str = ""


@dataclass
class CompositeArtifact:
    key: str
    channel_keys: tuple[str, ...]
    array: np.ndarray
    frac_bits: int
    quant_step: int
    parameter_key: str = ""


@dataclass
class RenderState:
    version: int = 0
    layers: dict[int, LayerArtifact] = field(default_factory=dict)
    partials: dict[int, LayerPartial] = field(default_factory=dict)
    channels: dict[str, ChannelArtifact] = field(default_factory=dict)
    composite: CompositeArtifact | None = None
    pending: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    ledger: list[dict[str, Any]] = field(default_factory=list)
    injection_hits: list[dict[str, Any]] = field(default_factory=list)
    last_preview: dict[str, Any] = field(default_factory=dict)
    last_sequence: int = 0
    effective_config_hash: str = ""
    last_texture_key: str = ""

    def has_pending(self) -> bool:
        return any(item.get("code") == "E_POINT_BUDGET_PENDING" for item in self.pending)

    def complete_layers(self) -> list[int]:
        return sorted(self.layers)

    def clear_render_products(self) -> None:
        self.layers.clear()
        self.partials.clear()
        self.channels.clear()
        self.composite = None
        self.pending.clear()
        self.failures.clear()
