"""Typed input and build models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import Issue


@dataclass(frozen=True)
class FixedPointConfig:
    bit_width: int
    fraction_bits: int
    quantization_level: int
    quantization_tiers: tuple[int, ...]
    rounding: dict[str, str]
    weight_fraction_bits: int = 12

    @property
    def scale(self) -> int:
        return 1 << self.fraction_bits

    def as_dict(self) -> dict[str, Any]:
        return {
            "bit_width": self.bit_width,
            "fraction_bits": self.fraction_bits,
            "quantization_level": self.quantization_level,
            "quantization_tiers": list(self.quantization_tiers),
            "rounding": dict(sorted(self.rounding.items())),
            "weight_fraction_bits": self.weight_fraction_bits,
        }


@dataclass(frozen=True)
class NoiseLayer:
    index: int
    name: str
    texture: str
    frequency: float
    amplitude: float
    as_dict_extra: dict[str, Any] = field(default_factory=dict, compare=False)

    def as_dict(self) -> dict[str, Any]:
        result = {
            "index": self.index,
            "name": self.name,
            "texture": self.texture,
            "frequency": self.frequency,
            "amplitude": self.amplitude,
        }
        result.update(self.as_dict_extra)
        return result


@dataclass(frozen=True)
class GenerationConfig:
    schema_version: int
    seed: int
    resolution_tiers: tuple[int, ...]
    resolution: int
    noise_layers: tuple[NoiseLayer, ...]
    material_mix_mode: str
    fixed_point: FixedPointConfig
    bit_width_ladder: tuple[int, ...]
    tolerances: dict[str, float]
    final_consistency_tolerance: float
    concurrency: int
    max_concurrency: int
    point_budget: int
    latency_budget_ms: int
    snapshot_commit_interval: int
    embed_pixel_diffs: bool = True
    max_boundary_samples: int = 10000

    @property
    def noise_layer_count(self) -> int:
        return len(self.noise_layers)

    @property
    def quantization_step(self) -> int:
        return self.fixed_point.quantization_level

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "resolution_tiers": list(self.resolution_tiers),
            "resolution": self.resolution,
            "noise_layer_count": self.noise_layer_count,
            "noise_layers": [layer.as_dict() for layer in self.noise_layers],
            "material_mix_mode": self.material_mix_mode,
            "fixed_point": self.fixed_point.as_dict(),
            "error_trace": {
                "bit_width_ladder": list(self.bit_width_ladder),
                "tolerance": dict(sorted(self.tolerances.items())),
                "final_consistency_tolerance": self.final_consistency_tolerance,
                "embed_pixel_diffs": self.embed_pixel_diffs,
            },
            "concurrency": self.concurrency,
            "max_concurrency": self.max_concurrency,
            "point_budget": self.point_budget,
            "latency_budget_ms": self.latency_budget_ms,
            "snapshot_commit_interval": self.snapshot_commit_interval,
            "max_boundary_samples": self.max_boundary_samples,
        }


@dataclass(frozen=True)
class MaterialLayer:
    layer_index: int
    weights: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {"noise_layer": self.layer_index, "weights": dict(sorted(self.weights.items()))}


@dataclass(frozen=True)
class Material:
    material_id: str
    label: str
    layers: tuple[MaterialLayer, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.material_id,
            "label": self.label,
            "layers": [layer.as_dict() for layer in self.layers],
        }


@dataclass(frozen=True)
class Command:
    command_id: str
    op: str
    payload: dict[str, Any]


@dataclass
class Texture:
    name: str
    path: str
    width: int
    height: int
    mode: str
    pixels: np.ndarray
    luma: np.ndarray
    content_hash: str
    cache_key: str


@dataclass
class LoadedInput:
    input_dir: str
    textures: dict[str, Texture]
    config: GenerationConfig | None
    materials: tuple[Material, ...]
    commands: tuple[Command, ...]
    issues: list[Issue]
    config_raw: dict[str, Any] = field(default_factory=dict)
    materials_raw: dict[str, Any] = field(default_factory=dict)
    commands_raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.config is not None and not any(issue.code.startswith("E_CONFIG") for issue in self.issues)


@dataclass
class LayerArtifact:
    index: int
    key: str
    texture_key: str
    upstream_layer_key: str
    frequency: float
    amplitude: float
    unquantized: np.ndarray
    quantized: np.ndarray
    residual: np.ndarray
    scale: int
    quantization_step: int
    reused: bool = False

    def content_hash(self) -> str:
        import hashlib

        digest = hashlib.sha256()
        digest.update(self.key.encode("ascii"))
        digest.update(np.ascontiguousarray(self.quantized, dtype=np.int64).tobytes())
        return digest.hexdigest()


@dataclass
class MaterialChannelArtifact:
    key: str
    material_id: str
    channel: str
    layer_keys: tuple[str, ...]
    layer_indices: tuple[int, ...]
    mixed: np.ndarray
    final: np.ndarray
    contributions: dict[int, np.ndarray]
    boundary_samples: list[dict[str, Any]]
    boundary_total: int
    reused: bool = False

    def content_hash(self) -> str:
        import hashlib

        digest = hashlib.sha256()
        digest.update(self.key.encode("ascii"))
        digest.update(np.ascontiguousarray(self.final, dtype=np.int64).tobytes())
        return digest.hexdigest()


@dataclass
class MaterialArtifact:
    material_id: str
    channels: dict[str, MaterialChannelArtifact]

    def as_summary(self, scale: int) -> dict[str, Any]:
        from .fixed import fixed_channel_to_uint8

        channel_summary: dict[str, Any] = {}
        for channel, artifact in self.channels.items():
            values = artifact.final.astype(np.float64) / float(scale)
            if channel == "normal_perturbation":
                display = (values + 1.0) * 0.5
            else:
                display = values
            channel_summary[channel] = {
                "key": artifact.key,
                "digest": artifact.content_hash(),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "uint8_digest": __import__("hashlib").sha256(
                    fixed_channel_to_uint8(artifact.final, scale, channel).tobytes()
                ).hexdigest(),
                "display_min": float(np.min(display)),
                "display_max": float(np.max(display)),
            }
        return channel_summary


@dataclass
class BuildResult:
    config: GenerationConfig
    materials: tuple[Material, ...]
    textures: dict[str, Texture]
    layers: list[LayerArtifact]
    material_artifacts: dict[str, MaterialArtifact]
    final_key: str
    final_reused: bool
    issues: list[Issue]
    ledger_entries: list[dict[str, Any]]
    scheduler_events: list[dict[str, Any]]
    recomputed: list[str]
    reused: list[str]
    affected_layers: list[int]
    affected_channels: list[str]
    batches_consumed: int
    build_elapsed_ms: float

    @property
    def scale(self) -> int:
        return self.config.fixed_point.scale

    @property
    def final_digest(self) -> str:
        import hashlib

        digest = hashlib.sha256()
        for material_id in sorted(self.material_artifacts):
            for channel in sorted(self.material_artifacts[material_id].channels):
                digest.update(self.material_artifacts[material_id].channels[channel].final.tobytes())
        return digest.hexdigest()

    def final_summary(self) -> dict[str, Any]:
        return {
            "final_key": self.final_key,
            "digest": self.final_digest,
            "scale": self.scale,
            "materials": {
                material_id: artifact.as_summary(self.scale)
                for material_id, artifact in sorted(self.material_artifacts.items())
            },
        }

    @property
    def completed_layer_count(self) -> int:
        return len(self.layers)

    @property
    def total_layer_count(self) -> int:
        return self.config.noise_layer_count

    def preview_summary(self) -> dict[str, Any]:
        return {
            "revision_digest": self.final_digest,
            "completed_layers": self.completed_layer_count,
            "total_layers": self.total_layer_count,
            "progress": 1.0 if self.completed_layer_count == self.total_layer_count else 0.0,
            "final_channels": self.final_summary(),
        }
