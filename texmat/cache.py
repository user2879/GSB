"""Content-fingerprint cache keys, dependency graph and precise invalidation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


from .config import CHANNELS, GenerationConfig, MaterialSpec
from .hashing import cache_key, combine_fingerprints, parameter_fingerprint
from .pipeline import WORK_BITS


@dataclass
class NodeRecord:
    kind: str                 # texture | layer | channel | final
    node_id: str
    cache_key: str
    depends_on: tuple[str, ...]
    references: tuple[str, ...]
    fingerprint: int
    payload_path: str | None = None


@dataclass
class InvalidationPlan:
    recompute: list[str] = field(default_factory=list)
    reuse: list[str] = field(default_factory=list)
    changed_inputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recompute": list(self.recompute),
            "reuse": list(self.reuse),
            "changed_inputs": list(self.changed_inputs),
        }


class DependencyGraph:
    """Holds every node key and the edges required for downstream invalidation."""

    def __init__(self) -> None:
        self.nodes: dict[str, NodeRecord] = {}

    def add(self, record: NodeRecord) -> None:
        self.nodes[record.node_id] = record

    def downstream(self, seeds: list[str]) -> list[str]:
        """Return all transitive dependents of the seed nodes, stable-sorted."""
        affected: set[str] = set(seeds)
        changed = True
        while changed:
            changed = False
            for node_id, record in self.nodes.items():
                if node_id in affected:
                    continue
                if any(dep in affected for dep in record.depends_on):
                    affected.add(node_id)
                    changed = True
        return sorted(affected)

    def plan(self, changed_inputs: list[str]) -> InvalidationPlan:
        affected = set(self.downstream(changed_inputs))
        recompute = sorted(node_id for node_id in self.nodes if node_id in affected)
        reuse = sorted(node_id for node_id in self.nodes if node_id not in affected)
        return InvalidationPlan(recompute=recompute, reuse=reuse,
                                changed_inputs=sorted(changed_inputs))


def texture_node_id(texture_name: str) -> str:
    return f"texture:{texture_name}"


def layer_node_id(index: int) -> str:
    return f"layer:layer{index:02d}"


def channel_node_id(material: str, channel: str) -> str:
    return f"channel:{material}:{channel}"


def final_node_id(material: str) -> str:
    return f"final:{material}"


def _quant_params(config: GenerationConfig) -> dict[str, Any]:
    return {
        "fractional_bits": config.fractional_bits,
        "quantize": config.quantize,
        "rounding": config.rounding,
        "resolution": config.active_tier,
        "work_bits": WORK_BITS,
    }


def _channel_params(material: MaterialSpec, channel: str) -> dict[str, Any]:
    spec = material.channels[channel]
    return {
        "material": material.name,
        "channel": channel,
        "blend_mode": material.blend_mode,
        "weights": list(material.weights),
        "layers": list(material.layers),
        "base": spec.base,
        "scale": spec.scale,
        "bias": spec.bias,
    }


def build_graph(config: GenerationConfig, materials: list[MaterialSpec],
                texture_fingerprints: dict[str, int]) -> DependencyGraph:
    graph = DependencyGraph()
    quant_fp = parameter_fingerprint(_quant_params(config))

    for texture_name, fp in sorted(texture_fingerprints.items()):
        node_id = texture_node_id(texture_name)
        graph.add(NodeRecord(
            kind="texture", node_id=node_id,
            cache_key=cache_key("texture", texture_name, fp),
            depends_on=(), references=(), fingerprint=fp,
            payload_path=f"textures/{texture_name}.npy",
        ))

    for spec in config.noise_layers:
        node_id = layer_node_id(spec.index)
        base = texture_node_id(spec.source_texture)
        layer_fp = combine_fingerprints(
            "layer", spec.index, texture_fingerprints[spec.source_texture], spec.fingerprint,
            spec.frequency, spec.amplitude, config.fractional_bits, WORK_BITS,
            config.quantize, config.rounding, config.active_tier,
        )
        graph.add(NodeRecord(
            kind="layer", node_id=node_id,
            cache_key=cache_key("layer", layer_fp, quant_fp),
            depends_on=(base,), references=(base,), fingerprint=layer_fp,
            payload_path=f"layers/layer{spec.index:02d}.npz",
        ))

    for material in materials:
        layer_deps = [layer_node_id(int(layer[5:])) for layer in material.layers]
        layer_fps = [graph.nodes[dep].fingerprint for dep in layer_deps]
        for channel in CHANNELS:
            node_id = channel_node_id(material.name, channel)
            params_fp = parameter_fingerprint(_channel_params(material, channel))
            channel_fp = combine_fingerprints("channel", material.name, channel,
                                              params_fp, *layer_fps)
            graph.add(NodeRecord(
                kind="channel", node_id=node_id,
                cache_key=cache_key("channel", channel_fp, quant_fp),
                depends_on=tuple(layer_deps), references=tuple(layer_deps),
                fingerprint=channel_fp,
                payload_path=f"channels/{material.name}/{channel}.npz",
            ))
        final_id = final_node_id(material.name)
        channel_deps = [channel_node_id(material.name, channel) for channel in CHANNELS]
        final_params_fp = parameter_fingerprint({
            "material": material.name,
            "blend_mode": material.blend_mode,
            "rounding": config.rounding,
            "bits": config.fractional_bits,
        })
        final_fp = combine_fingerprints(
            "final", material.name, final_params_fp,
            *[graph.nodes[dep].fingerprint for dep in channel_deps])
        graph.add(NodeRecord(
            kind="final", node_id=final_id,
            cache_key=cache_key("final", final_fp, quant_fp),
            depends_on=tuple(channel_deps), references=tuple(channel_deps),
            fingerprint=final_fp, payload_path=f"final/{material.name}.npz",
        ))
    return graph
