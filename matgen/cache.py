"""Content-addressed cache and exact invalidation key derivation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .fixed import CHANNELS
from .hashing import fingerprint, stable_float
from .models import FixedPointConfig, Material, MaterialLayer


@dataclass
class CacheEntry:
    kind: str
    key: str
    meta: dict[str, Any] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)


class CacheStore:
    def __init__(self) -> None:
        self.entries: dict[str, CacheEntry] = {}

    def get(self, key: str) -> CacheEntry | None:
        return self.entries.get(key)

    def contains(self, key: str) -> bool:
        return key in self.entries

    def put(
        self,
        key: str,
        *,
        kind: str,
        meta: dict[str, Any] | None = None,
        arrays: dict[str, np.ndarray] | None = None,
    ) -> CacheEntry:
        entry = CacheEntry(
            kind=kind,
            key=key,
            meta=dict(meta or {}),
            arrays={name: np.asarray(value).copy() for name, value in (arrays or {}).items()},
        )
        self.entries[key] = entry
        return entry

    def all_entries(self) -> list[CacheEntry]:
        return [self.entries[key] for key in sorted(self.entries)]


def texture_cache_key(texture: Any) -> str:
    return fingerprint(
        {
            "kind": "texture",
            "content_hash": texture.content_hash,
            "mode": texture.mode,
            "width": texture.width,
            "height": texture.height,
        }
    )


def noise_layer_key(
    *,
    config: Any,
    layer_index: int,
    texture_key: str,
    frequency: float,
    amplitude: float,
    upstream_layer_key: str,
) -> str:
    fp = config.fixed_point
    return fingerprint(
        {
            "kind": "noise_layer",
            "version": 1,
            "seed": config.seed,
            "resolution": config.resolution,
            "layer_index": layer_index,
            "texture_key": texture_key,
            "frequency": stable_float(frequency),
            "amplitude": stable_float(amplitude),
            "bit_width": fp.bit_width,
            "fraction_bits": fp.fraction_bits,
            "quantization_level": fp.quantization_level,
            "quantization_tiers": list(fp.quantization_tiers),
            "rounding": dict(sorted(fp.rounding.items())),
            "upstream_layer_key": upstream_layer_key,
        }
    )


def _positive_layers(material: Material, channel: str) -> list[MaterialLayer]:
    return [layer for layer in sorted(material.layers, key=lambda row: row.layer_index) if layer.weights[channel] > 0]


def material_channel_key(
    *,
    material: Material,
    channel: str,
    layer_keys: dict[int, str],
    config: Any,
) -> str:
    fp: FixedPointConfig = config.fixed_point
    positive = _positive_layers(material, channel)
    return fingerprint(
        {
            "kind": "material_channel",
            "version": 1,
            "material_id": material.material_id,
            "channel": channel,
            "mix_mode": config.material_mix_mode,
            "bit_width": fp.bit_width,
            "fraction_bits": fp.fraction_bits,
            "quantization_level": fp.quantization_level,
            "weight_fraction_bits": fp.weight_fraction_bits,
            "rounding": fp.rounding[channel],
            "references": [
                {
                    "layer_index": layer.layer_index,
                    "weight": stable_float(layer.weights[channel]),
                    "layer_key": layer_keys[layer.layer_index],
                }
                for layer in positive
            ],
        }
    )


def final_synthesis_key(
    *,
    channel_keys: dict[str, dict[str, str]],
    config: Any,
) -> str:
    fp = config.fixed_point
    return fingerprint(
        {
            "kind": "final_synthesis",
            "version": 1,
            "resolution": config.resolution,
            "mix_mode": config.material_mix_mode,
            "bit_width": fp.bit_width,
            "fraction_bits": fp.fraction_bits,
            "quantization_level": fp.quantization_level,
            "channels": list(CHANNELS),
            "materials": {
                material_id: {channel: channel_keys[material_id][channel] for channel in CHANNELS}
                for material_id in sorted(channel_keys)
            },
        }
    )
