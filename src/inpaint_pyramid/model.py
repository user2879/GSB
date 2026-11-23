"""Immutable pyramid specification and mutable diagnostic parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .pyramid import cumulative_dimensions


Rect = tuple[int, int, int, int]


@dataclass(frozen=True)
class PyramidSpec:
    base_width: int
    base_height: int
    levels: int
    factors: tuple[int, ...]
    cumulative_factors: tuple[int, ...]
    sizes: tuple[tuple[int, int], ...]
    tile_size: int
    fraction_bits: int
    tile_order: str = "row_major"

    @classmethod
    def create(
        cls,
        base_width: int,
        base_height: int,
        levels: int,
        factors: list[int] | tuple[int, ...],
        tile_size: int,
        fraction_bits: int,
        tile_order: str = "row_major",
    ) -> "PyramidSpec":
        factors_tuple = tuple(int(value) for value in factors)
        sizes = cumulative_dimensions(base_width, base_height, factors_tuple)
        cumulative: list[int] = []
        product = 1
        for factor in factors_tuple:
            product *= factor
            cumulative.append(product)
        return cls(
            base_width=base_width,
            base_height=base_height,
            levels=levels,
            factors=factors_tuple,
            cumulative_factors=tuple(cumulative),
            sizes=tuple(sizes),
            tile_size=tile_size,
            fraction_bits=fraction_bits,
            tile_order=tile_order,
        )

    @property
    def scale(self) -> int:
        return 1 << self.fraction_bits

    def tile_rows(self, level: int) -> int:
        return self.sizes[level][1] // self.tile_size

    def tile_cols(self, level: int) -> int:
        return self.sizes[level][0] // self.tile_size

    def tile_count(self, level: int) -> int:
        return self.tile_rows(level) * self.tile_cols(level)

    def all_tiles(self) -> list[tuple[int, int, int]]:
        return [
            (level, row, col)
            for level in range(self.levels)
            for row in range(self.tile_rows(level))
            for col in range(self.tile_cols(level))
        ]

    def tile_bounds(self, level: int, row: int, col: int) -> tuple[int, int, int, int]:
        x0 = col * self.tile_size
        y0 = row * self.tile_size
        return (x0, y0, x0 + self.tile_size, y0 + self.tile_size)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "base_size": [self.base_width, self.base_height],
            "levels": self.levels,
            "downsample_factors": list(self.factors),
            "cumulative_factors": list(self.cumulative_factors),
            "level_sizes": [list(size) for size in self.sizes],
            "tile_size": self.tile_size,
            "tile_order": self.tile_order,
            "fraction_bits": self.fraction_bits,
            "rounding": "half_up",
        }


@dataclass
class EffectiveConfig:
    mask_rects: list[list[Rect]]
    iterations_by_layer: list[int]
    guide_strength_q: list[int]
    style_weights_q: list[tuple[int, int, int]]
    quant_levels_by_layer: list[int]
    composition_weights_q: list[int]
    mask_fill: str
    style_inside: str
    style_outside: str

    def clone(self) -> "EffectiveConfig":
        return EffectiveConfig(
            mask_rects=[[tuple(int(v) for v in rect) for rect in rects] for rects in self.mask_rects],
            iterations_by_layer=list(self.iterations_by_layer),
            guide_strength_q=list(self.guide_strength_q),
            style_weights_q=[tuple(weights) for weights in self.style_weights_q],
            quant_levels_by_layer=list(self.quant_levels_by_layer),
            composition_weights_q=list(self.composition_weights_q),
            mask_fill=self.mask_fill,
            style_inside=self.style_inside,
            style_outside=self.style_outside,
        )

    def mask_array(self, level: int, spec: PyramidSpec) -> np.ndarray:
        width, height = spec.sizes[level]
        mask = np.zeros((height, width), dtype=bool)
        for x, y, w, h in self.mask_rects[level]:
            mask[y : y + h, x : x + w] = True
        return mask

    def all_masks(self, spec: PyramidSpec) -> list[np.ndarray]:
        return [self.mask_array(level, spec) for level in range(spec.levels)]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "mask_rects": [[list(rect) for rect in rects] for rects in self.mask_rects],
            "iterations_by_layer": list(self.iterations_by_layer),
            "guide_strength_q": list(self.guide_strength_q),
            "style_weights_q": [list(weights) for weights in self.style_weights_q],
            "quant_levels_by_layer": list(self.quant_levels_by_layer),
            "composition_weights_q": list(self.composition_weights_q),
            "mask_fill": self.mask_fill,
            "style_inside": self.style_inside,
            "style_outside": self.style_outside,
        }

    @classmethod
    def from_metadata(cls, payload: dict[str, Any]) -> "EffectiveConfig":
        return cls(
            mask_rects=[
                [tuple(int(v) for v in rect) for rect in rects]
                for rects in payload["mask_rects"]
            ],
            iterations_by_layer=[int(v) for v in payload["iterations_by_layer"]],
            guide_strength_q=[int(v) for v in payload["guide_strength_q"]],
            style_weights_q=[
                tuple(int(v) for v in weights) for weights in payload["style_weights_q"]
            ],
            quant_levels_by_layer=[int(v) for v in payload["quant_levels_by_layer"]],
            composition_weights_q=[int(v) for v in payload["composition_weights_q"]],
            mask_fill=str(payload["mask_fill"]),
            style_inside=str(payload["style_inside"]),
            style_outside=str(payload["style_outside"]),
        )


@dataclass(frozen=True)
class Change:
    changed: bool
    kind: str
    layer: int | None = None
    rect: Rect | None = None
    rects: tuple[Rect, ...] = ()


@dataclass
class ConfigBundle:
    spec: PyramidSpec
    initial: EffectiveConfig
    image_dir: Path
    source_name: str
    style_name: str
    source_path: Path
    style_path: Path
    source_rgb: np.ndarray
    style_rgb: np.ndarray
    source_fingerprint: str
    style_fingerprint: str
    base_config_hash: str
    threads: int
    recompute_tile_budget: int
    commit_every_commands: int
    raw_config: dict[str, Any]
    image_records: list[dict[str, Any]] = field(default_factory=list)

    def new_mutable(self) -> EffectiveConfig:
        return self.initial.clone()

    def effective_config_payload(self, mutable: EffectiveConfig) -> dict[str, Any]:
        return {
            "pyramid": self.spec.to_metadata(),
            "effective": mutable.to_metadata(),
            "source_fingerprint": self.source_fingerprint,
            "style_fingerprint": self.style_fingerprint,
        }

    def effective_config_hash(self, mutable: EffectiveConfig) -> str:
        from .util import sha256_json

        return sha256_json(self.effective_config_payload(mutable))


@dataclass
class CommandStream:
    path: Path
    commands: list[dict[str, Any]]
    commands_hash: str
