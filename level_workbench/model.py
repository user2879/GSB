"""Typed, immutable specification values and a mutable level artifact value."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .canonical import Coord, coord_to_list, grid_hash, list_to_coord, row_major_key


@dataclass(frozen=True)
class RegionSpec:
    id: str
    bounds: tuple[int, int, int, int]
    source: str = "bounds"
    explicit_tiles: tuple[Coord, ...] = ()

    @property
    def tiles(self) -> tuple[Coord, ...]:
        if self.explicit_tiles:
            return self.explicit_tiles
        x0, y0, x1, y1 = self.bounds
        return tuple((x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1))

    @property
    def anchor(self) -> Coord:
        return min(self.tiles, key=row_major_key)

    def to_dict(self) -> dict[str, Any]:
        if self.source == "tiles":
            return {"id": self.id, "tiles": [coord_to_list(c) for c in self.tiles]}
        return {"id": self.id, "bounds": list(self.bounds)}


@dataclass(frozen=True)
class LevelSpec:
    seed: int
    width: int
    height: int
    terrain_weights: dict[str, float]
    blocked_terrain: frozenset[str]
    regions: tuple[RegionSpec, ...]
    resource_min: int
    resource_max: int
    resource_allowed_terrain: frozenset[str]
    spawn: Coord
    target: Coord
    forbidden_adjacent: frozenset[tuple[str, str]]
    fixed_cells: dict[Coord, str] = field(default_factory=dict)
    normalized: dict[str, Any] = field(default_factory=dict)

    @property
    def passable_terrain(self) -> tuple[str, ...]:
        return tuple(sorted(name for name in self.terrain_weights if name not in self.blocked_terrain))

    @property
    def corridor_terrain(self) -> str:
        passable = self.passable_terrain
        if not passable:
            raise ValueError("spec has no passable terrain")
        return passable[0]

    @property
    def forbidden_pairs(self) -> frozenset[frozenset[str]]:
        return frozenset(frozenset(pair) for pair in self.forbidden_adjacent)

    def region_by_id(self, region_id: str) -> RegionSpec:
        for region in self.regions:
            if region.id == region_id:
                return region
        raise KeyError(region_id)


@dataclass
class Level:
    width: int
    height: int
    cells: list[list[str]]
    resources: list[Coord]
    spawn: Coord
    target: Coord

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Level":
        grid = value["grid"]
        width = int(grid["width"])
        height = int(grid["height"])
        cells = [[str(token) for token in row] for row in grid["cells"]]
        resources = [list_to_coord(item) for item in value.get("resources", [])]
        return cls(
            width=width,
            height=height,
            cells=cells,
            resources=resources,
            spawn=list_to_coord(value["spawn"]),
            target=list_to_coord(value["target"]),
        )

    def copy(self) -> "Level":
        return Level(
            width=self.width,
            height=self.height,
            cells=[row[:] for row in self.cells],
            resources=list(self.resources),
            spawn=self.spawn,
            target=self.target,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid": {
                "width": self.width,
                "height": self.height,
                "cells": [row[:] for row in self.cells],
            },
            "resources": [coord_to_list(c) for c in self.resources],
            "spawn": coord_to_list(self.spawn),
            "target": coord_to_list(self.target),
            "hashes": {"grid_sha256": grid_hash(self.cells)},
        }

    def terrain_at(self, coord: Coord) -> str:
        x, y = coord
        return self.cells[y][x]

    def set_terrain(self, coord: Coord, terrain: str) -> None:
        x, y = coord
        self.cells[y][x] = terrain

    def in_bounds(self, coord: Coord) -> bool:
        x, y = coord
        return 0 <= x < self.width and 0 <= y < self.height
