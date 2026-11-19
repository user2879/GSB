"""Strict YAML level-specification parser with stable validation errors."""

from __future__ import annotations

import math
import re
from typing import Any, Iterable

import yaml

from .canonical import Coord, coord_to_list, row_major_key
from .errors import SpecError
from .model import LevelSpec, RegionSpec

_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_REQUIRED_FIELDS = (
    "seed",
    "grid",
    "terrain",
    "regions",
    "resources",
    "spawn",
    "target",
    "forbidden_adjacent",
)


def load_spec_file(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        raise SpecError("E_SPEC_FIELD", f"cannot read specification: {exc}", "$") from exc
    return load_spec_text(text)


def load_spec_text(text: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SpecError("E_SPEC_FIELD", f"invalid YAML: {exc}", "$") from exc
    if not isinstance(value, dict):
        raise SpecError("E_SPEC_FIELD", "specification root must be a mapping", "$")
    return value


def parse_spec(raw: dict[str, Any]) -> LevelSpec:
    if not isinstance(raw, dict):
        raise SpecError("E_SPEC_FIELD", "specification root must be a mapping", "$")

    for key in _REQUIRED_FIELDS:
        if key not in raw:
            raise SpecError("E_SPEC_FIELD", f"missing required field '{key}'", key)

    seed = _require_int(raw, "seed", "seed", minimum=0)

    grid = _require_mapping(raw, "grid", "grid")
    width = _require_int(grid, "width", "grid.width", minimum=1, maximum=512)
    height = _require_int(grid, "height", "grid.height", minimum=1, maximum=512)

    terrain = _require_mapping(raw, "terrain", "terrain")
    weights_raw = _require_mapping(terrain, "weights", "terrain.weights")
    if not weights_raw:
        raise SpecError("E_SPEC_WEIGHT", "terrain weights must not be empty", "terrain.weights")
    weights: dict[str, float] = {}
    for name in sorted(weights_raw):
        if not isinstance(name, str) or not name or any(ch.isspace() for ch in name):
            raise SpecError("E_SPEC_FIELD", "terrain names must be non-empty tokens", "terrain.weights")
        weight = weights_raw[name]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise SpecError("E_SPEC_FIELD", "terrain weight must be numeric", f"terrain.weights.{name}")
        numeric = float(weight)
        if not math.isfinite(numeric) or numeric < 0:
            raise SpecError("E_SPEC_WEIGHT", "terrain weights must be finite and non-negative", f"terrain.weights.{name}")
        weights[name] = numeric
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 1e-9:
        raise SpecError(
            "E_SPEC_WEIGHT",
            f"terrain weights must sum to 1.0 (actual {total_weight:.12g})",
            "terrain.weights",
            {"sum": total_weight},
        )

    blocked_raw = terrain.get("blocked", [])
    if blocked_raw is None:
        blocked_raw = []
    if not isinstance(blocked_raw, list):
        raise SpecError("E_SPEC_FIELD", "terrain.blocked must be a list", "terrain.blocked")
    blocked: set[str] = set()
    for index, name in enumerate(blocked_raw):
        if not isinstance(name, str) or name not in weights:
            raise SpecError(
                "E_SPEC_FIELD",
                "blocked terrain must be a terrain name declared in weights",
                f"terrain.blocked[{index}]",
            )
        blocked.add(name)
    if len(blocked) == len(weights):
        raise SpecError("E_SPEC_WEIGHT", "at least one terrain type must be passable", "terrain.blocked")

    regions = _parse_regions(raw["regions"], width, height)

    resources = _require_mapping(raw, "resources", "resources")
    count_raw = resources.get("count")
    if count_raw is None:
        raise SpecError("E_SPEC_FIELD", "missing required field 'resources.count'", "resources.count")
    if not isinstance(count_raw, list) or len(count_raw) != 2:
        raise SpecError("E_SPEC_FIELD", "resources.count must be [minimum, maximum]", "resources.count")
    resource_min = _as_int(count_raw[0], "resources.count[0]", minimum=0)
    resource_max = _as_int(count_raw[1], "resources.count[1]", minimum=0)
    if resource_min > resource_max:
        raise SpecError("E_SPEC_BOUNDS", "resource minimum exceeds maximum", "resources.count")
    if resource_max > width * height:
        raise SpecError(
            "E_SPEC_BOUNDS",
            "resource maximum exceeds the number of grid cells",
            "resources.count",
            {"maximum": resource_max, "grid_cells": width * height},
        )

    allowed_raw = resources.get("allowed_terrain")
    if allowed_raw is None:
        raise SpecError("E_SPEC_FIELD", "missing required field 'resources.allowed_terrain'", "resources.allowed_terrain")
    if not isinstance(allowed_raw, list) or not allowed_raw:
        raise SpecError("E_SPEC_FIELD", "resources.allowed_terrain must be a non-empty list", "resources.allowed_terrain")
    allowed: set[str] = set()
    for index, name in enumerate(allowed_raw):
        if not isinstance(name, str) or name not in weights:
            raise SpecError(
                "E_SPEC_FIELD",
                "resource terrain must be declared in terrain.weights",
                f"resources.allowed_terrain[{index}]",
            )
        if name in blocked:
            raise SpecError(
                "E_SPEC_BOUNDS",
                "blocked terrain cannot host resources",
                f"resources.allowed_terrain[{index}]",
            )
        allowed.add(name)

    spawn = _parse_coord(raw["spawn"], "spawn", width, height)
    target = _parse_coord(raw["target"], "target", width, height)
    if spawn == target:
        raise SpecError("E_SPEC_FIELD", "spawn and target must be distinct", "target")

    forbidden = _parse_forbidden(raw["forbidden_adjacent"], weights)
    fixed_cells = _parse_fixed_cells(raw.get("fixed_cells", []), width, height, weights)

    normalized: dict[str, Any] = {
        "schema": "level-workbench/spec-v1",
        "seed": seed,
        "grid": {"width": width, "height": height},
        "terrain": {
            "weights": {name: weights[name] for name in sorted(weights)},
            "blocked": sorted(blocked),
        },
        "regions": [region.to_dict() for region in regions],
        "resources": {
            "count": [resource_min, resource_max],
            "allowed_terrain": sorted(allowed),
        },
        "spawn": coord_to_list(spawn),
        "target": coord_to_list(target),
        "forbidden_adjacent": [list(pair) for pair in sorted(forbidden)],
        "fixed_cells": [
            {"coord": coord_to_list(coord), "terrain": fixed_cells[coord]}
            for coord in sorted(fixed_cells, key=row_major_key)
        ],
    }

    return LevelSpec(
        seed=seed,
        width=width,
        height=height,
        terrain_weights=weights,
        blocked_terrain=frozenset(blocked),
        regions=tuple(regions),
        resource_min=resource_min,
        resource_max=resource_max,
        resource_allowed_terrain=frozenset(allowed),
        spawn=spawn,
        target=target,
        forbidden_adjacent=frozenset(forbidden),
        fixed_cells=fixed_cells,
        normalized=normalized,
    )


def _parse_regions(value: Any, width: int, height: int) -> list[RegionSpec]:
    if not isinstance(value, list):
        raise SpecError("E_SPEC_FIELD", "regions must be a list", "regions")
    result: list[RegionSpec] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(value):
        path = f"regions[{index}]"
        if not isinstance(item, dict):
            raise SpecError("E_SPEC_FIELD", "region must be a mapping", path)
        region_id = item.get("id")
        if not isinstance(region_id, str) or not _ID_RE.fullmatch(region_id):
            raise SpecError("E_SPEC_FIELD", "region.id must match [A-Za-z0-9_.-]+", f"{path}.id")
        if region_id in seen_ids:
            raise SpecError("E_SPEC_FIELD", "duplicate region id", f"{path}.id")
        seen_ids.add(region_id)

        has_bounds = "bounds" in item
        has_tiles = "tiles" in item
        if has_bounds == has_tiles:
            raise SpecError("E_SPEC_FIELD", "region needs exactly one of bounds or tiles", path)
        if has_bounds:
            bounds_raw = item["bounds"]
            if not isinstance(bounds_raw, list) or len(bounds_raw) != 4:
                raise SpecError("E_SPEC_FIELD", "region bounds must be [x0,y0,x1,y1]", f"{path}.bounds")
            x0 = _as_int(bounds_raw[0], f"{path}.bounds[0]")
            y0 = _as_int(bounds_raw[1], f"{path}.bounds[1]")
            x1 = _as_int(bounds_raw[2], f"{path}.bounds[2]")
            y1 = _as_int(bounds_raw[3], f"{path}.bounds[3]")
            if x0 > x1 or y0 > y1:
                raise SpecError("E_SPEC_BOUNDS", "region bounds are inverted", f"{path}.bounds")
            if x0 < 0 or y0 < 0 or x1 >= width or y1 >= height:
                raise SpecError(
                    "E_SPEC_BOUNDS",
                    "region bounds exceed the grid",
                    f"{path}.bounds",
                    {"grid": [width, height], "bounds": [x0, y0, x1, y1]},
                )
            result.append(RegionSpec(id=region_id, bounds=(x0, y0, x1, y1), source="bounds"))
        else:
            tiles_raw = item["tiles"]
            if not isinstance(tiles_raw, list) or not tiles_raw:
                raise SpecError("E_SPEC_FIELD", "region tiles must be a non-empty list", f"{path}.tiles")
            tiles: list[Coord] = []
            for tile_index, tile in enumerate(tiles_raw):
                tiles.append(_parse_coord(tile, f"{path}.tiles[{tile_index}]", width, height))
            tiles = sorted(set(tiles), key=row_major_key)
            xs = [coord[0] for coord in tiles]
            ys = [coord[1] for coord in tiles]
            result.append(
                RegionSpec(
                    id=region_id,
                    bounds=(min(xs), min(ys), max(xs), max(ys)),
                    source="tiles",
                    explicit_tiles=tuple(tiles),
                )
            )
    result.sort(key=lambda region: region.id)
    return result


def _parse_forbidden(value: Any, weights: dict[str, float]) -> list[tuple[str, str]]:
    if not isinstance(value, list):
        raise SpecError("E_SPEC_FIELD", "forbidden_adjacent must be a list", "forbidden_adjacent")
    result: set[tuple[str, str]] = set()
    for index, pair in enumerate(value):
        path = f"forbidden_adjacent[{index}]"
        if not isinstance(pair, list) or len(pair) != 2:
            raise SpecError("E_SPEC_FIELD", "forbidden pair must contain two terrain names", path)
        left, right = pair
        if not isinstance(left, str) or not isinstance(right, str):
            raise SpecError("E_SPEC_FIELD", "forbidden terrain names must be strings", path)
        if left not in weights or right not in weights:
            raise SpecError("E_SPEC_FIELD", "forbidden terrain must be declared in terrain.weights", path)
        if left == right:
            raise SpecError("E_SPEC_FIELD", "forbidden pair must use two distinct terrain names", path)
        result.add(tuple(sorted((left, right))))
    return sorted(result)


def _parse_fixed_cells(
    value: Any, width: int, height: int, weights: dict[str, float]
) -> dict[Coord, str]:
    if value is None:
        return {}
    if not isinstance(value, list):
        raise SpecError("E_SPEC_FIELD", "fixed_cells must be a list", "fixed_cells")
    result: dict[Coord, str] = {}
    for index, item in enumerate(value):
        path = f"fixed_cells[{index}]"
        if not isinstance(item, dict):
            raise SpecError("E_SPEC_FIELD", "fixed cell must be a mapping", path)
        if "coord" not in item or "terrain" not in item:
            raise SpecError("E_SPEC_FIELD", "fixed cell needs coord and terrain", path)
        coord = _parse_coord(item["coord"], f"{path}.coord", width, height)
        terrain = item["terrain"]
        if not isinstance(terrain, str) or terrain not in weights:
            raise SpecError("E_SPEC_FIELD", "fixed-cell terrain must be declared in terrain.weights", f"{path}.terrain")
        if coord in result:
            raise SpecError("E_SPEC_FIELD", "duplicate fixed-cell coordinate", f"{path}.coord")
        result[coord] = terrain
    return result


def _parse_coord(value: Any, path: str, width: int, height: int) -> Coord:
    if isinstance(value, dict):
        if "x" not in value or "y" not in value:
            raise SpecError("E_SPEC_FIELD", "coordinate mapping needs x and y", path)
        x_raw, y_raw = value["x"], value["y"]
    elif isinstance(value, list) and len(value) == 2:
        x_raw, y_raw = value
    else:
        raise SpecError("E_SPEC_FIELD", "coordinate must be [x,y] or {x:..., y:...}", path)
    x = _as_int(x_raw, f"{path}[0]")
    y = _as_int(y_raw, f"{path}[1]")
    if x < 0 or y < 0 or x >= width or y >= height:
        raise SpecError(
            "E_SPEC_BOUNDS",
            "coordinate exceeds the grid",
            path,
            {"coord": [x, y], "grid": [width, height]},
        )
    return (x, y)


def _as_int(value: Any, path: str, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecError("E_SPEC_FIELD", "value must be an integer", path)
    if minimum is not None and value < minimum:
        raise SpecError("E_SPEC_BOUNDS", f"value must be at least {minimum}", path)
    if maximum is not None and value > maximum:
        raise SpecError("E_SPEC_BOUNDS", f"value must be at most {maximum}", path)
    return value


def _require_int(
    obj: dict[str, Any],
    key: str,
    path: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if key not in obj:
        raise SpecError("E_SPEC_FIELD", f"missing required field '{key}'", path)
    return _as_int(obj[key], path, minimum=minimum, maximum=maximum)


def _require_mapping(obj: dict[str, Any], key: str, path: str) -> dict[str, Any]:
    if key not in obj:
        raise SpecError("E_SPEC_FIELD", f"missing required field '{key}'", path)
    value = obj[key]
    if not isinstance(value, dict):
        raise SpecError("E_SPEC_FIELD", f"{key} must be a mapping", path)
    return value
