"""Independent artifact verifier.

This module deliberately consumes only the normalized specification and the
serialized level document. It never reads generator instances, sub-seeds,
repair traces or cached reachability state.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

from .canonical import Coord, coord_to_list, grid_hash, row_major_key
from .errors import VerificationError
from .model import LevelSpec

_NEIGHBORS: tuple[Coord, ...] = ((0, 1), (1, 0), (0, -1), (-1, 0))
_ISSUE_PRIORITY = {
    "E_LEVEL_MALFORMED": 0,
    "E_GRID_INVALID": 1,
    "E_GRID_HASH": 2,
    "E_SPAWN_INVALID": 3,
    "E_TARGET_INVALID": 4,
    "E_CONNECT_UNSAT": 5,
    "E_RESOURCE_RANGE": 6,
    "E_RESOURCE_DUPLICATE": 7,
    "E_RESOURCE_BOUNDS": 8,
    "E_RESOURCE_TERRAIN": 9,
    "E_RESOURCE_ENDPOINT": 10,
    "E_FORBIDDEN_ADJACENT": 11,
}


def verify_artifact(spec: LevelSpec, level_doc: dict[str, Any]) -> dict[str, Any]:
    try:
        parsed = _parse_artifact(level_doc)
    except VerificationError as exc:
        issue = {
            "code": exc.code,
            "coord": [0, 0],
            "message": exc.message,
            "input_fragment": exc.details,
        }
        return {
            "schema": "level-workbench/verification-v1",
            "status": "fail",
            "constraints": [
                {
                    "constraint": "grid",
                    "status": "fail",
                    "details": {"issues": [issue]},
                }
            ],
            "issues": [issue],
            "first_violation": issue,
        }

    width = parsed["width"]
    height = parsed["height"]
    cells = parsed["cells"]
    resources = parsed["resources"]
    spawn = parsed["spawn"]
    target = parsed["target"]
    constraints: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    grid_issues = _verify_grid(spec, level_doc, width, height, cells, parsed["grid_hash"])
    issues.extend(grid_issues)
    constraints.append(
        {
            "constraint": "grid",
            "status": "fail" if grid_issues else "pass",
            "details": {
                "width": width,
                "height": height,
                "grid_sha256": parsed["grid_hash"],
                "issues": grid_issues,
            },
        }
    )

    spawn_issues = _verify_endpoint(spec, cells, spawn, "spawn")
    issues.extend(spawn_issues)
    constraints.append(
        {
            "constraint": "spawn",
            "status": "fail" if spawn_issues else "pass",
            "details": {
                "coord": coord_to_list(spawn),
                "terrain": _terrain_at(cells, spawn),
                "issues": spawn_issues,
            },
        }
    )

    target_issues = _verify_endpoint(spec, cells, target, "target")
    issues.extend(target_issues)
    constraints.append(
        {
            "constraint": "target",
            "status": "fail" if target_issues else "pass",
            "details": {
                "coord": coord_to_list(target),
                "terrain": _terrain_at(cells, target),
                "issues": target_issues,
            },
        }
    )

    connectivity = _verify_connectivity(spec, width, height, cells, spawn, target)
    issues.extend(connectivity["issues"])
    constraints.append(connectivity["entry"])

    resource_result = _verify_resources(spec, width, height, cells, resources, spawn, target)
    issues.extend(resource_result["issues"])
    constraints.append(resource_result["entry"])

    forbidden = _verify_forbidden(spec, width, height, cells)
    issues.extend(forbidden["issues"])
    constraints.append(forbidden["entry"])

    issues.sort(key=_issue_sort_key)
    first = issues[0] if issues else None
    return {
        "schema": "level-workbench/verification-v1",
        "status": "fail" if issues else "pass",
        "constraints": constraints,
        "issues": issues,
        "first_violation": first,
    }


def verify_level_file(spec: LevelSpec, level_doc: dict[str, Any]) -> dict[str, Any]:
    return verify_artifact(spec, level_doc)


def _parse_artifact(level_doc: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(level_doc, dict):
        raise VerificationError("E_LEVEL_MALFORMED", "level artifact root must be a mapping")
    grid = level_doc.get("grid")
    if not isinstance(grid, dict):
        raise VerificationError("E_LEVEL_MALFORMED", "level artifact is missing grid")
    width = grid.get("width")
    height = grid.get("height")
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise VerificationError("E_GRID_INVALID", "grid width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height < 1:
        raise VerificationError("E_GRID_INVALID", "grid height must be a positive integer")
    rows = grid.get("cells")
    if not isinstance(rows, list) or len(rows) != height:
        raise VerificationError("E_GRID_INVALID", "grid row count does not match height")
    cells: list[list[str]] = []
    for y, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != width:
            raise VerificationError("E_GRID_INVALID", f"grid row {y} does not match width")
        if any(not isinstance(token, str) for token in row):
            raise VerificationError("E_GRID_INVALID", f"grid row {y} contains a non-string terrain")
        cells.append(row[:])

    spawn = _parse_coord(level_doc.get("spawn"), "spawn")
    target = _parse_coord(level_doc.get("target"), "target")
    resources_raw = level_doc.get("resources")
    if not isinstance(resources_raw, list):
        raise VerificationError("E_LEVEL_MALFORMED", "resources must be a list")
    resources = [_parse_coord(item, f"resources[{index}]") for index, item in enumerate(resources_raw)]
    hashes = level_doc.get("hashes", {})
    declared_grid_hash = hashes.get("grid_sha256") if isinstance(hashes, dict) else None
    return {
        "width": width,
        "height": height,
        "cells": cells,
        "resources": resources,
        "spawn": spawn,
        "target": target,
        "grid_hash": grid_hash(cells),
        "declared_grid_hash": declared_grid_hash,
    }


def _parse_coord(value: Any, path: str) -> Coord:
    if not isinstance(value, list) or len(value) != 2:
        raise VerificationError("E_LEVEL_MALFORMED", f"{path} must be [x,y]")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise VerificationError("E_LEVEL_MALFORMED", f"{path} must contain integers")
    return (value[0], value[1])


def _verify_grid(
    spec: LevelSpec,
    level_doc: dict[str, Any],
    width: int,
    height: int,
    cells: list[list[str]],
    actual_hash: str,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if width != spec.width or height != spec.height:
        issues.append(
            {
                "code": "E_GRID_INVALID",
                "coord": [0, 0],
                "message": "artifact dimensions differ from the specification",
                "input_fragment": {
                    "expected": [spec.width, spec.height],
                    "actual": [width, height],
                },
            }
        )
    first_unknown: Coord | None = None
    for y, row in enumerate(cells):
        for x, token in enumerate(row):
            if token not in spec.terrain_weights:
                first_unknown = (x, y)
                break
        if first_unknown is not None:
            break
    if first_unknown is not None:
        token = cells[first_unknown[1]][first_unknown[0]]
        issues.append(
            {
                "code": "E_GRID_INVALID",
                "coord": coord_to_list(first_unknown),
                "message": "artifact contains terrain not declared by the specification",
                "input_fragment": {"terrain": token, "coord": coord_to_list(first_unknown)},
            }
        )
    hashes = level_doc.get("hashes")
    declared = hashes.get("grid_sha256") if isinstance(hashes, dict) else None
    if declared is not None and declared != actual_hash:
        issues.append(
            {
                "code": "E_GRID_HASH",
                "coord": [0, 0],
                "message": "declared grid hash does not match grid contents",
                "input_fragment": {"declared": declared, "actual": actual_hash},
            }
        )
    return issues


def _verify_endpoint(
    spec: LevelSpec, cells: list[list[str]], coord: Coord, kind: str
) -> list[dict[str, Any]]:
    code = "E_SPAWN_INVALID" if kind == "spawn" else "E_TARGET_INVALID"
    if not _in_bounds(coord, len(cells[0]), len(cells)):
        return [
            {
                "code": code,
                "coord": coord_to_list(coord),
                "message": f"{kind} is outside the grid",
                "input_fragment": {"coord": coord_to_list(coord)},
            }
        ]
    terrain = cells[coord[1]][coord[0]]
    if terrain in spec.blocked_terrain:
        return [
            {
                "code": code,
                "coord": coord_to_list(coord),
                "message": f"{kind} lies on blocked terrain",
                "input_fragment": {"coord": coord_to_list(coord), "terrain": terrain},
            }
        ]
    return []


def _verify_connectivity(
    spec: LevelSpec,
    width: int,
    height: int,
    cells: list[list[str]],
    spawn: Coord,
    target: Coord,
) -> dict[str, Any]:
    def passable(coord: Coord) -> bool:
        if not _in_bounds(coord, width, height):
            return False
        return cells[coord[1]][coord[0]] not in spec.blocked_terrain

    paths: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    goals: list[tuple[str, str, Coord]] = [("target", "target", target)]
    goals.extend(("region", region.id, region.anchor) for region in spec.regions)
    for kind, goal_id, coord in goals:
        path = _bfs(width, height, spawn, coord, passable)
        label = "target" if kind == "target" else f"region:{goal_id}"
        if path is None:
            issues.append(
                {
                    "code": "E_CONNECT_UNSAT",
                    "coord": coord_to_list(coord),
                    "goal_kind": kind,
                    "goal_id": goal_id,
                    "message": "goal is not reachable from spawn",
                    "input_fragment": {
                        "spawn": coord_to_list(spawn),
                        "goal": coord_to_list(coord),
                        "goal_label": label,
                    },
                }
            )
        else:
            paths.append(
                {
                    "from": "spawn",
                    "to": label,
                    "goal_kind": kind,
                    "goal_id": goal_id,
                    "tiles": [coord_to_list(item) for item in path],
                }
            )
    return {
        "entry": {
            "constraint": "connectivity",
            "status": "fail" if issues else "pass",
            "details": {"paths": paths, "issues": issues},
        },
        "issues": issues,
    }


def _verify_resources(
    spec: LevelSpec,
    width: int,
    height: int,
    cells: list[list[str]],
    resources: list[Coord],
    spawn: Coord,
    target: Coord,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    count = len(resources)
    if count < spec.resource_min or count > spec.resource_max:
        coords = [coord_to_list(coord) for coord in sorted(resources, key=row_major_key)]
        issues.append(
            {
                "code": "E_RESOURCE_RANGE",
                "coord": coords[0] if coords else [0, 0],
                "message": "resource count is outside the required range",
                "input_fragment": {
                    "range": [spec.resource_min, spec.resource_max],
                    "count": count,
                    "coords": coords,
                },
            }
        )
    seen: set[Coord] = set()
    for coord in resources:
        listed = coord_to_list(coord)
        if coord in seen:
            issues.append(
                {
                    "code": "E_RESOURCE_DUPLICATE",
                    "coord": listed,
                    "message": "duplicate resource coordinate",
                    "input_fragment": {"coord": listed},
                }
            )
            continue
        seen.add(coord)
        if not _in_bounds(coord, width, height):
            issues.append(
                {
                    "code": "E_RESOURCE_BOUNDS",
                    "coord": listed,
                    "message": "resource coordinate is outside the grid",
                    "input_fragment": {"coord": listed, "grid": [width, height]},
                }
            )
            continue
        terrain = cells[coord[1]][coord[0]]
        if terrain not in spec.resource_allowed_terrain or terrain in spec.blocked_terrain:
            issues.append(
                {
                    "code": "E_RESOURCE_TERRAIN",
                    "coord": listed,
                    "message": "resource lies on disallowed terrain",
                    "input_fragment": {"coord": listed, "terrain": terrain},
                }
            )
        if coord == spawn or coord == target:
            issues.append(
                {
                    "code": "E_RESOURCE_ENDPOINT",
                    "coord": listed,
                    "message": "resource overlaps spawn or target",
                    "input_fragment": {"coord": listed},
                }
            )
    return {
        "entry": {
            "constraint": "resources",
            "status": "fail" if issues else "pass",
            "details": {
                "range": [spec.resource_min, spec.resource_max],
                "count": count,
                "coords": [coord_to_list(coord) for coord in sorted(resources, key=row_major_key)],
                "allowed_terrain": sorted(spec.resource_allowed_terrain),
                "issues": issues,
            },
        },
        "issues": issues,
    }


def _verify_forbidden(
    spec: LevelSpec, width: int, height: int, cells: list[list[str]]
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for y in range(height):
        for x in range(width):
            left = cells[y][x]
            for nx, ny in ((x + 1, y), (x, y + 1)):
                if nx >= width or ny >= height:
                    continue
                right = cells[ny][nx]
                pair = frozenset((left, right))
                if left != right and pair in spec.forbidden_pairs:
                    issues.append(
                        {
                            "code": "E_FORBIDDEN_ADJACENT",
                            "coord": [x, y],
                            "neighbor": [nx, ny],
                            "message": "forbidden terrain pair is adjacent",
                            "input_fragment": {
                                "cells": [
                                    {"coord": [x, y], "terrain": left},
                                    {"coord": [nx, ny], "terrain": right},
                                ],
                                "pair": sorted((left, right)),
                            },
                        }
                    )
    return {
        "entry": {
            "constraint": "forbidden_adjacent",
            "status": "fail" if issues else "pass",
            "details": {
                "pairs": [list(pair) for pair in sorted(spec.forbidden_adjacent)],
                "checked_edges": width * (height - 1) + height * (width - 1),
                "violations": issues,
                "first_violation": issues[0] if issues else None,
            },
        },
        "issues": issues,
    }


def _bfs(
    width: int,
    height: int,
    start: Coord,
    goal: Coord,
    passable: Callable[[Coord], bool],
) -> list[Coord] | None:
    if not passable(start) or not passable(goal):
        return None
    queue: deque[Coord] = deque([start])
    parent: dict[Coord, Coord | None] = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            path: list[Coord] = []
            cursor: Coord | None = goal
            while cursor is not None:
                path.append(cursor)
                cursor = parent[cursor]
            path.reverse()
            return path
        x, y = current
        for dx, dy in _NEIGHBORS:
            candidate = (x + dx, y + dy)
            if candidate in parent or not passable(candidate):
                continue
            parent[candidate] = current
            queue.append(candidate)
    return None


def _in_bounds(coord: Coord, width: int, height: int) -> bool:
    x, y = coord
    return 0 <= x < width and 0 <= y < height


def _terrain_at(cells: list[list[str]], coord: Coord) -> str | None:
    x, y = coord
    if 0 <= y < len(cells) and 0 <= x < len(cells[0]):
        return cells[y][x]
    return None


def _issue_sort_key(issue: dict[str, Any]) -> tuple[int, int, int]:
    x, y = issue["coord"]
    return (y, x, _ISSUE_PRIORITY.get(issue["code"], 99))
