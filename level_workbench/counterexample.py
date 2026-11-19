"""Minimal, replayable counterexamples for the first verifier violation."""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Iterable

from .canonical import Coord, coord_to_list, row_major_key
from .model import LevelSpec


def build_minimal_counterexample(
    spec: LevelSpec, level_doc: dict[str, Any], verification: dict[str, Any]
) -> dict[str, Any] | None:
    violation = verification.get("first_violation")
    if not violation:
        return None
    code = violation.get("code")
    if code == "E_FORBIDDEN_ADJACENT":
        result = _forbidden_counterexample(spec, violation)
        return _attach_source_hashes(result, spec, level_doc)
    if code in {
        "E_RESOURCE_RANGE",
        "E_RESOURCE_DUPLICATE",
        "E_RESOURCE_BOUNDS",
        "E_RESOURCE_TERRAIN",
        "E_RESOURCE_ENDPOINT",
    }:
        result = _resource_counterexample(spec, violation)
        return _attach_source_hashes(result, spec, level_doc)
    if code == "E_CONNECT_UNSAT":
        result = _connectivity_counterexample(spec, level_doc, violation)
        return _attach_source_hashes(result, spec, level_doc)
    if code in {"E_SPAWN_INVALID", "E_TARGET_INVALID"}:
        result = _single_cell_counterexample(violation)
        return _attach_source_hashes(result, spec, level_doc)
    result = {
        "schema": "level-workbench/counterexample-v1",
        "violation_code": code,
        "first_violation": violation,
        "minimizable": False,
        "explanation": "this violation has no coordinate-local counterexample reduction",
        "elements": [],
        "reproduction": None,
        "checks": {"reproduces": False, "removals": []},
    }
    return _attach_source_hashes(result, spec, level_doc)


def _forbidden_counterexample(spec: LevelSpec, violation: dict[str, Any]) -> dict[str, Any]:
    fragment = violation.get("input_fragment", {})
    raw_cells = fragment.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) < 2:
        coord = tuple(violation["coord"])
        raw_cells = [{"coord": list(coord), "terrain": None}]
    elements = [
        {"coord": [int(item["coord"][0]), int(item["coord"][1])], "terrain": item["terrain"]}
        for item in raw_cells
    ]
    elements.sort(key=lambda item: row_major_key((item["coord"][0], item["coord"][1])))

    def predicate(items: list[dict[str, Any]]) -> bool:
        return _forbidden_present(spec, items)

    minimized, removal_checks = _minimize(elements, predicate)
    reproduces = predicate(minimized)
    return {
        "schema": "level-workbench/counterexample-v1",
        "violation_code": violation["code"],
        "first_violation": violation,
        "minimizable": True,
        "elements": minimized,
        "reproduction": {
            "kind": "standalone_tile_patch",
            "forbidden_pairs": [list(pair) for pair in sorted(spec.forbidden_adjacent)],
            "rule": "the supplied adjacent tile assignments alone contain a forbidden pair",
        },
        "checks": {
            "reproduces": reproduces,
            "removal_checks": removal_checks,
            "all_single_element_removals_disappear": all(
                not check["violation_present_after_removal"] for check in removal_checks
            )
            if removal_checks
            else False,
        },
        "explanation": "a two-tile patch is the smallest witness for an adjacent forbidden pair",
    }


def _forbidden_present(spec: LevelSpec, elements: list[dict[str, Any]]) -> bool:
    positions = {(item["coord"][0], item["coord"][1]): item["terrain"] for item in elements}
    for (x, y), terrain in sorted(positions.items(), key=lambda pair: row_major_key(pair[0])):
        for nx, ny in ((x + 1, y), (x, y + 1)):
            other = positions.get((nx, ny))
            if other is None:
                continue
            if terrain != other and frozenset((terrain, other)) in spec.forbidden_pairs:
                return True
    return False


def _resource_counterexample(spec: LevelSpec, violation: dict[str, Any]) -> dict[str, Any]:
    fragment = violation.get("input_fragment", {})
    code = violation["code"]
    if code == "E_RESOURCE_RANGE":
        coords = [tuple(item) for item in fragment.get("coords", [])]
        count = int(fragment.get("count", len(coords)))
        maximum = int(fragment.get("range", [spec.resource_min, spec.resource_max])[1])
        minimum = int(fragment.get("range", [spec.resource_min, spec.resource_max])[0])
        if count > maximum:
            elements = [
                {"coord": coord_to_list(coord), "role": "resource"}
                for coord in sorted(coords, key=row_major_key)[: maximum + 1]
            ]

            def predicate(items: list[dict[str, Any]]) -> bool:
                return len(items) > maximum

            minimized, removal_checks = _minimize(elements, predicate)
            return {
                "schema": "level-workbench/counterexample-v1",
                "violation_code": code,
                "first_violation": violation,
                "minimizable": True,
                "elements": minimized,
                "reproduction": {
                    "kind": "resource_count",
                    "range": [minimum, maximum],
                    "count": len(minimized),
                    "rule": "exactly maximum+1 resource coordinates exceed the allowed range",
                },
                "checks": {
                    "reproduces": predicate(minimized),
                    "removal_checks": removal_checks,
                    "all_single_element_removals_disappear": all(
                        not check["violation_present_after_removal"] for check in removal_checks
                    ),
                },
                "explanation": "maximum+1 resources form a minimal over-density witness",
            }
        if count < minimum:
            return {
                "schema": "level-workbench/counterexample-v1",
                "violation_code": code,
                "first_violation": violation,
                "minimizable": False,
                "elements": [
                    {"coord": coord_to_list(tuple(coord)), "role": "resource"} for coord in coords
                ],
                "reproduction": {
                    "kind": "resource_count",
                    "range": [minimum, maximum],
                    "count": count,
                },
                "checks": {"reproduces": count < minimum, "removal_checks": []},
                "explanation": "resource underflow cannot be minimized by removing elements",
            }
    if code == "E_RESOURCE_DUPLICATE":
        coord = tuple(violation["coord"])
        elements = [
            {"coord": coord_to_list(coord), "role": "resource"},
            {"coord": coord_to_list(coord), "role": "resource"},
        ]

        def duplicate_predicate(items: list[dict[str, Any]]) -> bool:
            coords = [(item["coord"][0], item["coord"][1]) for item in items]
            return len(coords) != len(set(coords))

        minimized, removal_checks = _minimize(elements, duplicate_predicate)
        return {
            "schema": "level-workbench/counterexample-v1",
            "violation_code": code,
            "first_violation": violation,
            "minimizable": True,
            "elements": minimized,
            "reproduction": {
                "kind": "duplicate_resource_pair",
                "coord": coord_to_list(coord),
                "rule": "two resource entries share one coordinate",
            },
            "checks": {
                "reproduces": duplicate_predicate(minimized),
                "removal_checks": removal_checks,
                "all_single_element_removals_disappear": all(
                    not check["violation_present_after_removal"] for check in removal_checks
                ),
            },
            "explanation": "two duplicate resource entries are the smallest duplicate witness",
        }
    coord = tuple(violation["coord"])
    element = {"coord": coord_to_list(coord), "role": "resource", "terrain": fragment.get("terrain")}
    return {
        "schema": "level-workbench/counterexample-v1",
        "violation_code": code,
        "first_violation": violation,
        "minimizable": True,
        "elements": [element],
        "reproduction": {"kind": "single_resource", "input_fragment": fragment},
        "checks": {"reproduces": True, "removal_checks": []},
        "explanation": "one resource coordinate is a minimal witness for this local resource violation",
    }


def _connectivity_counterexample(
    spec: LevelSpec, level_doc: dict[str, Any], violation: dict[str, Any]
) -> dict[str, Any]:
    grid = level_doc.get("grid", {})
    width = int(grid.get("width", spec.width))
    height = int(grid.get("height", spec.height))
    cells = grid.get("cells", [])
    spawn = tuple(level_doc.get("spawn", list(spec.spawn)))
    goal = tuple(violation.get("coord", []))
    if len(goal) != 2:
        return _not_minimizable_connectivity(violation, "goal coordinate is unavailable")
    if not _in_bounds(spawn, width, height) or not _in_bounds(goal, width, height):
        return _not_minimizable_connectivity(violation, "spawn or goal is outside the grid")

    blocked: list[Coord] = []
    for y in range(height):
        for x in range(width):
            terrain = cells[y][x]
            if terrain in spec.blocked_terrain:
                blocked.append((x, y))
    blocked = sorted(blocked, key=row_major_key)

    def predicate(items: list[dict[str, Any]]) -> bool:
        subset = {(item["coord"][0], item["coord"][1]) for item in items}
        return not _path_exists(width, height, spawn, goal, subset)

    elements = [{"coord": coord_to_list(coord), "terrain": _cell(cells, coord), "role": "blocked_cell"} for coord in blocked]
    if not predicate(elements):
        return _not_minimizable_connectivity(
            violation,
            "no blocked-cell cut reproduces the unreachable goal in the available grid",
        )
    minimized, removal_checks = _minimize(elements, predicate)
    minimal = all(not check["violation_present_after_removal"] for check in removal_checks)
    return {
        "schema": "level-workbench/counterexample-v1",
        "violation_code": violation["code"],
        "first_violation": violation,
        "minimizable": minimal,
        "elements": minimized,
        "reproduction": {
            "kind": "blocked_cut",
            "grid": [width, height],
            "spawn": coord_to_list(spawn),
            "goal": coord_to_list(goal),
            "blocked_cells": [item["coord"] for item in minimized],
            "all_other_cells": "passable",
            "rule": "the listed blocked cells separate spawn from goal; removing any one opens a path",
        },
        "checks": {
            "reproduces": predicate(minimized),
            "removal_checks": removal_checks,
            "all_single_element_removals_disappear": minimal,
        },
        "explanation": "minimal blocked-cell cut witnessing the connectivity violation",
    }


def _not_minimizable_connectivity(violation: dict[str, Any], explanation: str) -> dict[str, Any]:
    return {
        "schema": "level-workbench/counterexample-v1",
        "violation_code": violation["code"],
        "first_violation": violation,
        "minimizable": False,
        "elements": [],
        "reproduction": None,
        "checks": {"reproduces": False, "removal_checks": []},
        "explanation": explanation,
    }


def _single_cell_counterexample(violation: dict[str, Any]) -> dict[str, Any]:
    fragment = violation.get("input_fragment", {})
    coord = violation.get("coord", [0, 0])
    element = {"coord": [int(coord[0]), int(coord[1])], "terrain": fragment.get("terrain")}
    return {
        "schema": "level-workbench/counterexample-v1",
        "violation_code": violation["code"],
        "first_violation": violation,
        "minimizable": True,
        "elements": [element],
        "reproduction": {"kind": "single_cell", "input_fragment": fragment},
        "checks": {"reproduces": True, "removal_checks": []},
        "explanation": "one endpoint cell is the minimal witness",
    }


def _minimize(
    elements: list[dict[str, Any]], predicate: Callable[[list[dict[str, Any]]], bool]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    current = list(elements)
    checks: list[dict[str, Any]] = []
    changed = True
    while changed:
        changed = False
        for element in list(current):
            trial = [item for item in current if item is not element]
            if predicate(trial):
                current = trial
                changed = True
    # Final checks are evaluated against the converged set. For a true minimal
    # witness every single-element removal must make the predicate false.
    for element in current:
        trial = [item for item in current if item is not element]
        checks.append(
            {
                "removed": element,
                "violation_present_after_removal": predicate(trial),
            }
        )
    return current, checks


def _path_exists(
    width: int, height: int, start: Coord, goal: Coord, blocked: set[Coord]
) -> bool:
    if start in blocked or goal in blocked:
        return False
    queue: deque[Coord] = deque([start])
    visited = {start}
    while queue:
        current = queue.popleft()
        if current == goal:
            return True
        x, y = current
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            candidate = (x + dx, y + dy)
            if not _in_bounds(candidate, width, height):
                continue
            if candidate in visited or candidate in blocked:
                continue
            visited.add(candidate)
            queue.append(candidate)
    return False


def _in_bounds(coord: Coord, width: int, height: int) -> bool:
    x, y = coord
    return 0 <= x < width and 0 <= y < height


def _cell(cells: list[list[str]], coord: Coord) -> str | None:
    x, y = coord
    if 0 <= y < len(cells) and 0 <= x < len(cells[y]):
        return cells[y][x]
    return None


def replay_minimal_counterexample(spec: LevelSpec, counterexample: dict[str, Any]) -> dict[str, Any]:
    """Replay a serialized counterexample without regenerating the level."""

    source = counterexample.get("source_hashes", {})
    expected_spec_hash = source.get("spec_sha256")
    from .canonical import sha256_value

    if expected_spec_hash is not None and expected_spec_hash != sha256_value(spec.normalized):
        return {
            "reproduced": False,
            "reason": "counterexample was produced for a different normalized specification",
            "expected_spec_sha256": expected_spec_hash,
            "actual_spec_sha256": sha256_value(spec.normalized),
        }

    kind = counterexample.get("reproduction", {}).get("kind")
    elements = counterexample.get("elements", [])
    if kind == "standalone_tile_patch":
        reproduced = _forbidden_present(spec, elements)
        removals = [
            {
                "removed": element,
                "violation_present_after_removal": _forbidden_present(
                    spec, [item for item in elements if item is not element]
                ),
            }
            for element in elements
        ]
        return {
            "reproduced": reproduced,
            "kind": kind,
            "violation_code": counterexample.get("violation_code"),
            "removal_checks": removals,
            "minimal": reproduced and all(not item["violation_present_after_removal"] for item in removals),
        }

    if kind in {"resource_count", "single_resource", "duplicate_resource_pair"}:
        code = counterexample.get("violation_code")
        if kind == "resource_count":
            count = len(elements)
            minimum, maximum = counterexample["reproduction"]["range"]
            reproduced = count < minimum or count > maximum
            removals = [
                {
                    "removed": element,
                    "violation_present_after_removal": len(
                        [item for item in elements if item is not element]
                    )
                    < minimum
                    or len([item for item in elements if item is not element]) > maximum,
                }
                for element in elements
            ]
        elif kind == "duplicate_resource_pair":
            coords = [(item["coord"][0], item["coord"][1]) for item in elements]
            reproduced = len(coords) != len(set(coords))
            removals = [
                {
                    "removed": element,
                    "violation_present_after_removal": len(
                        [(item["coord"][0], item["coord"][1]) for item in elements if item is not element]
                    )
                    != len({(item["coord"][0], item["coord"][1]) for item in elements if item is not element}),
                }
                for element in elements
            ]
        else:
            fragment = counterexample.get("reproduction", {}).get("input_fragment", {})
            terrain = fragment.get("terrain")
            coord = (elements[0]["coord"][0], elements[0]["coord"][1]) if elements else (0, 0)
            if code == "E_RESOURCE_TERRAIN":
                reproduced = terrain not in spec.resource_allowed_terrain or terrain in spec.blocked_terrain
            elif code == "E_RESOURCE_BOUNDS":
                reproduced = not (0 <= coord[0] < spec.width and 0 <= coord[1] < spec.height)
            elif code == "E_RESOURCE_ENDPOINT":
                reproduced = coord in {spec.spawn, spec.target}
            else:
                reproduced = bool(elements)
            removals = [
                {"removed": element, "violation_present_after_removal": False}
                for element in elements
            ]
        return {
            "reproduced": reproduced,
            "kind": kind,
            "violation_code": code,
            "removal_checks": removals,
            "minimal": reproduced and all(not item["violation_present_after_removal"] for item in removals),
        }

    if kind == "blocked_cut":
        reproduction = counterexample["reproduction"]
        width, height = reproduction["grid"]
        spawn = tuple(reproduction["spawn"])
        goal = tuple(reproduction["goal"])
        blocked = {(item[0], item[1]) for item in reproduction["blocked_cells"]}
        reproduced = not _path_exists(width, height, spawn, goal, blocked)
        removals = []
        for coord in sorted(blocked, key=row_major_key):
            trial = set(blocked)
            trial.remove(coord)
            removals.append(
                {
                    "removed": {"coord": coord_to_list(coord), "role": "blocked_cell"},
                    "violation_present_after_removal": not _path_exists(width, height, spawn, goal, trial),
                }
            )
        return {
            "reproduced": reproduced,
            "kind": kind,
            "violation_code": counterexample.get("violation_code"),
            "removal_checks": removals,
            "minimal": reproduced and all(not item["violation_present_after_removal"] for item in removals),
        }

    if kind == "single_cell":
        fragment = counterexample.get("reproduction", {}).get("input_fragment", {})
        coord_raw = fragment.get("coord", counterexample.get("first_violation", {}).get("coord", [0, 0]))
        coord = (int(coord_raw[0]), int(coord_raw[1]))
        terrain = fragment.get("terrain")
        code = counterexample.get("violation_code")
        if code in {"E_SPAWN_INVALID", "E_TARGET_INVALID"}:
            reproduced = (
                not (0 <= coord[0] < spec.width and 0 <= coord[1] < spec.height)
                or terrain in spec.blocked_terrain
            )
        else:
            reproduced = bool(elements)
        return {
            "reproduced": reproduced,
            "kind": kind,
            "violation_code": code,
            "removal_checks": [
                {"removed": element, "violation_present_after_removal": False}
                for element in elements
            ],
            "minimal": reproduced and bool(elements),
        }

    return {"reproduced": False, "reason": "unsupported counterexample kind", "kind": kind}


def _attach_source_hashes(
    result: dict[str, Any], spec: LevelSpec, level_doc: dict[str, Any]
) -> dict[str, Any]:
    from .canonical import sha256_value

    result["source_hashes"] = {
        "spec_sha256": sha256_value(spec.normalized),
        "level_sha256": sha256_value(level_doc),
        "grid_sha256": level_doc.get("hashes", {}).get("grid_sha256"),
    }
    return result
