"""Deterministic constrained level generator with evidence and repair traces."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable

from .canonical import (
    Coord,
    canonical_json_bytes,
    coord_to_list,
    grid_hash,
    row_major_key,
    sha256_bytes,
    sorted_coords,
)
from .errors import InjectionError
from .graph import bfs_path, neighbors, reachable_set
from .injection import empty_injection
from .model import Level, LevelSpec
from .seeds import DeterministicRNG, derive_seed, seed_record

MAX_REPAIR_ROUNDS = 32


@dataclass
class GenerationOutcome:
    spec: LevelSpec
    level: Level | None
    status: str
    error: dict[str, Any] | None
    evidence: dict[str, Any]
    subseeds: list[dict[str, Any]] = field(default_factory=list)
    repair_trace: dict[str, Any] = field(default_factory=dict)
    stage_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    def level_document(self) -> dict[str, Any] | None:
        if self.level is None:
            return None
        core: dict[str, Any] = {
            "schema": "level-workbench/level-v1",
            "seed": self.spec.seed,
            "spec_sha256": sha256_bytes(canonical_json_bytes(self.spec.normalized)),
            "grid": {
                "width": self.level.width,
                "height": self.level.height,
                "cells": [row[:] for row in self.level.cells],
            },
            "resources": [coord_to_list(coord) for coord in self.level.resources],
            "spawn": coord_to_list(self.level.spawn),
            "target": coord_to_list(self.level.target),
            "generation": {
                "status": self.status,
                "error": self.error,
                "subseeds": self.subseeds,
                "stage_log": self.stage_log,
                "evidence": self.evidence,
                "repair_trace": self.repair_trace,
            },
        }
        content_hash = sha256_bytes(canonical_json_bytes(core))
        core["hashes"] = {
            "grid_sha256": grid_hash(self.level.cells),
            "content_sha256": content_hash,
        }
        return core


class _SeedFactory:
    def __init__(self, spec: LevelSpec, injection: dict[str, Any], records: list[dict[str, Any]]):
        self.spec = spec
        self.offsets = injection.get("pipeline", {}).get("subseed_offsets", {})
        self.records = records

    def rng(self, purpose: str, *parts: object) -> DeterministicRNG:
        offset = int(self.offsets.get(purpose, 0))
        seed = derive_seed(self.spec.seed, purpose, *parts, offset=offset)
        record_parts = {f"part_{index}": str(part) for index, part in enumerate(parts)}
        self.records.append(seed_record(self.spec.seed, purpose, seed, offset=offset, **record_parts))
        return DeterministicRNG(seed)


def generate_level(spec: LevelSpec, injection: dict[str, Any] | None = None) -> GenerationOutcome:
    injection = normalized_injection_or_empty(injection)
    subseeds: list[dict[str, Any]] = []
    seeds = _SeedFactory(spec, injection, subseeds)
    stage_log: list[dict[str, Any]] = []
    repair_trace: dict[str, Any] = {"forbidden_adjacent": [], "resources": []}

    cells = _generate_terrain(spec, seeds)
    fixed_changes = _apply_fixed_cells(spec, cells)
    stage_log.append(
        {
            "stage": "terrain",
            "purpose": "terrain",
            "fixed_cells_applied": len(fixed_changes),
        }
    )

    normalizations = _normalize_endpoints(spec, cells)
    stage_log.append({"stage": "endpoint_normalization", "changes": normalizations})
    spawn_decision = _sample_spawn_neighborhood(spec, cells, seeds)
    stage_log.append({"stage": "spawn_neighborhood", **spawn_decision})

    connectivity_paths, conflict = _ensure_connectivity(spec, cells)
    if conflict is not None:
        error = {
            "code": "E_CONNECT_UNSAT",
            "message": "required connectivity cannot be satisfied without changing immutable cells",
            "details": {
                "first_violation": conflict,
                "conflict": conflict,
            },
        }
        evidence = {
            "constraints": [
                {
                    "constraint": "connectivity",
                    "status": "fail",
                    "details": {"paths": connectivity_paths, "conflict": conflict},
                }
            ]
        }
        return GenerationOutcome(
            spec=spec,
            level=None,
            status="fail",
            error=error,
            evidence=evidence,
            subseeds=subseeds,
            repair_trace=repair_trace,
            stage_log=stage_log,
        )

    stage_log.append(
        {
            "stage": "connectivity_carving",
            "goals": len(connectivity_paths),
            "carved_cells": sorted(
                {
                    (coord[0], coord[1])
                    for path in connectivity_paths
                    for coord in path.get("carved", [])
                },
                key=row_major_key,
            ),
        }
    )

    level = Level(
        width=spec.width,
        height=spec.height,
        cells=cells,
        resources=[],
        spawn=spec.spawn,
        target=spec.target,
    )

    protected = {spec.spawn, spec.target}
    for path in connectivity_paths:
        protected.update((coord[0], coord[1]) for coord in path["tiles"])
    for region in spec.regions:
        protected.add(region.anchor)

    pre_repair_changes = _apply_forced_terrain(
        spec, level, injection["pipeline"]["pre_repair_force_terrain"], "pre_repair"
    )
    if pre_repair_changes:
        stage_log.append({"stage": "pre_repair_injection", "changes": pre_repair_changes})

    forbidden_trace, forbidden_pass = _repair_forbidden_adjacent(
        spec,
        level,
        seeds,
        protected,
        skip=bool(injection["pipeline"]["skip_local_repair"]),
    )
    repair_trace["forbidden_adjacent"] = forbidden_trace
    stage_log.append(
        {
            "stage": "forbidden_adjacent_repair",
            "status": forbidden_trace["status"],
            "rounds": len(forbidden_trace["rounds"]),
        }
    )

    resource_seed = seeds.rng("resources")
    desired = spec.resource_min + resource_seed.randbelow(spec.resource_max - spec.resource_min + 1)
    level.resources = _sample_resources(spec, level, desired, resource_seed)

    override = injection["pipeline"].get("resource_count_override")
    override_changes: dict[str, Any] | None = None
    if override is not None:
        override_changes = _set_resource_count(
            spec, level, int(override), seeds.rng("resource_count_override")
        )
        stage_log.append({"stage": "resource_count_override", **override_changes})

    resource_trace = _repair_resources(
        spec,
        level,
        seeds,
        skip=bool(injection["pipeline"]["skip_resource_repair"]),
        reason="after pipeline override" if override_changes is not None else "initial placement",
    )
    if override_changes is not None:
        resource_trace["override"] = override_changes
    repair_trace["resources"] = resource_trace
    stage_log.append(
        {
            "stage": "resource_placement",
            "desired_count": desired,
            "actual_count": len(level.resources),
            "repair_status": resource_trace["status"],
        }
    )

    post_changes = _apply_forced_terrain(
        spec, level, injection["pipeline"]["post_generation_force_terrain"], "post_generation"
    )
    if post_changes:
        stage_log.append({"stage": "post_generation_injection", "changes": post_changes})

    evidence = _build_final_evidence(spec, level)
    constraints = {entry["constraint"]: entry for entry in evidence["constraints"]}
    error: dict[str, Any] | None = None
    if constraints["spawn"]["status"] == "fail":
        error = _constraint_error("E_CONNECT_UNSAT", constraints["spawn"])
    elif constraints["target"]["status"] == "fail":
        error = _constraint_error("E_CONNECT_UNSAT", constraints["target"])
    elif constraints["connectivity"]["status"] == "fail":
        error = _constraint_error("E_CONNECT_UNSAT", constraints["connectivity"])
    elif constraints["resources"]["status"] == "fail":
        resource_status = repair_trace["resources"].get("status")
        error_code = "E_REPAIR_EXHAUST" if resource_status == "exhausted" else "E_RESOURCE_RANGE"
        error = _constraint_error(error_code, constraints["resources"])
        error["details"]["last_trace"] = repair_trace["resources"]
    elif constraints["forbidden_adjacent"]["status"] == "fail":
        error = _constraint_error("E_REPAIR_EXHAUST", constraints["forbidden_adjacent"])
        error["details"]["last_trace"] = repair_trace["forbidden_adjacent"]
    elif constraints["grid"]["status"] == "fail":
        error = _constraint_error("E_GRID_INVALID", constraints["grid"])

    status = "pass" if error is None else "fail"
    return GenerationOutcome(
        spec=spec,
        level=level,
        status=status,
        error=error,
        evidence=evidence,
        subseeds=subseeds,
        repair_trace=repair_trace,
        stage_log=stage_log,
    )


def normalized_injection_or_empty(injection: dict[str, Any] | None) -> dict[str, Any]:
    if injection is None:
        return empty_injection()
    return copy.deepcopy(injection)


def _generate_terrain(spec: LevelSpec, seeds: _SeedFactory) -> list[list[str]]:
    names = sorted(spec.terrain_weights)
    weights = [spec.terrain_weights[name] for name in names]
    rng = seeds.rng("terrain")
    cells: list[list[str]] = []
    for _y in range(spec.height):
        row: list[str] = []
        for _x in range(spec.width):
            row.append(names[rng.weighted_index(weights)])
        cells.append(row)
    return cells


def _apply_fixed_cells(spec: LevelSpec, cells: list[list[str]]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for coord in sorted(spec.fixed_cells, key=row_major_key):
        x, y = coord
        terrain = spec.fixed_cells[coord]
        if cells[y][x] != terrain:
            changes.append(
                {"coord": coord_to_list(coord), "from": cells[y][x], "to": terrain, "immutable": True}
            )
            cells[y][x] = terrain
    return changes


def _normalize_endpoints(spec: LevelSpec, cells: list[list[str]]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for label, coord in (("spawn", spec.spawn), ("target", spec.target)):
        x, y = coord
        terrain = cells[y][x]
        if terrain not in spec.blocked_terrain:
            continue
        if coord in spec.fixed_cells:
            continue
        replacement = spec.corridor_terrain
        cells[y][x] = replacement
        changes.append({"cell": label, "coord": coord_to_list(coord), "from": terrain, "to": replacement})
    return changes


def _sample_spawn_neighborhood(
    spec: LevelSpec, cells: list[list[str]], seeds: _SeedFactory
) -> dict[str, Any]:
    rng = seeds.rng("spawn_neighborhood")
    candidates = [coord for coord in neighbors(spec.spawn, spec.width, spec.height) if coord not in spec.fixed_cells]
    candidates.sort(key=row_major_key)
    if not candidates:
        return {"selected": None, "changed": False, "reason": "spawn has no mutable orthogonal neighbor"}
    selected = rng.choice(candidates)
    x, y = selected
    changed = False
    before = cells[y][x]
    if before in spec.blocked_terrain:
        cells[y][x] = spec.corridor_terrain
        changed = True
    return {
        "selected": coord_to_list(selected),
        "changed": changed,
        "from": before,
        "to": cells[y][x],
    }


def _ensure_connectivity(
    spec: LevelSpec, cells: list[list[str]]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    paths: list[dict[str, Any]] = []

    def passable(coord: Coord) -> bool:
        x, y = coord
        return cells[y][x] not in spec.blocked_terrain

    def carvable(coord: Coord) -> bool:
        x, y = coord
        if coord in spec.fixed_cells and cells[y][x] in spec.blocked_terrain:
            return False
        return True

    goals: list[tuple[str, str, Coord]] = [("target", "target", spec.target)]
    goals.extend(("region", region.id, region.anchor) for region in spec.regions)

    for goal_kind, goal_id, goal_coord in goals:
        existing = bfs_path(spec.width, spec.height, spec.spawn, goal_coord, passable)
        carved: list[Coord] = []
        if existing is None:
            candidate_path = bfs_path(spec.width, spec.height, spec.spawn, goal_coord, carvable)
            if candidate_path is None:
                blocked_cells = sorted(
                    (coord for coord in spec.fixed_cells if spec.fixed_cells[coord] in spec.blocked_terrain),
                    key=row_major_key,
                )
                conflict = {
                    "conflict_pair": ["spawn", "target" if goal_kind == "target" else f"region:{goal_id}"],
                    "coord": coord_to_list(goal_coord),
                    "goal_kind": goal_kind,
                    "goal_id": goal_id,
                    "reason": "no path exists through mutable cells; immutable blocked cells separate spawn from the goal",
                    "blocking_cells": [coord_to_list(coord) for coord in blocked_cells],
                    "blocking_cell_count": len(blocked_cells),
                }
                return paths, conflict
            for coord in candidate_path:
                x, y = coord
                if cells[y][x] in spec.blocked_terrain:
                    cells[y][x] = spec.corridor_terrain
                    carved.append(coord)
            existing = bfs_path(spec.width, spec.height, spec.spawn, goal_coord, passable)
            if existing is None:
                conflict = {
                    "conflict_pair": ["spawn", "target" if goal_kind == "target" else f"region:{goal_id}"],
                    "coord": coord_to_list(goal_coord),
                    "reason": "corridor carving failed to create a passable path",
                    "blocking_cells": [],
                }
                return paths, conflict

        paths.append(
            {
                "from": "spawn",
                "to": "target" if goal_kind == "target" else f"region:{goal_id}",
                "goal_kind": goal_kind,
                "goal_id": goal_id,
                "goal_tile": coord_to_list(goal_coord),
                "tiles": [coord_to_list(coord) for coord in existing],
                "carved": [coord_to_list(coord) for coord in carved],
            }
        )
    return paths, None


def _find_forbidden_violations(spec: LevelSpec, level: Level) -> list[dict[str, Any]]:
    forbidden = spec.forbidden_pairs
    violations: list[dict[str, Any]] = []
    for y in range(level.height):
        for x in range(level.width):
            left = level.cells[y][x]
            for nx, ny in ((x + 1, y), (x, y + 1)):
                if nx >= level.width or ny >= level.height:
                    continue
                right = level.cells[ny][nx]
                pair = frozenset((left, right))
                if pair in forbidden and left != right:
                    violations.append(
                        {
                            "coord": [x, y],
                            "neighbor": [nx, ny],
                            "pair": sorted((left, right)),
                            "coord_terrain": left,
                            "neighbor_terrain": right,
                        }
                    )
    return violations


def _repair_forbidden_adjacent(
    spec: LevelSpec,
    level: Level,
    seeds: _SeedFactory,
    protected: set[Coord],
    skip: bool,
) -> tuple[dict[str, Any], bool]:
    initial = _find_forbidden_violations(spec, level)
    trace: dict[str, Any] = {
        "initial": initial,
        "initial_count": len(initial),
        "rounds": [],
        "status": "pass" if not initial else "pending",
    }
    if not initial:
        return trace, True
    if skip:
        trace["status"] = "skipped"
        trace["last_violations"] = initial
        return trace, False

    immutable = set(spec.fixed_cells)
    candidates = sorted(spec.passable_terrain)
    for round_number in range(1, MAX_REPAIR_ROUNDS + 1):
        before = _find_forbidden_violations(spec, level)
        if not before:
            trace["status"] = "pass"
            trace["last_violations"] = []
            return trace, True
        offending = sorted(
            {(item["coord"][0], item["coord"][1]) for item in before}
            | {(item["neighbor"][0], item["neighbor"][1]) for item in before},
            key=row_major_key,
        )
        changes: list[dict[str, Any]] = []
        for coord in offending:
            if coord in immutable:
                continue
            old = level.terrain_at(coord)
            rng = seeds.rng("repair_forbidden", round_number, coord[0], coord[1])
            candidate_order = candidates[:]
            rng.shuffle(candidate_order)
            best: str | None = None
            best_count: int | None = None
            for candidate in candidate_order:
                if candidate == old:
                    continue
                local_count = _incident_violation_count(spec, level, coord, candidate)
                if best is None or best_count is None or local_count < best_count:
                    best = candidate
                    best_count = local_count
                    if local_count == 0:
                        break
            if best is not None and best != old:
                level.set_terrain(coord, best)
                changes.append({"coord": coord_to_list(coord), "from": old, "to": best})

        after = _find_forbidden_violations(spec, level)
        trace["rounds"].append(
            {
                "round": round_number,
                "before_count": len(before),
                "after_count": len(after),
                "offending_coords": [coord_to_list(coord) for coord in offending],
                "changes": changes,
                "first_before": before[0] if before else None,
                "first_after": after[0] if after else None,
            }
        )
        if not after:
            trace["status"] = "pass"
            trace["last_violations"] = []
            return trace, True
        if not changes:
            break

    trace["status"] = "exhausted"
    trace["last_violations"] = _find_forbidden_violations(spec, level)
    return trace, False


def _incident_violation_count(spec: LevelSpec, level: Level, coord: Coord, candidate: str) -> int:
    count = 0
    x, y = coord
    for nx, ny in neighbors(coord, level.width, level.height):
        other = level.cells[ny][nx]
        if frozenset((candidate, other)) in spec.forbidden_pairs and candidate != other:
            count += 1
    return count


def _resource_eligible(spec: LevelSpec, level: Level) -> list[Coord]:
    result: list[Coord] = []
    for y in range(level.height):
        for x in range(level.width):
            coord = (x, y)
            if coord == spec.spawn or coord == spec.target:
                continue
            terrain = level.cells[y][x]
            if terrain in spec.resource_allowed_terrain and terrain not in spec.blocked_terrain:
                result.append(coord)
    return result


def _sample_resources(spec: LevelSpec, level: Level, count: int, rng: DeterministicRNG) -> list[Coord]:
    eligible = _resource_eligible(spec, level)
    rng.shuffle(eligible)
    selected = eligible[: min(count, len(eligible))]
    return sorted(selected, key=row_major_key)


def _repair_resources(
    spec: LevelSpec,
    level: Level,
    seeds: _SeedFactory,
    skip: bool,
    reason: str,
) -> dict[str, Any]:
    before = list(level.resources)
    invalid = [
        coord
        for coord in before
        if not level.in_bounds(coord)
        or level.terrain_at(coord) not in spec.resource_allowed_terrain
        or level.terrain_at(coord) in spec.blocked_terrain
        or coord in {spec.spawn, spec.target}
    ]
    # Unique coordinates are part of resource legality.
    seen: set[Coord] = set()
    duplicates: list[Coord] = []
    for coord in before:
        if coord in seen:
            duplicates.append(coord)
        seen.add(coord)

    trace: dict[str, Any] = {
        "reason": reason,
        "before": [coord_to_list(coord) for coord in sorted_coords(before)],
        "invalid": [coord_to_list(coord) for coord in sorted_coords(invalid)],
        "duplicates": [coord_to_list(coord) for coord in sorted_coords(duplicates)],
        "rounds": [],
        "status": "pending",
    }
    if skip:
        trace["status"] = "skipped"
        trace["after"] = [coord_to_list(coord) for coord in sorted_coords(level.resources)]
        return trace

    round_number = 1
    changes: list[dict[str, Any]] = []
    if invalid or duplicates:
        level.resources = sorted(
            {coord for coord in before if coord not in set(invalid)},
            key=row_major_key,
        )
        changes.append(
            {
                "round": round_number,
                "action": "remove_invalid_or_duplicate",
                "removed": [coord_to_list(coord) for coord in sorted_coords(set(invalid) | set(duplicates))],
            }
        )

    if len(level.resources) < spec.resource_min:
        rng = seeds.rng("resource_repair_add", round_number)
        eligible = [coord for coord in _resource_eligible(spec, level) if coord not in set(level.resources)]
        rng.shuffle(eligible)
        need = spec.resource_min - len(level.resources)
        added = eligible[:need]
        level.resources.extend(added)
        level.resources = sorted(level.resources, key=row_major_key)
        changes.append(
            {
                "round": round_number,
                "action": "add_to_minimum",
                "added": [coord_to_list(coord) for coord in sorted(added, key=row_major_key)],
            }
        )
    elif len(level.resources) > spec.resource_max:
        rng = seeds.rng("resource_repair_remove", round_number)
        removable = level.resources[:]
        rng.shuffle(removable)
        excess = len(level.resources) - spec.resource_max
        removed = sorted(removable[:excess], key=row_major_key)
        removed_set = set(removed)
        level.resources = sorted(
            [coord for coord in level.resources if coord not in removed_set], key=row_major_key
        )
        changes.append(
            {
                "round": round_number,
                "action": "remove_to_maximum",
                "removed": [coord_to_list(coord) for coord in removed],
            }
        )

    trace["rounds"] = changes
    trace["after"] = [coord_to_list(coord) for coord in level.resources]
    valid = spec.resource_min <= len(level.resources) <= spec.resource_max and all(
        level.in_bounds(coord)
        and level.terrain_at(coord) in spec.resource_allowed_terrain
        and level.terrain_at(coord) not in spec.blocked_terrain
        and coord not in {spec.spawn, spec.target}
        for coord in level.resources
    )
    trace["status"] = "pass" if valid else "exhausted"
    return trace


def _set_resource_count(
    spec: LevelSpec, level: Level, target_count: int, rng: DeterministicRNG
) -> dict[str, Any]:
    before = len(level.resources)
    if target_count > before:
        eligible = [coord for coord in _resource_eligible(spec, level) if coord not in set(level.resources)]
        rng.shuffle(eligible)
        added = eligible[: target_count - before]
        level.resources.extend(added)
        level.resources = sorted(level.resources, key=row_major_key)
        return {
            "before": before,
            "requested": target_count,
            "after": len(level.resources),
            "added": [coord_to_list(coord) for coord in sorted(added, key=row_major_key)],
            "achieved": len(level.resources) == target_count,
        }
    if target_count < before:
        removable = level.resources[:]
        rng.shuffle(removable)
        removed = removable[: before - target_count]
        removed_set = set(removed)
        level.resources = sorted(
            [coord for coord in level.resources if coord not in removed_set], key=row_major_key
        )
        return {
            "before": before,
            "requested": target_count,
            "after": len(level.resources),
            "removed": [coord_to_list(coord) for coord in sorted(removed, key=row_major_key)],
            "achieved": len(level.resources) == target_count,
        }
    return {"before": before, "requested": target_count, "after": before, "achieved": True}


def _apply_forced_terrain(
    spec: LevelSpec, level: Level, entries: Iterable[dict[str, Any]], stage: str
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for entry in entries:
        x, y = entry["coord"]
        coord = (x, y)
        terrain = entry["terrain"]
        if terrain not in spec.terrain_weights:
            raise InjectionError(
                "E_INJECTION",
                f"forced terrain '{terrain}' is not declared in the spec",
                f"{stage}.force_terrain",
                {"coord": coord_to_list(coord), "terrain": terrain},
            )
        if not level.in_bounds(coord):
            raise InjectionError(
                "E_INJECTION",
                "forced terrain coordinate is outside the grid",
                f"{stage}.force_terrain",
                {"coord": coord_to_list(coord), "grid": [level.width, level.height]},
            )
        old = level.terrain_at(coord)
        if old != terrain:
            level.set_terrain(coord, terrain)
            changes.append({"coord": coord_to_list(coord), "from": old, "to": terrain})
    return changes


def _build_final_evidence(spec: LevelSpec, level: Level) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []

    grid_issues: list[dict[str, Any]] = []
    if level.width != spec.width or level.height != spec.height:
        grid_issues.append(
            {
                "code": "E_GRID_INVALID",
                "coord": [0, 0],
                "message": "grid dimensions do not match the specification",
            }
        )
    if len(level.cells) != level.height or any(len(row) != level.width for row in level.cells):
        grid_issues.append(
            {
                "code": "E_GRID_INVALID",
                "coord": [0, 0],
                "message": "grid rows are ragged or have the wrong height",
            }
        )
    unknown_terrain = sorted(
        {
            token
            for row in level.cells
            for token in row
            if token not in spec.terrain_weights
        }
    )
    if unknown_terrain:
        grid_issues.append(
            {
                "code": "E_GRID_INVALID",
                "coord": [0, 0],
                "message": "grid contains undeclared terrain",
                "input_fragment": {"terrain": unknown_terrain},
            }
        )
    entries.append(
        {
            "constraint": "grid",
            "status": "fail" if grid_issues else "pass",
            "details": {
                "width": level.width,
                "height": level.height,
                "grid_sha256": grid_hash(level.cells),
                "terrain_counts": _terrain_counts(level),
                "issues": grid_issues,
            },
        }
    )

    entries.append(_endpoint_evidence(spec, level, "spawn", level.spawn))
    entries.append(_endpoint_evidence(spec, level, "target", level.target))

    connectivity = _connectivity_evidence(spec, level)
    entries.append(connectivity)

    resource_issues = _resource_issues(spec, level)
    entries.append(
        {
            "constraint": "resources",
            "status": "fail" if resource_issues else "pass",
            "details": {
                "range": [spec.resource_min, spec.resource_max],
                "count": len(level.resources),
                "coords": [coord_to_list(coord) for coord in sorted_coords(level.resources)],
                "allowed_terrain": sorted(spec.resource_allowed_terrain),
                "issues": resource_issues,
            },
        }
    )

    forbidden = _find_forbidden_violations(spec, level)
    entries.append(
        {
            "constraint": "forbidden_adjacent",
            "status": "fail" if forbidden else "pass",
            "details": {
                "pairs": [list(pair) for pair in sorted(spec.forbidden_adjacent)],
                "checked_edges": level.width * (level.height - 1) + level.height * (level.width - 1),
                "violations": forbidden,
                "first_violation": forbidden[0] if forbidden else None,
            },
        }
    )
    return {"constraints": entries}


def _endpoint_evidence(spec: LevelSpec, level: Level, name: str, coord: Coord) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    if not level.in_bounds(coord):
        issues.append(
            {
                "code": "E_SPAWN_INVALID" if name == "spawn" else "E_TARGET_INVALID",
                "coord": coord_to_list(coord),
                "message": f"{name} is outside the grid",
            }
        )
    else:
        terrain = level.terrain_at(coord)
        if terrain in spec.blocked_terrain:
            issues.append(
                {
                    "code": "E_SPAWN_INVALID" if name == "spawn" else "E_TARGET_INVALID",
                    "coord": coord_to_list(coord),
                    "message": f"{name} lies on blocked terrain",
                    "input_fragment": {"coord": coord_to_list(coord), "terrain": terrain},
                }
            )
    return {
        "constraint": name,
        "status": "fail" if issues else "pass",
        "details": {
            "coord": coord_to_list(coord),
            "terrain": level.terrain_at(coord) if level.in_bounds(coord) else None,
            "issues": issues,
        },
    }


def _connectivity_evidence(spec: LevelSpec, level: Level) -> dict[str, Any]:
    def passable(coord: Coord) -> bool:
        return level.terrain_at(coord) not in spec.blocked_terrain

    paths: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    goals: list[tuple[str, str, Coord]] = [("target", "target", level.target)]
    goals.extend(("region", region.id, region.anchor) for region in spec.regions)
    reachable = reachable_set(level.width, level.height, [level.spawn], passable)
    for kind, goal_id, coord in goals:
        path = bfs_path(level.width, level.height, level.spawn, coord, passable)
        if path is None:
            issues.append(
                {
                    "code": "E_CONNECT_UNSAT",
                    "coord": coord_to_list(coord),
                    "goal_kind": kind,
                    "goal_id": goal_id,
                    "message": "goal is not reachable from spawn",
                    "input_fragment": {
                        "spawn": coord_to_list(level.spawn),
                        "goal": coord_to_list(coord),
                    },
                }
            )
        else:
            paths.append(
                {
                    "from": "spawn",
                    "to": "target" if kind == "target" else f"region:{goal_id}",
                    "goal_kind": kind,
                    "goal_id": goal_id,
                    "tiles": [coord_to_list(item) for item in path],
                }
            )
    return {
        "constraint": "connectivity",
        "status": "fail" if issues else "pass",
        "details": {
            "paths": paths,
            "issues": issues,
            "reachable_cell_count": len(reachable),
        },
    }


def _resource_issues(spec: LevelSpec, level: Level) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    count = len(level.resources)
    if count < spec.resource_min or count > spec.resource_max:
        first_coord = coord_to_list(level.resources[0]) if level.resources else [0, 0]
        issues.append(
            {
                "code": "E_RESOURCE_RANGE",
                "coord": first_coord,
                "message": "resource count is outside the required range",
                "input_fragment": {
                    "range": [spec.resource_min, spec.resource_max],
                    "count": count,
                    "coords": [coord_to_list(coord) for coord in sorted_coords(level.resources)],
                },
            }
        )
    seen: set[Coord] = set()
    for coord in sorted(level.resources, key=row_major_key):
        if coord in seen:
            issues.append(
                {
                    "code": "E_RESOURCE_DUPLICATE",
                    "coord": coord_to_list(coord),
                    "message": "duplicate resource coordinate",
                    "input_fragment": {"coord": coord_to_list(coord)},
                }
            )
            continue
        seen.add(coord)
        if not level.in_bounds(coord):
            issues.append(
                {
                    "code": "E_RESOURCE_BOUNDS",
                    "coord": coord_to_list(coord),
                    "message": "resource coordinate is outside the grid",
                    "input_fragment": {"coord": coord_to_list(coord), "grid": [level.width, level.height]},
                }
            )
            continue
        terrain = level.terrain_at(coord)
        if terrain not in spec.resource_allowed_terrain or terrain in spec.blocked_terrain:
            issues.append(
                {
                    "code": "E_RESOURCE_TERRAIN",
                    "coord": coord_to_list(coord),
                    "message": "resource lies on disallowed terrain",
                    "input_fragment": {"coord": coord_to_list(coord), "terrain": terrain},
                }
            )
        if coord in {spec.spawn, spec.target}:
            issues.append(
                {
                    "code": "E_RESOURCE_ENDPOINT",
                    "coord": coord_to_list(coord),
                    "message": "resource overlaps spawn or target",
                    "input_fragment": {"coord": coord_to_list(coord)},
                }
            )
    return sorted(issues, key=lambda issue: (issue["coord"][1], issue["coord"][0], issue["code"]))


def _terrain_counts(level: Level) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in level.cells:
        for terrain in row:
            counts[terrain] = counts.get(terrain, 0) + 1
    return {name: counts[name] for name in sorted(counts)}


def _constraint_error(code: str, entry: dict[str, Any]) -> dict[str, Any]:
    details = entry["details"]
    issues = details.get("issues") or details.get("violations") or []
    first = issues[0] if issues else None
    return {
        "code": code,
        "message": f"constraint '{entry['constraint']}' failed",
        "details": {
            "constraint": entry["constraint"],
            "first_violation": first,
            "issues": issues,
        },
    }
