"""Constraint definitions: precheck, deterministic alternatives and checks.

Each constraint supports four operations:

* :func:`precheck` - cheap feasibility test on the current state;
* :func:`enumerate_actions` - ordered, deterministic candidate modifications;
* :func:`apply_action` - mutate a grid and room ledger using one candidate;
* :func:`is_satisfied` - verify the constraint on the current state.

All randomness flows from an explicit :class:`DeterministicRNG`; the global
random state and system clock are never touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import E_CONSTRAINT, SolverError
from .grid import BLOCKED, WALKABLE, Grid, chebyshev, l_shaped_path
from .prng import DeterministicRNG


EMPTY_PLACEHOLDER = 0


@dataclass
class Action:
    """One concrete, reproducible grid modification."""

    cid: str
    ctype: str
    label: str
    data: dict[str, Any] = field(default_factory=dict)
    grid_snapshot: Grid | None = None
    rooms_snapshot: list[dict[str, int]] | None = None

    def fingerprint(self) -> tuple:
        if self.ctype == "min_rooms":
            rects = tuple((r["x"], r["y"], r["w"], r["h"]) for r in self.data["rooms"])
            return ("min_rooms", rects)
        return (self.ctype, tuple(self.data.get("points", ())))


# --------------------------------------------------------------------- checks
def rect_distance(first: dict[str, int], second: dict[str, int]) -> int:
    """Minimum Chebyshev distance between two axis-aligned rectangles."""
    gap_x = max(0, first["x"] - (second["x"] + second["w"] - 1), second["x"] - (first["x"] + first["w"] - 1))
    gap_y = max(0, first["y"] - (second["y"] + second["h"] - 1), second["y"] - (first["y"] + first["h"] - 1))
    return max(gap_x, gap_y)


def is_satisfied(spec, grid: Grid, rooms: list[dict[str, int]]) -> tuple[bool, str]:
    if spec.type == "min_rooms":
        if len(rooms) >= spec.params["count"]:
            return True, ""
        return False, f"only {len(rooms)} of {spec.params['count']} rooms placed"

    if spec.type == "room_spacing":
        minimum = spec.params["min_distance"]
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                if rect_distance(rooms[i], rooms[j]) < minimum:
                    return False, f"rooms {rooms[i]['id']} and {rooms[j]['id']} closer than {minimum}"
        return True, ""

    if spec.type == "max_region_area":
        maximum = spec.params["max_area"]
        components, _ = grid.components()
        for component in components:
            if len(component) > maximum:
                return False, f"region of {len(component)} cells exceeds max {maximum}"
        return True, ""

    if spec.type == "connectivity":
        components, _ = grid.components()
        if len(components) <= 1:
            return True, ""
        return False, f"{len(components)} disconnected walkable regions"

    if spec.type == "reachability":
        start = tuple(spec.params["start"])
        end = tuple(spec.params["end"])
        if grid.get(*start) != WALKABLE or grid.get(*end) != WALKABLE:
            return False, "endpoint is not walkable"
        if grid.connected_pair(start, end):
            return True, ""
        return False, "endpoints are not connected"

    raise SolverError(E_CONSTRAINT, f"unsupported constraint type {spec.type}")


# ------------------------------------------------------------------- precheck
def precheck(spec, grid: Grid, rooms: list[dict[str, int]]) -> None:
    """Raise :class:`SolverError` when no alternative can possibly work."""
    if spec.type == "min_rooms":
        needed = spec.params["count"] - len(rooms)
        if needed <= 0:
            return
        smallest = spec.params["min_room_w"] * spec.params["min_room_h"]
        if needed * smallest > grid.available_count():
            raise SolverError(
                E_CONSTRAINT,
                f"not enough free cells for {needed} more rooms in '{spec.cid}'",
            )

    elif spec.type == "room_spacing":
        minimum = spec.params["min_distance"]
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                if rect_distance(rooms[i], rooms[j]) < minimum:
                    raise SolverError(
                        E_CONSTRAINT,
                        f"existing rooms violate spacing in '{spec.cid}'",
                    )

    elif spec.type == "max_region_area":
        maximum = spec.params["max_area"]
        components, _ = grid.components()
        for component in components:
            if len(component) > maximum:
                raise SolverError(
                    E_CONSTRAINT,
                    f"existing region of {len(component)} cells already exceeds {maximum}",
                )

    elif spec.type == "connectivity":
        if grid.walkable_count() == 0:
            return
        # Blocked cells may make joining impossible; alternatives reveal it.

    elif spec.type == "reachability":
        start = tuple(spec.params["start"])
        end = tuple(spec.params["end"])
        for name, point in (("start", start), ("end", end)):
            if grid.get(*point) == BLOCKED:
                raise SolverError(
                    E_CONSTRAINT,
                    f"reachability '{spec.cid}' {name} point {point} is blocked",
                )


# ------------------------------------------------------------- enumeration
def _enumerate_room_actions(spec, grid: Grid, rooms, rng: DeterministicRNG, limit: int) -> list[Action]:
    needed = spec.params["count"] - len(rooms)
    if needed <= 0:
        return []
    actions: list[Action] = []
    seen: set[tuple] = set()
    for attempt in range(limit):
        candidate_rng = rng.substream(attempt)
        candidate_grid = grid.clone()
        candidate_rooms = list(rooms)
        ok = True
        for room_index in range(needed):
            room_w = candidate_rng.randint(spec.params["min_room_w"], spec.params["max_room_w"])
            room_h = candidate_rng.randint(spec.params["min_room_h"], spec.params["max_room_h"])
            positions = [
                (x, y)
                for x in range(grid.width - room_w + 1)
                for y in range(grid.height - room_h + 1)
                if candidate_grid.rect_fits(x, y, room_w, room_h)
            ]
            if not positions:
                ok = False
                break
            x, y = candidate_rng.choice(positions)
            candidate_grid.carve_rect(x, y, room_w, room_h)
            candidate_rooms.append(
                {
                    "id": len(candidate_rooms),
                    "x": x,
                    "y": y,
                    "w": room_w,
                    "h": room_h,
                }
            )
        if not ok:
            continue
        new_rooms = candidate_rooms[len(rooms) :]
        fingerprint = tuple((r["x"], r["y"], r["w"], r["h"]) for r in new_rooms)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        actions.append(
            Action(
                cid=spec.cid,
                ctype=spec.type,
                label=f"rooms-attempt-{attempt}",
                data={"rooms": new_rooms},
                grid_snapshot=candidate_grid,
                rooms_snapshot=candidate_rooms,
            )
        )
    return actions


def _corridor_feasible(grid: Grid, points: list[tuple[int, int]]) -> bool:
    return all(grid.in_bounds(x, y) and grid.get(x, y) != BLOCKED for x, y in points)


def _pick_other_component(
    grid: Grid,
    anchor: tuple[int, int],
    components: list[set[tuple[int, int]]],
    labels,
    prefer: tuple[int, int] | None,
) -> set[tuple[int, int]]:
    if prefer is not None and grid.get(*prefer) == WALKABLE:
        prefer_label = labels[prefer[1], prefer[0]]
        anchor_label = labels[anchor[1], anchor[0]]
        if prefer_label != anchor_label:
            for component in components:
                if prefer in component:
                    return component
    anchor_component = next(component for component in components if anchor in component)

    def distance(component: set[tuple[int, int]]) -> tuple[int, int, int]:
        best = min(chebyshev(a, b) for a in anchor_component for b in component)
        first = min(component)
        return best, first[1], first[0]

    others = [component for component in components if anchor not in component]
    return min(others, key=distance)


def _try_build_corridors(
    grid: Grid,
    start: tuple[int, int],
    goal: tuple[int, int] | None,
    rng: DeterministicRNG,
    jitter_budget: int,
) -> tuple[Grid, list[tuple[int, int]]] | None:
    """Greedily merge walkable components with L-shaped corridors."""
    candidate = grid.clone()
    if candidate.get(*start) != WALKABLE:
        candidate.carve_path([start])
    if goal is not None and candidate.get(*goal) == EMPTY_PLACEHOLDER:
        candidate.carve_path([goal])
    carved: list[tuple[int, int]] = [start]
    if goal is not None and goal != start and goal not in carved:
        carved.append(goal)
    while True:
        if goal is None:
            components, _ = candidate.components()
            if len(components) <= 1:
                return candidate, carved
        elif candidate.connected_pair(start, goal):
            return candidate, carved
        components, labels = candidate.components()
        if len(components) <= 1:
            if goal is None:
                return candidate, carved
            break
        target = _pick_other_component(candidate, start, components, labels, goal)
        anchor_component = next(component for component in components if start in component)
        start_choices = rng.shuffled(sorted(anchor_component))[:jitter_budget]
        end_choices = rng.shuffled(sorted(target))[:jitter_budget]
        chosen = None
        for endpoint_a in start_choices:
            for endpoint_b in end_choices:
                for bend in ("h", "v"):
                    path = l_shaped_path(endpoint_a, endpoint_b, bend)
                    if _corridor_feasible(candidate, path):
                        chosen = path
                        break
                if chosen is not None:
                    break
            if chosen is not None:
                break
        if chosen is None:
            return None
        candidate.carve_path(chosen)
        carved.extend(point for point in chosen if point not in carved)
    if goal is not None and candidate.connected_pair(start, goal):
        return candidate, carved
    return None


def _enumerate_corridor_actions(
    spec, grid: Grid, rooms, rng: DeterministicRNG, limit: int
) -> list[Action]:
    if spec.type == "connectivity":
        if grid.walkable_count() == 0:
            return []
        components, _ = grid.components()
        if len(components) <= 1:
            return []
        start = min(point for component in components for point in component)
        goal = None
    else:
        start = tuple(spec.params["start"])
        goal = tuple(spec.params["end"])
        if grid.get(*start) == BLOCKED or grid.get(*goal) == BLOCKED:
            return []
        if grid.connected_pair(start, goal):
            return []

    jitter_budget = min(4, max(1, grid.width, grid.height))
    actions: list[Action] = []
    seen: set[tuple] = set()
    for attempt in range(limit):
        attempt_rng = rng.substream(attempt)
        result = _try_build_corridors(grid, start, goal, attempt_rng, jitter_budget)
        if result is None:
            continue
        candidate_grid, points = result
        if goal is None:
            components, _ = candidate_grid.components()
            if len(components) > 1:
                continue
        elif not candidate_grid.connected_pair(start, goal):
            continue
        fingerprint = tuple(points)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        actions.append(
            Action(
                cid=spec.cid,
                ctype=spec.type,
                label=f"corridor-attempt-{attempt}",
                data={"points": points},
                grid_snapshot=candidate_grid,
                rooms_snapshot=list(rooms),
            )
        )
    return actions


def enumerate_actions(spec, grid: Grid, rooms, rng: DeterministicRNG, limit: int) -> list[Action]:
    """Return the ordered list of deterministic candidate actions."""
    if spec.type == "min_rooms":
        return _enumerate_room_actions(spec, grid, rooms, rng, limit)
    if spec.type in ("connectivity", "reachability"):
        return _enumerate_corridor_actions(spec, grid, rooms, rng, limit)
    # Pure checking constraints repair nothing; failure becomes a conflict.
    return []


def apply_action(grid: Grid, rooms: list[dict[str, int]], action: Action) -> None:
    if action.grid_snapshot is None:
        raise SolverError(E_CONSTRAINT, f"action '{action.label}' has no prepared grid")
    grid.copy_from(action.grid_snapshot)
    rooms.clear()
    rooms.extend(action.rooms_snapshot)
