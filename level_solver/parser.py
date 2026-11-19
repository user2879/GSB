"""Level specification parsing and validation.

Validation happens before any snapshot or output file is created.  Failures
raise :class:`~level_solver.errors.SolverError` with stable business codes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .errors import E_CONSTRAINT, E_DOMAIN, E_SCHEMA, SolverError

ALLOWED_CONSTRAINTS = {
    "min_rooms",
    "room_spacing",
    "max_region_area",
    "connectivity",
    "reachability",
}

DEFAULT_CONCURRENCY = 4
DEFAULT_MEMORY_BUDGET = 16
DEFAULT_MAX_ROLLBACK = 8
DEFAULT_ALTERNATIVE_LIMIT = 32


@dataclass
class ConstraintSpec:
    cid: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)
    order: int = 0


@dataclass
class LevelSpec:
    width: int
    height: int
    blocked: set[tuple[int, int]]
    constraints: list[ConstraintSpec]
    seed: int
    concurrency: int
    memory_budget: int
    max_rollback: int
    alternative_limit: int
    raw: dict[str, Any]

    @property
    def available_cells(self) -> int:
        return self.width * self.height - len(self.blocked)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: Any) -> bool:
    if isinstance(bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _require_object(raw: Any) -> dict:
    if not isinstance(raw, dict):
        raise SolverError(E_SCHEMA, "level specification must be a JSON object")
    return raw


def _require_int(raw: dict, key: str, *, minimum: int | None = None) -> int:
    if key not in raw:
        raise SolverError(E_SCHEMA, f"missing required field '{key}'")
    value = raw[key]
    if not _is_int(value):
        raise SolverError(E_DOMAIN, f"field '{key}' must be an integer")
    if minimum is not None and value < minimum:
        raise SolverError(E_DOMAIN, f"field '{key}' must be >= {minimum}")
    return value


def _optional_int(raw: dict, key: str, default: int, *, minimum: int) -> int:
    if key not in raw or raw[key] is None:
        return default
    value = raw[key]
    if not _is_int(value):
        raise SolverError(E_DOMAIN, f"field '{key}' must be an integer")
    if value < minimum:
        raise SolverError(E_DOMAIN, f"field '{key}' must be >= {minimum}")
    return value


def _point(value: Any, name: str, width: int, height: int) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2 or not all(_is_int(v) for v in value):
        raise SolverError(E_DOMAIN, f"{name} must be an [x, y] integer pair")
    x, y = int(value[0]), int(value[1])
    if not (0 <= x < width and 0 <= y < height):
        raise SolverError(E_DOMAIN, f"{name} ({x}, {y}) is outside the grid")
    return x, y


def _validate_constraint(raw: Any, index: int, width: int, height: int) -> ConstraintSpec:
    if not isinstance(raw, dict):
        raise SolverError(E_SCHEMA, f"constraint at index {index} must be an object")
    ctype = raw.get("type")
    if ctype not in ALLOWED_CONSTRAINTS:
        raise SolverError(
            E_CONSTRAINT,
            f"unknown constraint type '{ctype}' at index {index}",
            {"allowed": sorted(ALLOWED_CONSTRAINTS)},
        )
    cid = raw.get("id")
    if not isinstance(cid, str) or not cid:
        raise SolverError(E_SCHEMA, f"constraint at index {index} requires a non-empty string id")

    params = dict(raw.get("params", {}))
    if not isinstance(params, dict):
        raise SolverError(E_SCHEMA, f"constraint '{cid}' params must be an object")

    def int_param(name: str, minimum: int, default: int | None = None) -> int:
        if name not in params:
            if default is None:
                raise SolverError(E_DOMAIN, f"constraint '{cid}' missing integer param '{name}'")
            return default
        value = params[name]
        if not _is_int(value):
            raise SolverError(E_DOMAIN, f"constraint '{cid}' param '{name}' must be an integer")
        if value < minimum:
            raise SolverError(E_DOMAIN, f"constraint '{cid}' param '{name}' must be >= {minimum}")
        return value

    if ctype == "min_rooms":
        count = int_param("count", 1)
        min_w = int_param("min_room_w", 1)
        min_h = int_param("min_room_h", 1)
        max_w = int_param("max_room_w", min_w)
        max_h = int_param("max_room_h", min_h)
        if max_w < min_w or max_h < min_h:
            raise SolverError(E_DOMAIN, f"constraint '{cid}' room size range is inverted")
        if max_w > width or max_h > height:
            raise SolverError(E_DOMAIN, f"constraint '{cid}' room size exceeds grid dimensions")
    elif ctype == "room_spacing":
        int_param("min_distance", 0)
    elif ctype == "max_region_area":
        int_param("max_area", 1)
    elif ctype == "reachability":
        _point(params.get("start"), f"constraint '{cid}' start", width, height)
        _point(params.get("end"), f"constraint '{cid}' end", width, height)
    # connectivity has no numeric params

    return ConstraintSpec(cid=cid, type=ctype, params=params, order=index)


def parse_spec(raw: Any) -> LevelSpec:
    """Validate a decoded JSON specification and return a normalised form."""
    data = _require_object(raw)

    width = _require_int(data, "width", minimum=1)
    height = _require_int(data, "height", minimum=1)

    if "seed" not in data or not _is_int(data["seed"]):
        raise SolverError(E_DOMAIN, "field 'seed' must be an integer")
    seed = int(data["seed"])

    blocked: set[tuple[int, int]] = set()
    for index, item in enumerate(data.get("blocked", [])):
        point = _point(item, f"blocked[{index}]", width, height)
        if point in blocked:
            raise SolverError(E_DOMAIN, f"duplicate blocked cell {point}")
        blocked.add(point)

    raw_constraints = data.get("constraints")
    if not isinstance(raw_constraints, list) or not raw_constraints:
        raise SolverError(E_SCHEMA, "'constraints' must be a non-empty array")
    constraints = [
        _validate_constraint(item, index, width, height)
        for index, item in enumerate(raw_constraints)
    ]
    seen_ids: set[str] = set()
    for constraint in constraints:
        if constraint.cid in seen_ids:
            raise SolverError(E_DOMAIN, f"duplicate constraint id '{constraint.cid}'")
        seen_ids.add(constraint.cid)

    available = width * height - len(blocked)
    for constraint in constraints:
        if constraint.type == "min_rooms":
            count = constraint.params["count"]
            if count > available:
                raise SolverError(
                    E_DOMAIN,
                    f"constraint '{constraint.cid}' min rooms {count} exceeds "
                    f"available cells {available}",
                )
        if constraint.type == "room_spacing":
            distance = constraint.params["min_distance"]
            if distance > max(width, height):
                raise SolverError(
                    E_DOMAIN,
                    f"constraint '{constraint.cid}' spacing {distance} exceeds grid extent",
                )

    concurrency = _optional_int(data, "concurrency", DEFAULT_CONCURRENCY, minimum=1)
    memory_budget = _optional_int(data, "memory_budget", DEFAULT_MEMORY_BUDGET, minimum=1)
    max_rollback = _optional_int(data, "max_rollback", DEFAULT_MAX_ROLLBACK, minimum=0)
    alternative_limit = _optional_int(
        data, "alternative_limit", DEFAULT_ALTERNATIVE_LIMIT, minimum=1
    )
    if concurrency > memory_budget:
        raise SolverError(E_DOMAIN, "memory_budget must be >= concurrency")

    return LevelSpec(
        width=width,
        height=height,
        blocked=blocked,
        constraints=constraints,
        seed=seed,
        concurrency=concurrency,
        memory_budget=memory_budget,
        max_rollback=max_rollback,
        alternative_limit=alternative_limit,
        raw=data,
    )
