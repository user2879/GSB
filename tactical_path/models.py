"""Core immutable and mutable data models."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from .canonical import sha256_json
from .errors import ErrorCode, TacticalError


Coord = tuple[int, int]


def xy(cell: int, width: int) -> Coord:
    return cell % width, cell // width


def flat(coord: Coord, width: int) -> int:
    return coord[1] * width + coord[0]


@dataclass(frozen=True)
class MobilityRule:
    name: str
    cost_multiplier: int
    max_terrain: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "cost_multiplier": self.cost_multiplier,
            "max_terrain": self.max_terrain,
            "name": self.name,
        }


@dataclass
class MapData:
    tier: str
    seed: int
    width: int
    height: int
    region_size: int
    region_shape: tuple[int, int]
    sector_shape: tuple[int, int]
    terrain: np.ndarray
    passable: np.ndarray
    regions: np.ndarray
    sectors: np.ndarray
    mobility_rules: dict[str, MobilityRule]
    content_hash: str
    map_version: str

    @property
    def cells(self) -> int:
        return self.width * self.height

    @property
    def region_count(self) -> int:
        return int(self.region_shape[0] * self.region_shape[1])

    @property
    def sector_count(self) -> int:
        return int(self.sector_shape[0] * self.sector_shape[1])

    def validate(self) -> None:
        expected = (self.height, self.width)
        for name, array in (
            ("terrain", self.terrain),
            ("passable", self.passable),
            ("regions", self.regions),
            ("sectors", self.sectors),
        ):
            if tuple(array.shape) != expected:
                raise TacticalError(
                    ErrorCode.MAP_SHAPE,
                    "map layer shape mismatch",
                    context={
                        "layer": name,
                        "expected": list(expected),
                        "actual": list(array.shape),
                    },
                )
        if not np.all((self.terrain >= 1) & (self.terrain <= 9)):
            raise TacticalError(
                ErrorCode.TERRAIN_RANGE,
                "terrain costs must be in [1, 9]",
                context={"tier": self.tier},
            )
        if not set(np.unique(self.passable).tolist()).issubset({0, 1}):
            raise TacticalError(
                ErrorCode.MAP_SHAPE,
                "passability layer must be boolean or 0/1",
                context={"tier": self.tier},
            )

    def allowed_mask(self, mobility: str | MobilityRule) -> np.ndarray:
        rule = self.mobility_rules[mobility] if isinstance(mobility, str) else mobility
        return self.passable.astype(bool, copy=False) & (
            self.terrain <= rule.max_terrain
        )

    def entering_cost(self, cell: int, mobility: str | MobilityRule) -> int:
        rule = self.mobility_rules[mobility] if isinstance(mobility, str) else mobility
        return int(self.terrain.flat[cell]) * rule.cost_multiplier

    def region_of_cell(self, cell: int) -> int:
        return int(self.regions.flat[cell])

    def map_manifest(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "region_size": self.region_size,
            "region_shape": list(self.region_shape),
            "sector_shape": list(self.sector_shape),
            "region_count": self.region_count,
            "sector_count": self.sector_count,
            "content_hash": self.content_hash,
            "map_version": self.map_version,
            "mobility": {
                key: value.to_dict()
                for key, value in sorted(self.mobility_rules.items())
            },
        }

    def clone_with(self, *, terrain: np.ndarray, passable: np.ndarray) -> "MapData":
        content_hash = compute_map_hash(
            tier=self.tier,
            seed=self.seed,
            width=self.width,
            height=self.height,
            region_size=self.region_size,
            terrain=terrain,
            passable=passable,
            regions=self.regions,
            sectors=self.sectors,
            mobility=self.mobility_rules,
        )
        return MapData(
            tier=self.tier,
            seed=self.seed,
            width=self.width,
            height=self.height,
            region_size=self.region_size,
            region_shape=self.region_shape,
            sector_shape=self.sector_shape,
            terrain=terrain,
            passable=passable,
            regions=self.regions,
            sectors=self.sectors,
            mobility_rules=dict(self.mobility_rules),
            content_hash=content_hash,
            map_version=f"{self.tier}:{content_hash[:16]}",
        )


@dataclass(frozen=True)
class Unit:
    id: str
    tier: str
    cell: int
    faction: str
    mobility: str
    cost_multiplier: int
    max_terrain: int

    def validate(self, map_data: MapData) -> None:
        if not 0 <= self.cell < map_data.cells:
            raise TacticalError(
                ErrorCode.UNIT_INVALID,
                "unit cell is outside the map",
                context={"unit_id": self.id, "cell": self.cell},
            )
        rule = map_data.mobility_rules.get(self.mobility)
        if rule is None:
            raise TacticalError(
                ErrorCode.UNIT_INVALID,
                "unknown movement class",
                context={"unit_id": self.id, "mobility": self.mobility},
            )
        if self.cost_multiplier != rule.cost_multiplier:
            raise TacticalError(
                ErrorCode.UNIT_INVALID,
                "unit cost multiplier does not match movement rule",
                context={"unit_id": self.id, "mobility": self.mobility},
            )
        if not bool(map_data.passable.flat[self.cell]):
            raise TacticalError(
                ErrorCode.UNIT_INVALID,
                "unit is placed on an impassable cell",
                context={"unit_id": self.id, "cell": self.cell},
            )
        if int(map_data.terrain.flat[self.cell]) > rule.max_terrain:
            raise TacticalError(
                ErrorCode.UNIT_INVALID,
                "unit terrain exceeds movement capability",
                context={
                    "unit_id": self.id,
                    "cell": self.cell,
                    "terrain": int(map_data.terrain.flat[self.cell]),
                    "max_terrain": rule.max_terrain,
                },
            )

    def to_manifest(self, width: int) -> dict[str, Any]:
        x, y = xy(self.cell, width)
        return {
            "id": self.id,
            "tier": self.tier,
            "cell": self.cell,
            "x": x,
            "y": y,
            "faction": self.faction,
            "mobility": self.mobility,
            "cost_multiplier": self.cost_multiplier,
            "max_terrain": self.max_terrain,
        }


@dataclass(frozen=True)
class Query:
    id: str
    tier: str
    unit_id: str
    start: int
    target: int
    mobility: str

    def validate(self, map_data: MapData) -> None:
        mask = map_data.allowed_mask(self.mobility)
        for label, cell in (("start", self.start), ("target", self.target)):
            if not 0 <= cell < map_data.cells:
                raise TacticalError(
                    ErrorCode.UNIT_INVALID,
                    f"query {label} is outside the map",
                    context={"query_id": self.id, "cell": cell},
                )
            if not bool(mask.flat[cell]):
                raise TacticalError(
                    ErrorCode.UNIT_INVALID,
                    f"query {label} is not traversable by mobility",
                    context={
                        "query_id": self.id,
                        "cell": cell,
                        "mobility": self.mobility,
                    },
                )

    def to_manifest(self, width: int) -> dict[str, Any]:
        sx, sy = xy(self.start, width)
        tx, ty = xy(self.target, width)
        return {
            "id": self.id,
            "tier": self.tier,
            "unit_id": self.unit_id,
            "start": self.start,
            "start_xy": [sx, sy],
            "target": self.target,
            "target_xy": [tx, ty],
            "mobility": self.mobility,
        }


@dataclass(frozen=True)
class TierSpec:
    name: str
    size: int
    region_size: int
    unit_count: int
    query_count: int
    query_mode: str
    target_groups: int
    interior_walls: bool = True
    mobility_classes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Budgets:
    tier: str
    query_time_ms: float
    query_memory_mib: float
    total_memory_mib: float
    max_threads: int
    precompute_time_ms: float

    def validate(self) -> None:
        values = (
            self.query_time_ms,
            self.query_memory_mib,
            self.total_memory_mib,
            self.precompute_time_ms,
        )
        if any(value <= 0 for value in values) or self.max_threads <= 0:
            raise TacticalError(
                ErrorCode.BUDGET_DENIED,
                "budgets and thread limit must be positive",
                context={"tier": self.tier},
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "query_time_ms": self.query_time_ms,
            "query_memory_mib": self.query_memory_mib,
            "total_memory_mib": self.total_memory_mib,
            "max_threads": self.max_threads,
            "precompute_time_ms": self.precompute_time_ms,
        }


@dataclass(frozen=True)
class StrategyCandidate:
    strategy: str
    estimated_ms: float
    estimated_bytes: int
    feasible: bool
    error_upper_bound: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "estimated_ms": self.estimated_ms,
            "estimated_bytes": self.estimated_bytes,
            "feasible": self.feasible,
            "error_upper_bound": self.error_upper_bound,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class StrategyDecision:
    query_id: str
    strategy: str
    estimated_ms: float
    estimated_bytes: int
    error_upper_bound: float
    rationale: str
    candidates: tuple[StrategyCandidate, ...]
    signature: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "strategy": self.strategy,
            "estimated_ms": self.estimated_ms,
            "estimated_bytes": self.estimated_bytes,
            "error_upper_bound": self.error_upper_bound,
            "rationale": self.rationale,
            "candidates": [item.to_dict() for item in self.candidates],
            "signature": self.signature,
        }


@dataclass(frozen=True)
class Route:
    query_id: str
    strategy: str
    path: tuple[int, ...]
    cost: int
    expansions: int
    visited: int
    peak_frontier: int
    estimated_peak_bytes: int
    work_units: int
    flow_key: str | None = None
    jump_regions: tuple[int, ...] = ()
    route_regions: tuple[int, ...] = ()

    def sequence_xy(self, width: int) -> list[list[int]]:
        return [[cell % width, cell // width] for cell in self.path]

    def content_payload(self, width: int) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "strategy": self.strategy,
            "cost": self.cost,
            "path": self.sequence_xy(width),
        }

    def route_hash(self, width: int) -> str:
        return sha256_json(self.content_payload(width))

    def to_dict(self, width: int) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "strategy": self.strategy,
            "cost": self.cost,
            "path": self.sequence_xy(width),
            "route_hash": self.route_hash(width),
            "expansions": self.expansions,
            "visited": self.visited,
            "peak_frontier": self.peak_frontier,
            "estimated_peak_bytes": self.estimated_peak_bytes,
            "work_units": self.work_units,
            "flow_key": self.flow_key,
            "jump_regions": list(self.jump_regions),
            "route_regions": list(self.route_regions),
        }


@dataclass
class PlanResult:
    decisions: dict[str, StrategyDecision]
    recomputed: tuple[str, ...]
    reused: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recomputed": list(self.recomputed),
            "reused": list(self.reused),
            "decisions": [
                self.decisions[key].to_dict() for key in sorted(self.decisions)
            ],
        }


@dataclass
class SolveBatchResult:
    routes: dict[str, Route]
    workers_used: int
    downgrades: list[dict[str, Any]]
    cache_evicted: list[str]
    estimated_peak_bytes: int

    def to_dict(self, width: int) -> dict[str, Any]:
        return {
            "workers_used": self.workers_used,
            "downgrades": self.downgrades,
            "cache_evicted": self.cache_evicted,
            "estimated_peak_bytes": self.estimated_peak_bytes,
            "queries": [
                self.routes[key].to_dict(width) for key in sorted(self.routes)
            ],
        }


@dataclass
class MapBundle:
    map_data: MapData
    units: list[Unit]
    queries: list[Query]
    metadata: dict[str, Any]

    def unit_by_id(self) -> dict[str, Unit]:
        return {unit.id: unit for unit in self.units}

    def query_by_id(self) -> dict[str, Query]:
        return {query.id: query for query in self.queries}


def compute_map_hash(
    *,
    tier: str,
    seed: int,
    width: int,
    height: int,
    region_size: int,
    terrain: np.ndarray,
    passable: np.ndarray,
    regions: np.ndarray,
    sectors: np.ndarray,
    mobility: Mapping[str, MobilityRule],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"TACTICAL-MAP-V1\0")
    digest.update(
        struct.pack(
            "<QIIII",
            int(seed),
            int(width),
            int(height),
            int(region_size),
            len(mobility),
        )
    )
    digest.update(tier.encode("utf-8") + b"\0")
    for name, dtype in sorted(mobility.items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(struct.pack("<iI", dtype.cost_multiplier, dtype.max_terrain))
    layers = (
        ("passable", np.asarray(passable, dtype="<u1")),
        ("regions", np.asarray(regions, dtype="<u2")),
        ("sectors", np.asarray(sectors, dtype="<u2")),
        ("terrain", np.asarray(terrain, dtype="<u1")),
    )
    for name, array in layers:
        digest.update(name.encode("ascii") + b"\0")
        digest.update(struct.pack("<II", array.shape[0], array.shape[1]))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def region_cells(map_data: MapData, region_id: int) -> Iterable[int]:
    ys, xs = np.nonzero(map_data.regions == region_id)
    return (int(y) * map_data.width + int(x) for y, x in zip(ys, xs))


def region_content_hash(map_data: MapData, region_id: int) -> str:
    digest = hashlib.sha256()
    digest.update(b"REGION-V1\0")
    digest.update(struct.pack("<I", region_id))
    region_size = map_data.region_size
    cols = map_data.width // region_size
    x0 = (region_id % cols) * region_size
    y0 = (region_id // cols) * region_size
    x1 = x0 + region_size
    y1 = y0 + region_size
    digest.update(struct.pack("<IIII", x0, y0, x1, y1))
    digest.update(np.ascontiguousarray(map_data.terrain[y0:y1, x0:x1]).tobytes())
    digest.update(np.ascontiguousarray(map_data.passable[y0:y1, x0:x1]).tobytes())
    return digest.hexdigest()
