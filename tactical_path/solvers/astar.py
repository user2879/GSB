"""Deterministic four-neighbour A* on the weighted tactical grid."""

from __future__ import annotations

import heapq
import threading
from dataclasses import dataclass

import numpy as np

from ..errors import ErrorCode, TacticalError
from ..models import MapData, Query, Route


@dataclass(frozen=True)
class AStarConfig:
    heuristic: str = "manhattan"


class AStarSolver:
    def __init__(self, map_data: MapData, config: AStarConfig | None = None) -> None:
        self.map = map_data
        self.config = config or AStarConfig()
        self._allowed: dict[str, np.ndarray] = {}
        self._min_cost: dict[str, int] = {}
        self._cache_lock = threading.RLock()

    def _mask(self, mobility: str) -> np.ndarray:
        with self._cache_lock:
            if mobility not in self._allowed:
                mask = self.map.allowed_mask(mobility)
                if not mask.any():
                    self._min_cost[mobility] = 1
                else:
                    self._min_cost[mobility] = int(self.map.terrain[mask].min())
                self._allowed[mobility] = mask
            return self._allowed[mobility]

    def _heuristic(
        self, cell: int, target: int, width: int, min_cost: int, mobility: str
    ) -> int:
        if self.config.heuristic == "zero":
            return 0
        rule = self.map.mobility_rules[mobility]
        dx = abs(cell % width - target % width)
        dy = abs(cell // width - target // width)
        if self.config.heuristic == "euclidean_ceiling":
            # Keeping the heuristic integral preserves byte-stable tie breaking.
            squared = dx * dx + dy * dy
            distance = int(np.ceil(np.sqrt(squared))) if squared else 0
            return distance * min_cost * rule.cost_multiplier
        return (dx + dy) * min_cost * rule.cost_multiplier

    def solve(self, query: Query, strategy: str = "dense_astar") -> Route:
        width = self.map.width
        start = query.start
        target = query.target
        mobility = query.mobility
        mask = self._mask(mobility)
        min_cost = self._min_cost[mobility]
        rule = self.map.mobility_rules[mobility]
        multiplier = rule.cost_multiplier
        terrain = self.map.terrain.ravel()

        if start == target:
            path = (start,)
            return Route(
                query_id=query.id,
                strategy=strategy,
                path=path,
                cost=0,
                expansions=0,
                visited=1,
                peak_frontier=1,
                estimated_peak_bytes=128,
                work_units=0,
                route_regions=(self.map.region_of_cell(start),),
            )

        h0 = self._heuristic(start, target, width, min_cost, mobility)
        heap: list[tuple[int, int, int, int]] = [(h0, h0, 0, start)]
        best: dict[int, int] = {start: 0}
        parent: dict[int, int] = {}
        peak_frontier = 1
        expansions = 0

        while heap:
            f, h, g, cell = heapq.heappop(heap)
            if best.get(cell) != g:
                continue
            expansions += 1
            if cell == target:
                path_rev = [cell]
                current = cell
                while current != start:
                    current = parent[current]
                    path_rev.append(current)
                path_rev.reverse()
                path = tuple(path_rev)
                route_regions = tuple(
                    sorted({self.map.region_of_cell(node) for node in path})
                )
                return Route(
                    query_id=query.id,
                    strategy=strategy,
                    path=path,
                    cost=g,
                    expansions=expansions,
                    visited=len(best),
                    peak_frontier=peak_frontier,
                    estimated_peak_bytes=peak_frontier * 96 + len(best) * 72,
                    work_units=expansions,
                    route_regions=route_regions,
                )

            x = cell % width
            if x > 0:
                neighbor = cell - 1
                if mask.flat[neighbor]:
                    ng = g + int(terrain[neighbor]) * multiplier
                    if ng < best.get(neighbor, 1 << 62):
                        best[neighbor] = ng
                        parent[neighbor] = cell
                        nh = self._heuristic(
                            neighbor, target, width, min_cost, mobility
                        )
                        heapq.heappush(heap, (ng + nh, nh, ng, neighbor))
            if x + 1 < width:
                neighbor = cell + 1
                if mask.flat[neighbor]:
                    ng = g + int(terrain[neighbor]) * multiplier
                    if ng < best.get(neighbor, 1 << 62):
                        best[neighbor] = ng
                        parent[neighbor] = cell
                        nh = self._heuristic(
                            neighbor, target, width, min_cost, mobility
                        )
                        heapq.heappush(heap, (ng + nh, nh, ng, neighbor))
            y = cell // width
            if y > 0:
                neighbor = cell - width
                if mask.flat[neighbor]:
                    ng = g + int(terrain[neighbor]) * multiplier
                    if ng < best.get(neighbor, 1 << 62):
                        best[neighbor] = ng
                        parent[neighbor] = cell
                        nh = self._heuristic(
                            neighbor, target, width, min_cost, mobility
                        )
                        heapq.heappush(heap, (ng + nh, nh, ng, neighbor))
            if y + 1 < self.map.height:
                neighbor = cell + width
                if mask.flat[neighbor]:
                    ng = g + int(terrain[neighbor]) * multiplier
                    if ng < best.get(neighbor, 1 << 62):
                        best[neighbor] = ng
                        parent[neighbor] = cell
                        nh = self._heuristic(
                            neighbor, target, width, min_cost, mobility
                        )
                        heapq.heappush(heap, (ng + nh, nh, ng, neighbor))
            peak_frontier = max(peak_frontier, len(heap))

        raise TacticalError(
            ErrorCode.UNIT_INVALID,
            "query has no feasible route",
            context={"query_id": query.id, "start": start, "target": target},
        )


def path_cost(map_data: MapData, path: tuple[int, ...], mobility: str) -> int:
    multiplier = map_data.mobility_rules[mobility].cost_multiplier
    return sum(int(map_data.terrain.flat[cell]) * multiplier for cell in path[1:])
