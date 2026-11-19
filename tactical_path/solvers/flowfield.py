"""Region-scoped reverse Dijkstra flow fields."""

from __future__ import annotations

import heapq
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..canonical import sha256_json
from ..errors import ErrorCode, TacticalError
from ..models import MapData, Query, Route


INF = np.iinfo(np.int64).max // 4
FLOW_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))
FLOW_DIR_BYTES = (0, 1, 2, 3)


def flow_key(region_id: int, target: int, mobility: str) -> str:
    return f"flow:{region_id}:{target}:{mobility}"


@dataclass
class FlowField:
    region_id: int
    target: int
    mobility: str
    x0: int
    y0: int
    size: int
    dist: np.ndarray
    direction: np.ndarray
    expansions: int
    peak_frontier: int
    key: str
    content_hash: str

    @property
    def estimated_bytes(self) -> int:
        return int(self.dist.nbytes + self.direction.nbytes + self.peak_frontier * 96)

    def manifest(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "region_id": self.region_id,
            "target": self.target,
            "mobility": self.mobility,
            "x0": self.x0,
            "y0": self.y0,
            "size": self.size,
            "expansions": self.expansions,
            "peak_frontier": self.peak_frontier,
            "estimated_bytes": self.estimated_bytes,
            "content_hash": self.content_hash,
        }

    def solve(self, query: Query, width: int) -> Route:
        if self.target != query.target or self.mobility != query.mobility:
            raise TacticalError(
                ErrorCode.DEPENDENCY_STALE,
                "flow field does not match query target or mobility",
                context={"query_id": query.id, "flow_key": self.key},
            )
        start = query.start
        x = start % width
        y = start // width
        if not (self.x0 <= x < self.x0 + self.size and self.y0 <= y < self.y0 + self.size):
            raise TacticalError(
                ErrorCode.DEPENDENCY_STALE,
                "query start is outside the flow field region",
                context={"query_id": query.id, "flow_key": self.key},
            )
        local = (y - self.y0) * self.size + (x - self.x0)
        if self.dist[local] >= INF:
            raise TacticalError(
                ErrorCode.UNIT_INVALID,
                "query start is unreachable in the flow field",
                context={"query_id": query.id, "flow_key": self.key},
            )

        path: list[int] = [start]
        current = start
        current_local = local
        guard = self.size * self.size + 1
        steps = 0
        while current != self.target and steps <= guard:
            direction = int(self.direction[current_local])
            if direction == 255:
                break
            dx, dy = FLOW_OFFSETS[direction]
            nx = current % width + dx
            ny = current // width + dy
            current = ny * width + nx
            current_local = (ny - self.y0) * self.size + (nx - self.x0)
            path.append(current)
            steps += 1
        if current != self.target:
            raise TacticalError(
                ErrorCode.DEPENDENCY_STALE,
                "flow direction field is inconsistent",
                context={"query_id": query.id, "flow_key": self.key},
            )
        cost = int(self.dist[local])
        route_regions = tuple(sorted({self.region_id}))
        return Route(
            query_id=query.id,
            strategy="region_flow",
            path=tuple(path),
            cost=cost,
            expansions=self.expansions,
            visited=int(np.count_nonzero(self.dist < INF)),
            peak_frontier=self.peak_frontier,
            estimated_peak_bytes=self.estimated_bytes,
            work_units=self.expansions + len(path),
            flow_key=self.key,
            route_regions=route_regions,
        )


class FlowFieldBuilder:
    def __init__(self, map_data: MapData) -> None:
        self.map = map_data
        self._allowed: dict[str, np.ndarray] = {}
        self._cache_lock = threading.RLock()

    def _mask(self, mobility: str) -> np.ndarray:
        with self._cache_lock:
            if mobility not in self._allowed:
                self._allowed[mobility] = self.map.allowed_mask(mobility)
            return self._allowed[mobility]

    def region_bounds(self, region_id: int) -> tuple[int, int, int]:
        region_size = self.map.region_size
        cols = self.map.width // region_size
        x0 = (region_id % cols) * region_size
        y0 = (region_id // cols) * region_size
        return x0, y0, region_size

    def build(self, region_id: int, target: int, mobility: str) -> FlowField:
        x0, y0, size = self.region_bounds(region_id)
        width = self.map.width
        tx = target % width
        ty = target // width
        if not (x0 <= tx < x0 + size and y0 <= ty < y0 + size):
            raise TacticalError(
                ErrorCode.UNIT_INVALID,
                "flow target is outside its declared tactical region",
                context={"region_id": region_id, "target": target},
            )
        mask = self._mask(mobility)
        allowed = np.ascontiguousarray(mask[y0 : y0 + size, x0 : x0 + size]).ravel()
        terrain = np.ascontiguousarray(self.map.terrain[y0 : y0 + size, x0 : x0 + size]).ravel()
        multiplier = self.map.mobility_rules[mobility].cost_multiplier
        dist = np.full(size * size, INF, dtype=np.int64)
        target_local = (ty - y0) * size + (tx - x0)
        dist[target_local] = 0
        heap: list[tuple[int, int]] = [(0, target_local)]
        peak_frontier = 1
        expansions = 0
        while heap:
            value, local = heapq.heappop(heap)
            if value != dist[local]:
                continue
            expansions += 1
            x = local % size
            y = local // size
            enter_current = int(terrain[local]) * multiplier
            if x > 0:
                neighbor = local - 1
                if allowed[neighbor]:
                    candidate = value + enter_current
                    if candidate < dist[neighbor]:
                        dist[neighbor] = candidate
                        heapq.heappush(heap, (candidate, neighbor))
            if x + 1 < size:
                neighbor = local + 1
                if allowed[neighbor]:
                    candidate = value + enter_current
                    if candidate < dist[neighbor]:
                        dist[neighbor] = candidate
                        heapq.heappush(heap, (candidate, neighbor))
            if y > 0:
                neighbor = local - size
                if allowed[neighbor]:
                    candidate = value + enter_current
                    if candidate < dist[neighbor]:
                        dist[neighbor] = candidate
                        heapq.heappush(heap, (candidate, neighbor))
            if y + 1 < size:
                neighbor = local + size
                if allowed[neighbor]:
                    candidate = value + enter_current
                    if candidate < dist[neighbor]:
                        dist[neighbor] = candidate
                        heapq.heappush(heap, (candidate, neighbor))
            peak_frontier = max(peak_frontier, len(heap))

        direction = np.full(size * size, 255, dtype=np.uint8)
        for local in range(size * size):
            if local == target_local or not allowed[local] or dist[local] >= INF:
                continue
            x = local % size
            y = local // size
            best_cost = INF
            best_neighbor = -1
            for direction_index, (dx, dy) in enumerate(FLOW_OFFSETS):
                nx = x + dx
                ny = y + dy
                if nx < 0 or nx >= size or ny < 0 or ny >= size:
                    continue
                neighbor = ny * size + nx
                if not allowed[neighbor] or dist[neighbor] >= INF:
                    continue
                candidate = int(terrain[neighbor]) * multiplier + int(dist[neighbor])
                if candidate < best_cost or (
                    candidate == best_cost and (best_neighbor < 0 or neighbor < best_neighbor)
                ):
                    best_cost = candidate
                    best_neighbor = neighbor
                    best_direction = direction_index
            if best_neighbor < 0 or best_cost != int(dist[local]):
                raise TacticalError(
                    ErrorCode.DEPENDENCY_STALE,
                    "failed to construct a valid flow direction",
                    context={
                        "region_id": region_id,
                        "target": target,
                        "local": local,
                    },
                )
            direction[local] = FLOW_DIR_BYTES[best_direction]

        field_key = flow_key(region_id, target, mobility)
        content_hash = sha256_json(
            {
                "key": field_key,
                "region_id": region_id,
                "target": target,
                "mobility": mobility,
                "x0": x0,
                "y0": y0,
                "size": size,
                "dist": dist.astype("<i8").tobytes().hex(),
                "direction": direction.tobytes().hex(),
            }
        )
        return FlowField(
            region_id=region_id,
            target=target,
            mobility=mobility,
            x0=x0,
            y0=y0,
            size=size,
            dist=dist,
            direction=direction,
            expansions=expansions,
            peak_frontier=peak_frontier,
            key=field_key,
            content_hash=content_hash,
        )


def flow_eligible(map_data: MapData, query: Query) -> bool:
    return map_data.region_of_cell(query.start) == map_data.region_of_cell(query.target)
