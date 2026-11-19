"""Exact region-portal jump acceleration.

Every tactical region stores local shortest distances between its boundary
portals.  The query then searches a complete portal graph.  Because a global
shortest path can be decomposed at every region-boundary crossing, and every
within-region segment is replaced by an exact local shortest path, this is an
exact acceleration rather than a heuristic approximation.
"""

from __future__ import annotations

import heapq
import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from ..canonical import sha256_json
from ..errors import ErrorCode, TacticalError
from ..models import MapData, Query, Route


INF = np.iinfo(np.int64).max // 4
OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def jump_key(region_id: int, mobility: str) -> str:
    return f"jump:{region_id}:{mobility}"


@dataclass
class JumpData:
    region_id: int
    mobility: str
    map_width: int
    x0: int
    y0: int
    size: int
    portals: tuple[int, ...]
    distances: dict[tuple[int, int], int]
    directions: dict[int, np.ndarray]
    expansions: int
    peak_frontier: int
    content_hash: str

    @property
    def key(self) -> str:
        return jump_key(self.region_id, self.mobility)

    @property
    def estimated_bytes(self) -> int:
        return int(
            len(self.portals) * self.size * self.size * 9
            + self.peak_frontier * 96
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "region_id": self.region_id,
            "mobility": self.mobility,
            "x0": self.x0,
            "y0": self.y0,
            "size": self.size,
            "portals": list(self.portals),
            "portal_count": len(self.portals),
            "expansions": self.expansions,
            "peak_frontier": self.peak_frontier,
            "estimated_bytes": self.estimated_bytes,
            "content_hash": self.content_hash,
        }

    def portal_path(self, source: int, target: int) -> list[int]:
        """Reconstruct a precomputed portal-to-portal segment in O(length)."""
        width = self.map_width
        if source == target:
            return [source]
        directions = self.directions.get(source)
        if directions is None:
            raise TacticalError(
                ErrorCode.DEPENDENCY_STALE,
                "missing portal direction cache",
                context={"region_id": self.region_id, "source": source},
            )
        local = (target // width - self.y0) * self.size + (target % width - self.x0)
        result = [target]
        guard = self.size * self.size + 1
        for _ in range(guard):
            direction = int(directions[local])
            if direction == 255:
                break
            dx, dy = OFFSETS[direction]
            x = local % self.size + dx
            y = local // self.size + dy
            local = y * self.size + x
            result.append((self.y0 + y) * width + self.x0 + x)
        if not result or result[-1] != source:
            raise TacticalError(
                ErrorCode.DEPENDENCY_STALE,
                "portal direction cache does not reach source",
                context={
                    "region_id": self.region_id,
                    "source": source,
                    "target": target,
                },
            )
        result.reverse()
        return result


@dataclass
class LocalSearchStats:
    expansions: int = 0
    peak_frontier: int = 0

    def merge(self, other: "LocalSearchStats") -> None:
        self.expansions += other.expansions
        self.peak_frontier = max(self.peak_frontier, other.peak_frontier)


class JumpDataBuilder:
    def __init__(self, map_data: MapData) -> None:
        self.map = map_data
        self._allowed: dict[str, np.ndarray] = {}
        self._guard = threading.RLock()
        self._locks: dict[tuple[int, str], threading.RLock] = {}
        self._cache: dict[tuple[int, str], JumpData] = {}

    def _mask(self, mobility: str) -> np.ndarray:
        if mobility not in self._allowed:
            self._allowed[mobility] = self.map.allowed_mask(mobility)
        return self._allowed[mobility]

    def bounds(self, region_id: int) -> tuple[int, int, int]:
        size = self.map.region_size
        cols = self.map.width // size
        return (region_id % cols) * size, (region_id // cols) * size, size

    def cached(self, region_id: int, mobility: str) -> JumpData | None:
        with self._guard:
            return self._cache.get((region_id, mobility))

    def all_cached(self) -> list[JumpData]:
        with self._guard:
            return [self._cache[key] for key in sorted(self._cache)]

    def get(self, region_id: int, mobility: str) -> JumpData:
        key = (region_id, mobility)
        with self._guard:
            lock = self._locks.setdefault(key, threading.RLock())
        with lock:
            with self._guard:
                if key in self._cache:
                    return self._cache[key]
            data = self.build(region_id, mobility)
            with self._guard:
                self._cache[key] = data
            return data

    def invalidate(self, region_id: int | None = None) -> None:
        with self._guard:
            if region_id is None:
                self._cache.clear()
            else:
                for key in list(self._cache):
                    if key[0] == region_id:
                        del self._cache[key]

    def invalidate_key(self, region_id: int, mobility: str) -> None:
        with self._guard:
            self._cache.pop((region_id, mobility), None)

    def _local_arrays(
        self, region_id: int, mobility: str
    ) -> tuple[np.ndarray, np.ndarray, int, int, int, int]:
        x0, y0, size = self.bounds(region_id)
        mask = self._mask(mobility)
        allowed = np.ascontiguousarray(mask[y0 : y0 + size, x0 : x0 + size]).ravel()
        terrain = np.ascontiguousarray(
            self.map.terrain[y0 : y0 + size, x0 : x0 + size]
        ).ravel()
        multiplier = self.map.mobility_rules[mobility].cost_multiplier
        return allowed, terrain, x0, y0, size, multiplier

    def _find_portals(
        self, region_id: int, mobility: str, x0: int, y0: int, size: int
    ) -> tuple[int, ...]:
        width = self.map.width
        height = self.map.height
        mask = self._mask(mobility)
        portals: list[int] = []
        for local_y in range(size):
            for local_x in range(size):
                if local_x not in (0, size - 1) and local_y not in (0, size - 1):
                    continue
                x = x0 + local_x
                y = y0 + local_y
                cell = y * width + x
                if not mask[y, x]:
                    continue
                for dx, dy in OFFSETS:
                    nx = x + dx
                    ny = y + dy
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue
                    if int(self.map.regions[ny, nx]) == region_id:
                        continue
                    if mask[ny, nx]:
                        portals.append(cell)
                        break
        return tuple(sorted(set(portals)))

    def build(self, region_id: int, mobility: str) -> JumpData:
        allowed, terrain, x0, y0, size, multiplier = self._local_arrays(
            region_id, mobility
        )
        portals = self._find_portals(region_id, mobility, x0, y0, size)
        portal_index = {cell: index for index, cell in enumerate(portals)}
        local_index = {
            cell: (cell // self.map.width - y0) * size + (cell % self.map.width - x0)
            for cell in portals
        }
        distances: dict[tuple[int, int], int] = {}
        directions: dict[int, np.ndarray] = {}
        total_expansions = 0
        global_peak = 0
        for source in portals:
            portal_locals = set(local_index.values())
            dist = np.full(size * size, INF, dtype=np.int64)
            parent_dir = np.full(size * size, 255, dtype=np.uint8)
            start_local = local_index[source]
            dist[start_local] = 0
            heap: list[tuple[int, int]] = [(0, start_local)]
            peak = 1
            expansions = 0
            while heap:
                value, local = heapq.heappop(heap)
                if value != dist[local]:
                    continue
                expansions += 1
                if portal_locals:
                    portal_locals.discard(local)
                    if not portal_locals:
                        break
                x = local % size
                y = local // size
                if x > 0:
                    neighbor = local - 1
                    if allowed[neighbor]:
                        candidate = value + int(terrain[neighbor]) * multiplier
                        if candidate < dist[neighbor]:
                            dist[neighbor] = candidate
                            parent_dir[neighbor] = 1
                            heapq.heappush(heap, (candidate, neighbor))
                if x + 1 < size:
                    neighbor = local + 1
                    if allowed[neighbor]:
                        candidate = value + int(terrain[neighbor]) * multiplier
                        if candidate < dist[neighbor]:
                            dist[neighbor] = candidate
                            parent_dir[neighbor] = 0
                            heapq.heappush(heap, (candidate, neighbor))
                if y > 0:
                    neighbor = local - size
                    if allowed[neighbor]:
                        candidate = value + int(terrain[neighbor]) * multiplier
                        if candidate < dist[neighbor]:
                            dist[neighbor] = candidate
                            parent_dir[neighbor] = 3
                            heapq.heappush(heap, (candidate, neighbor))
                if y + 1 < size:
                    neighbor = local + size
                    if allowed[neighbor]:
                        candidate = value + int(terrain[neighbor]) * multiplier
                        if candidate < dist[neighbor]:
                            dist[neighbor] = candidate
                            parent_dir[neighbor] = 2
                            heapq.heappush(heap, (candidate, neighbor))
                peak = max(peak, len(heap))
            total_expansions += expansions
            global_peak = max(global_peak, peak)
            for destination in portals:
                destination_local = local_index[destination]
                distances[(source, destination)] = int(dist[destination_local])
            directions[source] = parent_dir

        direction_digest = hashlib.sha256()
        for source in portals:
            direction_digest.update(directions[source].tobytes())
        content_hash = sha256_json(
            {
                "region_id": region_id,
                "mobility": mobility,
                "x0": x0,
                "y0": y0,
                "size": size,
                "portals": list(portals),
                "distances": [
                    [source, destination, distances[(source, destination)]]
                    for source in portals
                    for destination in portals
                ],
                "directions_sha256": direction_digest.hexdigest(),
            }
        )
        return JumpData(
            region_id=region_id,
            mobility=mobility,
            map_width=self.map.width,
            x0=x0,
            y0=y0,
            size=size,
            portals=portals,
            distances=distances,
            directions=directions,
            expansions=total_expansions,
            peak_frontier=global_peak,
            content_hash=content_hash,
        )


class ChunkJumpSolver:
    def __init__(self, map_data: MapData, builder: JumpDataBuilder) -> None:
        self.map = map_data
        self.builder = builder
        self._allowed: dict[str, np.ndarray] = {}
        self._min_enter: dict[str, int] = {}
        self._cache_lock = threading.RLock()

    def _mask(self, mobility: str) -> np.ndarray:
        with self._cache_lock:
            if mobility not in self._allowed:
                mask = self.map.allowed_mask(mobility)
                self._min_enter[mobility] = int(self.map.terrain[mask].min()) * self.map.mobility_rules[mobility].cost_multiplier
                self._allowed[mobility] = mask
            return self._allowed[mobility]

    def _dijkstra_distances(
        self,
        data: JumpData,
        *,
        source: int | None,
        target: int | None,
        want_path: bool = False,
        stop_cells: Iterable[int] | None = None,
    ) -> tuple[np.ndarray, list[int] | None, LocalSearchStats]:
        width = self.map.width
        size = data.size
        mask = self._mask(data.mobility)
        terrain = self.map.terrain
        multiplier = self.map.mobility_rules[data.mobility].cost_multiplier
        dist = np.full(size * size, INF, dtype=np.int64)
        parent = np.full(size * size, -1, dtype=np.int64)
        root = source if source is not None else target
        if root is None:
            raise TacticalError(ErrorCode.DEPENDENCY_STALE, "missing local root")
        root_local = (root // width - data.y0) * size + (root % width - data.x0)
        dist[root_local] = 0
        heap: list[tuple[int, int]] = [(0, root_local)]
        stop_locals: set[int] = set()
        if stop_cells is not None:
            for cell in stop_cells:
                stop_locals.add(
                    (cell // width - data.y0) * size + (cell % width - data.x0)
                )
            stop_locals.add(root_local)
        stats = LocalSearchStats(peak_frontier=1)
        while heap:
            value, local = heapq.heappop(heap)
            if value != dist[local]:
                continue
            stats.expansions += 1
            if source is not None and target is not None:
                target_local = (target // width - data.y0) * size + (target % width - data.x0)
                if local == target_local:
                    break
            if stop_locals:
                stop_locals.discard(local)
                if not stop_locals:
                    break
            x = local % size
            y = local // size
            for dx, dy in OFFSETS:
                nx = x + dx
                ny = y + dy
                if nx < 0 or nx >= size or ny < 0 or ny >= size:
                    continue
                neighbor = ny * size + nx
                gx = data.x0 + nx
                gy = data.y0 + ny
                if not mask[gy, gx]:
                    continue
                if source is not None:
                    candidate = value + int(terrain[gy, gx]) * multiplier
                else:
                    # Reverse relaxation: distance-to-target of the current
                    # cell is added when moving backwards to its neighbour.
                    cx = data.x0 + x
                    cy = data.y0 + y
                    candidate = value + int(terrain[cy, cx]) * multiplier
                if candidate < dist[neighbor]:
                    dist[neighbor] = candidate
                    parent[neighbor] = local
                    heapq.heappush(heap, (candidate, neighbor))
            stats.peak_frontier = max(stats.peak_frontier, len(heap))

        path: list[int] | None = None
        if want_path and source is not None and target is not None:
            target_local = (target // width - data.y0) * size + (target % width - data.x0)
            if dist[target_local] >= INF:
                path = None
            else:
                chain = [target_local]
                cursor = target_local
                while cursor != root_local:
                    cursor = int(parent[cursor])
                    if cursor < 0:
                        path = None
                        break
                    chain.append(cursor)
                if path is None:
                    if chain and chain[-1] == root_local:
                        chain.reverse()
                        path = [
                            (data.y0 + local // size) * width + data.x0 + local % size
                            for local in chain
                        ]
                else:
                    chain.reverse()
                    path = [
                        (data.y0 + local // size) * width + data.x0 + local % size
                        for local in chain
                    ]
        return dist, path, stats

    def _local_path(
        self, data: JumpData, start: int, target: int
    ) -> tuple[list[int], int, LocalSearchStats]:
        dist, path, stats = self._dijkstra_distances(
            data, source=start, target=target, want_path=True
        )
        width = self.map.width
        target_local = (target // width - data.y0) * data.size + (target % width - data.x0)
        cost = int(dist[target_local])
        if path is None or cost >= INF:
            raise TacticalError(
                ErrorCode.UNIT_INVALID,
                "regional jump segment has no route",
                context={
                    "region_id": data.region_id,
                    "start": start,
                    "target": target,
                },
            )
        return path, cost, stats

    def _cross_portals(self, cell: int, mobility: str) -> tuple[int, ...]:
        width = self.map.width
        height = self.map.height
        mask = self._mask(mobility)
        x = cell % width
        y = cell // width
        region_id = int(self.map.regions[y, x])
        result: list[int] = []
        for dx, dy in OFFSETS:
            nx = x + dx
            ny = y + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if not mask[ny, nx]:
                continue
            if int(self.map.regions[ny, nx]) == region_id:
                continue
            result.append(ny * width + nx)
        return tuple(sorted(result))

    def solve(self, query: Query) -> Route:
        width = self.map.width
        start = query.start
        target = query.target
        mobility = query.mobility
        if start == target:
            return Route(
                query_id=query.id,
                strategy="chunk_jump",
                path=(start,),
                cost=0,
                expansions=0,
                visited=1,
                peak_frontier=1,
                estimated_peak_bytes=128,
                work_units=0,
                jump_regions=(self.map.region_of_cell(start),),
                route_regions=(self.map.region_of_cell(start),),
            )
        source_region = self.map.region_of_cell(start)
        target_region = self.map.region_of_cell(target)
        source_data = self.builder.get(source_region, mobility)
        target_data = self.builder.get(target_region, mobility)
        stats = LocalSearchStats()

        source_stop = list(source_data.portals)
        if source_region == target_region:
            source_stop.append(target)
        source_dist, _unused, source_stats = self._dijkstra_distances(
            source_data,
            source=start,
            target=None,
            stop_cells=source_stop if source_stop else None,
        )
        stats.merge(source_stats)
        target_stop = list(target_data.portals) or [target]
        target_dist, _unused2, target_stats = self._dijkstra_distances(
            target_data,
            source=None,
            target=target,
            stop_cells=target_stop,
        )
        stats.merge(target_stats)

        best_cost = INF
        best_chain: tuple[int, ...] | None = None
        direct = False
        if source_region == target_region:
            target_local = (target // width - source_data.y0) * source_data.size + (target % width - source_data.x0)
            if int(source_dist[target_local]) < best_cost:
                best_cost = int(source_dist[target_local])
                direct = True

        min_enter = self._min_enter[mobility]
        heap: list[tuple[int, int, int, int]] = []
        best: dict[int, int] = {}
        parent: dict[int, int | None] = {}
        initial: list[int] = []
        for portal in source_data.portals:
            local = (portal // width - source_data.y0) * source_data.size + (portal % width - source_data.x0)
            value = int(source_dist[local])
            if value < INF:
                initial.append(portal)
        for portal in sorted(initial, key=lambda p: (source_dist[(p // width - source_data.y0) * source_data.size + (p % width - source_data.x0)], p)):
            value = int(source_dist[(portal // width - source_data.y0) * source_data.size + (portal % width - source_data.x0)])
            best[portal] = value
            parent[portal] = None
            dx = abs(portal % width - target % width)
            dy = abs(portal // width - target // width)
            h = (dx + dy) * min_enter
            heapq.heappush(heap, (value + h, h, value, portal))

        expansions = 0
        peak_frontier = len(heap)
        while heap:
            f, h, g, portal = heapq.heappop(heap)
            if best.get(portal) != g:
                continue
            if f >= best_cost:
                break
            expansions += 1
            portal_region = self.map.region_of_cell(portal)
            if portal_region == target_region:
                local = (portal // width - target_data.y0) * target_data.size + (portal % width - target_data.x0)
                tail = int(target_dist[local])
                if tail < INF:
                    candidate = g + tail
                    if candidate < best_cost:
                        best_cost = candidate
                        chain: list[int] = []
                        cursor: int | None = portal
                        while cursor is not None:
                            chain.append(cursor)
                            cursor = parent.get(cursor)
                        chain.reverse()
                        best_chain = tuple(chain)
                        direct = False

            data = self.builder.get(portal_region, mobility)
            # Same-region portal jumps.
            for other in data.portals:
                if other == portal:
                    continue
                weight = data.distances[(portal, other)]
                if weight >= INF:
                    continue
                candidate = g + weight
                if candidate < best.get(other, INF):
                    best[other] = candidate
                    parent[other] = portal
                    dx = abs(other % width - target % width)
                    dy = abs(other // width - target // width)
                    oh = (dx + dy) * min_enter
                    heapq.heappush(heap, (candidate + oh, oh, candidate, other))
            # Cross-boundary one-cell jumps.
            for other in self._cross_portals(portal, mobility):
                weight = int(self.map.terrain.flat[other]) * self.map.mobility_rules[mobility].cost_multiplier
                candidate = g + weight
                if candidate < best.get(other, INF):
                    best[other] = candidate
                    parent[other] = portal
                    dx = abs(other % width - target % width)
                    dy = abs(other // width - target // width)
                    oh = (dx + dy) * min_enter
                    heapq.heappush(heap, (candidate + oh, oh, candidate, other))
            peak_frontier = max(peak_frontier, len(heap))

        if best_cost >= INF:
            raise TacticalError(
                ErrorCode.UNIT_INVALID,
                "query has no feasible chunk-jump route",
                context={"query_id": query.id},
            )

        jump_regions: set[int] = {source_region, target_region}
        if direct:
            path, direct_cost, path_stats = self._local_path(source_data, start, target)
            stats.merge(path_stats)
            if direct_cost != best_cost:
                raise TacticalError(
                    ErrorCode.DEPENDENCY_STALE,
                    "direct and flow distance disagree",
                    context={"query_id": query.id},
                )
        else:
            assert best_chain is not None
            path = []
            first = best_chain[0]
            segment, segment_cost, path_stats = self._local_path(source_data, start, first)
            stats.merge(path_stats)
            path.extend(segment)
            previous = first
            for portal in best_chain[1:]:
                previous_region = self.map.region_of_cell(previous)
                jump_regions.add(previous_region)
                if previous_region == self.map.region_of_cell(portal):
                    data = self.builder.get(previous_region, mobility)
                    segment = data.portal_path(previous, portal)
                    segment_cost = data.distances[(previous, portal)]
                else:
                    if abs(previous - portal) not in (1, width):
                        raise TacticalError(
                            ErrorCode.DEPENDENCY_STALE,
                            "cross-region jump is not adjacent",
                            context={"query_id": query.id},
                        )
                    segment = [previous, portal]
                    segment_cost = int(self.map.terrain.flat[portal]) * self.map.mobility_rules[mobility].cost_multiplier
                path.extend(segment[1:])
                previous = portal
            jump_regions.add(self.map.region_of_cell(previous))
            tail, tail_cost, path_stats = self._local_path(target_data, previous, target)
            stats.merge(path_stats)
            path.extend(tail[1:])

        actual_cost = sum(
            int(self.map.terrain.flat[cell]) * self.map.mobility_rules[mobility].cost_multiplier
            for cell in path[1:]
        )
        if actual_cost != best_cost:
            raise TacticalError(
                ErrorCode.DEPENDENCY_STALE,
                "chunk-jump reconstruction cost mismatch",
                context={
                    "query_id": query.id,
                    "expected": best_cost,
                    "actual": actual_cost,
                },
            )
        used_jump = sorted(
            region for region in jump_regions if self.builder.cached(region, mobility) is not None
        )
        route_regions = tuple(sorted({self.map.region_of_cell(cell) for cell in path}))
        return Route(
            query_id=query.id,
            strategy="chunk_jump",
            path=tuple(path),
            cost=best_cost,
            expansions=stats.expansions + expansions,
            visited=len(best),
            peak_frontier=max(peak_frontier, stats.peak_frontier),
            estimated_peak_bytes=peak_frontier * 96 + stats.peak_frontier * 96,
            work_units=stats.expansions + expansions,
            jump_regions=tuple(used_jump),
            route_regions=route_regions,
        )
