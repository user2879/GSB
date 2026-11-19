"""区域级流场求解。

设计要点
--------
1. 每个战术区域预计算一张方向场：以区域内全部可通行格为源做区域内
   Dijkstra（仅四邻域、仅同区域边），得到 ``dist[src][cell]`` 与
   ``next_step[src][cell]``（cell 朝 src 走的第一格）。
2. 区域之间通过“边界门户/跳点”连接：门户是跨区域的可通行相邻格对
   ``(a, b)``，门户穿越代价取进入格地形代价。
3. 跨区域路由在压缩门户图上做确定性 Dijkstra，节点为区域内门户端点，
   再用各区域方向场把门户端点串成完整格序列。
4. 流场内容（含距离表与方向表）参与内容哈希，地形变更后哈希随之改变，
   依赖图据此判断可否复用。
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .astar import Route
from .encoding import content_hash
from .errors import QueryUnreachableError
from .grid import Cell, TacticalGrid

INF = 10**18


@dataclass(frozen=True)
class Portal:
    portal_id: str
    a: Cell
    b: Cell

    def cells(self) -> Tuple[Cell, Cell]:
        return (self.a, self.b)

    def other(self, cell: Cell) -> Cell:
        if cell == self.a:
            return self.b
        if cell == self.b:
            return self.a
        raise KeyError(f"门户 {self.portal_id} 不包含格 {cell}")

    def to_payload(self) -> Dict[str, Any]:
        return {"portal_id": self.portal_id, "a": list(self.a), "b": list(self.b)}


@dataclass
class RegionFlowField:
    region_id: int
    cells: Tuple[Cell, ...]
    portals: Tuple[Portal, ...]
    dist: Dict[Cell, Dict[Cell, int]]
    next_step: Dict[Cell, Dict[Cell, Optional[Cell]]]
    internal_expanded: int

    def distance(self, src: Cell, dst: Cell) -> int:
        return int(self.dist[src].get(dst, INF))

    def first_step(self, src: Cell, dst: Cell) -> Optional[Cell]:
        if src == dst:
            return dst
        return self.next_step[src].get(dst)

    def content_payload(self) -> Dict[str, Any]:
        ordered_cells = list(self.cells)
        return {
            "region_id": self.region_id,
            "cells": [list(cell) for cell in ordered_cells],
            "portals": [portal.to_payload() for portal in self.portals],
            "dist": [
                [list(src), [[list(dst), self.dist[src].get(dst, INF)] for dst in ordered_cells]]
                for src in ordered_cells
            ],
            "next_step": [
                [
                    list(src),
                    [
                        [list(dst), list(nxt) if nxt is not None else None]
                        for dst, nxt in sorted(self.next_step[src].items())
                    ],
                ]
                for src in ordered_cells
            ],
            "internal_expanded": self.internal_expanded,
        }

    def content_hash(self) -> str:
        return content_hash(self.content_payload())


@dataclass
class JumpPointTable:
    """区域边界跳点（门户）表，只依赖边界两侧的通行与地形代价。"""

    portals: Tuple[Portal, ...]
    portal_hash_input: Dict[str, Any] = field(default_factory=dict)

    def portal_ids(self) -> List[str]:
        return [portal.portal_id for portal in self.portals]

    def by_id(self, portal_id: str) -> Portal:
        for portal in self.portals:
            if portal.portal_id == portal_id:
                return portal
        raise KeyError(portal_id)

    def content_payload(self) -> Dict[str, Any]:
        return {
            "portals": [portal.to_payload() for portal in self.portals],
            "hash_input": self.portal_hash_input,
        }

    def content_hash(self) -> str:
        return content_hash(self.content_payload())


class FlowFieldBuilder:
    def __init__(self, grid: TacticalGrid) -> None:
        self.grid = grid

    def build_jump_points(self) -> JumpPointTable:
        portals: List[Portal] = []
        hash_input: Dict[str, Any] = {}
        for index, (a, b) in enumerate(self.grid.boundary_portals()):
            portal = Portal(portal_id=f"P{index:04d}", a=a, b=b)
            portals.append(portal)
            hash_input[portal.portal_id] = {
                "a": list(a),
                "b": list(b),
                "cost_a": self.grid.cost_of(a),
                "cost_b": self.grid.cost_of(b),
                "pass_a": int(self.grid.is_passable(a)),
                "pass_b": int(self.grid.is_passable(b)),
            }
        return JumpPointTable(portals=tuple(portals), portal_hash_input=hash_input)

    def portals_touching_cell(self, jump_table: JumpPointTable, cell: Cell) -> List[Portal]:
        return [portal for portal in jump_table.portals if cell in portal.cells()]

    def build_region_flow(
        self, region_id: int, jump_table: JumpPointTable
    ) -> RegionFlowField:
        grid = self.grid
        cells = tuple(grid.cells_of_region(region_id))
        passable_cells = tuple(cell for cell in cells if grid.is_passable(cell))
        region_portals = tuple(
            portal
            for portal in jump_table.portals
            if grid.region_of_cell(portal.a) == region_id
            or grid.region_of_cell(portal.b) == region_id
        )
        dist: Dict[Cell, Dict[Cell, int]] = {}
        next_step: Dict[Cell, Dict[Cell, Optional[Cell]]] = {}
        expanded_total = 0
        for source in passable_cells:
            d, nxt, expanded = self._dijkstra_in_region(source, region_id)
            dist[source] = d
            next_step[source] = nxt
            expanded_total += expanded
        return RegionFlowField(
            region_id=region_id,
            cells=passable_cells,
            portals=region_portals,
            dist=dist,
            next_step=next_step,
            internal_expanded=expanded_total,
        )

    def _dijkstra_in_region(
        self, source: Cell, region_id: int
    ) -> Tuple[Dict[Cell, int], Dict[Cell, Optional[Cell]], int]:
        grid = self.grid
        dist = {source: 0}
        first: Dict[Cell, Optional[Cell]] = {source: source}
        heap: List[Tuple[int, int, Cell]] = [(0, 0, source)]
        counter = 0
        expanded = 0
        while heap:
            cost, _tie, current = heapq.heappop(heap)
            if cost != dist[current]:
                continue
            expanded += 1
            for neighbor in grid.neighbors(current):
                if grid.region_of_cell(neighbor) != region_id:
                    continue
                tentative = cost + grid.edge_cost(current, neighbor)
                if tentative < dist.get(neighbor, INF):
                    dist[neighbor] = tentative
                    first[neighbor] = neighbor if current == source else first[current]
                    counter += 1
                    heapq.heappush(heap, (tentative, counter, neighbor))
        result_next: Dict[Cell, Optional[Cell]] = {}
        for cell in dist:
            if cell == source:
                result_next[cell] = cell
            else:
                result_next[cell] = first[cell]
        return dist, result_next, expanded


@dataclass
class FlowFieldSolver:
    grid: TacticalGrid
    jump_table: JumpPointTable
    flows: Dict[int, RegionFlowField]
    route_expanded: int = 0

    def _portal_cross_cost(self, portal: Portal, entering: Cell) -> int:
        return self.grid.cost_of(entering)

    def solve(
        self,
        source: Cell,
        target: Cell,
        movement_budget: Optional[int] = None,
        cache_source: str = "none",
    ) -> Route:
        grid = self.grid
        if not grid.is_passable(source):
            from .errors import UnitInvalidError

            raise UnitInvalidError("起点不可通行", cell=list(source))
        if not grid.is_passable(target):
            raise QueryUnreachableError(
                "目标格不可通行",
                source=list(source),
                target=list(target),
            )

        expanded_holder = {"count": 0}
        start_region = grid.region_of_cell(source)
        target_region = grid.region_of_cell(target)

        if start_region == target_region:
            flow = self.flows[start_region]
            cost = flow.distance(source, target)
            if cost >= INF:
                raise QueryUnreachableError(
                    "同区域流场不可达",
                    source=list(source),
                    target=list(target),
                )
            cells = self._walk_within_region(flow, source, target)
            expanded_holder["count"] += flow.internal_expanded
            return self._route(cells, cost, expanded_holder["count"], movement_budget, cache_source)

        node_dist = {source: 0}
        node_parent: Dict[Cell, Tuple[Cell, Portal]] = {}
        heap: List[Tuple[int, int, Cell]] = [(0, 0, source)]
        counter = 0
        settled: set[Cell] = set()

        while heap:
            cost, _tie, node = heapq.heappop(heap)
            if node in settled:
                continue
            settled.add(node)
            expanded_holder["count"] += 1
            if node == target:
                break
            region_id = grid.region_of_cell(node)
            flow = self.flows[region_id]

            if region_id == target_region:
                inner = flow.distance(node, target)
                if inner < INF:
                    counter = self._relax(node_dist, node_parent, heap, node, target, inner, None, counter)
            else:
                for portal in flow.portals:
                    endpoint = portal.a if grid.region_of_cell(portal.a) == region_id else portal.b
                    other = portal.other(endpoint)
                    inner = flow.distance(node, endpoint)
                    if inner >= INF:
                        continue
                    cross = self._portal_cross_cost(portal, other)
                    counter = self._relax(
                        node_dist,
                        node_parent,
                        heap,
                        node,
                        other,
                        inner + cross,
                        portal,
                        counter,
                    )

        if target not in node_dist:
            raise QueryUnreachableError(
                "流场门户图不可达",
                source=list(source),
                target=list(target),
                expanded=expanded_holder["count"],
            )

        cells, total_cost = self._rebuild_path(source, target, node_parent, node_dist[target])
        return self._route(cells, total_cost, expanded_holder["count"], movement_budget, cache_source)

    def _relax(
        self,
        node_dist: Dict[Cell, int],
        node_parent: Dict[Cell, Tuple[Cell, Portal]],
        heap: List[Tuple[int, int, Cell]],
        node: Cell,
        nxt: Cell,
        added: int,
        portal: Optional[Portal],
        counter: int,
    ) -> int:
        tentative = node_dist[node] + added
        if tentative < node_dist.get(nxt, INF):
            node_dist[nxt] = tentative
            if portal is None:
                node_parent[nxt] = (node, Portal("__direct__", node, nxt))
            else:
                node_parent[nxt] = (node, portal)
            counter += 1
            heapq.heappush(heap, (tentative, counter, nxt))
        return counter

    def _rebuild_path(
        self,
        source: Cell,
        target: Cell,
        node_parent: Dict[Cell, Tuple[Cell, Portal]],
        total_cost: int,
    ) -> Tuple[List[Cell], int]:
        chain: List[Tuple[Cell, Cell, Portal]] = []
        cursor = target
        while cursor != source:
            prev, portal = node_parent[cursor]
            chain.append((prev, cursor, portal))
            cursor = prev
        chain.reverse()

        path: List[Cell] = [source]
        for prev, cursor, portal in chain:
            region_id = self.grid.region_of_cell(prev)
            flow = self.flows[region_id]
            if portal.portal_id == "__direct__":
                segment = self._walk_within_region(flow, prev, cursor)
            else:
                endpoint = portal.a if self.grid.region_of_cell(portal.a) == region_id else portal.b
                segment = self._walk_within_region(flow, prev, endpoint)
                if segment[-1] != endpoint:
                    raise QueryUnreachableError("流场方向场拼接失败", endpoint=list(endpoint))
                segment = segment + [portal.other(endpoint)]
            for cell in segment[1:]:
                if cell != path[-1]:
                    path.append(cell)
        if path[-1] != target:
            raise QueryUnreachableError("流场重建未抵达目标", target=list(target))
        return path, total_cost

    def _walk_within_region(self, flow: RegionFlowField, source: Cell, target: Cell) -> List[Cell]:
        if source == target:
            return [source]
        path = [source]
        seen = {source}
        cursor = source
        while cursor != target:
            nxt = flow.first_step(cursor, target)
            if nxt is None or nxt in seen:
                raise QueryUnreachableError(
                    "区域方向场出现断链",
                    source=list(source),
                    target=list(target),
                    cursor=list(cursor),
                )
            path.append(nxt)
            seen.add(nxt)
            cursor = nxt
        return path

    def _route(
        self,
        cells: List[Cell],
        cost: int,
        expanded: int,
        movement_budget: Optional[int],
        cache_source: str,
    ) -> Route:
        self.route_expanded = expanded
        within = movement_budget is None or cost <= movement_budget
        return Route(
            cells=tuple(cells),
            cost=int(cost),
            expanded=expanded,
            solver="flowfield:region",
            cache_source=cache_source,
            movement_budget=movement_budget,
            within_budget=within,
        )


def build_all_flows(
    grid: TacticalGrid, jump_table: JumpPointTable, region_ids: Optional[Sequence[int]] = None
) -> Dict[int, RegionFlowField]:
    builder = FlowFieldBuilder(grid)
    ids = list(region_ids) if region_ids is not None else grid.region_ids()
    return {region_id: builder.build_region_flow(region_id, jump_table) for region_id in sorted(ids)}
