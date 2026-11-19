"""四邻域 A* 求解器。

确定性保证：
- 启发函数可选 ``manhattan`` / ``zero``（Dijkstra）；
- 堆条目带单调序列号与坐标次序键，相同 f 值的弹出顺序与字典遍历无关；
- 距离只使用整数地形代价，结果完全确定。
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .errors import QueryUnreachableError
from .grid import Cell, TacticalGrid

HeuristicName = str


@dataclass(frozen=True)
class Route:
    cells: Tuple[Cell, ...]
    cost: int
    expanded: int
    solver: str
    cache_source: str
    movement_budget: Optional[int] = None
    within_budget: bool = True

    def path_hash(self) -> str:
        from .encoding import content_hash

        return content_hash({"cells": [list(cell) for cell in self.cells], "cost": self.cost})

    def to_payload(self) -> Dict[str, object]:
        return {
            "cells": [list(cell) for cell in self.cells],
            "cost": self.cost,
            "expanded": self.expanded,
            "solver": self.solver,
            "cache_source": self.cache_source,
            "movement_budget": self.movement_budget,
            "within_budget": self.within_budget,
            "path_hash": self.path_hash(),
        }


def manhattan_heuristic(grid: TacticalGrid) -> Callable[[Cell, Cell], int]:
    def heuristic(cell: Cell, target: Cell) -> int:
        return abs(cell[0] - target[0]) + abs(cell[1] - target[1])

    return heuristic


def zero_heuristic(grid: TacticalGrid) -> Callable[[Cell, Cell], int]:
    def heuristic(cell: Cell, target: Cell) -> int:
        return 0

    return heuristic


HEURISTICS: Dict[str, Callable[[TacticalGrid], Callable[[Cell, Cell], int]]] = {
    "manhattan": manhattan_heuristic,
    "zero": zero_heuristic,
}


@dataclass
class AStarSolver:
    grid: TacticalGrid
    heuristic_name: str = "manhattan"
    expanded_counter: int = 0
    _last_expanded: int = field(default=0, init=False)

    def _heuristic(self) -> Callable[[Cell, Cell], int]:
        if self.heuristic_name not in HEURISTICS:
            from .errors import MapShapeError

            raise MapShapeError(
                "未知启发函数",
                heuristic=self.heuristic_name,
                available=sorted(HEURISTICS),
            )
        return HEURISTICS[self.heuristic_name](self.grid)

    def solve(
        self,
        source: Cell,
        target: Cell,
        movement_budget: Optional[int] = None,
        cache_source: str = "none",
    ) -> Route:
        grid = self.grid
        grid.require_cell(*source)
        grid.require_cell(*target)
        if not grid.is_passable(source):
            from .errors import UnitInvalidError

            raise UnitInvalidError("起点不可通行", cell=list(source))
        if not grid.is_passable(target):
            raise QueryUnreachableError(
                "目标格不可通行",
                source=list(source),
                target=list(target),
            )

        heuristic = self._heuristic()
        dist: Dict[Cell, int] = {source: 0}
        parent: Dict[Cell, Cell] = {}
        counter = 0
        open_heap: List[Tuple[int, int, Cell]] = []
        heapq.heappush(open_heap, (heuristic(source, target), counter, source))
        closed: set[Cell] = set()

        while open_heap:
            _f, _tie, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)
            self.expanded_counter += 1
            if current == target:
                cells = _reconstruct(parent, source, target)
                expanded = len(closed)
                self._last_expanded = expanded
                within = movement_budget is None or dist[target] <= movement_budget
                return Route(
                    cells=tuple(cells),
                    cost=dist[target],
                    expanded=expanded,
                    solver=f"astar:{self.heuristic_name}",
                    cache_source=cache_source,
                    movement_budget=movement_budget,
                    within_budget=within,
                )
            for neighbor in grid.neighbors(current):
                if neighbor in closed:
                    continue
                tentative = dist[current] + grid.edge_cost(current, neighbor)
                if tentative < dist.get(neighbor, 10**18):
                    dist[neighbor] = tentative
                    parent[neighbor] = current
                    counter += 1
                    heapq.heappush(
                        open_heap,
                        (tentative + heuristic(neighbor, target), counter, neighbor),
                    )

        raise QueryUnreachableError(
            "A* 未找到可达路径",
            source=list(source),
            target=list(target),
            expanded=len(closed),
        )


def _reconstruct(parent: Dict[Cell, Cell], source: Cell, target: Cell) -> List[Cell]:
    path = [target]
    while path[-1] != source:
        path.append(parent[path[-1]])
    path.reverse()
    return path
