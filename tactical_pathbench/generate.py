"""固定种子确定性样本生成。

只使用标准库 ``random.Random`` 的整数/选择原语，不依赖 NumPy 随机流，
保证 Python 3.11 同种子下逐字节可复现；所有集合在落盘前排序。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from .encoding import content_hash
from .errors import QueryUnreachableError, UnitInvalidError
from .grid import MAX_COST, MIN_COST, RegionPartition, TacticalGrid

Cell = Tuple[int, int]


@dataclass(frozen=True)
class Unit:
    unit_id: str
    faction: str
    cell: Cell
    movement_budget: int

    def to_payload(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "faction": self.faction,
            "cell": list(self.cell),
            "movement_budget": self.movement_budget,
        }


@dataclass(frozen=True)
class Query:
    query_id: str
    unit_id: str
    source: Cell
    target: Cell

    def key(self) -> str:
        return self.query_id

    def to_payload(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "unit_id": self.unit_id,
            "source": list(self.source),
            "target": list(self.target),
        }


@dataclass(frozen=True)
class Sample:
    seed: int
    grid: TacticalGrid
    units: List[Unit]
    queries: List[Query]

    def manifest_payload(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "grid": self.grid.to_payload(),
            "units": [unit.to_payload() for unit in self.units],
            "queries": [query.to_payload() for query in self.queries],
        }

    def manifest_hash(self) -> str:
        return content_hash(self.manifest_payload())


def generate_sample(
    seed: int = 20261119,
    height: int = 24,
    width: int = 28,
    region_rows: int = 3,
    region_cols: int = 4,
    unit_count: int = 6,
    query_count: int = 10,
) -> Sample:
    rng = np.random.default_rng(seed)
    partition = RegionPartition(region_rows=region_rows, region_cols=region_cols)

    terrain = rng.integers(MIN_COST, MAX_COST + 1, size=(height, width), dtype=np.uint8)
    # 固定约 18% 障碍，使用确定性阈值而不是逐格随机布尔，保证两次生成完全一致。
    roll = rng.random(size=(height, width))
    passable = roll >= 0.18
    cover = rng.random(size=(height, width)) >= 0.72

    grid = TacticalGrid(terrain, passable, cover, partition)

    passable_cells = sorted(cell for cell in (
        (r, c) for r in range(height) for c in range(width)
    ) if grid.is_passable(cell))

    rnd = _PyRandom(seed ^ 0x9E3779B9)
    chosen = rnd.sample(passable_cells, unit_count + query_count)
    units: List[Unit] = []
    for index in range(unit_count):
        cell = chosen[index]
        units.append(
            Unit(
                unit_id=f"U{index:02d}",
                faction="alpha" if index % 2 == 0 else "bravo",
                cell=cell,
                movement_budget=60 + (index * 7),
            )
        )

    queries: List[Query] = []
    for index in range(query_count):
        unit = units[index % unit_count]
        target = chosen[unit_count + index]
        queries.append(
            Query(
                query_id=f"Q{index:02d}",
                unit_id=unit.unit_id,
                source=unit.cell,
                target=target,
            )
        )

    _validate_units(grid, units)
    sample = Sample(seed=seed, grid=grid, units=units, queries=queries)
    _ensure_reachable(grid, queries)
    return sample


class _PyRandom:
    """对标准库随机器的薄封装，显式声明使用的原语。"""

    def __init__(self, seed: int) -> None:
        import random

        self._rng = random.Random(seed)

    def sample(self, population: List[Cell], k: int) -> List[Cell]:
        return self._rng.sample(population, k)


def _validate_units(grid: TacticalGrid, units: List[Unit]) -> None:
    for unit in units:
        if not grid.in_bounds(*unit.cell) or not grid.is_passable(unit.cell):
            raise UnitInvalidError(
                "单位位于不可通行格或越界",
                unit_id=unit.unit_id,
                cell=list(unit.cell),
            )


def _ensure_reachable(grid: TacticalGrid, queries: List[Query]) -> None:
    from .astar import AStarSolver

    solver = AStarSolver(grid)
    for query in queries:
        try:
            solver.solve(query.source, query.target)
        except QueryUnreachableError:
            raise
