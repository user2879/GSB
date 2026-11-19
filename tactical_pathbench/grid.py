"""分层战术网格地图。

三层栅格（均为 NumPy 数组）：
- ``terrain_cost``：uint8 地形代价（合法区间 1..10）；
- ``passable``：bool 通行标记（False 即障碍）；
- ``cover``：bool 覆盖标记（影响内容哈希，供诊断使用）。

区域划分采用固定矩形分块（纯函数，仅取决于形状与分块参数），
因此同种子两次生成的区域划分逐字节一致，且可独立校验。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .encoding import canonical_encoding, content_hash
from .errors import MapShapeError, TerrainRangeError

Cell = Tuple[int, int]
MIN_COST = 1
MAX_COST = 10
NEIGHBOR4: Tuple[Cell, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass(frozen=True)
class RegionPartition:
    region_rows: int
    region_cols: int

    def region_of(self, row: int, col: int, height: int, width: int) -> int:
        band = max(1, int(np.ceil(height / self.region_rows)))
        stripe = max(1, int(np.ceil(width / self.region_cols)))
        rb = min(row // band, self.region_rows - 1)
        cb = min(col // stripe, self.region_cols - 1)
        return rb * self.region_cols + cb

    def region_count(self) -> int:
        return self.region_rows * self.region_cols

    def to_payload(self) -> Dict[str, int]:
        return {"region_rows": self.region_rows, "region_cols": self.region_cols}


class TacticalGrid:
    def __init__(
        self,
        terrain_cost: np.ndarray,
        passable: np.ndarray,
        cover: np.ndarray,
        partition: RegionPartition,
    ) -> None:
        cost = np.asarray(terrain_cost)
        pas = np.asarray(passable, dtype=bool)
        cov = np.asarray(cover, dtype=bool)
        if cost.ndim != 2:
            raise MapShapeError("地形代价层必须是二维数组", shape=list(cost.shape))
        if pas.shape != cost.shape or cov.shape != cost.shape:
            raise MapShapeError(
                "三层栅格形状不一致",
                cost_shape=list(cost.shape),
                passable_shape=list(pas.shape),
                cover_shape=list(cov.shape),
            )
        if cost.dtype != np.uint8:
            cost = cost.astype(np.uint8)
        bad = int(np.count_nonzero((cost < MIN_COST) | (cost > MAX_COST)))
        if bad:
            raise TerrainRangeError(
                "存在超出合法区间的地形代价",
                allowed=[MIN_COST, MAX_COST],
                bad_cell_count=bad,
            )
        self.terrain_cost = cost
        self.passable = pas
        self.cover = cov
        self.partition = partition
        self.height, self.width = cost.shape
        self._region_cells: Dict[int, List[Cell]] = {}
        self._cell_region: Dict[Cell, int] = {}
        self._build_region_index()

    # ---------- 基础访问 ----------
    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def require_cell(self, row: int, col: int) -> Cell:
        if not self.in_bounds(row, col):
            raise MapShapeError(
                "坐标越界",
                cell=[row, col],
                shape=[self.height, self.width],
            )
        return (row, col)

    def region_of_cell(self, cell: Cell) -> int:
        self.require_cell(*cell)
        return self._cell_region[cell]

    def cells_of_region(self, region_id: int) -> List[Cell]:
        if region_id not in self._region_cells:
            raise MapShapeError("未知区域", region_id=region_id, region_count=self.partition.region_count())
        return list(self._region_cells[region_id])

    def region_ids(self) -> List[int]:
        return sorted(self._region_cells.keys())

    def cost_of(self, cell: Cell) -> int:
        self.require_cell(*cell)
        return int(self.terrain_cost[cell[0], cell[1]])

    def is_passable(self, cell: Cell) -> bool:
        self.require_cell(*cell)
        return bool(self.passable[cell[0], cell[1]])

    def has_cover(self, cell: Cell) -> bool:
        self.require_cell(*cell)
        return bool(self.cover[cell[0], cell[1]])

    def neighbors(self, cell: Cell) -> List[Cell]:
        row, col = cell
        result: List[Cell] = []
        for dr, dc in NEIGHBOR4:
            nr, nc = row + dr, col + dc
            if self.in_bounds(nr, nc) and self.passable[nr, nc]:
                result.append((nr, nc))
        return result

    def edge_cost(self, src: Cell, dst: Cell) -> int:
        """进入 dst 的代价（四邻域移动代价 = 目标格地形代价）。"""
        if src == dst:
            return 0
        if dst not in self.neighbors(src):
            raise MapShapeError("两格不构成四邻域可通行边", src=list(src), dst=list(dst))
        return self.cost_of(dst)

    # ---------- 区域与边界 ----------
    def _build_region_index(self) -> None:
        for row in range(self.height):
            for col in range(self.width):
                region_id = self.partition.region_of(row, col, self.height, self.width)
                cell = (row, col)
                self._region_cells.setdefault(region_id, []).append(cell)
                self._cell_region[cell] = region_id
        for cells in self._region_cells.values():
            cells.sort()

    def is_boundary_cell(self, cell: Cell) -> bool:
        region_id = self.region_of_cell(cell)
        row, col = cell
        for dr, dc in NEIGHBOR4:
            nr, nc = row + dr, col + dc
            if self.in_bounds(nr, nc) and self.region_of_cell((nr, nc)) != region_id:
                return True
        return False

    def boundary_portals(self) -> List[Tuple[Cell, Cell]]:
        """跨区域的可通行相邻格对，按字典序去重，顺序确定。"""
        portals: set[Tuple[Cell, Cell]] = set()
        for row in range(self.height):
            for col in range(self.width):
                cell = (row, col)
                if not self.is_passable(cell):
                    continue
                for dr, dc in ((1, 0), (0, 1)):
                    other = (row + dr, col + dc)
                    if not self.in_bounds(*other) or not self.is_passable(other):
                        continue
                    if self.region_of_cell(cell) != self.region_of_cell(other):
                        pair = tuple(sorted((cell, other)))  # type: ignore[arg-type]
                        portals.add(pair)  # type: ignore[arg-type]
        return sorted(portals)

    def portals_touching_regions(self, region_ids: Iterable[int]) -> List[Tuple[Cell, Cell]]:
        wanted = set(region_ids)
        return [
            pair
            for pair in self.boundary_portals()
            if self.region_of_cell(pair[0]) in wanted or self.region_of_cell(pair[1]) in wanted
        ]

    def boundary_portals_of_cell(self, cell: Cell) -> List[Tuple[Cell, Cell]]:
        row, col = cell
        result: List[Tuple[Cell, Cell]] = []
        for pair in self.boundary_portals():
            if cell in pair:
                result.append(pair)
        return result

    # ---------- 变更 ----------
    def set_terrain_cost(self, cell: Cell, cost: int) -> None:
        self.require_cell(*cell)
        if not (MIN_COST <= int(cost) <= MAX_COST):
            raise TerrainRangeError(
                "非法地形代价",
                cell=list(cell),
                cost=int(cost),
                allowed=[MIN_COST, MAX_COST],
            )
        self.terrain_cost[cell[0], cell[1]] = np.uint8(cost)

    def set_passable(self, cell: Cell, value: bool) -> None:
        self.require_cell(*cell)
        self.passable[cell[0], cell[1]] = bool(value)

    def set_cover(self, cell: Cell, value: bool) -> None:
        self.require_cell(*cell)
        self.cover[cell[0], cell[1]] = bool(value)

    # ---------- 序列化 / 哈希 ----------
    def cell_records(self) -> List[Tuple[int, int, int, int, int, int]]:
        records: List[Tuple[int, int, int, int, int, int]] = []
        for row in range(self.height):
            for col in range(self.width):
                records.append(
                    (
                        row,
                        col,
                        int(self.terrain_cost[row, col]),
                        int(self.passable[row, col]),
                        int(self.cover[row, col]),
                        self.region_of_cell((row, col)),
                    )
                )
        return records

    def content_payload(self) -> Dict[str, object]:
        return {
            "height": self.height,
            "width": self.width,
            "partition": self.partition.to_payload(),
            "cells": self.cell_records(),
        }

    def content_hash(self) -> str:
        return content_hash(self.content_payload())

    def region_checksum_report(self) -> Dict[str, str]:
        report: Dict[str, str] = {}
        for region_id in self.region_ids():
            payload = [
                (cell, self.cost_of(cell), int(self.is_passable(cell)), int(self.has_cover(cell)))
                for cell in self.cells_of_region(region_id)
            ]
            report[str(region_id)] = content_hash(payload)
        return report

    def region_partition_signature(self) -> str:
        mapping = [[self.region_of_cell((r, c)) for c in range(self.width)] for r in range(self.height)]
        return content_hash(mapping)

    def to_payload(self) -> Dict[str, object]:
        return {
            "content_hash": self.content_hash(),
            "partition_signature": self.region_partition_signature(),
            "region_checksums": self.region_checksum_report(),
            "grid": self.content_payload(),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_encoding(self.content_payload())

    @classmethod
    def from_payload(cls, payload: Dict[str, object]) -> "TacticalGrid":
        grid_payload = payload["grid"] if "grid" in payload else payload  # type: ignore[assignment]
        assert isinstance(grid_payload, dict)
        height = int(grid_payload["height"])  # type: ignore[arg-type]
        width = int(grid_payload["width"])  # type: ignore[arg-type]
        part = grid_payload["partition"]  # type: ignore[index]
        assert isinstance(part, dict)
        partition = RegionPartition(int(part["region_rows"]), int(part["region_cols"]))
        cost = np.ones((height, width), dtype=np.uint8)
        pas = np.ones((height, width), dtype=bool)
        cover = np.zeros((height, width), dtype=bool)
        for row, col, c, p, cv, _region in grid_payload["cells"]:  # type: ignore[misc]
            cost[row, col] = np.uint8(c)
            pas[row, col] = bool(p)
            cover[row, col] = bool(cv)
        return cls(cost, pas, cover, partition)


def cells_touching_portals(portals: Sequence[Tuple[Cell, Cell]]) -> List[Cell]:
    return sorted({cell for pair in portals for cell in pair})
