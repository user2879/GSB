"""Grid map model and grid-level geometric helpers.

Cell states:
    0 = empty, 1 = blocked (unavailable), 2 = walkable (room or corridor).

Only the empty/walkable distinction matters for connectivity; blocked cells
can never be modified by the solver.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

EMPTY = 0
BLOCKED = 1
WALKABLE = 2

_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


class Grid:
    """Mutable rectangular grid with blocked, empty and walkable cells."""

    def __init__(self, width: int, height: int, blocked: Iterable[tuple[int, int]] = ()) -> None:
        self.width = int(width)
        self.height = int(height)
        self.cells = np.zeros((self.height, self.width), dtype=np.uint8)
        for x, y in blocked:
            self.cells[int(y), int(x)] = BLOCKED

    # ------------------------------------------------------------------ basic
    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def get(self, x: int, y: int) -> int:
        return int(self.cells[y, x])

    def is_available(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and self.get(x, y) == EMPTY

    def clone(self) -> "Grid":
        other = Grid(self.width, self.height)
        other.cells = self.cells.copy()
        return other

    def copy_from(self, other: "Grid") -> None:
        if (other.width, other.height) != (self.width, self.height):
            raise ValueError("grid dimension mismatch")
        self.cells = other.cells.copy()

    def to_list(self) -> list[list[int]]:
        return self.cells.tolist()

    @classmethod
    def from_list(cls, rows: list[list[int]]) -> "Grid":
        height = len(rows)
        width = len(rows[0]) if height else 0
        grid = cls(width, height)
        grid.cells = np.asarray(rows, dtype=np.uint8).copy()
        return grid

    # --------------------------------------------------------------- geometry
    def walkable_count(self) -> int:
        return int(np.count_nonzero(self.cells == WALKABLE))

    def available_count(self) -> int:
        return int(np.count_nonzero(self.cells == EMPTY))

    def rect_fits(self, x: int, y: int, room_w: int, room_h: int) -> bool:
        """All cells of an axis-aligned rectangle must be empty and in bounds."""
        if room_w <= 0 or room_h <= 0:
            return False
        if not (self.in_bounds(x, y) and self.in_bounds(x + room_w - 1, y + room_h - 1)):
            return False
        view = self.cells[y : y + room_h, x : x + room_w]
        return bool(np.all(view == EMPTY))

    def carve_rect(self, x: int, y: int, room_w: int, room_h: int) -> None:
        self.cells[y : y + room_h, x : x + room_w] = WALKABLE

    def carve_path(self, points: Iterable[tuple[int, int]]) -> None:
        for x, y in points:
            if self.get(x, y) == EMPTY:
                self.cells[y, x] = WALKABLE

    # ----------------------------------------------------------- connectivity
    def components(self) -> tuple[list[set[tuple[int, int]]], np.ndarray]:
        """Return 4-connected walkable components and a label matrix.

        Label 0 marks non-walkable cells; component labels start at 1.
        """
        labels = np.zeros((self.height, self.width), dtype=np.int32)
        components: list[set[tuple[int, int]]] = []
        next_label = 1
        for start_y in range(self.height):
            for start_x in range(self.width):
                if self.cells[start_y, start_x] != WALKABLE or labels[start_y, start_x] != 0:
                    continue
                members: set[tuple[int, int]] = set()
                queue = deque([(start_x, start_y)])
                labels[start_y, start_x] = next_label
                while queue:
                    x, y = queue.popleft()
                    members.add((x, y))
                    for dx, dy in _NEIGHBOURS:
                        nx, ny = x + dx, y + dy
                        if (
                            self.in_bounds(nx, ny)
                            and labels[ny, nx] == 0
                            and self.cells[ny, nx] == WALKABLE
                        ):
                            labels[ny, nx] = next_label
                            queue.append((nx, ny))
                components.append(members)
                next_label += 1
        return components, labels

    def component_of(self, x: int, y: int) -> set[tuple[int, int]] | None:
        if not self.in_bounds(x, y) or self.get(x, y) != WALKABLE:
            return None
        visited = {(x, y)}
        queue = deque([(x, y)])
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in _NEIGHBOURS:
                nx, ny = cx + dx, cy + dy
                if (
                    self.in_bounds(nx, ny)
                    and (nx, ny) not in visited
                    and self.get(nx, ny) == WALKABLE
                ):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return visited

    def connected_pair(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        component = self.component_of(*a)
        return component is not None and b in component


def chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def l_shaped_path(
    start: tuple[int, int],
    end: tuple[int, int],
    bend_first: str,
) -> list[tuple[int, int]]:
    """Cells of an axis-aligned L path between two points, both ends included.

    ``bend_first`` is ``"h"`` (horizontal leg first) or ``"v"``.
    """
    sx, sy = start
    ex, ey = end
    points: list[tuple[int, int]] = []
    if bend_first == "h":
        step_x = 1 if ex >= sx else -1
        for x in range(sx, ex + step_x, step_x):
            points.append((x, sy))
        step_y = 1 if ey >= sy else -1
        for y in range(sy + step_y, ey + step_y, step_y):
            points.append((ex, y))
    else:
        step_y = 1 if ey >= sy else -1
        for y in range(sy, ey + step_y, step_y):
            points.append((sx, y))
        step_x = 1 if ex >= sx else -1
        for x in range(sx + step_x, ex + step_x, step_x):
            points.append((x, ey))
    return points
