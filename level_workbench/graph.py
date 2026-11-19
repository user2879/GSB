"""Generator-side deterministic graph traversal helpers."""

from __future__ import annotations

from collections import deque
from typing import Callable, Iterable

from .canonical import Coord, row_major_key

# Right, down, left, up: deterministic and compatible with row-major evidence.
NEIGHBOR_STEPS: tuple[Coord, ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))


def neighbors(coord: Coord, width: int, height: int) -> Iterable[Coord]:
    x, y = coord
    for dx, dy in NEIGHBOR_STEPS:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            yield (nx, ny)


def bfs_path(
    width: int,
    height: int,
    start: Coord,
    goal: Coord,
    passable: Callable[[Coord], bool],
) -> list[Coord] | None:
    if not passable(start) or not passable(goal):
        return None
    queue: deque[Coord] = deque([start])
    parent: dict[Coord, Coord | None] = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            path: list[Coord] = []
            cursor: Coord | None = goal
            while cursor is not None:
                path.append(cursor)
                cursor = parent[cursor]
            path.reverse()
            return path
        for candidate in neighbors(current, width, height):
            if candidate in parent or not passable(candidate):
                continue
            parent[candidate] = current
            queue.append(candidate)
    return None


def reachable_set(
    width: int,
    height: int,
    starts: Iterable[Coord],
    passable: Callable[[Coord], bool],
) -> set[Coord]:
    queue: deque[Coord] = deque()
    visited: set[Coord] = set()
    for start in sorted(set(starts), key=row_major_key):
        if passable(start) and start not in visited:
            visited.add(start)
            queue.append(start)
    while queue:
        current = queue.popleft()
        for candidate in neighbors(current, width, height):
            if candidate in visited or not passable(candidate):
                continue
            visited.add(candidate)
            queue.append(candidate)
    return visited
