"""Global window lattice with center-region ownership and cross-tile context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import BusinessEvent, E_WINDOW_CONTEXT_UNAVAILABLE
from .manifest import Manifest
from .tiles import Mosaic


@dataclass(frozen=True)
class WindowSpec:
    window_id: str
    core_row: int
    core_col: int
    core_size: int
    core_rect: tuple[int, int, int, int]
    context_rect: tuple[int, int, int, int]
    owner_tile_id: str
    source_tile_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "core_row": self.core_row,
            "core_col": self.core_col,
            "core_rect": list(self.core_rect),
            "context_rect": list(self.context_rect),
            "owner_tile_id": self.owner_tile_id,
            "source_tile_ids": list(self.source_tile_ids),
        }

    def order_key(self) -> tuple[int, int, int, int]:
        return (
            self.core_rect[0],
            self.core_rect[1],
            self.core_rect[2],
            self.core_rect[3],
        )


def _intersecting_tiles(
    rect: tuple[int, int, int, int],
    manifest: Manifest,
    tile_size: int,
) -> list[str]:
    top, left, bottom, right = rect
    found = []
    for tile in manifest.tiles:
        tile_top = tile.row * tile_size
        tile_left = tile.col * tile_size
        tile_bottom = tile_top + tile_size
        tile_right = tile_left + tile_size
        if top < tile_bottom and bottom > tile_top and left < tile_right and right > tile_left:
            found.append(tile.tile_id)
    return sorted(found)


def generate_windows(
    manifest: Manifest,
    mosaic: Mosaic,
    valid_tile_ids: set[str],
    config: dict[str, Any],
) -> tuple[list[WindowSpec], list[BusinessEvent]]:
    window_size = int(config["window"]["size"])
    step = int(config["window"]["step"])
    overlap = int(config["window"]["context_overlap"])
    core_size = window_size - 2 * overlap
    tile_size = int(config["tile"]["size"])
    by_position = manifest.by_position()
    windows: list[WindowSpec] = []
    events: list[BusinessEvent] = []

    for core_row in range(0, mosaic.height - core_size + 1, step):
        for core_col in range(0, mosaic.width - core_size + 1, step):
            core_rect = (core_row, core_col, core_row + core_size, core_col + core_size)
            context_rect = (
                core_row - overlap,
                core_col - overlap,
                core_row + core_size + overlap,
                core_col + core_size + overlap,
            )
            center_row = core_row + core_size // 2
            center_col = core_col + core_size // 2
            owner_row = center_row // tile_size
            owner_col = center_col // tile_size
            owner = by_position.get((owner_row, owner_col))
            window_id = f"w_r{core_row:04d}_c{core_col:04d}"
            if owner is None or owner.tile_id not in valid_tile_ids:
                events.append(
                    BusinessEvent(
                        tile_id=owner.tile_id if owner is not None else None,
                        stage="window_generation",
                        code=E_WINDOW_CONTEXT_UNAVAILABLE,
                        message="window owner tile is unavailable",
                        detail={"window_id": window_id, "owner_position": [owner_row, owner_col]},
                    )
                )
                continue
            source_tile_ids = _intersecting_tiles(context_rect, manifest, tile_size)
            unavailable = sorted(set(source_tile_ids) - valid_tile_ids)
            if unavailable:
                events.append(
                    BusinessEvent(
                        tile_id=owner.tile_id,
                        stage="window_generation",
                        code=E_WINDOW_CONTEXT_UNAVAILABLE,
                        message="window context intersects an unavailable tile",
                        detail={"window_id": window_id, "unavailable_tiles": unavailable},
                    )
                )
                continue
            windows.append(
                WindowSpec(
                    window_id=window_id,
                    core_row=core_row,
                    core_col=core_col,
                    core_size=core_size,
                    core_rect=core_rect,
                    context_rect=context_rect,
                    owner_tile_id=owner.tile_id,
                    source_tile_ids=tuple(source_tile_ids),
                )
            )
    windows.sort(key=lambda window: window.order_key())
    return windows, events


def windows_by_tile(windows: list[WindowSpec]) -> dict[str, list[WindowSpec]]:
    grouped: dict[str, list[WindowSpec]] = {}
    for window in windows:
        grouped.setdefault(window.owner_tile_id, []).append(window)
    for tile_windows in grouped.values():
        tile_windows.sort(key=lambda window: window.order_key())
    return grouped

