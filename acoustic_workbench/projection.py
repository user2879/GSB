"""Fixed equirectangular tile boundaries, energy assembly, and deterministic tone mapping."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .spherical import equirectangular_directions


@dataclass(frozen=True)
class TileSpec:
    tile_id: str
    row: int
    col: int
    x0: int
    x1: int
    y0: int
    y1: int
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "tile_id": self.tile_id,
            "row": self.row,
            "col": self.col,
            "pixel_bounds": [self.x0, self.x1, self.y0, self.y1],
            "lon_lat_bounds": [self.lon_min, self.lon_max, self.lat_min, self.lat_max],
        }


def tile_layout(scene: dict[str, Any]) -> list[TileSpec]:
    grid = scene["grid"]
    width = int(grid["width"])
    height = int(grid["height"])
    rows = int(grid["tile_rows"])
    cols = int(grid["tile_cols"])
    lon_min = float(grid["lon_min"])
    lon_max = float(grid["lon_max"])
    lat_min = float(grid["lat_min"])
    lat_max = float(grid["lat_max"])
    tiles: list[TileSpec] = []
    for row in range(rows):
        y0 = row * height // rows
        y1 = (row + 1) * height // rows
        row_lat_max = lat_max - (lat_max - lat_min) * y0 / height
        row_lat_min = lat_max - (lat_max - lat_min) * y1 / height
        for col in range(cols):
            x0 = col * width // cols
            x1 = (col + 1) * width // cols
            col_lon_min = lon_min + (lon_max - lon_min) * x0 / width
            col_lon_max = lon_min + (lon_max - lon_min) * x1 / width
            tiles.append(
                TileSpec(
                    tile_id=f"tile:r{row}c{col}",
                    row=row,
                    col=col,
                    x0=x0,
                    x1=x1,
                    y0=y0,
                    y1=y1,
                    lon_min=col_lon_min,
                    lon_max=col_lon_max,
                    lat_min=row_lat_min,
                    lat_max=row_lat_max,
                )
            )
    return tiles


def grid_geometry(scene: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = scene["grid"]
    bounds = (
        float(grid["lon_min"]),
        float(grid["lon_max"]),
        float(grid["lat_min"]),
        float(grid["lat_max"]),
    )
    return equirectangular_directions(int(grid["width"]), int(grid["height"]), bounds)


def channel_support_in_tile(channel_id: int, directions: np.ndarray, tile: TileSpec, width: int) -> bool:
    from .spherical import support_mask

    block = directions.reshape((int(directions.shape[0] // width), width, 3))[tile.y0 : tile.y1, tile.x0 : tile.x1]
    return bool(np.any(support_mask(channel_id, block.reshape(-1, 3))))


def compose_energy_block(
    scene: dict[str, Any],
    tile: TileSpec,
    channel_fields: Iterable[tuple[int, np.ndarray, np.ndarray]],
    directions: np.ndarray | None = None,
) -> np.ndarray:
    width = int(scene["grid"]["width"])
    height = int(scene["grid"]["height"])
    rows = tile.y1 - tile.y0
    cols = tile.x1 - tile.x0
    energy = np.zeros((rows, cols), dtype=np.float64)
    if directions is None:
        directions, _lon, _lat = grid_geometry(scene)
    shape = directions.reshape((height, width, 3))
    block_directions = shape[tile.y0 : tile.y1, tile.x0 : tile.x1].reshape(-1, 3)
    for channel_id, coefficients, order in channel_fields:
        from .spherical import evaluate_channel

        field = evaluate_channel(coefficients, int(order), channel_id, block_directions)
        energy += field.reshape(rows, cols)
    return energy


_COLORMAP = np.array(
    [
        [4, 8, 28],
        [12, 38, 82],
        [8, 88, 128],
        [18, 142, 126],
        [103, 184, 74],
        [224, 190, 48],
        [248, 112, 38],
        [220, 32, 32],
    ],
    dtype=np.float64,
)


def tone_map(energy: np.ndarray) -> np.ndarray:
    db = 10.0 * np.log10(np.maximum(np.asarray(energy, dtype=np.float64), 1e-12))
    normalized = np.clip((db + 100.0) / 90.0, 0.0, 1.0)
    position = normalized * (len(_COLORMAP) - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, len(_COLORMAP) - 1)
    fraction = (position - lower)[..., np.newaxis]
    rgb = _COLORMAP[lower] * (1.0 - fraction) + _COLORMAP[upper] * fraction
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def energy_to_tile_bytes(energy: np.ndarray) -> bytes:
    return tone_map(energy).tobytes(order="C")


def assemble_preview_png(scene: dict[str, Any], tile_rgb: dict[str, np.ndarray]) -> bytes:
    grid = scene["grid"]
    width = int(grid["width"])
    height = int(grid["height"])
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for tile in tile_layout(scene):
        rgb = tile_rgb[tile.tile_id]
        image[tile.y0 : tile.y1, tile.x0 : tile.x1, :] = rgb
    output = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(output, format="PNG", compress_level=9, optimize=False)
    return output.getvalue()


def rgb_from_tile_bytes(data: bytes, tile: TileSpec) -> np.ndarray:
    expected = (tile.y1 - tile.y0) * (tile.x1 - tile.x0) * 3
    if len(data) != expected:
        raise ValueError(f"tile byte count {len(data)} != {expected}")
    return np.frombuffer(data, dtype=np.uint8).reshape(tile.y1 - tile.y0, tile.x1 - tile.x0, 3).copy()


def tile_support_channels(
    scene: dict[str, Any],
    channel_count: int = 4,
    directions: np.ndarray | None = None,
) -> dict[str, list[int]]:
    width = int(scene["grid"]["width"])
    if directions is None:
        directions, _lon, _lat = grid_geometry(scene)
    support: dict[str, list[int]] = {}
    for tile in tile_layout(scene):
        support[tile.tile_id] = [
            channel_id
            for channel_id in range(channel_count)
            if channel_support_in_tile(channel_id, directions, tile, width)
        ]
    return support
