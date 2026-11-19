"""Tile index validation and integer-preserving multi-band tile loading.

Pixels stay as unsigned 8-bit integers up to the feature stage. The only
non-integer path is an explicit floating-point band patch, which is rejected
with ``E_NON_FINITE_BAND`` as soon as a non-finite value is observed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import BAND_ORDER
from .errors import BusinessError, ErrorCode


@dataclass(frozen=True)
class TileRecord:
    row: int
    col: int
    file: str
    bounds: tuple[int, int, int, int]
    neighbors: tuple[tuple[int, int], ...]

    @property
    def key(self) -> tuple[int, int]:
        return (self.row, self.col)

    @property
    def name(self) -> str:
        return f"tile_r{self.row:03d}_c{self.col:03d}"

    def to_dict(self) -> dict:
        return {
            "row": self.row,
            "col": self.col,
            "file": self.file,
            "bounds": list(self.bounds),
            "neighbors": [list(n) for n in self.neighbors],
        }


@dataclass(frozen=True)
class TileIndex:
    band_order_version: str
    band_order: tuple[str, ...]
    tile_size: int
    band_count: int
    tiles: tuple[TileRecord, ...]
    bounds: tuple[int, int, int, int]

    def tile_map(self) -> dict[tuple[int, int], TileRecord]:
        return {tile.key: tile for tile in self.tiles}

    def ordered_keys(self) -> list[tuple[int, int]]:
        return sorted((tile.row, tile.col) for tile in self.tiles)

    def to_dict(self) -> dict:
        xs = [t.bounds[0] for t in self.tiles] + [t.bounds[2] for t in self.tiles]
        ys = [t.bounds[1] for t in self.tiles] + [t.bounds[3] for t in self.tiles]
        return {
            "band_order_version": self.band_order_version,
            "band_order": list(self.band_order),
            "tile_size": self.tile_size,
            "bands": self.band_count,
            "global_bounds": [min(xs), min(ys), max(xs), max(ys)],
            "tiles": [t.to_dict() for t in self.tiles],
        }


def load_index(index_path: str | Path, *, injection_id: str | None = None) -> TileIndex:
    path = Path(index_path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise BusinessError(ErrorCode.E_INDEX_INVALID, f"cannot read index: {exc}",
                            injection_id=injection_id) from exc
    return _parse_index(raw, injection_id=injection_id, base_dir=path.parent)


def _parse_index(raw: dict, *, injection_id: str | None, base_dir: Path) -> TileIndex:
    try:
        band_order = tuple(raw["band_order"])
        band_order_version = str(raw["band_order_version"])
        tile_size = int(raw["tile_size"])
        band_count = int(raw["bands"])
        raw_tiles = raw["tiles"]
    except (KeyError, TypeError, ValueError) as exc:
        raise BusinessError(ErrorCode.E_INDEX_INVALID, f"index missing fields: {exc}",
                            injection_id=injection_id) from exc

    if band_order != tuple(BAND_ORDER) or band_count != len(BAND_ORDER):
        raise BusinessError(ErrorCode.E_INDEX_INVALID,
                            f"band order/version mismatch: {band_order}",
                            injection_id=injection_id)

    seen: set[tuple[int, int]] = set()
    tiles: list[TileRecord] = []
    for entry in raw_tiles:
        try:
            key = (int(entry["row"]), int(entry["col"]))
            neighbors = tuple((int(n[0]), int(n[1])) for n in entry["neighbors"])
            bounds = tuple(int(v) for v in entry["bounds"])
            file_name = str(entry["file"])
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise BusinessError(ErrorCode.E_INDEX_INVALID,
                                f"malformed tile entry {entry}: {exc}",
                                injection_id=injection_id) from exc
        if key in seen:
            raise BusinessError(ErrorCode.E_INDEX_INVALID,
                                f"duplicate tile row/col {key}", injection_id=injection_id)
        seen.add(key)
        if len(bounds) != 4 or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            raise BusinessError(ErrorCode.E_INDEX_INVALID,
                                f"invalid spatial bounds for {key}", injection_id=injection_id)
        tiles.append(TileRecord(key[0], key[1], file_name, bounds, neighbors))  # type: ignore[arg-type]

    tile_map = {tile.key: tile for tile in tiles}
    for tile in tiles:
        for neighbor in tile.neighbors:
            if neighbor not in tile_map:
                raise BusinessError(ErrorCode.E_INDEX_INVALID,
                                    f"tile {tile.key} references unknown neighbor {neighbor}",
                                    injection_id=injection_id)
            reverse = tile_map[neighbor]
            if tile.key not in reverse.neighbors:
                raise BusinessError(
                    ErrorCode.E_INDEX_INVALID,
                    f"asymmetric adjacency between {tile.key} and {neighbor}",
                    injection_id=injection_id,
                )

    if not tiles:
        raise BusinessError(ErrorCode.E_INDEX_INVALID, "index contains no tiles",
                            injection_id=injection_id)

    xs = [t.bounds[0] for t in tiles] + [t.bounds[2] for t in tiles]
    ys = [t.bounds[1] for t in tiles] + [t.bounds[3] for t in tiles]
    return TileIndex(
        band_order_version=band_order_version,
        band_order=band_order,
        tile_size=tile_size,
        band_count=band_count,
        tiles=tuple(sorted(tiles, key=lambda t: t.key)),
        bounds=(min(xs), min(ys), max(xs), max(ys)),
    )


def load_tile_bands(
    tile: TileRecord,
    sample_dir: str | Path,
    index: TileIndex,
    *,
    patches_dir: str | Path | None = None,
    injection_id: str | None = None,
) -> np.ndarray:
    """Return an ``uint8`` HxWxB array; raises a business error on bad input."""

    sample_dir = Path(sample_dir)
    png_path = sample_dir / tile.file
    try:
        with Image.open(png_path) as image:
            array = np.array(image)
    except (OSError, ValueError, Image.UnidentifiedImageError) as exc:  # pragma: no cover - exercised via injection
        raise BusinessError(ErrorCode.E_TILE_READ_FAILED,
                            f"cannot decode PNG for {tile.key}: {exc}",
                            injection_id=injection_id) from exc

    if array.ndim != 3 or array.shape[2] != index.band_count:
        raise BusinessError(ErrorCode.E_TILE_SHAPE_MISMATCH,
                            f"band count mismatch for {tile.key}: shape={array.shape}",
                            injection_id=injection_id)
    if array.shape[0] != index.tile_size or array.shape[1] != index.tile_size:
        raise BusinessError(ErrorCode.E_TILE_SHAPE_MISMATCH,
                            f"tile size mismatch for {tile.key}: shape={array.shape}",
                            injection_id=injection_id)
    if array.dtype != np.uint8:
        raise BusinessError(ErrorCode.E_TILE_SHAPE_MISMATCH,
                            f"integer precision violated for {tile.key}: dtype={array.dtype}",
                            injection_id=injection_id)

    if patches_dir is not None:
        array = _apply_patches(array, tile, Path(patches_dir), injection_id=injection_id)
    return array


def _apply_patches(array: np.ndarray, tile: TileRecord, patches_dir: Path,
                   *, injection_id: str | None) -> np.ndarray:
    full_patch = patches_dir / f"{tile.name}.npy"
    if full_patch.exists():
        patch = np.load(full_patch)
        _guard_finite(patch, tile, injection_id)
        if patch.shape != array.shape:
            raise BusinessError(ErrorCode.E_TILE_SHAPE_MISMATCH,
                                f"patch shape mismatch for {tile.key}: {patch.shape}",
                                injection_id=injection_id)
        if np.issubdtype(patch.dtype, np.integer):
            array = patch.astype(np.uint8, copy=True)
        else:
            raise BusinessError(ErrorCode.E_NON_FINITE_BAND,
                                f"float full-patch rejected for {tile.key}",
                                injection_id=injection_id)
    for band in range(array.shape[2]):
        band_patch = patches_dir / f"{tile.name}_b{band}.npy"
        if not band_patch.exists():
            continue
        patch = np.load(band_patch)
        _guard_finite(patch, tile, injection_id)
        if patch.shape != array.shape[:2]:
            raise BusinessError(ErrorCode.E_TILE_SHAPE_MISMATCH,
                                f"band patch shape mismatch for {tile.key} band {band}",
                                injection_id=injection_id)
        if np.issubdtype(patch.dtype, np.integer):
            array[:, :, band] = patch.astype(np.uint8)
        else:
            raise BusinessError(ErrorCode.E_NON_FINITE_BAND,
                                f"float band patch rejected for {tile.key} band {band}",
                                injection_id=injection_id)
    return array


def _guard_finite(patch: np.ndarray, tile: TileRecord, injection_id: str | None) -> None:
    if np.issubdtype(patch.dtype, np.floating) and not np.all(np.isfinite(patch)):
        raise BusinessError(
            ErrorCode.E_NON_FINITE_BAND,
            f"non-finite band values detected for {tile.key}",
            injection_id=injection_id,
        )
