"""Layered tile loading and content-addressed tile fingerprints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from .errors import BusinessEvent, E_TILE_FORMAT
from .manifest import Manifest, TileSpec
from .util import sha256_file, sha256_json


@dataclass
class TileData:
    spec: TileSpec
    arrays: dict[str, np.ndarray]
    band_hashes: dict[str, str]
    content_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tile_id": self.spec.tile_id,
            "row": self.spec.row,
            "col": self.spec.col,
            "size": self.spec.size,
            "content_key": self.content_key,
            "band_hashes": dict(self.band_hashes),
            "band_set_version": self.spec.band_set_version,
        }


@dataclass
class Mosaic:
    width: int
    height: int
    arrays: dict[str, np.ndarray]
    tile_origins: dict[str, tuple[int, int, int, int]]
    valid_positions: set[tuple[int, int]]

    def extraction(self, band: str, top: int, left: int, height: int, width: int) -> np.ndarray:
        rows = np.clip(np.arange(top, top + height), 0, self.height - 1)
        cols = np.clip(np.arange(left, left + width), 0, self.width - 1)
        return self.arrays[band][np.ix_(rows, cols)]


def _tile_event(tile_id: str, message: str, **detail: Any) -> BusinessEvent:
    return BusinessEvent(
        tile_id=tile_id,
        stage="tile_load",
        code=E_TILE_FORMAT,
        message=message,
        detail=detail,
    )


def _load_layer(path: Path, band: str) -> np.ndarray:
    with Image.open(path) as image:
        array = np.array(image)
    if array.ndim != 2:
        raise ValueError(f"band layer {band} must decode to a two-dimensional array")
    return array.astype(np.float64)


def load_tiles(
    manifest: Manifest,
    tiles_dir: Path,
    config: dict[str, Any],
    injections: dict[str, Any] | None = None,
    skip_tile_ids: set[str] | None = None,
) -> tuple[dict[str, TileData], list[BusinessEvent]]:
    tile_faults = (injections or {}).get("tile_faults", {})
    expected_bands = list(config["tile"]["bands"])
    loaded: dict[str, TileData] = {}
    events: list[BusinessEvent] = []
    skipped = skip_tile_ids or set()

    for spec in sorted(manifest.tiles, key=lambda item: (item.row, item.col, item.tile_id)):
        if spec.tile_id in skipped:
            continue
        fault = tile_faults.get(spec.tile_id, {})
        if fault.get("kind") == "quant_gear_out_of_range":
            events.append(
                BusinessEvent(
                    tile_id=spec.tile_id,
                    stage="quantization_validation",
                    code=E_TILE_FORMAT,
                    message="quantization gear is outside the configured set",
                    detail={
                        "gear": fault.get("gear"),
                        "allowed_gears": config["quantization"]["allowed_gears"],
                    },
                )
            )
            continue
        if len(spec.bands) != len(expected_bands) or set(spec.bands) != set(expected_bands):
            events.append(
                _tile_event(
                    spec.tile_id,
                    "declared band count does not match configuration",
                    declared=sorted(spec.bands.keys()),
                    expected=expected_bands,
                )
            )
            continue
        arrays: dict[str, np.ndarray] = {}
        band_hashes: dict[str, str] = {}
        tile_ok = True
        resolutions: set[tuple[int, int]] = set()
        for band in expected_bands:
            relative_path = spec.bands.get(band)
            if not relative_path:
                events.append(_tile_event(spec.tile_id, "band path is missing", band=band))
                tile_ok = False
                break
            band_path = (tiles_dir / relative_path).resolve()
            try:
                array = _load_layer(band_path, band)
                band_hashes[band] = sha256_file(band_path)
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                events.append(
                    _tile_event(
                        spec.tile_id,
                        "band layer cannot be loaded",
                        band=band,
                        path=str(band_path),
                        error=str(exc),
                    )
                )
                tile_ok = False
                break
            arrays[band] = array
            resolutions.add((int(array.shape[0]), int(array.shape[1])))
        if not tile_ok:
            continue
        if len(resolutions) != 1:
            events.append(
                _tile_event(
                    spec.tile_id,
                    "band resolutions are inconsistent",
                    resolutions=sorted(resolutions),
                )
            )
            continue
        height, width = next(iter(resolutions))
        if height != spec.size or width != spec.size:
            events.append(
                _tile_event(
                    spec.tile_id,
                    "band resolution differs from manifest tile size",
                    resolution=[height, width],
                    declared_size=spec.size,
                )
            )
            continue

        if fault.get("kind") == "nonfinite_band":
            band = str(fault["band"])
            row = int(fault["row"])
            col = int(fault["col"])
            value_text = str(fault.get("value", "nan")).lower()
            value = float("inf") if value_text in {"inf", "+inf"} else float("nan")
            arrays[band][row, col] = value

        nonfinite_locations: list[dict[str, Any]] = []
        for band, array in arrays.items():
            if not np.all(np.isfinite(array)):
                positions = np.argwhere(~np.isfinite(array))
                first = positions[0].tolist() if len(positions) else []
                nonfinite_locations.append(
                    {"band": band, "count": int(len(positions)), "first": first}
                )
        if nonfinite_locations:
            events.append(
                _tile_event(
                    spec.tile_id,
                    "band data contains non-finite values",
                    locations=nonfinite_locations,
                )
            )
            continue

        content_key = sha256_json(
            {
                "tile": spec.to_dict(),
                "band_hashes": band_hashes,
                "tile_content_version": "tile-content-v1",
            }
        )
        loaded[spec.tile_id] = TileData(
            spec=spec,
            arrays=arrays,
            band_hashes=band_hashes,
            content_key=content_key,
        )
    return loaded, events


def build_mosaic(
    tiles: dict[str, TileData],
    config: dict[str, Any],
    manifest: Manifest | None = None,
) -> Mosaic:
    tile_size = int(config["tile"]["size"])
    if manifest is not None and manifest.grid:
        max_row = int(manifest.grid.get("rows", 1)) - 1
        max_col = int(manifest.grid.get("cols", 1)) - 1
    else:
        specs = manifest.tiles if manifest is not None else [tile.spec for tile in tiles.values()]
        max_row = max(spec.row for spec in specs)
        max_col = max(spec.col for spec in specs)
    height = (max_row + 1) * tile_size
    width = (max_col + 1) * tile_size
    arrays = {
        band: np.full((height, width), np.nan, dtype=np.float64)
        for band in config["tile"]["bands"]
    }
    origins: dict[str, tuple[int, int, int, int]] = {}
    valid_positions: set[tuple[int, int]] = set()
    for tile in tiles.values():
        top = tile.spec.row * tile_size
        left = tile.spec.col * tile_size
        bottom = top + tile_size
        right = left + tile_size
        origins[tile.spec.tile_id] = (top, left, bottom, right)
        valid_positions.add((tile.spec.row, tile.spec.col))
        for band in config["tile"]["bands"]:
            arrays[band][top:bottom, left:right] = tile.arrays[band]
    return Mosaic(
        width=width,
        height=height,
        arrays=arrays,
        tile_origins=origins,
        valid_positions=valid_positions,
    )
