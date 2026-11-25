"""Manifest parsing and tile-level structural validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import BusinessEvent, E_TILE_FORMAT
from .util import load_structured, sha256_json


@dataclass
class TileSpec:
    tile_id: str
    row: int
    col: int
    size: int
    band_set_version: str
    bands: dict[str, str]
    neighbors: list[str]
    raw: dict[str, Any] = field(default_factory=dict)
    effective_window_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.tile_id,
            "row": self.row,
            "col": self.col,
            "size": self.size,
            "band_set_version": self.band_set_version,
            "bands": dict(self.bands),
            "neighbors": list(self.neighbors),
            "effective_window_size": self.effective_window_size,
        }


@dataclass
class Manifest:
    path: Path
    seed: int | None
    grid: dict[str, Any]
    tiles: list[TileSpec]
    raw: dict[str, Any]

    def by_id(self) -> dict[str, TileSpec]:
        return {tile.tile_id: tile for tile in self.tiles}

    def by_position(self) -> dict[tuple[int, int], TileSpec]:
        return {(tile.row, tile.col): tile for tile in self.tiles}

    def content_key(self) -> str:
        return sha256_json(self.raw)


def _tile_issue(tile_id: str | None, message: str, **detail: Any) -> BusinessEvent:
    return BusinessEvent(
        tile_id=tile_id,
        stage="manifest_validation",
        code=E_TILE_FORMAT,
        message=message,
        detail=detail,
    )


def load_manifest(
    path: Path,
    config: dict[str, Any],
    injections: dict[str, Any] | None = None,
) -> tuple[Manifest, list[BusinessEvent], set[str]]:
    raw = load_structured(path)
    if not isinstance(raw, dict) or not isinstance(raw.get("tiles"), list):
        raise ValueError("manifest must contain a tiles list")
    tile_faults = (injections or {}).get("tile_faults", {})
    declared_size = int(config["tile"]["size"])
    allowed_versions = set(config["tile"]["band_set_versions"])
    allowed_bands = list(config["tile"]["bands"])

    parsed: list[TileSpec] = []
    events: list[BusinessEvent] = []
    invalid: set[str] = set()
    identifier_counts: dict[str, int] = {}
    position_counts: dict[tuple[int, int], int] = {}

    for raw_tile in raw["tiles"]:
        tile_id = str(raw_tile.get("id", ""))
        identifier_counts[tile_id] = identifier_counts.get(tile_id, 0) + 1
        position = (int(raw_tile.get("row", -1)), int(raw_tile.get("col", -1)))
        position_counts[position] = position_counts.get(position, 0) + 1
        bands = dict(raw_tile.get("bands", {}))
        neighbors = list(raw_tile.get("neighbors", []))
        fault = tile_faults.get(tile_id, {})
        if fault.get("kind") == "self_neighbor":
            neighbors.append(str(fault["neighbor"]))
        if fault.get("kind") == "band_count_mismatch":
            bands.pop(str(fault["band"]), None)
        effective_window_size = int(config["window"]["size"])
        if fault.get("kind") == "window_not_divisible":
            effective_window_size = int(fault["window_size"])
        parsed.append(
            TileSpec(
                tile_id=tile_id,
                row=position[0],
                col=position[1],
                size=int(raw_tile.get("size", -1)),
                band_set_version=str(raw_tile.get("band_set_version", "")),
                bands=bands,
                neighbors=neighbors,
                raw=raw_tile,
                effective_window_size=effective_window_size,
            )
        )

    for tile in parsed:
        if not tile.tile_id:
            events.append(_tile_issue(None, "tile id is empty"))
            invalid.add(tile.tile_id)
            continue
        if identifier_counts[tile.tile_id] > 1:
            events.append(
                _tile_issue(tile.tile_id, "duplicate tile id", count=identifier_counts[tile.tile_id])
            )
            invalid.add(tile.tile_id)
        if position_counts[(tile.row, tile.col)] > 1:
            events.append(
                _tile_issue(
                    tile.tile_id,
                    "duplicate tile row and column",
                    row=tile.row,
                    col=tile.col,
                )
            )
            invalid.add(tile.tile_id)
        if tile.size != declared_size:
            events.append(
                _tile_issue(
                    tile.tile_id,
                    "tile size differs from configuration",
                    declared=tile.size,
                    configured=declared_size,
                )
            )
            invalid.add(tile.tile_id)
        if tile.band_set_version not in allowed_versions:
            events.append(
                _tile_issue(
                    tile.tile_id,
                    "band set version is not declared by configuration",
                    band_set_version=tile.band_set_version,
                    allowed=sorted(allowed_versions),
                )
            )
            invalid.add(tile.tile_id)
        if len(tile.bands) != len(allowed_bands) or set(tile.bands) != set(allowed_bands):
            events.append(
                _tile_issue(
                    tile.tile_id,
                    "manifest band set does not match configuration",
                    bands=sorted(tile.bands.keys()),
                    expected=allowed_bands,
                )
            )
            invalid.add(tile.tile_id)
        if tile.effective_window_size is None or tile.size % tile.effective_window_size != 0:
            events.append(
                _tile_issue(
                    tile.tile_id,
                    "window size does not divide tile edge",
                    window_size=tile.effective_window_size,
                    tile_size=tile.size,
                )
            )
            invalid.add(tile.tile_id)
        if tile.tile_id in tile.neighbors:
            events.append(
                _tile_issue(
                    tile.tile_id,
                    "tile has a self-referencing neighbor",
                    neighbors=tile.neighbors,
                )
            )
            invalid.add(tile.tile_id)
        if int(tile.row) < 0 or int(tile.col) < 0:
            events.append(
                _tile_issue(tile.tile_id, "tile row and column must be non-negative")
            )
            invalid.add(tile.tile_id)

    by_id = {tile.tile_id: tile for tile in parsed if tile.tile_id}
    for tile in parsed:
        for neighbor_id in tile.neighbors:
            if neighbor_id == tile.tile_id:
                continue
            neighbor = by_id.get(neighbor_id)
            if neighbor is None:
                events.append(
                    _tile_issue(
                        tile.tile_id,
                        "neighbor is not present in manifest",
                        neighbor=neighbor_id,
                    )
                )
                invalid.add(tile.tile_id)
                continue
            if tile.tile_id not in neighbor.neighbors:
                events.append(
                    _tile_issue(
                        tile.tile_id,
                        "neighbor relation is not symmetric",
                        neighbor=neighbor_id,
                    )
                )
                invalid.add(tile.tile_id)
                invalid.add(neighbor_id)

    unique_tiles: list[TileSpec] = []
    seen_ids: set[str] = set()
    for tile in parsed:
        if tile.tile_id in seen_ids:
            continue
        seen_ids.add(tile.tile_id)
        unique_tiles.append(tile)
    manifest = Manifest(
        path=path,
        seed=raw.get("seed"),
        grid=dict(raw.get("grid", {})),
        tiles=unique_tiles,
        raw=raw,
    )
    return manifest, events, invalid
