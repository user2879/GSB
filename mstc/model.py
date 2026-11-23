"""Domain data model: tiles, bands and adjacency."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Class codes are fixed integers so tie-breaking by "lowest class code" is
# well-defined and deterministic across runs.
CLASSES = ("WATER", "VEGETATION", "BUILT", "BARE")
CLASS_INDEX = {name: idx for idx, name in enumerate(CLASSES)}

# Band set definitions.  BS_V1 has the five canonical bands; BS_V2 adds a SWIR
# band so the version is a real part of the fingerprint.
BAND_SETS: Dict[str, Tuple[str, ...]] = {
    "BS_V1": ("R", "G", "B", "NIR", "DEM"),
    "BS_V2": ("R", "G", "B", "NIR", "SWIR", "DEM"),
}


@dataclass
class TileSpec:
    tile_id: str
    row: int
    col: int
    bandset: str
    bands: Dict[str, str]  # band name -> windows relative path
    declared_resolution: int
    neighbors: List[str] = field(default_factory=list)


@dataclass
class Tile:
    spec: TileSpec
    data: Dict[str, np.ndarray]  # band name -> 2D uint8 (DEM may be uint16)

    @property
    def tile_id(self) -> str:
        return self.spec.tile_id

    @property
    def edge(self) -> int:
        return next(iter(self.data.values())).shape[0]


@dataclass
class Manifest:
    tile_edge: int
    overlap: int
    tiles: List[TileSpec]

    def by_id(self) -> Dict[str, TileSpec]:
        return {t.tile_id: t for t in self.tiles}

    def sorted_tiles(self) -> List[TileSpec]:
        return sorted(self.tiles, key=lambda t: (t.row, t.col, t.tile_id))
