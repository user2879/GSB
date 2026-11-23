"""Tile, manifest and band loading with validation.

Validation failures are collected into a failed list; a failing tile never
aborts the overall run.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import errors


@dataclass
class TileSpec:
    tile_id: str
    row: int
    col: int
    neighbors: list
    band_set_version: str

    def sort_key(self):
        return (self.row, self.col, self.tile_id)


@dataclass
class Manifest:
    tile_size: int
    overlap_px: int
    bands: list
    tiles: dict = field(default_factory=dict)  # tile_id -> TileSpec
    raw: dict = field(default_factory=dict)


def load_manifest(samples_dir) -> Manifest:
    raw = json.loads((Path(samples_dir) / "manifest.json").read_text(encoding="utf-8"))
    tiles = {}
    for entry in raw["tiles"]:
        spec = TileSpec(
            tile_id=entry["id"],
            row=int(entry["row"]),
            col=int(entry["col"]),
            neighbors=list(entry["neighbors"]),
            band_set_version=entry["band_set_version"],
        )
        tiles[spec.tile_id] = spec
    return Manifest(
        tile_size=int(raw["tile_size"]),
        overlap_px=int(raw["overlap_px"]),
        bands=list(raw["bands"]),
        tiles=tiles,
        raw=raw,
    )


def validate_manifest(manifest: Manifest, declared_band_versions):
    """Structural validation. Returns (failed_entries, valid_tile_ids)."""
    failed = []
    seen_raw_ids = [t["id"] for t in manifest.raw["tiles"]]
    dupes = sorted({tid for tid in seen_raw_ids if seen_raw_ids.count(tid) > 1})
    invalid = set(dupes)
    for tid in dupes:
        failed.append({"tile_id": tid, "stage": "load",
                       "code": errors.E_TILE_DUPLICATE, "detail": "duplicate tile id"})
    for tid, spec in sorted(manifest.tiles.items()):
        if tid in invalid:
            continue
        if spec.band_set_version not in declared_band_versions:
            invalid.add(tid)
            failed.append({"tile_id": tid, "stage": "load",
                           "code": errors.E_BAND_VERSION_UNKNOWN,
                           "detail": f"band_set_version {spec.band_set_version} not declared"})
    for tid, spec in sorted(manifest.tiles.items()):
        if tid in invalid:
            continue
        for nb in spec.neighbors:
            if nb not in manifest.tiles:
                invalid.add(tid)
                failed.append({"tile_id": tid, "stage": "load",
                               "code": errors.E_ADJ_UNKNOWN_REF,
                               "detail": f"neighbor {nb} does not exist"})
        for nb in spec.neighbors:
            if nb in manifest.tiles and tid not in manifest.tiles[nb].neighbors:
                invalid.add(tid)
                failed.append({"tile_id": tid, "stage": "load",
                               "code": errors.E_ADJ_ASYMMETRIC,
                               "detail": f"{tid} lists {nb} but not vice versa"})
    valid = [tid for tid in sorted(manifest.tiles) if tid not in invalid]
    return failed, valid


class BandLoadError(Exception):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code
        self.detail = detail


def band_path(samples_dir, tile_id, band):
    return Path(samples_dir) / "tiles" / tile_id / f"{band}.png"


def load_tile_bands(samples_dir, tile_id, bands, tile_size, band_reader=None):
    """Load layered band files. Returns dict band -> ndarray (uint8 or float)."""
    reader = band_reader or _read_band_file
    out = {}
    for band in bands:
        path = band_path(samples_dir, tile_id, band)
        if not path.exists():
            raise BandLoadError(errors.E_BAND_MISSING, f"missing band file {path}")
        arr = reader(path, tile_id, band)
        if arr.shape != (tile_size, tile_size):
            raise BandLoadError(
                errors.E_TILE_SHAPE,
                f"band {band} shape {arr.shape} != ({tile_size}, {tile_size})")
        out[band] = arr
    return out


def _read_band_file(path, tile_id, band):
    try:
        with Image.open(path) as img:
            return np.asarray(img)
    except Exception as exc:  # truncated / corrupt bytes
        raise BandLoadError(errors.E_BAND_CORRUPT, f"band {band} undecodable: {exc}")


def adjacency_pairs(manifest: Manifest, valid_ids):
    """Canonical neighbor pairs sorted by ((rA,cA),(rB,cB)); each pair once."""
    valid = set(valid_ids)
    pairs = []
    for tid in sorted(valid):
        spec = manifest.tiles[tid]
        for nb in spec.neighbors:
            if nb not in valid:
                continue
            other = manifest.tiles[nb]
            key = (spec.sort_key(), other.sort_key())
            if key[0] < key[1]:
                pairs.append((tid, nb))
    pairs.sort(key=lambda p: (manifest.tiles[p[0]].sort_key(),
                              manifest.tiles[p[1]].sort_key()))
    return pairs


def pair_key(manifest: Manifest, id_a, id_b) -> str:
    return f"{id_a}::{id_b}"


def pair_sort_key(manifest: Manifest, id_a, id_b):
    return (manifest.tiles[id_a].sort_key(), manifest.tiles[id_b].sort_key())
