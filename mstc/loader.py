"""Manifest + tile loading and the full validation layer.

Validation is deliberately *per tile*: a bad tile produces a structured
:class:`Failure` (truncated file, missing band, bad resolution, dangling
neighbour, ...) and is excluded from processing, while every other valid tile
still loads and classifies normally.  Manifest-level structural problems that
make the document unusable raise a fatal code instead.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .config import Config
from .errors import Code, FailureList, PipelineFatal
from .model import BAND_SETS, Manifest, Tile, TileSpec
from .paths import to_path


def load_manifest(path: str | Path, root: str | Path) -> Manifest:
    p = Path(path)
    if not p.exists():
        raise PipelineFatal(Code.E_FILE_NOT_FOUND, f"manifest not found: {p}")
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PipelineFatal(Code.E_MANIFEST_PARSE, f"invalid manifest JSON: {exc}") from exc
    try:
        edge = int(doc["tile_edge"])
        overlap = int(doc["overlap"])
        raw_tiles = doc["tiles"]
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineFatal(Code.E_MANIFEST_PARSE, f"manifest missing fields: {exc}") from exc

    specs: List[TileSpec] = []
    for item in raw_tiles:
        specs.append(
            TileSpec(
                tile_id=item["tile_id"],
                row=int(item["row"]),
                col=int(item["col"]),
                bandset=item["bandset"],
                bands=dict(item["bands"]),
                declared_resolution=int(item["resolution"]),
                neighbors=list(item.get("neighbors", [])),
            )
        )
    return Manifest(tile_edge=edge, overlap=overlap, tiles=specs)


def validate_and_load(
    manifest: Manifest, config: Config, root: str | Path
) -> Tuple[Dict[str, Tile], FailureList]:
    failures = FailureList()

    # --- tile id uniqueness (manifest level) ---
    seen_ids: set[str] = set()
    unique_specs: List[TileSpec] = []
    for spec in manifest.sorted_tiles():
        if spec.tile_id in seen_ids:
            failures.add(Code.E_TILE_ID_DUPLICATE, spec.tile_id, "duplicate tile identifier")
            continue
        seen_ids.add(spec.tile_id)
        unique_specs.append(spec)

    id_set = {s.tile_id for s in unique_specs}

    # --- adjacency: dangling references + symmetry ---
    adjacency: Dict[str, set[str]] = {s.tile_id: set() for s in unique_specs}
    for spec in unique_specs:
        for nb in spec.neighbors:
            if nb not in id_set:
                failures.add(
                    Code.E_ADJ_DANGLING,
                    spec.tile_id,
                    f"neighbour {nb} does not exist",
                    detail={"neighbor": nb},
                )
            else:
                adjacency[spec.tile_id].add(nb)
    for tile_id, nbs in adjacency.items():
        for nb in nbs:
            if tile_id not in adjacency.get(nb, set()):
                failures.add(
                    Code.E_ADJ_NOT_SYMMETRIC,
                    tile_id,
                    f"adjacency {tile_id}->{nb} not reciprocated",
                    detail={"neighbor": nb},
                )

    # --- window divisibility (global config vs tile edge) ---
    win = config.window
    edge = manifest.tile_edge
    if win.size > edge:
        failures.add(Code.E_WINDOW_LARGER_TILE, "<config>", f"window {win.size} > tile edge {edge}")
    if edge % win.size != 0:
        failures.add(
            Code.E_WINDOW_NOT_DIVISIBLE,
            "<config>",
            f"window size {win.size} does not divide tile edge {edge}",
        )

    tiles: Dict[str, Tile] = {}
    for spec in unique_specs:
        tile = _load_one_tile(spec, manifest, config, root, failures)
        if tile is not None:
            tiles[spec.tile_id] = tile
    return tiles, failures


def _load_one_tile(
    spec: TileSpec, manifest: Manifest, config: Config, root: str | Path, failures: FailureList
) -> Tile | None:
    # --- band set version must be in the allowed set ---
    if spec.bandset not in config.allowed_bandsets:
        failures.add(
            Code.E_BANDSET_UNKNOWN,
            spec.tile_id,
            f"bandset {spec.bandset} not in allowed {list(config.allowed_bandsets)}",
        )
        return None
    expected_bands = BAND_SETS.get(spec.bandset)
    if expected_bands is None:
        failures.add(Code.E_BANDSET_UNKNOWN, spec.tile_id, f"unknown bandset {spec.bandset}")
        return None

    data: Dict[str, np.ndarray] = {}
    tile_failed = False

    # band count vs manifest declaration
    if set(spec.bands.keys()) != set(expected_bands):
        missing = sorted(set(expected_bands) - set(spec.bands.keys()))
        extra = sorted(set(spec.bands.keys()) - set(expected_bands))
        failures.add(
            Code.E_BAND_COUNT_MISMATCH,
            spec.tile_id,
            "declared band set does not match band files",
            detail={"missing": missing, "extra": extra},
        )
        tile_failed = True

    for band_name in expected_bands:
        rel = spec.bands.get(band_name)
        if rel is None:
            failures.add(Code.E_BAND_MISSING, spec.tile_id, f"band file missing: {band_name}")
            tile_failed = True
            continue
        native = to_path(rel, root)
        if not native.exists():
            failures.add(
                Code.E_BAND_MISSING,
                spec.tile_id,
                f"band file not found: {band_name}",
                detail={"path": rel},
            )
            tile_failed = True
            continue
        try:
            arr = np.load(native)
        except Exception as exc:  # truncated / corrupt .npy
            failures.add(
                Code.E_BAND_TRUNCATED,
                spec.tile_id,
                f"band {band_name} unreadable/truncated: {exc}",
                detail={"path": rel},
            )
            tile_failed = True
            continue
        if arr.ndim != 2:
            failures.add(
                Code.E_RESOLUTION_MISMATCH,
                spec.tile_id,
                f"band {band_name} is not 2D",
            )
            tile_failed = True
            continue
        if arr.shape[0] != spec.declared_resolution or arr.shape[1] != spec.declared_resolution:
            failures.add(
                Code.E_RESOLUTION_MISMATCH,
                spec.tile_id,
                f"band {band_name} resolution {arr.shape} != {spec.declared_resolution}",
            )
            tile_failed = True
            continue
        data[band_name] = arr

    if tile_failed:
        return None

    # cross-band resolution consistency (all square, same edge)
    shapes = {name: arr.shape for name, arr in data.items()}
    if len(set(shapes.values())) != 1:
        failures.add(Code.E_RESOLUTION_MISMATCH, spec.tile_id, f"inconsistent band shapes {shapes}")
        return None
    if data[expected_bands[0]].shape[0] != manifest.tile_edge:
        failures.add(
            Code.E_RESOLUTION_MISMATCH,
            spec.tile_id,
            "tile edge does not match manifest tile_edge",
        )
        return None

    return Tile(spec=spec, data=data)
