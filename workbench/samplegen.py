"""Deterministic local sample generator.

Generates layered band tiles (R, G, B, ELEV), a tile manifest, a scheduling
config, an edit command stream and a fault-injection config.  Every byte is a
pure function of the seeds, so regeneration is bit-exact.
"""
from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

GLOBAL_SEED = 20261123
TILE_SIZE = 64
OVERLAP = 8
GRID_ROWS = 3
GRID_COLS = 3
BANDS = ["R", "G", "B", "ELEV"]

# per-class band means (R, G, B, ELEV) and noise amplitude
CLASS_PARAMS = {
    0: {"mean": (60, 140, 50, 100), "noise": 8, "elev_noise": 8},    # farmland
    1: {"mean": (70, 95, 165, 80), "noise": 5, "elev_noise": 2},     # water
    2: {"mean": (150, 150, 150, 125), "noise": 25, "elev_noise": 40},  # building
    3: {"mean": (110, 110, 115, 105), "noise": 12, "elev_noise": 4},  # road
}


def _hash32(*ints) -> int:
    data = b"".join(struct.pack("<q", int(v)) for v in ints)
    return zlib.crc32(data) & 0xFFFFFFFF


def _noise(seed, band_idx, g_r, g_c, amp) -> float:
    h = _hash32(seed, band_idx, g_r, g_c)
    return ((h % 2001) / 1000.0 - 1.0) * amp


WORLD_TILES = 8  # fixed world size so tile ground truth is grid-size independent


def global_class_map(rows=GRID_ROWS, cols=GRID_COLS, tile_size=TILE_SIZE,
                     overlap=OVERLAP, seed=GLOBAL_SEED) -> np.ndarray:
    """Coherent global ground-truth class map from a seeded control grid."""
    step = tile_size - overlap
    size_r = rows * step + overlap
    size_c = cols * step + overlap
    world = _world_class_map(tile_size, overlap, seed)
    return world[:size_r, :size_c]


def _world_class_map(tile_size=TILE_SIZE, overlap=OVERLAP, seed=GLOBAL_SEED) -> np.ndarray:
    step = tile_size - overlap
    size_r = WORLD_TILES * step + overlap
    size_c = WORLD_TILES * step + overlap
    ctrl = 8
    rng_vals = np.zeros((4, ctrl, ctrl))
    for cls in range(4):
        for i in range(ctrl):
            for j in range(ctrl):
                rng_vals[cls, i, j] = _hash32(seed, cls, i, j) / 0xFFFFFFFF
    rr = np.linspace(0, ctrl - 1, size_r)
    cc = np.linspace(0, ctrl - 1, size_c)
    gi, gj = np.meshgrid(rr, cc, indexing="ij")
    i0 = np.floor(gi).astype(int).clip(0, ctrl - 2)
    j0 = np.floor(gj).astype(int).clip(0, ctrl - 2)
    fi = (gi - i0)[None, :, :]
    fj = (gj - j0)[None, :, :]
    v00 = rng_vals[:, i0, j0]
    v01 = rng_vals[:, i0, j0 + 1]
    v10 = rng_vals[:, i0 + 1, j0]
    v11 = rng_vals[:, i0 + 1, j0 + 1]
    interp = (v00 * (1 - fi) * (1 - fj) + v01 * (1 - fi) * fj +
              v10 * fi * (1 - fj) + v11 * fi * fj)
    return np.argmax(interp, axis=0).astype(np.uint8)


def tile_ground_truth(row, col, tile_size=TILE_SIZE, overlap=OVERLAP,
                      rows=GRID_ROWS, cols=GRID_COLS, seed=GLOBAL_SEED):
    gmap = _world_class_map(tile_size, overlap, seed)
    step = tile_size - overlap
    r0, c0 = row * step, col * step
    return gmap[r0:r0 + tile_size, c0:c0 + tile_size]


def synthesize_bands(gt: np.ndarray, row: int, col: int, tile_size=TILE_SIZE,
                     overlap=OVERLAP, seed=GLOBAL_SEED) -> dict:
    """Pure function: band arrays for a tile from its ground-truth map."""
    step = tile_size - overlap
    g_r0, g_c0 = row * step, col * step
    out = {name: np.zeros((tile_size, tile_size), dtype=np.uint8) for name in BANDS}
    for cls, params in CLASS_PARAMS.items():
        mask = gt == cls
        if not mask.any():
            continue
        for b_idx, band in enumerate(BANDS):
            mean = params["mean"][b_idx]
            amp = params["elev_noise"] if band == "ELEV" else params["noise"]
            arr = out[band].astype(np.float64)
            idx = np.argwhere(mask)
            noise = np.array([_noise(seed, b_idx, g_r0 + r, g_c0 + c, amp)
                              for r, c in idx])
            vals = np.clip(mean + noise, 0, 255)
            arr[mask] = vals
            out[band] = np.rint(arr).astype(np.uint8)
    # buildings get sharp rooftop bumps -> high slope/texture
    bump = (gt == 2)
    if bump.any():
        elev = out["ELEV"].astype(np.int64)
        elev[bump] = np.clip(elev[bump] + 20, 0, 255)
        out["ELEV"] = elev.astype(np.uint8)
    return out


def tile_id_of(row, col):
    return f"tile_r{row}_c{col}"


def generate_tile(samples_dir, row, col, band_set_version="bs-v1",
                  tile_size=TILE_SIZE, overlap=OVERLAP,
                  rows=GRID_ROWS, cols=GRID_COLS, seed=GLOBAL_SEED):
    tid = tile_id_of(row, col)
    gt = tile_ground_truth(row, col, tile_size, overlap, rows, cols, seed)
    bands = synthesize_bands(gt, row, col, tile_size, overlap, seed)
    tdir = Path(samples_dir) / "tiles" / tid
    tdir.mkdir(parents=True, exist_ok=True)
    for band, arr in bands.items():
        Image.fromarray(arr, mode="L").save(tdir / f"{band}.png")
    Image.fromarray(gt, mode="L").save(tdir / "ground_truth.png")
    return tid


def default_config() -> dict:
    return {
        "window_size": 16,
        "stride": 16,
        "context_overlap": 2,
        "index_weights": {
            "farmland": {"veg": 6, "water": -3, "texture": -2, "slope": -1},
            "water": {"veg": -2, "water": 6, "texture": -1, "slope": -1},
            "building": {"veg": -1, "water": -1, "texture": 2, "slope": 3},
            "road": {"veg": -1, "water": 0, "texture": 4, "slope": -2},
        },
        "min_connected_pixels": 6,
        "fusion_strategy_version": "fs-v1",
        "declared_fusion_versions": ["fs-v1", "fs-v2"],
        "fixed_point_frac_bits": 12,
        "quantum": 16,
        "thread_pool_size": 4,
        "global_pixel_budget": 10 ** 9,
        "per_tile_quota": 10 ** 9,
        "merge_batch_limit": 10 ** 6,
        "snapshot_commit_interval": 2,
        "declared_band_set_versions": ["bs-v1", "bs-v2"],
    }


def default_edits() -> list:
    return [
        {"op": "replace_band", "tile": tile_id_of(0, 1), "band": "G", "seed_delta": 7},
        {"op": "set_index_weights", "weights": {
            "farmland": {"veg": 7, "water": -3, "texture": -2, "slope": -1},
            "water": {"veg": -2, "water": 6, "texture": -1, "slope": -1},
            "building": {"veg": -1, "water": -1, "texture": 2, "slope": 3},
            "road": {"veg": -1, "water": 0, "texture": 4, "slope": -2}}},
        {"op": "set_fusion_strategy", "version": "fs-v2"},
        {"op": "set_adjacency", "tile": tile_id_of(0, 0),
         "neighbors": [tile_id_of(0, 1), tile_id_of(1, 0)]},
        {"op": "add_tile", "row": 3, "col": 0, "band_set_version": "bs-v2",
         "neighbors": [tile_id_of(2, 0)]},
        {"op": "remove_tile", "tile": tile_id_of(0, 2)},
    ]


def default_injections() -> list:
    return [
        {"inj_id": "INJ-001", "tile": tile_id_of(1, 1), "stage": "load",
         "fault": "truncate_band_bytes", "band": "R", "keep_bytes": 32},
        {"inj_id": "INJ-002", "tile": tile_id_of(0, 1), "stage": "classify",
         "fault": "nonfinite_band_value", "band": "B", "position": [3, 3]},
        {"inj_id": "INJ-003", "tile": tile_id_of(0, 0), "stage": "load",
         "fault": "adjacency_unknown_ref", "ghost": "tile_ghost"},
        {"inj_id": "INJ-004", "tile": "*", "stage": "config",
         "fault": "negative_min_connected", "value": -5},
    ]


def generate_all(out_dir, rows=GRID_ROWS, cols=GRID_COLS, tile_size=TILE_SIZE,
                 overlap=OVERLAP, seed=GLOBAL_SEED):
    out = Path(out_dir)
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    tiles = []
    for r in range(rows):
        for c in range(cols):
            tid = generate_tile(out, r, c, tile_size=tile_size, overlap=overlap,
                                rows=rows, cols=cols, seed=seed)
            neighbors = []
            for dr, dc in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbors.append(tile_id_of(nr, nc))
            tiles.append({"id": tid, "row": r, "col": c,
                          "neighbors": sorted(neighbors),
                          "band_set_version": "bs-v1"})
    manifest = {"tile_size": tile_size, "overlap_px": overlap,
                "bands": BANDS, "tiles": tiles}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "config.json").write_text(json.dumps(default_config(), indent=2))
    (out / "edits.json").write_text(json.dumps(default_edits(), indent=2))
    (out / "inject.json").write_text(json.dumps(default_injections(), indent=2))
    gmap = global_class_map(rows, cols, tile_size, overlap, seed)
    np.save(out / "ground_truth.npy", gmap)
    return out
