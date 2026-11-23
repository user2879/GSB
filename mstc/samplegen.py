"""Deterministic local sample generation.

No external download is involved.  A continuous latent "ground truth" field is
defined over the *global* grid so neighbouring tiles share structure across
their overlap.  Each tile then renders its own acquisition of that field with a
seeded RNG (small sensor noise), guaranteeing bit-identical regeneration for a
given seed while giving the fusion layer slightly different per-tile views.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .model import BAND_SETS
from .paths import to_path, to_win

SEED = 20261124


def _latent_class(global_r: np.ndarray, global_c: np.ndarray) -> np.ndarray:
    """Smooth 4-class latent field from deterministic sinusoids."""
    x = global_c.astype(np.float64)
    y = global_r.astype(np.float64)
    water = np.sin(0.35 * x) * np.cos(0.30 * y)
    veg = np.sin(0.25 * (x + y)) + 0.5
    built = np.cos(0.40 * x - 0.20 * y)
    bare = np.sin(0.55 * x) * np.cos(0.45 * y) - 0.3
    stack = np.stack([water, veg, built, bare], axis=0)
    return np.argmax(stack, axis=0).astype(np.uint8)


def _render_bands(latent: np.ndarray, rng: np.random.Generator, edge: int) -> Dict[str, np.ndarray]:
    bands: Dict[str, np.ndarray] = {}
    noise = lambda scale: rng.integers(-scale, scale + 1, size=(edge, edge), dtype=np.int16)

    base = np.full_like(latent, 110, dtype=np.int16)
    r = base + noise(18)
    g = base + noise(18)
    b = base + noise(18)
    nir = base + noise(22)
    swir = base + noise(20)
    dem = np.full((edge, edge), 100, dtype=np.int16) + rng.integers(-12, 13, size=(edge, edge))

    # class 0 WATER: low NIR, blue dominant, low elevation
    m = latent == 0
    r[m], g[m], b[m], nir[m], swir[m], dem[m] = 35, 60, 150, 25, 40, 40
    # class 1 VEGETATION: high NIR, green dominant
    m = latent == 1
    r[m], g[m], b[m], nir[m], swir[m], dem[m] = 55, 140, 60, 205, 90, 120
    # class 2 BUILT: high texture / grey, elevated
    m = latent == 2
    r[m], g[m], b[m], nir[m], swir[m], dem[m] = 175, 170, 165, 130, 160, 170
    # class 3 BARE: red dominant, low NIR
    m = latent == 3
    r[m], g[m], b[m], nir[m], swir[m], dem[m] = 190, 140, 95, 110, 175, 110

    for name, arr in (("R", r), ("G", g), ("B", b), ("NIR", nir), ("SWIR", swir)):
        bands[name] = np.clip(arr, 0, 255).astype(np.uint8)
    bands["DEM"] = np.clip(dem, 0, 4095).astype(np.uint16)
    return bands


def generate_sample(
    root: str | Path,
    grid_rows: int = 3,
    grid_cols: int = 3,
    edge: int = 16,
    overlap: int = 4,
    bandset: str = "BS_V1",
    seed: int = SEED,
) -> Tuple[Path, dict]:
    """Generate tiles + manifest.  Returns (manifest native path, manifest dict)."""
    root = Path(root)
    tiles_dir = root / "samples" / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    band_names = BAND_SETS[bandset]
    tile_specs: List[dict] = []

    for tr in range(grid_rows):
        for tc in range(grid_cols):
            tile_id = f"t{tr}{tc}"
            r0 = tr * (edge - overlap)
            c0 = tc * (edge - overlap)
            rr = r0 + np.arange(edge)[:, None]
            cc = c0 + np.arange(edge)[None, :]
            latent = _latent_class(np.broadcast_to(rr, (edge, edge)), np.broadcast_to(cc, (edge, edge)))
            rng = np.random.default_rng(seed + tr * 1000 + tc)
            bands = _render_bands(latent, rng, edge)
            tile_dir = tiles_dir / tile_id
            tile_dir.mkdir(parents=True, exist_ok=True)
            band_paths: Dict[str, str] = {}
            for name in band_names:
                native = tile_dir / f"{name}.npy"
                np.save(native, bands[name])
                band_paths[name] = to_win(f"samples\\tiles\\{tile_id}\\{name}.npy")
            neighbors = []
            if tr > 0:
                neighbors.append(f"t{tr-1}{tc}")
            if tr < grid_rows - 1:
                neighbors.append(f"t{tr+1}{tc}")
            if tc > 0:
                neighbors.append(f"t{tr}{tc-1}")
            if tc < grid_cols - 1:
                neighbors.append(f"t{tr}{tc+1}")
            tile_specs.append(
                {
                    "tile_id": tile_id,
                    "row": tr,
                    "col": tc,
                    "bandset": bandset,
                    "resolution": edge,
                    "neighbors": neighbors,
                    "bands": band_paths,
                }
            )

    manifest = {
        "tile_edge": edge,
        "overlap": overlap,
        "band_order": list(band_names),
        "tiles": tile_specs,
    }
    manifest_native = root / "samples" / "manifest.json"
    manifest_native.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_native, manifest
