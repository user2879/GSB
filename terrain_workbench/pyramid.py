"""Multi-resolution heightfield pyramid.

Downsampling filter (declared, fixed): separable 3x3 tent filter applied at
stride 2, i.e. coarse[i, j] is the weighted mean of fine[2i-1:2i+2, 2j-1:2j+2]
with weights [[1,2,1],[2,4,2],[1,2,1]] / 16, border-replicated, rounded
half-up in integer arithmetic. This keeps the 2**n+1 vertex layout aligned
across levels (coarse vertices coincide with fine vertices).

For every level we record the max absolute deviation between the level and
the bilinearly reconstructed previous level, plus the cumulative deviation
bound vs. level 0 (valid because bilinear upsampling is a convex operator,
so |up(a) - up(b)| <= max|a - b| pointwise).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .heightfield import bilinear_upsample, log2_cells

FILTER_NAME = "tent3x3_stride2_round_half_up"


def downsample_tent(fine: np.ndarray) -> np.ndarray:
    """One pyramid level down using the declared tent filter (uint16 in/out)."""
    f = fine.astype(np.int32)
    padded = np.pad(f, 1, mode="edge")
    acc = (
        4 * padded[1::2, 1::2]
        + 2
        * (
            padded[0:-2:2, 1::2]
            + padded[2::2, 1::2]
            + padded[1::2, 0:-2:2]
            + padded[1::2, 2::2]
        )
        + padded[0:-2:2, 0:-2:2]
        + padded[0:-2:2, 2::2]
        + padded[2::2, 0:-2:2]
        + padded[2::2, 2::2]
    )
    return ((acc + 8) >> 4).astype(np.uint16)


def build_pyramid(level0: np.ndarray) -> tuple[list[np.ndarray], dict]:
    """Build all levels down to 3x3. Returns (levels, metadata)."""
    n = log2_cells(level0.shape[0])
    if level0.shape[0] != level0.shape[1]:
        raise ValueError("heightfield must be square")
    levels = [np.ascontiguousarray(level0)]
    while levels[-1].shape[0] > 3:
        levels.append(downsample_tent(levels[-1]))

    level_meta = []
    cumulative = 0
    for idx in range(1, len(levels)):
        recon = bilinear_upsample(levels[idx].astype(np.float64), levels[idx - 1].shape[0])
        dev = int(np.abs(recon - levels[idx - 1].astype(np.float64)).max())
        cumulative += dev
        level_meta.append(
            {
                "level": idx,
                "size": int(levels[idx].shape[0]),
                "max_dev_vs_prev": dev,
                "cumulative_dev_vs_level0": cumulative,
            }
        )
    meta = {
        "filter": FILTER_NAME,
        "levels": level_meta,
        "cumulative_dev": [0] + [m["cumulative_dev_vs_level0"] for m in level_meta],
        "level_count": len(levels),
        "base_size": int(level0.shape[0]),
        "base_log2": n,
    }
    return levels, meta


def level_hashes(levels: list[np.ndarray]) -> list[str]:
    return [
        hashlib.sha256(np.ascontiguousarray(level).tobytes()).hexdigest() for level in levels
    ]


def save_pyramid(levels: list[np.ndarray], meta: dict, out_dir: str | Path, tag: str) -> dict:
    """Serialize levels as raw uint16 (.u16) plus a meta.json; returns meta with hashes."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hashes = []
    for idx, level in enumerate(levels):
        path = out / f"{tag}_L{idx}.u16"
        path.write_bytes(np.ascontiguousarray(level).tobytes())
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    full_meta = dict(meta)
    full_meta["tag"] = tag
    full_meta["sha256"] = hashes
    meta_path = out / f"{tag}_meta.json"
    meta_path.write_text(json.dumps(full_meta, indent=2, sort_keys=True) + "\n")
    return full_meta


def load_pyramid(out_dir: str | Path, tag: str) -> tuple[list[np.ndarray], dict]:
    out = Path(out_dir)
    meta = json.loads((out / f"{tag}_meta.json").read_text())
    levels = []
    for idx in range(meta["level_count"]):
        size = ((meta["base_size"] - 1) >> idx) + 1
        raw = (out / f"{tag}_L{idx}.u16").read_bytes()
        levels.append(np.frombuffer(raw, dtype=np.uint16).reshape(size, size).copy())
    return levels, meta
