"""Global raster assembly and small-component removal."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from .classifier import TileResult
from .fusion import FusionResult
from .model import CLASSES


@dataclass
class RemovedComponent:
    component_id: int
    class_code: int
    pixel_count: int
    cells: List[List[int]]
    filled_with: int


@dataclass
class GlobalRaster:
    raster: np.ndarray
    height: int
    width: int
    removed: List[RemovedComponent] = field(default_factory=list)
    overlap_writes: int = 0


def assemble_global(
    tile_results: Dict[str, TileResult],
    fusions: List[FusionResult],
    grid_rows: int,
    grid_cols: int,
    edge: int,
    overlap: int,
) -> GlobalRaster:
    step = edge - overlap
    height = grid_rows * step + overlap
    width = grid_cols * step + overlap
    raster = np.full((height, width), -1, dtype=np.int8)
    overlap_writes = 0

    # 1) paint every tile's full raster at its global origin
    for tr in sorted(tile_results.values(), key=lambda t: (t.row, t.col)):
        r0, c0 = tr.row * step, tr.col * step
        raster[r0 : r0 + edge, c0 : c0 + edge] = tr.class_raster

    # 2) overwrite overlap cells with the fused winner
    for fusion in sorted(fusions, key=lambda f: f.pair_key):
        geom = fusion.geometry
        depth, span = geom["depth"], geom["span"]
        for s in range(span):
            for d in range(depth):
                gr, gc = fusion.votes[s * depth + d].global_row, fusion.votes[s * depth + d].global_col
                raster[gr, gc] = fusion.winner_raster[d, s]
                overlap_writes += 1

    return GlobalRaster(raster=raster, height=height, width=width, overlap_writes=overlap_writes)


def _connected_components(raster: np.ndarray) -> np.ndarray:
    """4-connectivity labelling, deterministic scan order."""
    h, w = raster.shape
    labels = np.full((h, w), -1, dtype=np.int32)
    next_label = 0
    for r in range(h):
        for c in range(w):
            if labels[r, c] != -1:
                continue
            target = raster[r, c]
            stack = [(r, c)]
            labels[r, c] = next_label
            while stack:
                cr, cc = stack.pop()
                for dr, dc in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] == -1 and raster[nr, nc] == target:
                        labels[nr, nc] = next_label
                        stack.append((nr, nc))
            next_label += 1
    return labels


def remove_small_components(global_ras: GlobalRaster, min_pixels: int) -> GlobalRaster:
    raster = global_ras.raster
    labels = _connected_components(raster)
    h, w = raster.shape
    removed: List[RemovedComponent] = []

    for label_id in sorted(np.unique(labels).tolist()):
        cells = np.argwhere(labels == label_id)
        if len(cells) >= min_pixels:
            continue
        class_code = int(raster[cells[0][0], cells[0][1]])
        # find dominant 4-neighbour class outside the component; deterministic
        # row-major accumulation, tie -> lowest class code.  Unfilled cells
        # (class -1, produced by tiles that failed validation) are ignored.
        neighbour_votes = {k: 0 for k in range(len(CLASSES))}
        cell_set = {(int(r), int(c)) for r, c in cells}
        for r, c in cells:
            for dr, dc in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cell_set:
                    neighbour = int(raster[nr, nc])
                    if neighbour in neighbour_votes:
                        neighbour_votes[neighbour] += 1
        if any(neighbour_votes.values()):
            fill = min(neighbour_votes, key=lambda k: (-neighbour_votes[k], k))
        elif class_code in neighbour_votes:
            fill = class_code
        else:
            fill = 0  # component had only unfilled (-1) neighbours
        for r, c in cells:
            raster[int(r), int(c)] = fill
        removed.append(
            RemovedComponent(
                component_id=int(label_id),
                class_code=class_code,
                pixel_count=int(len(cells)),
                cells=[[int(r), int(c)] for r, c in cells],
                filled_with=int(fill),
            )
        )

    global_ras.removed = removed
    return global_ras
