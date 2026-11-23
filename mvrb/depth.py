"""深度图：按观测投影像素栅格化并量化到固定档位，输出确定性摘要。"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .utils import combine_fingerprint, round_half_up


@dataclass
class DepthMap:
    view: int
    resolution: tuple
    quant_levels: int
    cells: list            # [(row, col, quantized_depth, count)]
    summary: dict

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DepthMap":
        return cls(**d)


def build_depth_map(view: int, points: list, image_size: tuple,
                    resolution: tuple, quant_levels: int) -> DepthMap:
    img_w, img_h = image_size
    rows, cols = resolution
    grid: dict = {}
    for p in points:
        for (obs_view, _fid, u, v, depth) in p.observations:
            if obs_view != view or depth <= 0:
                continue
            col = min(cols - 1, int(u / img_w * cols))
            row = min(rows - 1, int(v / img_h * rows))
            grid.setdefault((row, col), []).append(depth)
    if grid:
        dmin = min(min(v) for v in grid.values())
        dmax = max(max(v) for v in grid.values())
    else:
        dmin = dmax = 0.0
    span = (dmax - dmin) or 1.0
    cells = []
    for (row, col), depths in sorted(grid.items()):
        mean_depth = float(np.mean(depths))
        q = round_half_up((mean_depth - dmin) / span * (quant_levels - 1)) if grid else 0
        cells.append((row, col, int(q), len(depths)))
    quant_values = [c[2] for c in cells]
    summary = {
        "view": view,
        "cell_count": len(cells),
        "depth_min": round(dmin, 6),
        "depth_max": round(dmax, 6),
        "quant_sum": int(sum(quant_values)),
        "quant_histogram": _histogram(quant_values, quant_levels),
    }
    return DepthMap(view, (rows, cols), quant_levels, cells, summary)


def _histogram(values: list, levels: int) -> list:
    hist = [0] * levels
    for v in values:
        hist[int(v)] += 1
    return hist


def build_all_depth_maps(points: list, view_ids: list, image_size: tuple,
                         resolution: tuple, quant_levels: int) -> dict:
    return {
        view: build_depth_map(view, points, image_size, resolution, quant_levels)
        for view in sorted(view_ids)
    }


def depth_cache_key(points_fp: str, cfg_fp: str) -> str:
    return combine_fingerprint("depth", points_fp, cfg_fp)
