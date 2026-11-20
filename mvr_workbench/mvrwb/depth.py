"""深度图生成：重建点投影、量化离散与固定次数确定性邻域平滑。"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import ReconstructionConfig
from .util import content_fingerprint, fingerprint_many, quantize_scalar


def _convolve3(grid: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """零填充 3x3 卷积（不做归一化），等价于逐像素 3x3 邻域归约。"""
    padded = np.pad(grid, 1, mode="constant", constant_values=0.0)
    out = np.zeros_like(grid, dtype=np.float64)
    for dy in range(3):
        for dx in range(3):
            out += kernel[dy, dx] * padded[dy:dy + grid.shape[0], dx:dx + grid.shape[1]]
    return out


def _gather_view_points(view_id: int, match_artifacts: list[dict], frac_bits: int):
    """收集引用该视图的全部重建点，按 point_key 稳定排序。"""
    gathered: list[dict] = []
    for artifact in sorted(match_artifacts, key=lambda item: item["pair_id"]):
        content = artifact["content"]
        if not content.get("accepted") or view_id not in (content["view_a"], content["view_b"]):
            continue
        for point in content["points"]:
            if view_id not in point["source_view_ids"]:
                continue
            xyz = np.asarray(
                [value / float(1 << frac_bits) for value in point["xyz_q"]],
                dtype=np.float64,
            )
            gathered.append({"point_key": point["point_key"], "pair_id": point["pair_id"], "xyz": xyz})
    gathered.sort(key=lambda item: item["point_key"])
    return gathered


def build_depth_map(
    view_id: int,
    camera,
    match_artifacts: list[dict],
    cfg: ReconstructionConfig,
) -> dict[str, Any]:
    frac_bits = cfg.fixed_point_frac_bits
    grid_h, grid_w = cfg.depth_resolution
    points = _gather_view_points(view_id, match_artifacts, frac_bits)

    grid = np.full((grid_h, grid_w), np.nan, dtype=np.float64)
    cell_sources: dict[tuple[int, int], list[dict]] = {}
    scale_x = grid_w / float(camera.width)
    scale_y = grid_h / float(camera.height)

    for item in points:
        pixels, depth, _ = camera.project(item["xyz"].reshape(1, 3))
        if depth[0] <= 0.0:
            continue
        gx = int(pixels[0, 0] * scale_x)
        gy = int(pixels[0, 1] * scale_y)
        if not (0 <= gx < grid_w and 0 <= gy < grid_h):
            continue
        cell = (gy, gx)
        cell_sources.setdefault(cell, []).append(item)
        # 同格多点按 point_key 稳定排序，取最小深度（确定性）
        if np.isnan(grid[cell]) or depth[0] < grid[cell]:
            grid[cell] = float(depth[0])

    valid = np.isfinite(grid)
    if valid.any():
        dmin = float(np.nanmin(grid))
        dmax = float(np.nanmax(grid))
    else:
        dmin, dmax = 0.0, 1.0
    span = (dmax - dmin) if dmax > dmin else 1.0

    quant = np.full((grid_h, grid_w), -1, dtype=np.int64)
    quant[valid] = np.floor((grid[valid] - dmin) / span * cfg.depth_quant_levels)
    quant = np.clip(quant, 0, cfg.depth_quant_levels - 1)

    # 固定次数确定性平滑：仅对有效格做 3x3 有效邻域均值（与逐格循环同值），
    # 用零填充卷积求邻域和，再除以有效邻域计数，结果与线程调度无关
    smoothed = quant.astype(np.float64)
    smoothed[~valid] = np.nan
    kernel = np.ones((3, 3), dtype=np.float64)
    for _ in range(cfg.depth_smooth_passes):
        filled = np.where(np.isfinite(smoothed), smoothed, 0.0)
        mask = np.isfinite(smoothed).astype(np.float64)
        neighbor_sum = _convolve3(filled, kernel)
        neighbor_count = _convolve3(mask, kernel)
        with np.errstate(invalid="ignore", divide="ignore"):
            updated = np.where(
                np.isfinite(smoothed) & (neighbor_count > 0),
                neighbor_sum / np.where(neighbor_count > 0, neighbor_count, 1.0),
                np.nan,
            )
        smoothed = updated
    final_quant = np.full((grid_h, grid_w), -1, dtype=np.int64)
    final_valid = np.isfinite(smoothed)
    final_quant[final_valid] = np.rint(smoothed[final_valid]).astype(np.int64)

    entries: list[dict[str, Any]] = []
    for y in range(grid_h):
        for x in range(grid_w):
            if final_quant[y, x] < 0:
                continue
            sources = sorted(
                (source["point_key"] for source in cell_sources.get((y, x), [])),
            )
            entries.append(
                {
                    "gy": y,
                    "gx": x,
                    "depth_q": quantize_scalar(float(grid[y, x]), frac_bits),
                    "quant_bin": int(final_quant[y, x]),
                    "point_keys": sources,
                }
            )

    content = {
        "view_id": view_id,
        "depth_resolution": [grid_h, grid_w],
        "depth_quant_levels": cfg.depth_quant_levels,
        "depth_smooth_passes": cfg.depth_smooth_passes,
        "depth_range_q": [quantize_scalar(dmin, frac_bits), quantize_scalar(dmax, frac_bits)],
        "entries": entries,
    }
    return {
        "artifact_id": f"depth-{view_id:04d}",
        "stage": "depth",
        "view_id": view_id,
        "content": content,
        "content_fingerprint": content_fingerprint(content),
    }


def depth_artifact_key(view_id: int, upstream_fps: list[str], param_fp: str) -> str:
    return fingerprint_many("depth", view_id, *sorted(upstream_fps), param_fp)
