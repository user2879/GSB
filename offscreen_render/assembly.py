"""行序拼装灰度图、误差报告与原子图像写出（业务要求 3、8）。"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from .errors import E_IO, RenderError
from .persistence import atomic_write_text
from .renderer import TileResult
from .scene import RenderConfig


def assemble_supersampled(tiles: list[TileResult], cfg: RenderConfig
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """把分块按行序拼回完整超采样网格。"""
    grid_w = cfg.width * cfg.supersample
    grid_h = cfg.height * cfg.supersample
    gray = np.zeros((grid_h, grid_w), dtype=np.float64)
    diff = np.zeros((grid_h, grid_w), dtype=np.float64)
    steps = np.zeros((grid_h, grid_w), dtype=np.int32)
    reasons = np.full((grid_h, grid_w), "completed", dtype="<U16")
    for tile in sorted(tiles, key=lambda t: (t.y0, t.x0)):
        gray[tile.y0:tile.y0 + tile.height, tile.x0:tile.x0 + tile.width] = tile.grayscale
        diff[tile.y0:tile.y0 + tile.height, tile.x0:tile.x0 + tile.width] = tile.diff_map
        steps[tile.y0:tile.y0 + tile.height, tile.x0:tile.x0 + tile.width] = tile.steps_map
        reasons[tile.y0:tile.y0 + tile.height, tile.x0:tile.x0 + tile.width] = tile.reason_map
    return gray, diff, steps, reasons


def downsample(grid: np.ndarray, supersample: int) -> np.ndarray:
    """对超采样网格做 N x N 平均，回到最终像素分辨率。"""
    if supersample == 1:
        return grid.copy()
    h, w = grid.shape
    n = supersample
    if h % n or w % n:
        raise RenderError(E_IO, "超采样网格尺寸无法整除回像素分辨率")
    reshaped = grid.reshape(h // n, n, w // n, n)
    return reshaped.mean(axis=(1, 3))


def quantize_grayscale(pixel_radiance: np.ndarray) -> np.ndarray:
    """线性灰度归一化并量化到 0..255（确定性 round-half-even 由 numpy 完成）。"""
    finite = np.isfinite(pixel_radiance)
    if not np.all(finite):
        pixel_radiance = np.where(finite, pixel_radiance, 0.0)
    peak = float(pixel_radiance.max())
    normalized = pixel_radiance / peak if peak > 0.0 else pixel_radiance
    normalized = np.clip(normalized, 0.0, 1.0)
    return np.rint(normalized * 255.0).astype(np.uint8), peak


def write_pgm(path: str, image: np.ndarray) -> None:
    """以二进制 PGM（P5）原子写出灰度图。"""
    if image.ndim != 2 or image.dtype != np.uint8:
        raise RenderError(E_IO, "PGM 输出需要二维 uint8 图像")
    h, w = image.shape
    header = f"P5\n{w} {h}\n255\n".encode("ascii")
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "wb") as fh:
            fh.write(header)
            fh.write(image.tobytes())
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RenderError(E_IO, f"图像原子写出失败: {path}",
                          {"message": str(exc)}) from exc


def build_report(tiles: list[TileResult], cfg: RenderConfig,
                 pixel_image: np.ndarray, peak_radiance: float,
                 version_id: str, stability: dict[str, Any] | None = None) -> dict[str, Any]:
    _, diff_grid, _, reason_grid = assemble_supersampled(tiles, cfg)
    total_rays = reason_grid.size
    cutoff = int(np.count_nonzero(reason_grid == "cutoff"))
    max_steps = int(np.count_nonzero(reason_grid == "max_steps"))
    completed = int(np.count_nonzero(reason_grid == "completed"))

    first_exceeds: list[dict[str, Any]] = []
    for tile in sorted(tiles, key=lambda t: t.tile_index):
        first_exceeds.extend(tile.first_exceeds)

    report = {
        "format": "offscreen-render-report",
        "format_version": 1,
        "version_id": version_id,
        "image_size": [int(cfg.width), int(cfg.height)],
        "supersample": int(cfg.supersample),
        "rays_per_pixel": int(cfg.supersample ** 2),
        "total_rays": int(total_rays),
        "termination_stats": {
            "completed": completed,
            "cutoff": cutoff,
            "max_steps": max_steps,
            "cutoff_ratio": cutoff / total_rays,
            "max_steps_ratio": max_steps / total_rays,
        },
        "precision": {
            "fp64_reference": "grayscale source",
            "fp32_mantissa_bits": cfg.fp32_mantissa_bits,
            "transmittance_tolerance": cfg.transmittance_tolerance,
            "max_abs_transmittance_diff": float(diff_grid.max()),
            "mean_abs_transmittance_diff": float(diff_grid.mean()),
            "first_exceed_count": len(first_exceeds),
            "first_exceeds": first_exceeds[:64],
        },
        "quantization": {"peak_radiance": peak_radiance,
                         "grayscale_range": [int(pixel_image.min()), int(pixel_image.max())]},
    }
    if stability is not None:
        report["stability"] = stability
    return report


def write_report(path: str, report: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
