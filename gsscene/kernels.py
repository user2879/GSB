"""Deterministic per-view, per-level Gaussian kernel initialization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .config import SceneConfig, kernel_grid_size, level_fingerprint, projection_matrix_for_level
from .models import KernelGroup, KernelSpec, ViewRecord
from .util import array_fingerprint, arrays_fingerprint, hash_value


@dataclass
class KernelPlan:
    view_index: int
    level: int
    grid_size: tuple[int, int]
    requested_count: int
    allowed_count: int
    kernel_keys: list[str]
    level_fingerprint: str
    view_content_fingerprint: str
    camera_fingerprint: str
    projection: np.ndarray
    projection_seed: int

    @property
    def cache_key(self) -> str:
        return group_cache_key(self.view_index, self.level, self.kernel_keys, self.grid_size)

    def specs(self) -> list[KernelSpec]:
        width, _height = self.grid_size
        return [
            KernelSpec(
                view_index=self.view_index,
                level=self.level,
                row=index // width,
                col=index % width,
                kernel_index=index,
                cache_key=self.kernel_keys[index],
            )
            for index in range(len(self.kernel_keys))
        ]


def group_cache_key(view_index: int, level: int, kernel_keys: Iterable[str], grid_size: tuple[int, int]) -> str:
    return "kernels:" + hash_value(
        {
            "grid": list(grid_size),
            "kernel_keys": list(kernel_keys),
            "level": int(level),
            "view": int(view_index),
        }
    )


def _kernel_key(
    view: ViewRecord,
    level: int,
    row: int,
    col: int,
    kernel_index: int,
    level_fp: str,
    mods: list[dict[str, Any]],
) -> str:
    payload = {
        "camera_fingerprint": view.camera_fingerprint,
        "col": int(col),
        "kernel_index": int(kernel_index),
        "level": int(level),
        "level_fingerprint": level_fp,
        "row": int(row),
        "sh_mods": mods,
        "view_content_fingerprint": view.content_fingerprint,
        "view_index": int(view.view_index),
    }
    return "kernel:" + hash_value(payload)


def plan_kernel_group(
    config: SceneConfig,
    view: ViewRecord,
    level: int,
    sh_mods: list[dict[str, Any]],
) -> KernelPlan:
    grid = kernel_grid_size(config, level)
    requested = int(grid[0] * grid[1])
    budget = int(config.kernel_budgets[level])
    allowed = min(requested, budget)
    level_fp = level_fingerprint(config, level, grid, budget)
    projection, seed = projection_matrix_for_level(config, level)
    keys: list[str] = []
    width, _height = grid
    for kernel_index in range(allowed):
        row = kernel_index // width
        col = kernel_index % width
        mods_for_kernel = _mods_for(sh_mods, view.view_index, level, kernel_index)
        keys.append(_kernel_key(view, level, row, col, kernel_index, level_fp, mods_for_kernel))
    return KernelPlan(
        view_index=view.view_index,
        level=level,
        grid_size=grid,
        requested_count=requested,
        allowed_count=allowed,
        kernel_keys=keys,
        level_fingerprint=level_fp,
        view_content_fingerprint=view.content_fingerprint,
        camera_fingerprint=view.camera_fingerprint,
        projection=projection,
        projection_seed=seed,
    )


def _mods_for(sh_mods: list[dict[str, Any]], view_index: int, level: int, kernel_index: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for mod in sh_mods:
        if int(mod.get("view_index", -1)) != view_index:
            continue
        if level not in [int(item) for item in mod.get("levels", [])]:
            continue
        target = int(mod.get("kernel_index", -1))
        if target not in (-1, kernel_index):
            continue
        result.append(
            {
                "channel": int(mod["channel"]),
                "coefficient": int(mod["coefficient"]),
                "delta": float(mod["delta"]),
                "kernel_index": int(mod["kernel_index"]),
                "levels": [int(item) for item in mod["levels"]],
                "seq": int(mod.get("seq", 0)),
                "view_index": int(mod["view_index"]),
            }
        )
    return sorted(result, key=lambda item: item["seq"])


def _bilinear(array: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    height, width = array.shape[:2]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    x0 = np.clip(x0, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    fx = (x - x0).astype(np.float64)
    fy = (y - y0).astype(np.float64)
    if array.ndim == 2:
        top = array[y0, x0] * (1.0 - fx) + array[y0, x1] * fx
        bottom = array[y1, x0] * (1.0 - fx) + array[y1, x1] * fx
        return top * (1.0 - fy) + bottom * fy
    top = array[y0, x0, :] * (1.0 - fx)[:, None] + array[y0, x1, :] * fx[:, None]
    bottom = array[y1, x0, :] * (1.0 - fx)[:, None] + array[y1, x1, :] * fx[:, None]
    return top * (1.0 - fy)[:, None] + bottom * fy[:, None]


def load_view_arrays(view: ViewRecord) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(view.image_path) as image:
        image.load()
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    depth = np.load(view.depth_path, allow_pickle=False)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth = np.asarray(depth, dtype=np.float64)
    return np.ascontiguousarray(rgb), np.ascontiguousarray(depth)


def _sh_basis(direction: np.ndarray, degree: int) -> np.ndarray:
    x, y, z = (float(direction[0]), float(direction[1]), float(direction[2]))
    values = [0.28209479177387814]
    if degree >= 1:
        values.extend([0.4886025119029199 * y, 0.4886025119029199 * z, 0.4886025119029199 * x])
    if degree >= 2:
        values.extend(
            [
                1.0925484305920792 * x * y,
                1.0925484305920792 * y * z,
                0.31539156525252005 * (3.0 * z * z - 1.0),
                1.0925484305920792 * x * z,
                0.5462742152960396 * (x * x - y * y),
            ]
        )
    if degree >= 3:
        values.extend(
            [
                0.5900435899266435 * y * (3.0 * x * x - y * y),
                2.890611442640554 * x * y * z,
                0.4570457994644658 * y * (5.0 * z * z - 1.0),
                0.3731763325901154 * z * (5.0 * z * z - 3.0),
                0.4570457994644658 * x * (5.0 * z * z - 1.0),
                1.445305721320277 * z * (x * x - y * y),
                0.5900435899266435 * x * (x * x - 3.0 * y * y),
            ]
        )
    return np.asarray(values[: (degree + 1) ** 2], dtype=np.float64)


def evaluate_sh(sh: np.ndarray, direction: np.ndarray, degree: int) -> np.ndarray:
    basis = _sh_basis(direction, degree)
    values = np.tensordot(sh, basis, axes=([0], [0]))
    return np.clip(values, 0.0, 1.0)


def _compute_subset(
    config: SceneConfig,
    view: ViewRecord,
    level: int,
    plan: KernelPlan,
    indices: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    sh_mods: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    grid_w, grid_h = plan.grid_size
    height, width = rgb.shape[:2]
    rows = indices // grid_w
    cols = indices % grid_w
    u = (cols.astype(np.float64) + 0.5) * (float(width) / grid_w) - 0.5
    v = (rows.astype(np.float64) + 0.5) * (float(height) / grid_h) - 0.5
    sampled_rgb = _bilinear(rgb, np.clip(u, 0.0, width - 1.0), np.clip(v, 0.0, height - 1.0))
    sampled_depth = _bilinear(depth, np.clip(u, 0.0, width - 1.0), np.clip(v, 0.0, height - 1.0))
    sampled_depth = np.clip(sampled_depth, 1e-4, None)
    depth_norm = sampled_depth / (1.0 + sampled_depth)
    u_norm = (u + 0.5) / width
    v_norm = (v + 0.5) / height
    projected = (
        plan.projection[0, 0] * u_norm
        + plan.projection[0, 1] * v_norm
        + plan.projection[0, 2] * depth_norm
        + plan.projection[0, 3]
    )
    projected_y = (
        plan.projection[1, 0] * u_norm
        + plan.projection[1, 1] * v_norm
        + plan.projection[1, 2] * depth_norm
        + plan.projection[1, 3]
    )
    projected_z = (
        plan.projection[2, 0] * u_norm
        + plan.projection[2, 1] * v_norm
        + plan.projection[2, 2] * depth_norm
        + plan.projection[2, 3]
    )
    projected_w = (
        plan.projection[3, 0] * u_norm
        + plan.projection[3, 1] * v_norm
        + plan.projection[3, 2] * depth_norm
        + plan.projection[3, 3]
    )
    camera = view.camera
    fx = camera.intrinsic[0, 0]
    fy = camera.intrinsic[1, 1]
    cx = camera.intrinsic[0, 2]
    cy = camera.intrinsic[1, 2]
    camera_xyz = np.stack(
        [
            (u - cx) / fx * sampled_depth + 0.015 * projected,
            (v - cy) / fy * sampled_depth + 0.015 * projected_y,
            sampled_depth + 0.01 * projected_z + 0.002 * projected_w,
        ],
        axis=-1,
    )
    delta = camera_xyz - camera.translation[None, :]
    rotation = camera.rotation
    # Explicit fixed-order accumulation is bitwise independent of the number of
    # rows being recomputed.  A batched BLAS matrix multiply can change its
    # reduction order between partial and full kernel-group rebuilds.
    positions = np.stack(
        [
            delta[:, 0] * rotation[0, 0] + delta[:, 1] * rotation[1, 0] + delta[:, 2] * rotation[2, 0],
            delta[:, 0] * rotation[0, 1] + delta[:, 1] * rotation[1, 1] + delta[:, 2] * rotation[2, 1],
            delta[:, 0] * rotation[0, 2] + delta[:, 1] * rotation[1, 2] + delta[:, 2] * rotation[2, 2],
        ],
        axis=-1,
    )

    grid_cell = min(float(width) / grid_w, float(height) / grid_h)
    base_sigma = max(0.32, float(config.effective_scales()[level]) * grid_cell * 0.82)
    scale_variation = 1.0 + 0.035 * projected_w
    scales = np.stack(
        [
            np.maximum(0.2, base_sigma * scale_variation),
            np.maximum(0.2, base_sigma * (1.0 + 0.025 * projected_y)),
            np.maximum(0.2, base_sigma * (0.7 + 0.1 * depth_norm)),
        ],
        axis=-1,
    )
    opacity = np.clip(0.34 + 0.58 * (1.0 - depth_norm) + 0.04 * projected, 0.02, 0.98)

    coefficient_count = (config.sh_degree + 1) ** 2
    sh = np.zeros((indices.size, coefficient_count, 3), dtype=np.float64)
    sh[:, 0, :] = sampled_rgb
    for coefficient in range(1, coefficient_count):
        a = plan.projection[(coefficient - 1) % 4, cols % 4]
        b = plan.projection[(coefficient + 2) % 4, rows % 3]
        for channel in range(3):
            sh[:, coefficient, channel] = (
                0.055 * (a + 0.5 * b) * (sampled_rgb[:, channel] - 0.5)
                + 0.003 * float(coefficient * (channel + 1))
            )
    for local_index, kernel_index in enumerate(indices.tolist()):
        for mod in _mods_for(sh_mods, view.view_index, level, int(kernel_index)):
            coefficient = int(mod["coefficient"])
            if 0 <= coefficient < coefficient_count:
                sh[local_index, coefficient, int(mod["channel"])] += float(mod["delta"])
    return {
        "positions": positions.astype(np.float64, copy=False),
        "scales": scales.astype(np.float64, copy=False),
        "opacity": opacity.astype(np.float64, copy=False),
        "sh": sh.astype(np.float64, copy=False),
        "grid": np.stack([rows, cols], axis=-1).astype(np.int64),
        "depth": sampled_depth.astype(np.float64, copy=False),
        "source_uv": np.stack([u, v], axis=-1).astype(np.float64, copy=False),
    }


def _empty_arrays(plan: KernelPlan, coefficient_count: int) -> dict[str, np.ndarray]:
    count = plan.allowed_count
    return {
        "positions": np.zeros((count, 3), dtype=np.float64),
        "scales": np.zeros((count, 3), dtype=np.float64),
        "opacity": np.zeros((count,), dtype=np.float64),
        "sh": np.zeros((count, coefficient_count, 3), dtype=np.float64),
        "grid": np.stack(
            [np.arange(count, dtype=np.int64) // plan.grid_size[0], np.arange(count, dtype=np.int64) % plan.grid_size[0]],
            axis=-1,
        ),
        "depth": np.zeros((count,), dtype=np.float64),
        "source_uv": np.zeros((count, 2), dtype=np.float64),
        "grid_size": np.asarray(plan.grid_size, dtype=np.int64),
    }


def compute_kernel_group(
    config: SceneConfig,
    view: ViewRecord,
    level: int,
    plan: KernelPlan,
    sh_mods: list[dict[str, Any]],
    previous_arrays: dict[str, np.ndarray] | None = None,
    previous_keys: list[str] | None = None,
    previous_level_fingerprint: str | None = None,
) -> tuple[KernelGroup, list[int], list[int]]:
    """Compute only rows whose content key changed; reuse all other rows verbatim."""

    coefficient_count = (config.sh_degree + 1) ** 2
    arrays: dict[str, np.ndarray]
    if previous_arrays is None or previous_keys is None or len(previous_keys) != plan.allowed_count:
        arrays = _empty_arrays(plan, coefficient_count)
        indices_to_compute = np.arange(plan.allowed_count, dtype=np.int64)
        indices_reused: list[int] = []
    else:
        arrays = {name: np.array(value, copy=True) for name, value in previous_arrays.items()}
        if arrays["sh"].shape != (plan.allowed_count, coefficient_count, 3):
            arrays = _empty_arrays(plan, coefficient_count)
            indices_to_compute = np.arange(plan.allowed_count, dtype=np.int64)
            indices_reused = []
        else:
            changed = [index for index, key in enumerate(plan.kernel_keys) if index >= len(previous_keys) or previous_keys[index] != key]
            indices_to_compute = np.asarray(changed, dtype=np.int64)
            indices_reused = [
                index
                for index, key in enumerate(plan.kernel_keys)
                if index < len(previous_keys) and previous_keys[index] == key
            ]
    if indices_to_compute.size:
        rgb, depth = load_view_arrays(view)
        computed = _compute_subset(config, view, level, plan, indices_to_compute, rgb, depth, sh_mods)
        for name, value in computed.items():
            arrays[name][indices_to_compute] = value
    content_fingerprints = []
    for index in range(plan.allowed_count):
        row_arrays = {
            "positions": arrays["positions"][index],
            "scales": arrays["scales"][index],
            "opacity": arrays["opacity"][index],
            "sh": arrays["sh"][index],
            "depth": arrays["depth"][index],
        }
        content_fingerprints.append(arrays_fingerprint(row_arrays))
    group = KernelGroup(
        view_index=view.view_index,
        level=level,
        grid_size=plan.grid_size,
        cache_key=plan.cache_key,
        kernel_keys=list(plan.kernel_keys),
        kernel_content_fingerprints=content_fingerprints,
        arrays=arrays,
        level_fingerprint=plan.level_fingerprint,
        view_content_fingerprint=plan.view_content_fingerprint,
        camera_fingerprint=plan.camera_fingerprint,
        projection_matrix_fingerprint=array_fingerprint(plan.projection),
    )
    return group, indices_to_compute.tolist(), indices_reused


def kernel_center_radius(config: SceneConfig, level: int, grid_size: tuple[int, int], row: int, col: int) -> tuple[float, float, float]:
    width, height = config.image_size
    grid_w, grid_h = grid_size
    cell_w = float(width) / grid_w
    cell_h = float(height) / grid_h
    cx = (float(col) + 0.5) * cell_w
    cy = (float(row) + 0.5) * cell_h
    sigma = max(0.2, float(config.effective_scales()[level]) * min(cell_w, cell_h) * 0.82)
    return cx, cy, max(1.0, 3.0 * sigma)


def tiles_for_kernel(config: SceneConfig, level: int, grid_size: tuple[int, int], row: int, col: int) -> list[tuple[int, int]]:
    width, height = config.image_size
    tile_size = config.tile_size
    tile_cols = int(math.ceil(width / tile_size))
    tile_rows = int(math.ceil(height / tile_size))
    cx, cy, radius = kernel_center_radius(config, level, grid_size, row, col)
    col0 = max(0, int(math.floor((cx - radius) / tile_size)))
    col1 = min(tile_cols - 1, int(math.floor((cx + radius) / tile_size)))
    row0 = max(0, int(math.floor((cy - radius) / tile_size)))
    row1 = min(tile_rows - 1, int(math.floor((cy + radius) / tile_size)))
    return [(tile_row, tile_col) for tile_row in range(row0, row1 + 1) for tile_col in range(col0, col1 + 1)]


def tile_kernel_refs(config: SceneConfig, level: int, grid_size: tuple[int, int], row: int, col: int, count: int) -> list[int]:
    refs: list[int] = []
    grid_w = grid_size[0]
    for index in range(count):
        kernel_row = index // grid_w
        kernel_col = index % grid_w
        if (row, col) in tiles_for_kernel(config, level, grid_size, kernel_row, kernel_col):
            refs.append(index)
    return refs
