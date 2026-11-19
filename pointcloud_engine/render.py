"""Point projection, occlusion compositing and tile rendering.

Determinism contract
--------------------
* Points are always visited in ascending point-cloud index (file order).
* Within a tile, affected samples are visited in deterministic row/column
  order and a nearer point wins only on a **strictly smaller** depth, so an
  equal-depth tie keeps the earlier point - exactly matching serial output.
* Tiles own disjoint sample buffers, therefore thread scheduling order can
  never change the resulting image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import Camera
from .errors import EngineError
from .scene import Scene


@dataclass
class PreparedPoints:
    index: np.ndarray          # original point indices that survive culling
    center_x: np.ndarray       # supersample-grid center coordinate (float)
    center_y: np.ndarray
    radius_px: np.ndarray      # footprint radius in supersample pixels
    depth: np.ndarray          # positive view-axis distance (world units)
    gray: np.ndarray           # unattenuated grayscale [0,1]
    min_x: np.ndarray          # inclusive integer bounds on the grid
    max_x: np.ndarray
    min_y: np.ndarray
    max_y: np.ndarray
    total_points: int
    culled_frustum: int
    behind_near: int


def prepare_points(scene: Scene) -> PreparedPoints:
    """Project every world point and derive its supersample footprint."""
    assert scene.cloud is not None and scene.camera is not None
    camera: Camera = scene.camera
    world = scene.world_points()
    total = world.shape[0]

    cam = camera.to_camera(world)
    depth = -cam[:, 2]
    behind_near = int(np.count_nonzero(depth <= camera.near))

    ndc_x, ndc_y, depth_pos = camera.project(cam)
    radius_px = camera.screen_radius(scene.point_radius, depth, scene.grid_height * 0.5)

    center_x = (ndc_x + 1.0) * 0.5 * scene.grid_width
    center_y = (1.0 - ndc_y) * 0.5 * scene.grid_height

    # Non-finite coordinates (behind-camera points) get safe sentinel bounds;
    # they are removed by the visibility mask before any splatting.
    with np.errstate(invalid="ignore"):
        min_x = np.floor(np.nan_to_num(center_x - radius_px, nan=-1e18)).astype(np.int64)
        max_x = np.ceil(np.nan_to_num(center_x + radius_px, nan=-1e18)).astype(np.int64)
        min_y = np.floor(np.nan_to_num(center_y - radius_px, nan=-1e18)).astype(np.int64)
        max_y = np.ceil(np.nan_to_num(center_y + radius_px, nan=-1e18)).astype(np.int64)

    visible = depth > camera.near
    in_grid = (
        (max_x >= 0)
        & (min_x < scene.grid_width)
        & (max_y >= 0)
        & (min_y < scene.grid_height)
    )
    keep = visible & in_grid & np.isfinite(center_x) & np.isfinite(center_y) & np.isfinite(radius_px)
    culled_frustum = int(np.count_nonzero(~keep) - behind_near)

    idx = np.nonzero(keep)[0].astype(np.int64)
    return PreparedPoints(
        index=idx,
        center_x=center_x[keep],
        center_y=center_y[keep],
        radius_px=radius_px[keep],
        depth=depth[keep],
        gray=scene.cloud.grays[keep],
        min_x=min_x[keep],
        max_x=max_x[keep],
        min_y=min_y[keep],
        max_y=max_y[keep],
        total_points=total,
        culled_frustum=max(culled_frustum, 0),
        behind_near=behind_near,
    )


def _attenuated_gray(scene: Scene, gray: np.ndarray, depth: np.ndarray) -> np.ndarray:
    # Normalize depth by the camera-to-target reference distance so that a
    # uniform world-scale change leaves attenuation (and gray ordering)
    # invariant while depth occlusion stays in unified world units.
    assert scene.camera is not None
    normalized_depth = depth / scene.camera.reference_distance
    factor = scene.decay.factor(normalized_depth)
    return np.clip(gray * factor, 0.0, 1.0).astype(np.float32)


def _splat_into_tile(
    scene: Scene,
    prepared: PreparedPoints,
    point_indices: np.ndarray,
    tile_x0: int,
    tile_y0: int,
    tile_w: int,
    tile_h: int,
):
    """Composite the given points into one private tile buffer.

    Returns ``(gray_buffer, depth_buffer, hit_count)``. The same routine is
    used by the serial baseline (one tile spanning the whole grid) and by the
    parallel scheduler, guaranteeing pixel-identical results.
    """
    gray_buf = np.zeros((tile_h, tile_w), dtype=np.float32)
    depth_buf = np.zeros((tile_h, tile_w), dtype=np.float32)
    hit_count = 0

    tile_x1 = tile_x0 + tile_w
    tile_y1 = tile_y0 + tile_h

    for order, pi in enumerate(point_indices):
        px0 = max(int(prepared.min_x[pi]), tile_x0)
        px1 = min(int(prepared.max_x[pi]), tile_x1 - 1)
        py0 = max(int(prepared.min_y[pi]), tile_y0)
        py1 = min(int(prepared.max_y[pi]), tile_y1 - 1)
        if px0 > px1 or py0 > py1:
            continue

        cx = float(prepared.center_x[pi])
        cy = float(prepared.center_y[pi])
        r = float(prepared.radius_px[pi])
        point_depth = float(prepared.depth[pi])
        r2 = r * r

        local_x = np.arange(px0 - tile_x0, px1 - tile_x0 + 1, dtype=np.int64)
        global_x = local_x.astype(np.float64) + tile_x0
        dx2 = (global_x - cx) ** 2
        for ly in range(py0 - tile_y0, py1 - tile_y0 + 1):
            gy = ly + tile_y0
            dy2 = (gy - cy) ** 2
            inside = (dx2 + dy2) <= r2
            if not np.any(inside):
                continue
            lx_hits = local_x[inside]
            current_depth = depth_buf[ly, lx_hits]
            unwritten = current_depth == 0.0
            nearer = point_depth < current_depth
            update = unwritten | nearer
            if not np.any(update):
                continue
            target_lx = lx_hits[update]
            value = float(_attenuated_gray(scene, prepared.gray[pi:pi + 1],
                                           prepared.depth[pi:pi + 1])[0])
            depth_before = depth_buf[ly, target_lx]
            gray_buf[ly, target_lx] = value
            depth_buf[ly, target_lx] = np.float32(point_depth)
            hit_count += int(np.count_nonzero(depth_before == 0.0))

    return gray_buf, depth_buf, int(hit_count)


def render_tile(scene: Scene, prepared: PreparedPoints, tile_x0: int, tile_y0: int) -> dict:
    """Render one tile using only points whose footprint overlaps it."""
    tw, th = scene.tile_width, scene.tile_height
    overlaps = (
        (prepared.max_x >= tile_x0)
        & (prepared.min_x < tile_x0 + tw)
        & (prepared.max_y >= tile_y0)
        & (prepared.min_y < tile_y0 + th)
    )
    compact_positions = np.nonzero(overlaps)[0].astype(np.int64)
    gray_buf, depth_buf, hit_count = _splat_into_tile(
        scene, prepared, compact_positions, tile_x0, tile_y0, tw, th
    )
    return {
        "gray": gray_buf,
        "depth": depth_buf,
        "hit_count": hit_count,
        "tile_x0": tile_x0,
        "tile_y0": tile_y0,
    }


def render_serial(scene: Scene, prepared: PreparedPoints):
    """Whole-grid serial baseline used as the deterministic ground truth."""
    gray, depth, hit_count = _splat_into_tile(
        scene,
        prepared,
        np.arange(prepared.index.shape[0], dtype=np.int64),
        0,
        0,
        scene.grid_width,
        scene.grid_height,
    )
    return gray, depth, hit_count


def tile_origin(scene: Scene, tile_index: int) -> tuple[int, int]:
    ty = tile_index // scene.tiles_x
    tx = tile_index % scene.tiles_x
    return tx * scene.tile_width, ty * scene.tile_height


def downsample(scene: Scene, grid_gray: np.ndarray, grid_depth: np.ndarray):
    """Box-average supersample buffers into the final resolution images."""
    h, w = scene.height, scene.width
    s = scene.supersample
    gray = grid_gray.reshape(h, s, w, s).mean(axis=(1, 3)).astype(np.float32)
    depth = grid_depth.reshape(h, s, w, s).mean(axis=(1, 3)).astype(np.float32)
    return gray, depth


def inject_runtime_error(scene: Scene, tile_index: int) -> None:
    for directive in scene.injections:
        if directive["type"] == "tile_runtime_error":
            target = int(directive.get("tile_index", 0))
            if tile_index == target:
                raise EngineError(
                    "E_TILE_RUNTIME",
                    f"injected runtime failure at tile {tile_index}",
                    tile_index=tile_index,
                )
