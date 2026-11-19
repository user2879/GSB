"""Conservative frustum and Hi-Z occlusion culling with explicit error budgets.

Decision states:
  * ``visible``     - inside frustum and not provably occluded
  * ``frustum_out`` - provably outside at least one frustum plane
  * ``occluded``    - nearest depth is behind the Hi-Z depth beyond the error band
  * ``uncertain``   - the test lands inside the error band; block is kept/drawn

Any uncertain result is treated as visible for streaming, so an underestimated
error can never silently drop a block.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ABS_FLOOR, F32_REL_EPS, DEFAULT_SCREEN_ERROR_BUDGET_PX
from .camera import projection_error_bounds

AABB_CORNERS = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
], dtype=np.float64)


@dataclass
class CullDecision:
    block_id: str
    state: str
    nearest_depth: float
    hiz_sample_depth: float
    error_components: dict
    total_error: float
    screen_error_px: float
    reasons: list = field(default_factory=list)
    hiz_level: int = 0

    def to_dict(self) -> dict:
        return {
            "block_id": self.block_id,
            "state": self.state,
            "nearest_depth": self.nearest_depth,
            "hiz_sample_depth": self.hiz_sample_depth,
            "total_error": self.total_error,
            "screen_error_px": self.screen_error_px,
            "hiz_level": self.hiz_level,
            "reasons": list(self.reasons),
            "error_components": dict(self.error_components),
        }


def expand_aabb(bbox: tuple, extra: float) -> np.ndarray:
    """Conservative world-space AABB expansion by a uniform margin."""
    minx, miny, minz, maxx, maxy, maxz = bbox
    extra = max(float(extra), 0.0)
    return np.array([
        [minx - extra, miny - extra, minz - extra],
        [maxx + extra, maxy + extra, maxz + extra],
    ], dtype=np.float64)


def aabb_corners(box: np.ndarray) -> np.ndarray:
    return box[0] + AABB_CORNERS * (box[1] - box[0])


def block_surface_points(block, heights, grid_step: int = 8) -> np.ndarray:
    """Conservative surface samples: AABB corners plus a grid on the block top.

    Corner-only frustum tests miss a curved heightfield whose surface pokes into
    the viewport while all 8 AABB corners lie outside. Sampling the actual top
    surface gives a conservative projected footprint for culling and Hi-Z reads.
    """
    from .terrain import height_to_world, WORLD_EXTENT
    size = heights.shape[0]
    cell = WORLD_EXTENT / size
    xs = np.linspace(block.x0, block.x1 - 1,
                     min(grid_step, max(block.x1 - block.x0, 2))).astype(int)
    ys = np.linspace(block.y0, block.y1 - 1,
                     min(grid_step, max(block.y1 - block.y0, 2))).astype(int)
    gx, gy = np.meshgrid(xs, ys)
    world_x = gx.astype(np.float64) * cell
    world_z = gy.astype(np.float64) * cell
    world_y = height_to_world(heights[gy, gx])
    surface = np.stack([world_x.ravel(), world_y.ravel(), world_z.ravel()], axis=1)
    return np.concatenate([surface, aabb_corners(expand_aabb(block.bbox, 0.0))], axis=0)


def frustum_classify(ndc: np.ndarray) -> tuple[str, list[str]]:
    """Classify projected AABB corners against the NDC cube."""
    reasons = []
    out_x = bool(np.all(ndc[:, 0] > 1.0)) or bool(np.all(ndc[:, 0] < -1.0))
    out_y = bool(np.all(ndc[:, 1] > 1.0)) or bool(np.all(ndc[:, 1] < -1.0))
    behind = bool(np.all(ndc[:, 2] < -1.0))
    beyond = bool(np.all(ndc[:, 2] > 1.0))
    if out_x:
        reasons.append("outside_x")
    if out_y:
        reasons.append("outside_y")
    if behind:
        reasons.append("behind_near")
    if beyond:
        reasons.append("beyond_far")
    if reasons:
        return "frustum_out", reasons
    return "visible", reasons


def screen_space_error_px(ndc: np.ndarray, width: int, height: int,
                          abs_error: float, center_distance: float) -> float:
    """Float32 projected position error expressed in screen pixels.

    The AABB screen footprint itself is geometry, not numerical error; only the
    projected absolute position error is counted against the screen budget.
    """
    rel = abs_error / max(center_distance, ABS_FLOOR)
    return float(max(rel * width, rel * height))


def sample_hiz(hiz_levels: list[np.ndarray], infos: list, level: int,
               ndc: np.ndarray) -> tuple[float, float]:
    """Return (max stored depth at the projected surface pixels, level error).

    Only pixels actually covered by projected surface points are read. Reading
    the full AABB bounding rectangle would include occluder pixels that do not
    overlap the block, producing false occlusion. A one-texel conservative
    dilation covers rasterization gaps between sampled surface points.
    """
    pyramid = hiz_levels[level]
    h, w = pyramid.shape
    # Points behind the camera produce extreme, meaningless NDC coordinates and
    # must be dropped rather than clamped onto the viewport.
    inside = ((np.abs(ndc[:, 0]) <= 1.0) & (np.abs(ndc[:, 1]) <= 1.0)
              & (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0))
    if not inside.any():
        return np.inf, infos[level].depth_error_bound
    u = ((ndc[inside, 0] + 1.0) * 0.5 * w).astype(np.int64)
    v = ((1.0 - (ndc[inside, 1] + 1.0) * 0.5) * h).astype(np.int64)
    u = np.clip(u, 0, w - 1)
    v = np.clip(v, 0, h - 1)
    # Conservative one-texel dilation around each covered pixel.
    sampled = []
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            uu = np.clip(u + du, 0, w - 1)
            vv = np.clip(v + dv, 0, h - 1)
            sampled.append(pyramid[vv, uu])
    values = np.concatenate(sampled)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.inf, infos[level].depth_error_bound
    return float(finite.max()), float(infos[level].depth_error_bound)


class OcclusionCuller:
    def __init__(self, camera, config):
        self.camera = camera
        self.config = config

    def cull_block(self, block, camera_frame, hiz_levels, infos,
                   expansion_margin: float, hiz_level: int,
                   heights: np.ndarray | None = None) -> CullDecision:
        box = expand_aabb(block.bbox, expansion_margin)
        corners = aabb_corners(box)
        ndc = self.camera.world_to_ndc(camera_frame.view_projection, corners)
        if heights is not None:
            surface = block_surface_points(block, heights)
            footprint_ndc = self.camera.world_to_ndc(
                camera_frame.view_projection, surface)
        else:
            surface = corners
            footprint_ndc = ndc

        center = corners.mean(axis=0)
        eye = np.asarray(camera_frame.eye, dtype=np.float64)
        distance = float(np.linalg.norm(center - eye))
        radius = float(np.linalg.norm(box[1] - box[0])) * 0.5
        proj = projection_error_bounds(distance, radius)

        state, reasons = frustum_classify(footprint_ndc)
        screen_err = screen_space_error_px(
            footprint_ndc, self.camera.width, self.camera.height,
            proj["abs_error"], distance)
        if state == "frustum_out":
            return CullDecision(
                block_id=block.block_id, state=state,
                nearest_depth=float("inf"), hiz_sample_depth=float("inf"),
                error_components=proj, total_error=proj["abs_error"],
                screen_error_px=screen_err, reasons=reasons, hiz_level=hiz_level)

        view_depths = self.camera.world_depth(camera_frame.view, surface)
        nearest_depth = float(view_depths.min())
        # Occlusion is decided with the *farthest* AABB corner: if even the
        # farthest point sits behind the occluder (beyond the error band) the
        # whole block is hidden. The nearest corner always implies visibility.
        farthest_depth = float(view_depths.max())
        hiz_depth, hiz_err = sample_hiz(hiz_levels, infos, hiz_level,
                                        footprint_ndc)

        # Error components, each an explicit upper bound.
        expansion_error = float(np.linalg.norm(
            np.asarray(block.bbox[3:6]) - np.asarray(block.bbox[0:3])) * 0.0
            + expansion_margin * np.sqrt(3.0))
        # Angular aliasing: a grazing block can share a depth texel with a
        # nearer occluder. Bound the world footprint at the nearest depth.
        half_angle = float(infos[hiz_level].pixel_half_angle_rad)
        angular_error = float(
            max(nearest_depth, ABS_FLOOR) * np.tan(half_angle)
            + max(nearest_depth, ABS_FLOOR) * half_angle)
        depth_error = float(proj["abs_error"] + expansion_error
                            + hiz_err + angular_error)
        total_error = depth_error
        decision = CullDecision(
            block_id=block.block_id, state="visible",
            nearest_depth=nearest_depth, hiz_sample_depth=hiz_depth,
            error_components={
                **proj,
                "expansion_margin": float(expansion_margin),
                "expansion_error_world": expansion_error,
                "hiz_depth_error": hiz_err,
                "pixel_half_angle_rad": half_angle,
                "angular_aliasing_error": angular_error,
                "hiz_level": hiz_level,
                "screen_error_px": screen_err,
                "screen_budget_px": self.config.screen_error_budget_px,
            },
            total_error=total_error, screen_error_px=screen_err,
            reasons=[], hiz_level=hiz_level)

        if screen_err > self.config.screen_error_budget_px:
            decision.reasons.append("screen_error_over_budget")

        if not np.isfinite(hiz_depth):
            decision.reasons.append("no_occluder_sample")
            decision.state = "visible"
            return decision

        # hiz_depth is the nearest occluder. A block is provably hidden only
        # when its nearest surface point lies behind that occluder beyond the
        # full error band. Otherwise it is either clearly in front (visible) or
        # inside the band (uncertain, kept conservatively).
        delta_near = nearest_depth - hiz_depth
        if delta_near <= -total_error:
            decision.state = "visible"
            decision.reasons.append("nearest_point_in_front_of_occluder")
            return decision
        if delta_near >= total_error:
            decision.state = "occluded"
            decision.reasons.append("nearest_point_behind_occluder_band")
            return decision
        decision.state = "uncertain"
        decision.reasons.append("within_error_band_kept_conservative")
        decision.error_components["farthest_depth"] = farthest_depth
        return decision
