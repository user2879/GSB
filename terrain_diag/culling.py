from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .camera import ProjectionResult
from .hiz import DepthRender, HiZPyramid, validate_depth_bias
from .quadtree import QuadtreeManifest


@dataclass(frozen=True)
class CullingPolicy:
    hiz_level_bias: int = 0
    extra_screen_expand_px: int = 0
    extra_world_expand: float = 0.0
    depth_bias: float = 0.0

    def __post_init__(self) -> None:
        if not 0 <= int(self.hiz_level_bias) <= 4:
            raise ValueError("hiz_level_bias must be in [0, 4]")
        if not 0 <= int(self.extra_screen_expand_px) <= 8:
            raise ValueError("extra_screen_expand_px must be in [0, 8]")
        if float(self.extra_world_expand) < 0.0:
            raise ValueError("extra_world_expand must be non-negative")
        validate_depth_bias(self.depth_bias)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hiz_level_bias": self.hiz_level_bias,
            "extra_screen_expand_px": self.extra_screen_expand_px,
            "extra_world_expand": self.extra_world_expand,
            "depth_bias": self.depth_bias,
        }


@dataclass(frozen=True)
class BlockDecision:
    block_id: str
    status: str
    reason: str
    parameter_stamp: str
    projection: ProjectionResult | None
    owner_pixels: int
    hiz_level: int
    hiz_delta: float | None
    errors: dict[str, float]
    input_parameters: dict[str, Any]
    hiz_lookup: dict[str, Any] | None

    def to_report(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "status": self.status,
            "reason": self.reason,
            "parameter_stamp": self.parameter_stamp,
            "owner_pixels": self.owner_pixels,
            "hiz_level": self.hiz_level,
            "hiz_delta": self.hiz_delta,
            "errors": self.errors,
            "input": self.input_parameters,
            "projection": self.projection.to_report() if self.projection is not None else None,
            "hiz_lookup": self.hiz_lookup,
        }


@dataclass(frozen=True)
class CullingFrameResult:
    decisions: tuple[BlockDecision, ...]
    visible_block_ids: tuple[str, ...]
    visible_count: int
    frustum_culled_count: int
    occlusion_culled_count: int
    uncertain_count: int
    error_values: tuple[float, ...]
    max_error_bound: float
    mean_error_bound: float
    max_hiz_level_used: int

    def to_stats(self) -> dict[str, Any]:
        return {
            "total_blocks": len(self.decisions),
            "visible_blocks": self.visible_count,
            "frustum_culled_blocks": self.frustum_culled_count,
            "occlusion_culled_blocks": self.occlusion_culled_count,
            "culled_blocks": self.frustum_culled_count + self.occlusion_culled_count,
            "uncertain_blocks": self.uncertain_count,
        }


def _hiz_level_for_bbox(
    projection: ProjectionResult,
    pyramid: HiZPyramid,
    bias: int,
) -> int:
    left, top, right, bottom = projection.screen_bbox
    extent = max(1, right - left, bottom - top)
    base_level = int(math.floor(math.log2(max(1.0, extent / 4.0))))
    return min(pyramid.max_level, max(0, base_level + int(bias)))


def _hiz_lookup_bounds(
    projection: ProjectionResult,
    pyramid: HiZPyramid,
    level: int,
    expand_px: int,
) -> tuple[int, int, int, int]:
    level_data = pyramid.levels[level]
    base_width = pyramid.base_width
    base_height = pyramid.base_height
    left, top, right, bottom = projection.screen_bbox
    left = max(0, left - expand_px)
    right = min(base_width, right + expand_px)
    top = max(0, top - expand_px)
    bottom = min(base_height, bottom + expand_px)
    scale_x = level_data.width / float(base_width)
    scale_y = level_data.height / float(base_height)
    x0 = max(0, min(level_data.width - 1, int(math.floor(left * scale_x))))
    x1 = max(x0 + 1, min(level_data.width, int(math.ceil(right * scale_x))))
    y0 = max(0, min(level_data.height - 1, int(math.floor(top * scale_y))))
    y1 = max(y0 + 1, min(level_data.height, int(math.ceil(bottom * scale_y))))
    return x0, y0, x1, y1


def _owner_pixels_in_bbox(
    depth_render: DepthRender,
    projection: ProjectionResult,
    block_id: str,
    expand_px: int,
) -> int:
    if block_id not in depth_render.block_ordinals:
        return 0
    ordinal = depth_render.block_ordinals.index(block_id)
    left, top, right, bottom = projection.screen_bbox
    left = max(0, left - expand_px)
    right = min(depth_render.width, right + expand_px)
    top = max(0, top - expand_px)
    bottom = min(depth_render.height, bottom + expand_px)
    if left >= right or top >= bottom:
        return 0
    region = depth_render.owner[top:bottom, left:right]
    return int(np.count_nonzero(region == ordinal))


def cull_blocks(
    manifest: QuadtreeManifest,
    projections: dict[str, ProjectionResult],
    depth_render: DepthRender,
    pyramid: HiZPyramid,
    policy: CullingPolicy,
    parameter_stamp: str,
    base_world_expand: float,
) -> CullingFrameResult:
    decisions: list[BlockDecision] = []
    visible: list[str] = []
    error_values: list[float] = []
    frustum_culled = 0
    occlusion_culled = 0
    uncertain_count = 0
    max_level_used = 0

    for node in sorted(manifest.leaves, key=lambda item: item.block_id or ""):
        if node.block_id is None:
            continue
        projection = projections[node.block_id]
        owner_pixels = _owner_pixels_in_bbox(
            depth_render,
            projection,
            node.block_id,
            policy.extra_screen_expand_px,
        )
        input_parameters = {
            "height_min_raw": node.height_min_raw,
            "height_max_raw": node.height_max_raw,
            "height_range_raw": node.height_range_raw,
            "height_range_world": node.height_range_world,
            "bounds": node.bounds.to_list(),
            "base_world_expand": base_world_expand,
            "extra_world_expand": policy.extra_world_expand,
            "extra_screen_expand_px": policy.extra_screen_expand_px,
            "depth_bias": policy.depth_bias,
            "hiz_level_bias": policy.hiz_level_bias,
        }

        if not projection.intersects_frustum:
            frustum_culled += 1
            decisions.append(
                BlockDecision(
                    block_id=node.block_id,
                    status="FRUSTUM_CULLED",
                    reason="frustum_outside",
                    parameter_stamp=parameter_stamp,
                    projection=projection,
                    owner_pixels=owner_pixels,
                    hiz_level=0,
                    hiz_delta=None,
                    errors={
                        "projection_float": 0.0,
                        "bbox_expand": 0.0,
                        "hiz_sampling": 0.0,
                        "depth_bias": 0.0,
                        "total": 0.0,
                    },
                    input_parameters=input_parameters,
                    hiz_lookup=None,
                )
            )
            continue

        level = _hiz_level_for_bbox(projection, pyramid, policy.hiz_level_bias)
        max_level_used = max(max_level_used, level)
        x0, y0, x1, y1 = _hiz_lookup_bounds(projection, pyramid, level, policy.extra_screen_expand_px)
        level_data = pyramid.levels[level]
        hiz_depth = float(level_data.depth[y0:y1, x0:x1].max())

        left, top, right, bottom = projection.screen_bbox
        pixel_height = max(1.0, float(bottom - top))
        depth_span = max(1.0e-9, abs(projection.far_depth - projection.near_depth))
        depth_per_pixel = depth_span / pixel_height

        projection_component = (
            projection.error_components_px["float_roundoff_px"]
            + projection.error_components_px["subpixel_footprint_px"]
        ) * depth_per_pixel
        bbox_expand_component = (
            policy.extra_screen_expand_px * depth_per_pixel
            + (base_world_expand + policy.extra_world_expand)
            / max(1.0, projection.center_distance)
        )
        hiz_component = float(level_data.depth_error_upper_bound)
        total_error = (
            projection_component
            + bbox_expand_component
            + hiz_component
            + float(policy.depth_bias)
        )
        near_depth = projection.near_depth - (
            policy.extra_screen_expand_px * depth_per_pixel
            + (base_world_expand + policy.extra_world_expand) / max(1.0, projection.center_distance)
        )
        near_depth = max(0.0, float(near_depth))
        delta = float(near_depth - hiz_depth)
        error_values.append(float(total_error))
        lookup = {
            "level_width": level_data.width,
            "level_height": level_data.height,
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "hiz_depth": hiz_depth,
            "near_depth": near_depth,
            "threshold_depth": hiz_depth + total_error,
        }
        errors = {
            "projection_float": float(projection_component),
            "bbox_expand": float(bbox_expand_component),
            "hiz_sampling": float(hiz_component),
            "depth_bias": float(policy.depth_bias),
            "total": float(total_error),
        }

        if owner_pixels > 0:
            status = "VISIBLE"
            reason = "base_depth_owner_evidence"
        elif left >= right or top >= bottom:
            status = "UNCERTAIN"
            reason = "empty_screen_footprint_retained"
            uncertain_count += 1
        elif (right - left) * (bottom - top) < 4:
            status = "UNCERTAIN"
            reason = "small_screen_footprint_retained"
            uncertain_count += 1
        elif delta > total_error:
            status = "OCCLUDED"
            reason = "beyond_hiz_error_upper_bound"
        elif delta < -total_error:
            status = "VISIBLE"
            reason = "in_front_of_hiz_error_lower_bound"
        else:
            status = "UNCERTAIN"
            reason = "inside_hiz_error_interval_retained"
            uncertain_count += 1

        if status in ("VISIBLE", "UNCERTAIN"):
            visible.append(node.block_id)
        elif status == "OCCLUDED":
            occlusion_culled += 1

        decisions.append(
            BlockDecision(
                block_id=node.block_id,
                status=status,
                reason=reason,
                parameter_stamp=parameter_stamp,
                projection=projection,
                owner_pixels=owner_pixels,
                hiz_level=level,
                hiz_delta=delta,
                errors=errors,
                input_parameters=input_parameters,
                hiz_lookup=lookup,
            )
        )

    visible_ids = tuple(sorted(visible))
    max_error = max(error_values) if error_values else 0.0
    mean_error = float(sum(error_values) / len(error_values)) if error_values else 0.0
    return CullingFrameResult(
        decisions=tuple(decisions),
        visible_block_ids=visible_ids,
        visible_count=len(visible_ids),
        frustum_culled_count=frustum_culled,
        occlusion_culled_count=occlusion_culled,
        uncertain_count=uncertain_count,
        error_values=tuple(error_values),
        max_error_bound=float(max_error),
        mean_error_bound=float(mean_error),
        max_hiz_level_used=int(max_level_used),
    )


def projection_trigger_blocks(
    projections: dict[str, ProjectionResult],
    budget_px: float,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            block_id
            for block_id, result in projections.items()
            if result.intersects_frustum and result.projection_error_px > budget_px
        )
    )
