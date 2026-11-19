from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import BlockPlan, CameraRays, sample_trilinear
from .spec import TaskSpec, TransferFunction


@dataclass(frozen=True)
class RenderedBlock:
    gray: np.ndarray  # (block_height, block_width), float64
    depth: np.ndarray  # (block_height, block_width), float64 millimeters
    has_nonzero_samples: bool
    sampled_ray_count: int


def render_block(
    task: TaskSpec,
    volume: np.ndarray,
    transfer_function: TransferFunction,
    rays: CameraRays,
    plan: BlockPlan,
) -> RenderedBlock:
    """Render one block using only immutable volume/TF/camera snapshots."""

    height = plan.y1 - plan.y0
    width = plan.x1 - plan.x0
    pixel_count = height * width
    gray = np.full((height, width), task.background_gray, dtype=np.float64)
    depth = np.full((height, width), rays.max_depth_mm, dtype=np.float64)

    directions = rays.directions[plan.y0 : plan.y1, plan.x0 : plan.x1].reshape(-1, 3)
    t_near = rays.t_near[plan.y0 : plan.y1, plan.x0 : plan.x1].reshape(-1).copy()
    t_far = rays.t_far[plan.y0 : plan.y1, plan.x0 : plan.x1].reshape(-1).copy()
    hit = rays.hit[plan.y0 : plan.y1, plan.x0 : plan.x1].reshape(-1)
    t_near = np.where(hit, t_near, 0.0)
    distances = np.where(hit, np.maximum(t_far - t_near, 0.0), 0.0)
    counts = np.floor(distances / task.sampling_step_mm + 1e-12).astype(np.int64) + 1
    counts = np.where(hit, counts, 0)
    max_steps = int(counts.max(initial=0))
    if max_steps == 0:
        return RenderedBlock(
            gray=gray,
            depth=depth,
            has_nonzero_samples=False,
            sampled_ray_count=0,
        )

    # The scheduler reserves 64 bytes per sample. These arrays stay below that
    # bound and are never shared between concurrently computing blocks.
    t_values = t_near[:, None] + task.sampling_step_mm * np.arange(
        max_steps, dtype=np.float64
    )[None, :]
    active = np.arange(max_steps, dtype=np.int64)[None, :] < counts[:, None]
    positions = rays.origin[None, None, :] + directions[:, None, :] * t_values[:, :, None]
    spacing = np.asarray(task.volume.spacing_mm, dtype=np.float64)
    positions /= spacing[None, None, :]
    scalars = sample_trilinear(volume, positions)

    nonzero_sample = active & (scalars != 0.0)
    has_nonzero = bool(np.any(nonzero_sample))
    opacity = np.interp(
        scalars,
        transfer_function.scalars,
        transfer_function.opacities,
        left=float(transfer_function.opacities[0]),
        right=float(transfer_function.opacities[-1]),
    )
    sample_gray = np.interp(
        scalars,
        transfer_function.scalars,
        transfer_function.grays,
        left=float(transfer_function.grays[0]),
        right=float(transfer_function.grays[-1]),
    )
    opacity = np.where(nonzero_sample, opacity, 0.0)
    sample_gray = np.where(nonzero_sample, sample_gray, 0.0)

    color = np.zeros(pixel_count, dtype=np.float64)
    transmittance = np.ones(pixel_count, dtype=np.float64)
    depth_set = np.zeros(pixel_count, dtype=bool)
    color_step = color
    transmittance_step = transmittance
    depth_out = depth.reshape(-1)
    for step in range(max_steps):
        step_opacity = opacity[:, step]
        contributing = active[:, step] & (step_opacity > 0.0)
        first = contributing & (~depth_set)
        depth_out[first] = t_values[first, step]
        depth_set |= first
        color_step = color_step + transmittance_step * step_opacity * sample_gray[:, step]
        transmittance_step = transmittance_step * (1.0 - step_opacity)
        if bool(np.all((~active[:, step]) | (transmittance_step <= 1e-14))):
            break

    gray[...] = np.clip(color_step.reshape(height, width), 0.0, 1.0)
    depth[...] = depth_out.reshape(height, width)
    return RenderedBlock(
        gray=gray,
        depth=depth,
        has_nonzero_samples=has_nonzero,
        sampled_ray_count=int(np.count_nonzero(counts)),
    )
