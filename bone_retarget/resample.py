"""Timeline resampling with explicit no-extrapolation behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import ErrorCode, PipelineError
from .math3d import interpolate_quaternions
from .models import Animation

BOUNDARY_EPSILON = 1e-12


def sample_animation(animation: Animation, sample_times: np.ndarray) -> Animation:
    requested = np.asarray(sample_times, dtype=np.float64)
    if requested.ndim != 1 or requested.size == 0:
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "sample times must be a non-empty one-dimensional array",
            field="sample_times",
        )
    if not np.all(np.isfinite(requested)):
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "sample times contain non-finite values",
            field="sample_times",
        )
    if requested.size > 1 and np.any(np.diff(requested) <= 0.0):
        frame = int(np.nonzero(np.diff(requested) <= 0.0)[0][0] + 1)
        raise PipelineError(
            ErrorCode.E_TIME_NONMONOTONIC,
            "sample times must be strictly increasing",
            frame=frame,
            field="sample_times",
        )
    source_start = float(animation.times[0])
    source_end = float(animation.times[-1])
    below = np.nonzero(requested < source_start - BOUNDARY_EPSILON)[0]
    if below.size:
        frame = int(below[0])
        raise PipelineError(
            ErrorCode.E_TIME_NONMONOTONIC,
            "sample time falls before the source timeline; extrapolation is forbidden",
            frame=frame,
            field="sample_times",
            details={"sample_time": float(requested[frame]), "source_start": source_start},
        )
    above = np.nonzero(requested > source_end + BOUNDARY_EPSILON)[0]
    if above.size:
        frame = int(above[0])
        raise PipelineError(
            ErrorCode.E_TIME_NONMONOTONIC,
            "sample time falls after the source timeline; extrapolation is forbidden",
            frame=frame,
            field="sample_times",
            details={"sample_time": float(requested[frame]), "source_end": source_end},
        )
    requested = np.clip(requested, source_start, source_end)
    rotations = np.empty(
        (requested.size, animation.joint_count, 4), dtype=np.float64
    )
    roots = np.empty((requested.size, 3), dtype=np.float64)
    for sample_index, time_value in enumerate(requested):
        left = int(np.searchsorted(animation.times, time_value, side="right") - 1)
        left = max(left, 0)
        right = min(left + 1, animation.frame_count - 1)
        if right == left:
            alpha = 0.0
        else:
            span = float(animation.times[right] - animation.times[left])
            alpha = float((time_value - animation.times[left]) / span)
        for joint_index in range(animation.joint_count):
            rotations[sample_index, joint_index] = interpolate_quaternions(
                animation.rotations[left, joint_index],
                animation.rotations[right, joint_index],
                alpha,
                animation.interpolation,
            )
        if animation.interpolation == "step":
            roots[sample_index] = animation.root_translation[left]
        else:
            roots[sample_index] = (
                (1.0 - alpha) * animation.root_translation[left]
                + alpha * animation.root_translation[right]
            )
    return Animation(
        joint_names=animation.joint_names,
        times=requested,
        rotations=rotations,
        root_translation=roots,
        interpolation=animation.interpolation,
        fps=animation.fps,
    )


@dataclass
class ResampleReport:
    source_frame_count: int
    target_frame_count: int
    source_start: float
    source_end: float
    target_start: float
    target_end: float
    target_fps: float
    extrapolated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_frame_count": self.source_frame_count,
            "target_frame_count": self.target_frame_count,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "target_start": self.target_start,
            "target_end": self.target_end,
            "target_fps": self.target_fps,
            "extrapolated": self.extrapolated,
        }


def resample_to_fps(animation: Animation, target_fps: float) -> tuple[Animation, ResampleReport]:
    fps = float(target_fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise PipelineError(
            ErrorCode.E_FIELD_MISSING,
            "target fps must be finite and positive",
            field="target_fps",
        )
    start = float(animation.times[0])
    end = float(animation.times[-1])
    duration = end - start
    count = int(np.floor(duration * fps + 1e-9)) + 1
    if count < 1:
        count = 1
    target_times = start + np.arange(count, dtype=np.float64) / fps
    target_times = target_times[target_times <= end + BOUNDARY_EPSILON]
    target_times = np.clip(target_times, start, end)
    sampled = sample_animation(animation, target_times)
    report = ResampleReport(
        source_frame_count=animation.frame_count,
        target_frame_count=sampled.frame_count,
        source_start=start,
        source_end=end,
        target_start=float(sampled.times[0]),
        target_end=float(sampled.times[-1]),
        target_fps=fps,
        extrapolated=False,
    )
    return Animation(
        joint_names=sampled.joint_names,
        times=sampled.times,
        rotations=sampled.rotations,
        root_translation=sampled.root_translation,
        interpolation=sampled.interpolation,
        fps=fps,
    ), report
