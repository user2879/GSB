"""Timeline resampling onto a fixed target frame rate.

No extrapolation: output samples cover ``[t_first, t_last]`` on a grid
``t = t_first + k / fps`` and samples past the last keyframe are dropped.
Interpolation follows the declared mode (``slerp``, ``nlerp`` or
``step``); root translation uses linear interpolation (or step).
"""

from __future__ import annotations

import numpy as np

from .animation import Animation, check_time_monotonic
from .quat import nlerp, normalize, slerp


def output_timestamps(times: np.ndarray, fps: float) -> np.ndarray:
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    check_time_monotonic(times)
    start = float(times[0])
    end = float(times[-1])
    count = int(np.floor((end - start) * fps + 1e-12)) + 1
    grid = start + np.arange(count, dtype=np.float64) / fps
    grid = grid[grid <= end + 1e-12]
    grid[-1] = end  # pin the boundary exactly, no extrapolation
    return grid


def _interp_quat(
    q0: np.ndarray, q1: np.ndarray, u: float, mode: str
) -> np.ndarray:
    if mode == "step":
        return normalize(q0)
    if mode == "nlerp":
        return nlerp(q0, q1, u)
    return slerp(q0, q1, u)


def resample_frame_block(
    source: Animation, fps: float, target_times: np.ndarray,
    frame_range: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Pure function: resample output frames ``[start, end)`` of the grid."""

    start, end = frame_range
    frames = end - start
    joints = source.joint_count
    rotations = np.zeros((frames, joints, 4), dtype=np.float64)
    roots = np.zeros((frames, 3), dtype=np.float64)
    times = np.zeros(frames, dtype=np.float64)

    src_times = source.times
    for local_frame, out_index in enumerate(range(start, end)):
        t = float(target_times[out_index])
        times[local_frame] = t
        right = int(np.searchsorted(src_times, t, side="left"))
        if right == 0:
            right = 1
        if right >= len(src_times):
            right = len(src_times) - 1
        left = right - 1
        span = float(src_times[right] - src_times[left])
        u = 0.0 if span == 0.0 else (t - float(src_times[left])) / span
        u = float(np.clip(u, 0.0, 1.0))
        for joint in range(joints):
            rotations[local_frame, joint] = _interp_quat(
                source.rotations[left, joint],
                source.rotations[right, joint],
                u,
                source.interp,
            )
        if source.interp == "step" or u == 0.0:
            roots[local_frame] = source.root_translations[left]
        else:
            roots[local_frame] = (
                (1.0 - u) * source.root_translations[left]
                + u * source.root_translations[right]
            )
    return {
        "times": times,
        "rotations": rotations,
        "root_translations": roots,
    }


def resample(source: Animation, fps: float) -> Animation:
    from .quat import align_sequence

    grid = output_timestamps(source.times, fps)
    block = resample_frame_block(source, fps, grid, (0, grid.size))
    return Animation(
        joint_names=source.joint_names,
        times=block["times"],
        rotations=align_sequence(block["rotations"]),
        root_translations=block["root_translations"],
        interp=source.interp,
        quat_order=source.quat_order,
        parameter_version=source.parameter_version,
    )
