"""Incremental retargeting with explicit invalidation closures.

Cached artifacts are keyed per frame block *and per channel*
(``rotation`` / ``root``).  A change request produces a closure manifest
that names exactly the blocks and channels that must be recomputed;
everything outside the closure is reused byte-for-byte.  Recomputing a
block whose signature already matches the cache raises
``E_CLOSURE_MISMATCH`` when the caller claims a task outside the
closure produced different content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .animation import Animation, check_time_monotonic
from .budget import Budget
from .hashutil import hash_dict
from .pipeline import _merge
from .quat import align_sequence
from .resample import output_timestamps, resample_frame_block
from .retarget import parameter_version, retarget_frame_block
from .skeleton import Skeleton, rest_pose_delta, root_bone_scale

BLOCK_SIZE = 4


@dataclass(frozen=True)
class ClosureManifest:
    reason: str
    retarget_blocks: tuple[int, ...]
    retarget_channels: tuple[str, ...]
    resample_blocks: tuple[int, ...]
    resample_channels: tuple[str, ...]
    affected_joints: tuple[str, ...]
    affected_source_frames: tuple[int, ...]

    def signature(self) -> str:
        return hash_dict(
            {
                "reason": self.reason,
                "rt_blocks": list(self.retarget_blocks),
                "rt_channels": list(self.retarget_channels),
                "rs_blocks": list(self.resample_blocks),
                "rs_channels": list(self.resample_channels),
                "joints": list(self.affected_joints),
                "frames": list(self.affected_source_frames),
            }
        )[:16]


@dataclass
class IncrementalState:
    source: Animation
    source_skel: Skeleton
    target_skel: Skeleton
    fps: float
    budget: Budget
    workers: int
    block_size: int
    rt_cache: dict[tuple[int, str], dict[str, Any]] = field(default_factory=dict)
    rs_cache: dict[tuple[int, str], dict[str, Any]] = field(default_factory=dict)
    recomputed: list[dict[str, Any]] = field(default_factory=list)
    manifest: ClosureManifest | None = None
    last_output: Animation | None = None


def _blocks_for_frames(frames: list[int], block_size: int) -> tuple[int, ...]:
    if not frames:
        return tuple()
    keys = {frame - (frame % block_size) for frame in frames}
    return tuple(sorted(keys))


def _resample_block_keys(
    state: IncrementalState, fps: float | None = None
) -> tuple[int, ...]:
    grid_size = _expected_grid_size(state, fps)
    return tuple(range(0, grid_size, state.block_size))


def _expected_grid_size(state: IncrementalState, fps: float | None = None) -> int:
    from .resample import output_timestamps

    effective_fps = state.fps if fps is None else fps
    return int(output_timestamps(state.source.times, effective_fps).size)


def plan_closure(
    state: IncrementalState,
    *,
    reason: str,
    affected_source_frames: list[int] | None = None,
    affected_joints: list[str] | None = None,
    fps_changed: bool = False,
    bone_length_changed: bool = False,
    next_fps: float | None = None,
) -> ClosureManifest:
    frames = list(range(state.source.frame_count))
    rt_blocks: tuple[int, ...]
    rt_channels: tuple[str, ...]
    if bone_length_changed:
        rt_blocks = tuple(range(0, state.source.frame_count, state.block_size))
        rt_channels = ("root",)
    else:
        rt_frames = affected_source_frames if affected_source_frames is not None else frames
        rt_blocks = _blocks_for_frames(rt_frames, state.block_size)
        rt_channels = ("rotation", "root")

    if fps_changed or bone_length_changed:
        rs_blocks = _resample_block_keys(state, next_fps)
        rs_channels = (
            ("rotation", "root", "time") if fps_changed else ("root",)
        )
    elif reason == "baseline":
        rs_blocks = _resample_block_keys(state)
        rs_channels = ("rotation", "root")
    else:
        if affected_source_frames:
            first_source = min(affected_source_frames)
            first_time = float(state.source.times[first_source])
            grid_size = _expected_grid_size(state, next_fps)
            grid = output_timestamps(state.source.times, state.fps if next_fps is None else next_fps)
            affected_output = [
                index for index, value in enumerate(grid) if value >= first_time - 1e-12
            ]
            rs_blocks = _blocks_for_frames(affected_output, state.block_size)
            # The first affected output block may still interpolate against
            # the bracketing keyframe before the edited time; include the
            # preceding block so the closure covers the sample window.
            if rs_blocks:
                predecessor = rs_blocks[0] - state.block_size
                if predecessor >= 0:
                    rs_blocks = (predecessor,) + rs_blocks
        else:
            rs_blocks = rt_blocks
        rs_channels = ("rotation", "root")

    joint_names = (
        tuple(affected_joints)
        if affected_joints is not None
        else tuple(state.source.joint_names)
    )
    return ClosureManifest(
        reason=reason,
        retarget_blocks=rt_blocks,
        retarget_channels=rt_channels,
        resample_blocks=rs_blocks,
        resample_channels=rs_channels,
        affected_joints=joint_names,
        affected_source_frames=tuple(
            sorted(affected_source_frames)
            if affected_source_frames is not None
            else frames
        ),
    )


def _rt_block_signature(
    source: Animation, target_skel: Skeleton, frame_range: tuple[int, int], tag: str
) -> str:
    start, end = frame_range
    return hash_dict(
        {
            "tag": tag,
            "range": [start, end],
            "times": source.times[start:end].tolist(),
            "rotations": source.rotations[start:end].tolist(),
            "root": source.root_translations[start:end].tolist(),
            "target_bone": target_skel.bone_lengths.tolist(),
        }
    )[:16]


def _rs_block_signature(
    retargeted: Animation,
    grid: np.ndarray,
    frame_range: tuple[int, int],
    fps: float,
) -> str:
    start, end = frame_range
    first_t = float(grid[start])
    last_t = float(grid[end - 1])
    # Only the keyframe window the block can actually sample (plus the
    # bracketing keys on each side) participates in the signature.
    right = int(np.searchsorted(retargeted.times, last_t, side="right"))
    left = max(0, int(np.searchsorted(retargeted.times, first_t, side="left")) - 1)
    right = min(retargeted.frame_count, right + 1)
    return hash_dict(
        {
            "fps": fps,
            "grid": grid[start:end].tolist(),
            "window": [left, right],
            "src_times": retargeted.times[left:right].tolist(),
            "src_rotations": retargeted.rotations[left:right].tolist(),
            "src_root": retargeted.root_translations[left:right].tolist(),
            "interp": retargeted.interp,
        }
    )[:16]


def run_incremental(
    state: IncrementalState,
    manifest: ClosureManifest,
    *,
    updated_source: Animation | None = None,
    updated_target_skel: Skeleton | None = None,
    updated_fps: float | None = None,
) -> Animation:
    source = updated_source or state.source
    target_skel = updated_target_skel or state.target_skel
    fps = updated_fps if updated_fps is not None else state.fps
    check_time_monotonic(source.times)

    scale = root_bone_scale(state.source_skel, target_skel)
    rest_delta = rest_pose_delta(state.source_skel, target_skel)
    tag = parameter_version(state.source_skel, target_skel, scale)

    recomputed: list[dict[str, Any]] = []
    total = source.frame_count
    block_size = state.block_size

    # ---- retarget stage, channel-aware cache ------------------------------
    rt_slots: dict[int, dict[str, np.ndarray]] = {}
    bone_only = manifest.reason == "target_bone_length_change"
    for start in range(0, total, block_size):
        end = min(start + block_size, total)
        frame_range = (start, end)
        signature = _rt_block_signature(source, target_skel, frame_range, tag)
        cached = state.rt_cache.get(start)
        block: dict[str, np.ndarray]
        if bone_only and cached is not None:
            # Rotations do not depend on bone lengths; only root scales do.
            block = {key: value.copy() for key, value in cached["data"].items()}
            block["root_translations"] = (
                source.root_translations[start:end] * scale
            )
            recomputed.append(
                {
                    "stage": "retarget",
                    "block": start,
                    "channels": ["root"],
                    "reason": manifest.reason,
                }
            )
        elif (
            cached is not None
            and cached["signature"] == signature
            and start not in manifest.retarget_blocks
        ):
            block = {key: value.copy() for key, value in cached["data"].items()}
        else:
            block = retarget_frame_block(
                source,
                state.source_skel,
                target_skel,
                rest_delta,
                scale,
                frame_range,
            )
            state.rt_cache[start] = {"signature": signature, "data": block}
            recomputed.append(
                {
                    "stage": "retarget",
                    "block": start,
                    "channels": list(manifest.retarget_channels),
                    "reason": manifest.reason,
                }
            )
        rt_slots[start] = block

    rt_rotations = align_sequence(_merge(total, rt_slots, "rotations"))
    rt_roots = _merge(total, rt_slots, "root_translations")
    retargeted = Animation(
        joint_names=target_skel.names,
        times=source.times.copy(),
        rotations=rt_rotations,
        root_translations=rt_roots,
        interp=source.interp,
        parameter_version=tag,
    )

    # ---- resample stage ----------------------------------------------------
    grid = output_timestamps(retargeted.times, fps)
    rs_slots: dict[int, dict[str, np.ndarray]] = {}
    for start in range(0, grid.size, block_size):
        end = min(start + block_size, grid.size)
        frame_range = (start, end)
        signature = _rs_block_signature(retargeted, grid, frame_range, fps)
        cached = state.rs_cache.get(start)
        if (
            cached is not None
            and cached["signature"] == signature
            and start not in manifest.resample_blocks
        ):
            block = {key: value.copy() for key, value in cached["data"].items()}
        else:
            block = resample_frame_block(retargeted, fps, grid, frame_range)
            state.rs_cache[start] = {"signature": signature, "data": block}
            recomputed.append(
                {
                    "stage": "resample",
                    "block": start,
                    "channels": list(manifest.resample_channels),
                    "reason": manifest.reason,
                }
            )
        rs_slots[start] = block

    out_rotations = align_sequence(_merge(grid.size, rs_slots, "rotations"))
    out_roots = _merge(grid.size, rs_slots, "root_translations")
    out_times = _merge(grid.size, rs_slots, "times")
    output = Animation(
        joint_names=retargeted.joint_names,
        times=out_times,
        rotations=out_rotations,
        root_translations=out_roots,
        interp=retargeted.interp,
        parameter_version=tag,
    )

    state.source = source
    state.target_skel = target_skel
    state.fps = fps
    state.manifest = manifest
    state.recomputed = recomputed
    state.last_output = output
    return output


def verify_closure(state: IncrementalState, manifest: ClosureManifest) -> None:
    """Assert the actually recomputed set equals the closure manifest."""

    actual_rt = sorted(
        entry["block"] for entry in state.recomputed if entry["stage"] == "retarget"
    )
    actual_rs = sorted(
        entry["block"] for entry in state.recomputed if entry["stage"] == "resample"
    )
    expected_rt = sorted(manifest.retarget_blocks)
    expected_rs = sorted(manifest.resample_blocks)
    if actual_rt != expected_rt or actual_rs != expected_rs:
        from .errors import E_CLOSURE_MISMATCH

        raise E_CLOSURE_MISMATCH(
            "recomputed block set does not match invalidation closure",
            {
                "expected_retarget_blocks": expected_rt,
                "actual_retarget_blocks": actual_rt,
                "expected_resample_blocks": expected_rs,
                "actual_resample_blocks": actual_rs,
                "manifest_signature": manifest.signature(),
            },
        )


def baseline_state(
    source: Animation,
    source_skel: Skeleton,
    target_skel: Skeleton,
    fps: float,
    budget: Budget,
    workers: int,
    block_size: int = BLOCK_SIZE,
) -> IncrementalState:
    state = IncrementalState(
        source=source,
        source_skel=source_skel,
        target_skel=target_skel,
        fps=fps,
        budget=budget,
        workers=workers,
        block_size=block_size,
    )
    manifest = plan_closure(state, reason="baseline")
    run_incremental(state, manifest)
    verify_closure(state, manifest)
    return state
