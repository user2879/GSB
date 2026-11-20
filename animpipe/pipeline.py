"""End-to-end orchestration: budget planning + block scheduling + merge.

The two heavy stages (retarget, resample) are sliced into frame blocks,
planned against the resource budget, executed on the worker pool and
merged by block start key.  Sign alignment runs once on the fully merged
array so output bytes never depend on block boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .animation import Animation
from .budget import Budget, BudgetPlanner
from .quat import align_sequence
from .resample import output_timestamps, resample_frame_block
from .retarget import RetargetResult, retarget_frame_block
from .scheduler import FrameBlockScheduler, ScheduleReport
from .skeleton import Skeleton, rest_pose_delta, root_bone_scale


def _merge(total_frames: int, slots: dict[int, dict[str, np.ndarray]], field: str) -> np.ndarray:
    ordered = [slots[key][field] for key in sorted(slots)]
    merged = np.concatenate(ordered, axis=0)
    if merged.shape[0] != total_frames:
        raise AssertionError("merge lost frames")
    return merged


@dataclass
class PipelineOutput:
    retarget: RetargetResult
    output: Animation
    frame_tags: tuple[str, ...]
    reports: list[ScheduleReport] = field(default_factory=list)
    budget_decisions: list[dict[str, Any]] = field(default_factory=list)


def run_pipeline(
    source: Animation,
    source_skel: Skeleton,
    target_skel: Skeleton,
    fps: float,
    *,
    budget: Budget,
    workers: int,
    block_size: int,
    divergence_tolerance: float = 1e-6,
    parameter_tag: str | None = None,
) -> PipelineOutput:
    from .retarget import parameter_version

    planner = BudgetPlanner(budget)
    scale = root_bone_scale(source_skel, target_skel)
    rest_delta = rest_pose_delta(source_skel, target_skel)
    tag = parameter_tag or parameter_version(source_skel, target_skel, scale)
    decisions: list[dict[str, Any]] = []
    reports: list[ScheduleReport] = []

    # ---- stage 1: retarget ------------------------------------------------
    decision_rt = planner.plan(
        requested_workers=workers,
        requested_block_size=block_size,
        total_frames=source.frame_count,
        joint_count=source.joint_count,
        stage="retarget",
    )
    decisions.append(
        {
            "stage": "retarget",
            "workers": decision_rt.workers,
            "block_size": decision_rt.block_size,
            "estimated_buffer_bytes": decision_rt.estimated_buffer_bytes,
            "degradation": decision_rt.degradation,
            "denied": decision_rt.denied,
        }
    )

    def rt_block(frame_range: tuple[int, int]) -> dict[str, np.ndarray]:
        return retarget_frame_block(
            source, source_skel, target_skel, rest_delta, scale, frame_range
        )

    scheduler_rt = FrameBlockScheduler(decision_rt.workers)
    rt_slots, rt_report, _ = scheduler_rt.run(
        "retarget", source.frame_count, decision_rt.block_size, rt_block
    )
    rt_report.degradation = decision_rt.degradation
    rt_report.denied = decision_rt.denied
    rt_report.workers_effective = decision_rt.workers
    rt_report.block_size_effective = decision_rt.block_size
    reports.append(rt_report)

    rt_rotations = align_sequence(
        _merge(source.frame_count, rt_slots, "rotations")
    )
    rt_roots = _merge(source.frame_count, rt_slots, "root_translations")
    rt_residual = _merge(source.frame_count, rt_slots, "residual")
    rt_deviation = _merge(source.frame_count, rt_slots, "deviation")

    retargeted = Animation(
        joint_names=target_skel.names,
        times=source.times.copy(),
        rotations=rt_rotations,
        root_translations=rt_roots,
        interp=source.interp,
        quat_order="wxyz",
        parameter_version=tag,
    )
    from .errors import E_RETARGET_DIVERGE

    max_deviation = float(np.max(rt_deviation))
    if max_deviation > divergence_tolerance:
        frame, joint = np.unravel_index(
            int(np.argmax(rt_deviation)), rt_deviation.shape
        )
        raise E_RETARGET_DIVERGE(
            "retarget global accumulation paths diverged beyond tolerance",
            {
                "frame": int(frame),
                "joint": int(joint),
                "joint_name": target_skel.names[joint],
                "deviation_rad": max_deviation,
                "tolerance_rad": divergence_tolerance,
                "parameter_version": tag,
            },
        )

    rt_result = RetargetResult(
        animation=retargeted,
        per_joint_residual=np.mean(rt_residual, axis=0),
        frame_joint_residual=rt_residual,
        global_deviation={
            "max_deviation_rad": max_deviation,
            "mean_deviation_rad": float(np.mean(rt_deviation)),
            "max_residual_rad": float(np.max(rt_residual)),
            "root_scale": scale,
            "tolerance_rad": divergence_tolerance,
        },
        parameter_version=tag,
        rest_delta=rest_delta,
        scale=scale,
    )

    # ---- stage 2: resample ------------------------------------------------
    grid = output_timestamps(retargeted.times, fps)
    decision_rs = planner.plan(
        requested_workers=workers,
        requested_block_size=block_size,
        total_frames=grid.size,
        joint_count=retargeted.joint_count,
        stage="resample",
    )
    decisions.append(
        {
            "stage": "resample",
            "workers": decision_rs.workers,
            "block_size": decision_rs.block_size,
            "estimated_buffer_bytes": decision_rs.estimated_buffer_bytes,
            "degradation": decision_rs.degradation,
            "denied": decision_rs.denied,
        }
    )

    def rs_block(frame_range: tuple[int, int]) -> dict[str, np.ndarray]:
        return resample_frame_block(retargeted, fps, grid, frame_range)

    scheduler_rs = FrameBlockScheduler(decision_rs.workers)
    rs_slots, rs_report, _ = scheduler_rs.run(
        "resample", grid.size, decision_rs.block_size, rs_block
    )
    rs_report.degradation = decision_rs.degradation
    rs_report.denied = decision_rs.denied
    rs_report.workers_effective = decision_rs.workers
    rs_report.block_size_effective = decision_rs.block_size
    reports.append(rs_report)

    out_rotations = align_sequence(_merge(grid.size, rs_slots, "rotations"))
    out_roots = _merge(grid.size, rs_slots, "root_translations")
    out_times = _merge(grid.size, rs_slots, "times")

    output = Animation(
        joint_names=retargeted.joint_names,
        times=out_times,
        rotations=out_rotations,
        root_translations=out_roots,
        interp=retargeted.interp,
        quat_order="wxyz",
        parameter_version=tag,
    )

    from .hashutil import frame_hash

    frame_tags = tuple(
        frame_hash(
            index,
            output.rotations[index],
            output.root_translations[index],
            {"parameter_version": tag, "stage": "final"},
        )
        for index in range(output.frame_count)
    )
    return PipelineOutput(
        retarget=rt_result,
        output=output,
        frame_tags=frame_tags,
        reports=reports,
        budget_decisions=decisions,
    )
