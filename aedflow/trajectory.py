"""Probability smoothing, event trajectory reconstruction and filtering."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from .config import CANONICAL_CLASSES
from .errors import failure_entry
from .models import BoundaryDecision, LoadedShard, Trajectory

SMOOTH_STAGE = "trajectory_smoothing"
BUILD_STAGE = "trajectory_build"


def _smooth_one(
    grid: np.ndarray,
    decision: BoundaryDecision,
    window: int,
    method: str,
) -> dict[str, Any]:
    boundary_frame = decision.sort_key_ms // 10
    left_indexes = list(range(max(0, boundary_frame - window + 1), boundary_frame + 1))
    right_indexes = list(
        range(boundary_frame + 1, min(grid.shape[0], boundary_frame + 1 + window))
    )
    if method != "linear_crossfade":
        raise ValueError(f"unsupported smoothing method {method}")
    left_values = grid[left_indexes].copy()
    right_values = grid[right_indexes].copy()
    count_left = len(left_indexes)
    count_right = len(right_indexes)
    for local, frame_index in enumerate(left_indexes):
        weight_right = (local + 1) / (count_left + 1)
        # Mirror partner on the right side of the boundary.
        partner_index = right_indexes[min(local, count_right - 1)]
        grid[frame_index] = (1 - weight_right) * left_values[local] + weight_right * grid[partner_index]
    for local, frame_index in enumerate(right_indexes):
        weight_left = (count_right - local) / (count_right + 1)
        partner_index = left_indexes[max(0, count_left - 1 - local)]
        grid[frame_index] = weight_left * grid[partner_index] + (1 - weight_left) * right_values[local]
    return {
        "boundary_id": decision.boundary_id,
        "sort_key_frame": boundary_frame,
        "method": method,
        "window_frames": window,
        "left_frames": left_indexes,
        "right_frames": right_indexes,
    }


def smooth_boundaries(
    global_grid: np.ndarray,
    decisions: list[BoundaryDecision],
    config: dict[str, Any],
    *,
    inject_failure_trajectory: str | None = None,
) -> dict[str, Any]:
    window = int(config["smoothing"]["window_frames"])
    method = str(config["smoothing"]["method"])
    workers = max(1, int(config["runtime"]["max_workers"]))
    smoothed_grid = global_grid.copy()
    continuous = [decision for decision in decisions if decision.continuous]
    failures: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []

    def runner(decision: BoundaryDecision) -> dict[str, Any]:
        if inject_failure_trajectory == decision.boundary_id:
            raise RuntimeError("injected smoothing task failure")
        return _smooth_one(smoothed_grid, decision, window, method)

    if workers == 1:
        for decision in continuous:
            try:
                applied.append(runner(decision))
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    failure_entry(
                        decision.boundary_id,
                        SMOOTH_STAGE,
                        "E_TASK_FAILURE",
                        str(exc),
                        exception_type=type(exc).__name__,
                    )
                )
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(runner, decision): decision for decision in continuous}
            for future, decision in future_map.items():
                try:
                    applied.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    failures.append(
                        failure_entry(
                            decision.boundary_id,
                            SMOOTH_STAGE,
                            "E_TASK_FAILURE",
                            str(exc),
                            exception_type=type(exc).__name__,
                        )
                    )
    applied.sort(key=lambda item: item["sort_key_frame"])
    return {"grid": smoothed_grid, "smoothing_records": applied, "failures": failures}


def _owner_at(frame_index: int, ownership: list[str | None]) -> str:
    owner = ownership[frame_index]
    return owner if owner is not None else "unowned"


def build_trajectories(
    frame_decisions: list[str],
    ownership: list[str | None],
    loaded_shards: list[LoadedShard],
    boundary_decisions: list[BoundaryDecision],
    smoothing_records: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    shift = int(config["audio"]["frame_shift_ms"])
    threshold = float(config["thresholds"]["frame_posterior"])
    min_gap_ms = int(config["thresholds"]["min_gap_ms"])
    min_trajectory_ms = int(config["thresholds"]["min_trajectory_ms"])
    smoothing_window = int(config["smoothing"]["window_frames"])
    smoothing_method = str(config["smoothing"]["method"])

    continuous_by_frame = {
        decision.sort_key_ms // shift: decision for decision in boundary_decisions if decision.continuous
    }
    smoothing_by_boundary = {record["boundary_id"]: record for record in smoothing_records}
    shard_by_id = {shard.shard_id: shard for shard in loaded_shards}

    # Run-length encode frame labels, ignoring background.
    segments: list[dict[str, Any]] = []
    index = 0
    while index < len(frame_decisions):
        label = frame_decisions[index]
        end = index
        while end + 1 < len(frame_decisions) and frame_decisions[end + 1] == label:
            end += 1
        if label != "background":
            boundary_ids = [
                decision.boundary_id
                for frame, decision in sorted(continuous_by_frame.items())
                if index <= frame < end and decision.dominant_class == label
            ]
            segments.append(
                {
                    "label": label,
                    "first": index,
                    "last": end,
                    "boundary_ids": boundary_ids,
                }
            )
        index = end + 1

    merged: list[dict[str, Any]] = []
    for segment in segments:
        if not merged:
            merged.append(segment)
            continue
        previous = merged[-1]
        gap_frames = segment["first"] - previous["last"] - 1
        gap_ms = max(0, gap_frames * shift)
        boundary_frame = previous["last"]
        # A continuous boundary joins adjacent same-label pieces even when the
        # RLE pieces touch directly (zero background frames between them).
        crosses = (
            previous["label"] == segment["label"]
            and boundary_frame in continuous_by_frame
            and gap_ms <= min_gap_ms
        )
        if crosses:
            decision = continuous_by_frame[boundary_frame]
            previous["last"] = segment["last"]
            previous.setdefault("boundary_ids", []).append(decision.boundary_id)
            previous["boundary_ids"] = sorted(set(previous["boundary_ids"]))
        else:
            merged.append(segment)

    # Merge same-class trajectories separated only by a short background gap.
    gap_merged: list[dict[str, Any]] = []
    for segment in merged:
        if not gap_merged:
            gap_merged.append(segment)
            continue
        previous = gap_merged[-1]
        gap_ms = (segment["first"] - previous["last"] - 1) * shift
        if previous["label"] == segment["label"] and 0 <= gap_ms < min_gap_ms:
            previous["last"] = segment["last"]
            previous.setdefault("gap_merged", []).append(gap_ms)
        else:
            gap_merged.append(segment)

    trajectories: list[Trajectory] = []
    dropped_short: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for ordinal, segment in enumerate(gap_merged):
        first_frame, last_frame = segment["first"], segment["last"]
        start_ms, end_ms = first_frame * shift, (last_frame + 1) * shift
        duration_ms = end_ms - start_ms
        if duration_ms < min_trajectory_ms:
            dropped_short.append(
                {
                    "class": segment["label"],
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_ms": duration_ms,
                    "reason": "below min_trajectory_ms",
                }
            )
            continue
        owners = sorted(
            {
                _owner_at(frame_index, ownership)
                for frame_index in range(first_frame, last_frame + 1)
            }
        )
        boundary_ids = segment.get("boundary_ids", [])
        evidence = []
        for boundary_frame, decision in continuous_by_frame.items():
            if decision.boundary_id in boundary_ids:
                evidence.append(
                    {
                        "boundary_id": decision.boundary_id,
                        "criterion": decision.criterion,
                        "dominant_class": decision.dominant_class,
                        "continuous": decision.continuous,
                    }
                )
        applied_boundaries = [
            boundary_id for boundary_id in boundary_ids if boundary_id in smoothing_by_boundary
        ]
        failed_boundaries = [
            boundary_id for boundary_id in boundary_ids if boundary_id not in smoothing_by_boundary
        ]
        smoothing_meta = {
            "applied": bool(applied_boundaries),
            "method": smoothing_method,
            "window_frames": smoothing_window,
            "boundaries": [
                {
                    "boundary_id": boundary_id,
                    "left_frames": smoothing_by_boundary[boundary_id]["left_frames"],
                    "right_frames": smoothing_by_boundary[boundary_id]["right_frames"],
                }
                for boundary_id in applied_boundaries
            ],
            "failed_boundary_ids": failed_boundaries,
        }
        trajectory = Trajectory(
            trajectory_id=f"traj_{ordinal:03d}_{segment['label']}",
            sort_key_frame=first_frame,
            event_class=segment["label"],
            first_frame=first_frame,
            last_frame=last_frame,
            start_ms=start_ms,
            end_ms=end_ms,
            source_shard_ids=owners,
            crossed_boundaries=boundary_ids,
            boundary_evidence=evidence,
            smoothing=smoothing_meta,
            owning_shard_id=owners[0],
        )
        trajectories.append(trajectory)

    # Same start + same class: keep lexicographically smallest owning shard.
    deduped: list[Trajectory] = []
    for trajectory in trajectories:
        if deduped and (
            deduped[-1].start_ms == trajectory.start_ms
            and deduped[-1].event_class == trajectory.event_class
        ):
            keeper, challenger = deduped[-1], trajectory
            if challenger.owning_shard_id < keeper.owning_shard_id:
                duplicates.append(
                    {
                        "kept": challenger.trajectory_id,
                        "dropped": keeper.trajectory_id,
                        "rule": "owning shard id lexicographic ascending",
                    }
                )
                deduped[-1] = challenger
            else:
                duplicates.append(
                    {
                        "kept": keeper.trajectory_id,
                        "dropped": challenger.trajectory_id,
                        "rule": "owning shard id lexicographic ascending",
                    }
                )
            continue
        deduped.append(trajectory)

    deduped.sort(key=lambda item: (item.sort_key_frame, item.event_class, item.trajectory_id))
    for ordinal, trajectory in enumerate(deduped):
        trajectory.trajectory_id = f"traj_{ordinal:03d}_{trajectory.event_class}"
    counts = {name: 0 for name in CANONICAL_CLASSES}
    for trajectory in deduped:
        counts[trajectory.event_class] += 1
    return {
        "trajectories": deduped,
        "counts": counts,
        "dropped_short": dropped_short,
        "duplicates": duplicates,
        "frame_threshold": threshold,
    }
