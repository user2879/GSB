"""Trajectory-to-ground-truth matching and deterministic evaluation."""

from __future__ import annotations

from typing import Any

from .models import Trajectory

EVAL_STAGE = "match_evaluation"


def _overlap_ms(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def evaluate(
    trajectories: list[Trajectory],
    truth_events: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    ratio_threshold = float(config["thresholds"]["overlap_ratio"])
    candidates: list[dict[str, Any]] = []
    trajectory_by_id = {trajectory.trajectory_id: trajectory for trajectory in trajectories}

    for trajectory in trajectories:
        for event in truth_events:
            if trajectory.event_class != event["class"]:
                continue
            overlap = _overlap_ms(
                trajectory.start_ms,
                trajectory.end_ms,
                int(event["start_ms"]),
                int(event["end_ms"]),
            )
            if overlap <= 0:
                continue
            truth_span = int(event["end_ms"]) - int(event["start_ms"])
            traj_span = trajectory.end_ms - trajectory.start_ms
            union = traj_span + truth_span - overlap
            ratio = overlap / union if union > 0 else 0.0
            if ratio < ratio_threshold:
                continue
            candidates.append(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "event_id": event["event_id"],
                    "class": trajectory.event_class,
                    "overlap_ms": overlap,
                    "overlap_ratio": round(ratio, 6),
                    "truth_start_ms": int(event["start_ms"]),
                    "trajectory_start_ms": trajectory.start_ms,
                }
            )

    # Conflict resolution: overlap desc, truth start asc, trajectory start asc,
    # then ids asc for a fully deterministic total order.
    candidates.sort(
        key=lambda item: (
            -item["overlap_ms"],
            item["truth_start_ms"],
            item["trajectory_start_ms"],
            item["event_id"],
            item["trajectory_id"],
        )
    )

    matched_trajectories: set[str] = set()
    matched_events: set[str] = set()
    matches: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate["trajectory_id"] in matched_trajectories:
            continue
        if candidate["event_id"] in matched_events:
            continue
        trajectory = trajectory_by_id[candidate["trajectory_id"]]
        event = next(item for item in truth_events if item["event_id"] == candidate["event_id"])
        record = {
            "trajectory_id": candidate["trajectory_id"],
            "event_id": candidate["event_id"],
            "class": candidate["class"],
            "overlap_ms": candidate["overlap_ms"],
            "overlap_ratio": candidate["overlap_ratio"],
            "start_boundary_error_ms": int(trajectory.start_ms - int(event["start_ms"])),
            "end_boundary_error_ms": int(trajectory.end_ms - int(event["end_ms"])),
        }
        matches.append(record)
        matched_trajectories.add(candidate["trajectory_id"])
        matched_events.add(candidate["event_id"])

    matches.sort(
        key=lambda item: (
            trajectory_by_id[item["trajectory_id"]].sort_key_frame,
            item["event_id"],
        )
    )
    unmatched_trajectories = sorted(
        [trajectory.trajectory_id for trajectory in trajectories if trajectory.trajectory_id not in matched_trajectories]
    )
    unmatched_truth = sorted(
        [event["event_id"] for event in truth_events if event["event_id"] not in matched_events]
    )
    return {
        "overlap_ratio_threshold": ratio_threshold,
        "matches": matches,
        "unmatched_trajectories": unmatched_trajectories,
        "unmatched_truth_events": unmatched_truth,
        "matched_trajectory_count": len(matches),
        "truth_event_count": len(truth_events),
    }
