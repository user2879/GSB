"""Precise dependency closures for incremental metric recomputation."""

from __future__ import annotations

from typing import Any

from .constants import METRIC_ENCODING_VERSION
from .errors import BusinessError
from .quality import policy_hash, record_content_hash
from .util import hash_object


def _shot_map(shots: list[dict]) -> dict[int, int]:
    result: dict[int, int] = {}
    for shot in shots:
        for frame_index in range(int(shot["start_frame"]), int(shot["end_frame"]) + 1):
            result[frame_index] = int(shot["shot_id"])
    return result


def _jump(timestamp_jumps: dict[Any, Any], frame_index: int) -> float:
    if str(frame_index) in timestamp_jumps:
        return float(timestamp_jumps[str(frame_index)])
    return float(timestamp_jumps.get(frame_index, 0.0))


def build_context(
    content_hash: str,
    encoding_params: dict[str, Any],
    policy: dict[str, Any],
    shots: list[dict],
    injection: dict[str, Any],
    frame_count: int,
) -> dict[str, Any]:
    shots_by_frame = _shot_map(shots)
    dropped = {int(value) for value in injection.get("dropped_frames", [])}
    timestamp_jumps = dict(injection.get("timestamp_jumps", {}))
    timestamp_tolerance = float(injection.get("timestamp_tolerance_ms", 1.0))
    encoding_hash = hash_object(encoding_params)
    policy_digest = policy_hash(policy)
    frame_deps: dict[int, str] = {}
    for frame_index in range(frame_count):
        frame_deps[frame_index] = hash_object(
            {
                "content_hash": content_hash,
                "encoding_hash": encoding_hash,
                "policy_hash": policy_digest,
                "shot_id": int(shots_by_frame[frame_index]),
                "video_fault": {
                    "dropped": frame_index in dropped,
                    "timestamp_jump_ms": _jump(timestamp_jumps, frame_index),
                    "timestamp_tolerance_ms": timestamp_tolerance,
                },
                "metric_encoding_version": METRIC_ENCODING_VERSION,
            }
        )
    return {
        "content_hash": content_hash,
        "encoding_hash": encoding_hash,
        "policy_hash": policy_digest,
        "shots_hash": hash_object(shots),
        "video_fault_hash": hash_object(
            {
                "dropped_frames": sorted(dropped),
                "timestamp_jumps": timestamp_jumps,
                "timestamp_tolerance_ms": timestamp_tolerance,
            }
        ),
        "frame_count": frame_count,
        "frame_deps": frame_deps,
    }


def segment_deps(
    shots: list[dict],
    frames: list[dict],
    context: dict[str, Any],
) -> dict[int, str]:
    frame_by_index = {int(frame["frame_index"]): frame for frame in frames}
    result: dict[int, str] = {}
    for shot in shots:
        start = int(shot["start_frame"])
        end = int(shot["end_frame"])
        members = [frame_by_index[index] for index in range(start, end + 1) if index in frame_by_index]
        member_signatures = [
            {
                "frame_index": int(frame["frame_index"]),
                "record_hash": record_content_hash(frame),
                "deps_hash": frame.get("deps_hash"),
            }
            for frame in members
        ]
        result[int(shot["shot_id"])] = hash_object(
            {
                "content_hash": context["content_hash"],
                "encoding_hash": context["encoding_hash"],
                "policy_hash": context["policy_hash"],
                "shot": shot,
                "members": member_signatures,
            }
        )
    return result


def compute_frame_closure(
    prior_frames: list[dict] | None,
    context: dict[str, Any],
) -> list[int]:
    if prior_frames is None:
        return list(range(int(context["frame_count"])))
    prior = {int(frame["frame_index"]): frame for frame in prior_frames}
    return [
        frame_index
        for frame_index in range(int(context["frame_count"]))
        if frame_index not in prior or prior[frame_index].get("deps_hash") != context["frame_deps"][frame_index]
    ]


def compute_segment_closure(
    prior_segments: list[dict] | None,
    shots: list[dict],
    frames: list[dict],
    context: dict[str, Any],
) -> list[int]:
    new_deps = segment_deps(shots, frames, context)
    if prior_segments is None:
        return sorted(new_deps)
    prior = {int(segment["shot_id"]): segment for segment in prior_segments}
    affected = set()
    for shot_id, dependency in new_deps.items():
        if shot_id not in prior or prior[shot_id].get("deps_hash") != dependency:
            affected.add(shot_id)
    for shot_id in prior:
        if shot_id not in new_deps:
            affected.add(shot_id)
    return sorted(affected)


def changed_inputs(prior_meta: dict[str, Any] | None, context: dict[str, Any]) -> list[str]:
    if prior_meta is None:
        return ["full_rebuild"]
    prior_context = prior_meta.get("context", {})
    changed: list[str] = []
    for key, label in (
        ("content_hash", "content"),
        ("encoding_hash", "encoding_params"),
        ("policy_hash", "metric_policy"),
        ("shots_hash", "shots"),
        ("video_fault_hash", "video_faults"),
    ):
        if prior_context.get(key) != context.get(key):
            changed.append(label)
    return changed


def assert_closure_match(
    expected_frames: list[int],
    actual_frames: list[int],
    expected_segments: list[int],
    actual_segments: list[int],
    force_mismatch: bool = False,
) -> None:
    actual_frames_set = set(actual_frames)
    actual_segments_set = set(actual_segments)
    if force_mismatch:
        actual_frames_set.add(-1)
    if set(expected_frames) != actual_frames_set or set(expected_segments) != actual_segments_set:
        raise BusinessError(
            "E_CLOSURE_MISMATCH",
            "executed recomputation set differs from invalidation closure",
            {
                "expected_frames": sorted(expected_frames),
                "actual_recomputed_frames": sorted(actual_frames_set),
                "expected_segments": sorted(expected_segments),
                "actual_recomputed_segments": sorted(actual_segments_set),
            },
        )


def closure_report(
    prior_meta: dict[str, Any] | None,
    context: dict[str, Any],
    expected_frames: list[int],
    actual_frames: list[int],
    expected_segments: list[int],
    actual_segments: list[int],
) -> dict[str, Any]:
    report = {
        "schema_version": "invalidation-closure/1.0",
        "content_hash": context["content_hash"],
        "prior_revision": prior_meta.get("revision") if prior_meta else None,
        "changed_inputs": changed_inputs(prior_meta, context),
        "reason": "full_rebuild" if prior_meta is None else ("no_changes" if not expected_frames and not expected_segments else "incremental"),
        "frames": sorted(expected_frames),
        "segments": sorted(expected_segments),
        "actual_recomputed_frames": sorted(actual_frames),
        "actual_recomputed_segments": sorted(actual_segments),
        "match": set(expected_frames) == set(actual_frames) and set(expected_segments) == set(actual_segments),
        "inputs": {
            "content_hash": context["content_hash"],
            "encoding_hash": context["encoding_hash"],
            "policy_hash": context["policy_hash"],
            "shots_hash": context["shots_hash"],
            "video_fault_hash": context["video_fault_hash"],
        },
    }
    report["digest"] = hash_object(report)
    return report
