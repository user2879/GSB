"""Historical snapshot replay and first-difference diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import sha256_file
from .errors import ReplayError
from .models import CURRENT_METRIC_VERSION, Interval, ProfilePolicy
from .session import ProfileSession, profile_file
from .snapshot import load_snapshot


@dataclass(frozen=True, slots=True)
class ReplayOutcome:
    session: ProfileSession
    report: dict[str, Any]


def _replay_error(message: str, *, field: str, details: dict[str, Any]) -> ReplayError:
    return ReplayError(message, field=field, details=details)


def _first_dict_difference(expected: Any, actual: Any, prefix: str = "") -> dict[str, Any] | None:
    if type(expected) is not type(actual):
        return {
            "field": prefix or "$",
            "expected": expected,
            "actual": actual,
        }
    if isinstance(expected, dict):
        keys = sorted(set(expected) | set(actual))
        for key in keys:
            child = f"{prefix}.{key}" if prefix else key
            if key not in expected:
                return {"field": child, "expected": "<absent>", "actual": actual[key]}
            if key not in actual:
                return {"field": child, "expected": expected[key], "actual": "<absent>"}
            difference = _first_dict_difference(expected[key], actual[key], child)
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return {
                "field": f"{prefix}.length" if prefix else "length",
                "expected": len(expected),
                "actual": len(actual),
            }
        for index, (left, right) in enumerate(zip(expected, actual)):
            difference = _first_dict_difference(left, right, f"{prefix}[{index}]")
            if difference is not None:
                return difference
        return None
    if expected != actual:
        return {"field": prefix or "$", "expected": expected, "actual": actual}
    return None


def _first_frame_difference(snapshot: dict[str, Any], session: ProfileSession) -> dict[str, Any] | None:
    expected_items = snapshot.get("folded_result", {}).get("frames", [])
    if not isinstance(expected_items, list):
        return {"frame": None, "field": "folded_result.frames", "expected": "list", "actual": type(expected_items).__name__}
    expected = {
        item.get("frame"): item
        for item in expected_items
        if isinstance(item, dict) and isinstance(item.get("frame"), str)
    }
    actual = {stat.frame: stat.to_dict() for stat in session.aggregation.frame_stats}
    fields = (
        "self_time_ms",
        "cumulative_time_ms",
        "participant_weight",
        "occurrence_count",
    )
    for frame in sorted(set(expected) | set(actual)):
        if frame not in expected:
            return {
                "frame": frame,
                "field": "frame_presence",
                "expected": "absent",
                "actual": "present",
            }
        if frame not in actual:
            return {
                "frame": frame,
                "field": "frame_presence",
                "expected": "present",
                "actual": "absent",
            }
        for field in fields:
            if expected[frame].get(field) != actual[frame].get(field):
                return {
                    "frame": frame,
                    "field": field,
                    "expected": expected[frame].get(field),
                    "actual": actual[frame].get(field),
                }
    return None


def _first_rank_difference(snapshot: dict[str, Any], session: ProfileSession) -> dict[str, Any] | None:
    expected = snapshot.get("hotspot_ranking", [])
    actual = session.ranking_report["ranking"]
    if not isinstance(expected, list):
        return {"rank": None, "frame": None, "field": "hotspot_ranking", "expected": "list", "actual": type(expected).__name__}
    limit = min(len(expected), len(actual))
    for index in range(limit):
        left = expected[index]
        right = actual[index]
        if not isinstance(left, dict):
            return {
                "rank": index + 1,
                "frame": None,
                "field": "hotspot_ranking.entry",
                "expected": "object",
                "actual": type(left).__name__,
            }
        if left.get("frame") != right.get("frame"):
            return {
                "rank": index + 1,
                "frame": left.get("frame"),
                "field": "rank.frame",
                "expected": left.get("frame"),
                "actual": right.get("frame"),
            }
        for field in ("self_time_ms", "cumulative_time_ms", "participant_weight", "self_ratio_fixed"):
            if left.get(field) != right.get(field):
                return {
                    "rank": index + 1,
                    "frame": right["frame"],
                    "field": field,
                    "expected": left.get(field),
                    "actual": right.get(field),
                }
        difference = _first_dict_difference(
            {k: v for k, v in left.items() if k not in {"rank", "frame"}},
            {k: v for k, v in right.items() if k not in {"rank", "frame"}},
            "ranking_entry",
        )
        if difference is not None:
            return {
                "rank": index + 1,
                "frame": right["frame"],
                "field": difference["field"],
                "expected": difference["expected"],
                "actual": difference["actual"],
            }
    if len(expected) != len(actual):
        return {
            "rank": limit + 1,
            "frame": expected[limit].get("frame") if len(expected) > limit else actual[limit]["frame"],
            "field": "ranking_length",
            "expected": len(expected),
            "actual": len(actual),
        }
    return None


def replay_snapshot(
    snapshot_path: str | Path,
    samples_path: str | Path,
    *,
    current_interval_version: str | None = None,
) -> ReplayOutcome:
    snapshot = load_snapshot(snapshot_path)
    snapshot_metric_version = snapshot["metric_version"]
    if snapshot_metric_version != CURRENT_METRIC_VERSION:
        raise _replay_error(
            "metric methodology version differs; snapshot is not reproducible",
            field="metric_version",
            details={
                "mismatch": "metric_version",
                "snapshot": snapshot_metric_version,
                "current": CURRENT_METRIC_VERSION,
            },
        )

    try:
        interval = Interval.from_dict(snapshot["interval"])
    except Exception as exc:
        if isinstance(exc, ReplayError):
            raise
        raise _replay_error(
            "snapshot interval parameters are invalid",
            field="interval",
            details={"mismatch": "interval_parameters", "error": str(exc)},
        ) from exc
    if (
        current_interval_version is not None
        and current_interval_version != interval.interval_version
    ):
        raise _replay_error(
            "interval parameter version differs; snapshot is not reproducible",
            field="interval.interval_version",
            details={
                "mismatch": "interval_version",
                "snapshot": interval.interval_version,
                "current": current_interval_version,
            },
        )

    policy_value = snapshot.get("policy")
    if not isinstance(policy_value, dict):
        raise _replay_error(
            "snapshot policy is missing; replay parameters are unavailable",
            field="policy",
            details={"mismatch": "policy"},
        )
    try:
        policy = ProfilePolicy.from_dict(policy_value)
    except Exception as exc:
        raise _replay_error(
            "snapshot policy is invalid; replay parameters are unavailable",
            field="policy",
            details={"mismatch": "policy", "error": str(exc)},
        ) from exc
    policy = ProfilePolicy(
        max_depth=policy.max_depth,
        full_unique_stack_threshold=policy.full_unique_stack_threshold,
        full_total_weight_threshold=policy.full_total_weight_threshold,
        scale=policy.scale,
        top_k=policy.top_k,
        metric_version=CURRENT_METRIC_VERSION,
    )

    recorded_path = snapshot.get("aggregation", {}).get("path")
    if recorded_path not in {"full", "streaming"}:
        raise _replay_error(
            "snapshot aggregation path is missing or invalid",
            field="aggregation.path",
            details={"mismatch": "aggregation_path", "snapshot": recorded_path},
        )

    sample_meta = snapshot.get("sample", {})
    expected_file_sha = sample_meta.get("file_sha256")
    if not isinstance(expected_file_sha, str) or not expected_file_sha:
        raise _replay_error(
            "snapshot sample digest is missing; replay baseline is unavailable",
            field="sample.file_sha256",
            details={"mismatch": "sample_sha256", "snapshot": expected_file_sha},
        )
    actual_file_sha = sha256_file(str(samples_path))
    if actual_file_sha != expected_file_sha:
        raise _replay_error(
            "sample content digest differs; snapshot is not reproducible",
            field="sample.file_sha256",
            details={
                "mismatch": "sample_sha256",
                "expected": expected_file_sha,
                "actual": actual_file_sha,
            },
        )

    session = profile_file(
        samples_path,
        policy,
        interval,
        force_path=recorded_path,
    )

    final_file_sha = sha256_file(str(samples_path))
    if final_file_sha != expected_file_sha:
        raise _replay_error(
            "sample file changed during replay",
            field="sample.file_sha256",
            details={
                "mismatch": "sample_sha256_changed_during_replay",
                "expected": expected_file_sha,
                "actual": final_file_sha,
            },
        )
    if session.metadata is not None:
        expected_selected_sha = sample_meta.get("selected_samples_sha256")
        if expected_selected_sha != session.metadata.selected_samples_sha256:
            raise _replay_error(
                "selected sample digest differs; snapshot is not reproducible",
                field="sample.selected_samples_sha256",
                details={
                    "mismatch": "selected_samples_sha256",
                    "expected": expected_selected_sha,
                    "actual": session.metadata.selected_samples_sha256,
                },
            )

    frame_difference = _first_frame_difference(snapshot, session)
    if frame_difference is not None:
        raise _replay_error(
            "replayed folded frame differs from snapshot",
            field="folded_result.frames",
            details={"mismatch": "folded_result", "first_mismatch": frame_difference},
        )
    rank_difference = _first_rank_difference(snapshot, session)
    if rank_difference is not None:
        raise _replay_error(
            "replayed hotspot ranking differs from snapshot",
            field="hotspot_ranking",
            details={"mismatch": "hotspot_ranking", "first_mismatch": rank_difference},
        )
    annotation_difference = _first_dict_difference(
        snapshot.get("error_annotation"),
        session.ranking_report["error_annotation"],
        "error_annotation",
    )
    if annotation_difference is not None:
        raise _replay_error(
            "replayed error annotation differs from snapshot",
            field="error_annotation",
            details={"mismatch": "error_annotation", "first_mismatch": annotation_difference},
        )

    expected_conclusion = snapshot.get("conclusion_sha256")
    if expected_conclusion is not None and expected_conclusion != session.conclusion_sha256:
        raise _replay_error(
            "replayed conclusion digest differs from snapshot",
            field="conclusion_sha256",
            details={
                "mismatch": "conclusion_sha256",
                "expected": expected_conclusion,
                "actual": session.conclusion_sha256,
            },
        )

    report = {
        "ok": True,
        "status": "reproducible",
        "replay_key": {
            "sample_sha256": actual_file_sha,
            "selected_samples_sha256": (
                session.metadata.selected_samples_sha256 if session.metadata else None
            ),
            "metric_version": CURRENT_METRIC_VERSION,
            "interval": interval.to_dict(),
            "aggregation_path": recorded_path,
        },
        "conclusion_sha256": session.conclusion_sha256,
    }
    return ReplayOutcome(session=session, report=report)
