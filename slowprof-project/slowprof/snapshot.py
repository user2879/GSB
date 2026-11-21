"""Versioned snapshot replacement and historical replay verification.

A snapshot is a single JSON document (plus a sidecar SHA-256) capturing:

* the sample-stream content digest and observation-interval parameter version;
* the chosen aggregation path and threshold boundaries;
* the full folded stacks, frame metrics, ranks, contexts and error annotations.

On reopen, any mismatch of sample digest or interval version triggers a
*whole-snapshot replacement* - never an incremental merge. Replay then
re-loads the sample stream with the recorded interval/path and recomputes
everything; the first divergent frame/field is reported as ``E_REPLAY``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

from .errors import ReplayError
from .folding import FoldResult
from .fixedpoint import ShareReport
from .ranking import FrameContext, FrameRank
from .samples import Interval, sample_stream_digest

SNAPSHOT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ReplayOutcome:
    matched: bool
    checked_frames: int
    mismatch: Optional[object] = None


def _result_to_payload(result: FoldResult) -> dict:
    return {
        "stacks": [
            {
                "frames": list(entry.frames),
                "weight": entry.weight,
                "leaf": entry.leaf,
            }
            for entry in result.stacks
        ],
        "frame_metrics": [
            {
                "frame": metric.frame,
                "self_weight": metric.self_weight,
                "cumulative_weight": metric.cumulative_weight,
            }
            for metric in result.frame_metrics
        ],
        "total_weight": result.total_weight,
        "unique_stack_count": result.unique_stack_count,
        "path_used": result.path_used,
        "path_reason": result.path_reason,
        "unique_stack_threshold": result.unique_stack_threshold,
        "total_weight_threshold": result.total_weight_threshold,
        "canonical_sha256": hashlib.sha256(
            result.canonical_bytes()).hexdigest(),
    }


def build_snapshot(*, session_id: str, sample_path: str,
                   interval: Optional[Interval], interval_version: int,
                   result: FoldResult, ranks: Sequence[FrameRank],
                   contexts: Mapping[str, FrameContext],
                   report: ShareReport,
                   force_path: Optional[str]) -> dict:
    digest = sample_stream_digest(sample_path, interval)
    return {
        "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
        "session_id": session_id,
        "interval_version": interval_version,
        "interval_canonical": (
            interval.canonical() if interval is not None else "-..-"),
        "sample_digest": digest,
        "force_path": force_path,
        "result": _result_to_payload(result),
        "ranks": [rank.__dict__ for rank in ranks],
        "contexts": {
            frame: {
                "callers": [edge.__dict__ for edge in ctx.callers],
                "callees": [edge.__dict__ for edge in ctx.callees],
            }
            for frame, ctx in contexts.items()
        },
        "share_report": {
            "scale": report.scale,
            "total_weight": report.total_weight,
            "self_shares": [s.__dict__ for s in report.self_shares],
            "cumulative_shares": [
                s.__dict__ for s in report.cumulative_shares],
            "aggregate_error_bound": [
                report.aggregate_error_bound_num,
                report.aggregate_error_bound_den,
            ],
            "top_k_contains_true_top_k":
                report.top_k_contains_true_top_k,
            "top_k_basis": report.top_k_basis,
            "error_affected_pairs": list(report.error_affected_pairs),
        },
    }


def write_snapshot(snapshot_dir: str | Path, snapshot: dict) -> Path:
    directory = Path(snapshot_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "snapshot.json"
    body = json.dumps(snapshot, sort_keys=True, indent=2).encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    tmp = directory / "snapshot.json.tmp"
    tmp.write_bytes(body)
    (directory / "snapshot.sha256").write_text(digest + "\n")
    tmp.replace(target)  # atomic on local filesystems
    return target


def read_snapshot(snapshot_dir: str | Path, *,
                  verify_digest: bool = True) -> Optional[dict]:
    target = Path(snapshot_dir) / "snapshot.json"
    if not target.exists():
        return None
    body = target.read_bytes()
    record = json.loads(body.decode("utf-8"))
    sidecar = Path(snapshot_dir) / "snapshot.sha256"
    if verify_digest and sidecar.exists():
        expected = sidecar.read_text().strip()
        if hashlib.sha256(body).hexdigest() != expected:
            raise ValueError("snapshot digest mismatch (tampered file?)")
    return record


def should_replace(existing: Optional[dict], *, sample_digest: str,
                   interval_version: int) -> Tuple[bool, str]:
    """Whole-snapshot replacement decision (no incremental merge)."""

    if existing is None:
        return True, "no existing snapshot"
    if existing.get("sample_digest") != sample_digest:
        return True, "sample stream digest differs -> full replacement"
    if existing.get("interval_version") != interval_version:
        return True, "interval parameter version differs -> full replacement"
    if (existing.get("snapshot_format_version")
            != SNAPSHOT_FORMAT_VERSION):
        return True, "snapshot format version differs -> full replacement"
    return False, "digest and interval version match -> reusable"


def verify_replay(recorded: dict, recomputed: FoldResult,
                  recomputed_ranks: Sequence[FrameRank]) -> ReplayOutcome:
    """Compare recorded conclusions against fresh recomputation.

    Frames are visited in the recorded name order; the first differing field
    raises :class:`ReplayError` (``E_REPLAY``) with expected vs actual values.
    """

    recorded_metrics = {
        item["frame"]: item
        for item in recorded["result"]["frame_metrics"]
    }
    actual_metrics = recomputed.metric_map()

    all_frames = sorted(set(recorded_metrics) | set(actual_metrics))
    checked = 0
    for frame in all_frames:
        checked += 1
        if frame not in recorded_metrics:
            raise ReplayError.from_mismatch(
                frame, "presence", "recorded", "missing_in_snapshot")
        if frame not in actual_metrics:
            raise ReplayError.from_mismatch(
                frame, "presence", "present", "missing_on_replay")
        expected = recorded_metrics[frame]
        actual = actual_metrics[frame]
        for field in ("self_weight", "cumulative_weight"):
            if expected[field] != getattr(actual, field):
                raise ReplayError.from_mismatch(
                    frame, field, expected[field], getattr(actual, field))

    if recorded["result"]["total_weight"] != recomputed.total_weight:
        raise ReplayError.from_mismatch(
            "<global>", "total_weight",
            recorded["result"]["total_weight"],
            recomputed.total_weight,
        )
    recorded_canonical = recorded["result"]["canonical_sha256"]
    actual_canonical = hashlib.sha256(
        recomputed.canonical_bytes()).hexdigest()
    if recorded_canonical != actual_canonical:
        raise ReplayError.from_mismatch(
            "<global>", "canonical_sha256",
            recorded_canonical, actual_canonical)

    recorded_ranks = [item["frame"] for item in recorded["ranks"]]
    actual_ranks = [rank.frame for rank in recomputed_ranks]
    if recorded_ranks != actual_ranks:
        for index, (exp, act) in enumerate(
                zip(recorded_ranks, actual_ranks), start=1):
            if exp != act:
                raise ReplayError.from_mismatch(
                    f"rank#{index}", "frame", exp, act)
        raise ReplayError.from_mismatch(
            "rank", "length", len(recorded_ranks), len(actual_ranks))

    return ReplayOutcome(matched=True, checked_frames=checked)
