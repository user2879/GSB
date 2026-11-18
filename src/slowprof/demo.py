"""End-to-end deterministic demonstration used by CLI and examples."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .aggregate import aggregate_file
from .errors import E_DEPTH, E_IO, E_PARAM, E_REPLAY, E_SAMPLE, E_VERSION, SlowProfError
from .logging import StructuredLogger
from .models import Interval, ProfilePolicy
from .ranking import build_ranking, canonical_core_bytes
from .replay import replay_snapshot
from .session import ProfileSession, profile_file
from .snapshot import restore_snapshot, save_snapshot

EXIT_CODES = {
    E_SAMPLE: 2,
    E_DEPTH: 3,
    E_VERSION: 4,
    E_REPLAY: 5,
    E_PARAM: 6,
    E_IO: 7,
}


@dataclass(frozen=True, slots=True)
class DemoOutcome:
    summary: dict[str, Any]
    exit_code: int


def _parity_check(
    samples_path: Path,
    policy: ProfilePolicy,
    interval: Interval,
    dataset: str,
) -> dict[str, Any]:
    full = aggregate_file(samples_path, policy, interval, force_path="full")
    streaming = aggregate_file(samples_path, policy, interval, force_path="streaming")
    full_ranking = build_ranking(full, policy)
    streaming_ranking = build_ranking(streaming, policy)
    full_bytes = canonical_core_bytes(full, full_ranking)
    streaming_bytes = canonical_core_bytes(streaming, streaming_ranking)
    return {
        "dataset": dataset,
        "full_path": full.selection.path,
        "streaming_path": streaming.selection.path,
        "byte_equal": full_bytes == streaming_bytes,
        "canonical_sha256": hashlib.sha256(full_bytes).hexdigest(),
        "frame_count": len(full.frame_stats),
        "ranking_length": len(full_ranking["ranking"]),
    }


def _save_and_replay(
    session: ProfileSession,
    samples_path: Path,
    snapshot_path: Path,
    logger: Any,
    session_name: str,
) -> dict[str, Any]:
    save_snapshot(session, snapshot_path)
    logger.event(
        "snapshot_saved",
        session=session_name,
        path=str(snapshot_path),
        conclusion_sha256=session.conclusion_sha256,
        aggregation_path=session.aggregation.selection.path,
    )
    replay = replay_snapshot(snapshot_path, samples_path)
    logger.event(
        "replay_verified",
        session=session_name,
        conclusion_sha256=replay.report["conclusion_sha256"],
        path=snapshot_path.as_posix(),
    )
    return {
        "session": session_name,
        "snapshot": str(snapshot_path),
        "status": replay.report["status"],
        "conclusion_sha256": replay.report["conclusion_sha256"],
    }


def run_demo(
    samples_dir: str | Path = "samples",
    runtime_dir: str | Path = "runtime",
    *,
    logger: Any | None = None,
    inject: str = "none",
) -> DemoOutcome:
    samples_dir = Path(samples_dir)
    runtime_dir = Path(runtime_dir)
    sessions_dir = runtime_dir / "sessions"
    logs_dir = runtime_dir / "logs"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    active_logger = logger or StructuredLogger(logs_dir / "slowprof.jsonl")

    manifest_path = samples_dir / "manifest.json"
    if not manifest_path.exists():
        raise SlowProfError(
            E_IO,
            "sample manifest is missing; run scripts/generate_samples.py first",
            field="samples_dir.manifest",
            details={"path": str(manifest_path)},
        )

    policy = ProfilePolicy()
    small_interval = Interval(1_700_000_000_000, 1_700_000_020_000, "obs-v1")
    large_interval = Interval(1_710_000_000_000, 1_710_001_000_000, "obs-v1")
    changed_interval = Interval(1_700_000_004_000, 1_700_000_020_000, "obs-narrow-v2")

    small_path = samples_dir / "small.ndjson"
    large_path = samples_dir / "large.ndjson"

    session_specs = (
        ("session-1-small-full", small_path, small_interval),
        ("session-2-large-streaming", large_path, large_interval),
        ("session-3-interval-changed", small_path, changed_interval),
    )
    sessions: list[ProfileSession] = []
    session_summaries: list[dict[str, Any]] = []
    snapshot_paths: list[Path] = []
    replay_summaries: list[dict[str, Any]] = []

    for name, path, interval in session_specs:
        session = profile_file(path, policy, interval)
        sessions.append(session)
        snapshot_path = sessions_dir / f"{name}.json"
        snapshot_paths.append(snapshot_path)
        summary = {"session": name, **session.to_summary()}
        session_summaries.append(summary)
        replay_summaries.append(
            _save_and_replay(session, path, snapshot_path, active_logger, name)
        )
        active_logger.event(
            "profile_completed",
            session=name,
            samples=str(path),
            aggregation_path=session.aggregation.selection.path,
            selected_row_count=session.metadata.selected_row_count if session.metadata else 0,
            total_weight=session.aggregation.total_weight,
            conclusion_sha256=session.conclusion_sha256,
        )

    parity = [
        _parity_check(small_path, policy, small_interval, "small"),
        _parity_check(large_path, policy, large_interval, "large"),
    ]
    for item in parity:
        active_logger.event("parity_checked", **item)
        if not item["byte_equal"]:
            raise AssertionError(f"full/streaming parity failed for {item['dataset']}")

    restore_mismatch = restore_snapshot(snapshot_paths[0], sessions[2])
    active_logger.event(
        "snapshot_restore",
        strategy=restore_mismatch.strategy,
        mismatch_count=len(restore_mismatch.mismatches),
        archived_snapshot_path=restore_mismatch.archived_snapshot_path,
        changed_frame_count=len(restore_mismatch.changed_frames),
        rank_move_count=len(restore_mismatch.rank_moves),
    )
    restore_consistent = restore_snapshot(snapshot_paths[2], sessions[2])
    replacement_snapshot = sessions_dir / "session-3-restored-replacement.json"
    save_snapshot(restore_mismatch.active_session, replacement_snapshot)
    archived_mode = None
    if restore_mismatch.archived_snapshot_path is not None:
        archived_mode = os.stat(restore_mismatch.archived_snapshot_path).st_mode
        write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        if archived_mode & write_bits:
            raise AssertionError("archived old snapshot is not read-only")

    summary: dict[str, Any] = {
        "ok": True,
        "sessions": session_summaries,
        "path_parity": parity,
        "replay_checks": replay_summaries,
        "restore_mismatch": restore_mismatch.to_dict(),
        "restore_consistent": restore_consistent.to_dict(),
        "replacement_snapshot": str(replacement_snapshot),
        "archived_snapshot_mode": archived_mode,
        "log_path": str(active_logger.path) if hasattr(active_logger, "path") else None,
    }

    if inject == "none":
        return DemoOutcome(summary=summary, exit_code=0)

    try:
        if inject == "zero-weight":
            profile_file(
                samples_dir / "anomalies" / "zero_weight.ndjson",
                policy,
                Interval(1_720_000_000_000, 1_720_000_000_100, "obs-v1"),
            )
        elif inject == "depth-exceeded":
            profile_file(
                samples_dir / "anomalies" / "depth_exceeded.ndjson",
                policy,
                Interval(1_720_000_001_000, 1_720_000_001_100, "obs-v1"),
            )
        elif inject == "missing-version":
            restore_snapshot(
                samples_dir / "anomalies" / "snapshot_missing_version.json",
                sessions[0],
            )
        else:
            raise SlowProfError(
                E_PARAM,
                "unknown anomaly injection",
                field="inject",
                details={"actual": inject},
            )
    except SlowProfError as error:
        active_logger.error(error, command="demo", injected=inject)
        summary["ok"] = False
        summary["injected_anomaly"] = error.to_dict()
        summary["injected_anomaly"]["injection"] = inject
        return DemoOutcome(summary=summary, exit_code=EXIT_CODES.get(error.code, 1))

    active_logger.error(
        SlowProfError(
            E_PARAM,
            "injected anomaly did not produce the expected business error",
            field="inject",
        ),
        command="demo",
        injected=inject,
    )
    raise AssertionError(f"anomaly injection {inject!r} unexpectedly succeeded")
