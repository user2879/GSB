"""High-level orchestration used by the CLI and demo scripts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes, canonical_text, first_diff
from .engine import (
    PATH_DIFF,
    PATH_FULL,
    PATH_INCREMENTAL,
    run_diff_synthesis,
    run_full,
    run_incremental,
)
from .errors import DivergenceError
from .injection import StructuredLogger
from .ledger import Ledger, write_snapshot
from .planner import change_ratio_from_diffs, select_path
from .recovery import compare_path_outputs, replay_from_ledger, run_session_with_crash


def all_path_results(scene, extra_requests: list[dict[str, Any]] | None = None):
    full = run_full(scene)
    incremental = run_incremental(scene, extra_requests=extra_requests)
    diff_synthesized, diffs = run_diff_synthesis(scene, extra_requests=extra_requests)
    outputs = {
        PATH_FULL: full,
        PATH_INCREMENTAL: incremental,
        PATH_DIFF: diff_synthesized,
    }
    return outputs, diffs


def assert_paths_identical(scene, extra_requests=None) -> dict[str, Any]:
    outputs, diffs = all_path_results(scene, extra_requests)
    compare_path_outputs(outputs)
    baseline = outputs[PATH_FULL]
    comparison = {
        name: [canonical_text(conclusion) for conclusion in conclusions]
        for name, conclusions in outputs.items()
    }
    for name in (PATH_INCREMENTAL, PATH_DIFF):
        for index, serialized in enumerate(comparison[name]):
            if serialized != comparison[PATH_FULL][index]:
                diff = first_diff(baseline[index], outputs[name][index])
                raise DivergenceError(
                    f"byte mismatch on path {name}",
                    f"rounds[{index}]",
                    {"first_diff": diff},
                )
    ratio = change_ratio_from_diffs(diffs, len(scene.instances))
    plan = select_path(scene.scale(), ratio)
    return {
        "rounds": len(baseline),
        "canonical_bytes_per_round": [
            len(canonical_bytes(conclusion)) for conclusion in baseline
        ],
        "change_ratio": ratio,
        "planner": plan,
        "diffs": diffs,
        "outputs": outputs,
    }


def run_and_persist(
    scene,
    db_path: str | Path,
    path_name: str = PATH_INCREMENTAL,
    snapshot_dir: str | Path | None = None,
) -> dict[str, Any]:
    ledger = Ledger(db_path)
    try:
        recovery = ledger.initialize_session(scene, path_name)
        from .engine import run_path

        conclusions = run_path(path_name, scene)
        digests = []
        for conclusion in conclusions:
            ledger.begin_round(conclusion["round_index"], path_name)
            digests.append(
                ledger.commit_round(conclusion["round_index"], path_name, conclusion)
            )
        snapshot = None
        if snapshot_dir is not None:
            snapshot = str(write_snapshot(snapshot_dir, scene, ledger))
        return {
            "path": path_name,
            "committed_rounds": ledger.committed_rounds(),
            "digests": digests,
            "recovery": asdict(recovery),
            "snapshot": snapshot,
        }
    finally:
        ledger.close()


def crash_and_recover(
    scene,
    db_path: str | Path,
    crash_round: int,
    log_path: str | Path,
    snapshot_dir: str | Path,
    path_name: str = PATH_INCREMENTAL,
) -> dict[str, Any]:
    logger = StructuredLogger(log_path)
    crash_ledger = Ledger(db_path)
    try:
        crash_result = run_session_with_crash(
            scene, crash_ledger, path_name, crash_round
        )
    finally:
        crash_ledger.close()
    logger.log(
        "WARN",
        "E_SNAPSHOT",
        "simulated forced termination before round commit",
        crash_round=crash_round,
        pending_round=crash_result["pending_round"],
    )

    recover_ledger = Ledger(db_path)
    try:
        recovery = recover_ledger.initialize_session(scene, path_name)
        report = replay_from_ledger(scene, recover_ledger, path_name)
        snapshot = write_snapshot(snapshot_dir, scene, recover_ledger)
    finally:
        recover_ledger.close()
    logger.log(
        "INFO",
        "OK",
        "recovery replay completed",
        verified_rounds=report.verified_rounds,
        rerun_rounds=report.rerun_rounds,
    )
    return {
        "crash": {
            "committed_rounds": crash_result["committed_rounds"],
            "pending_round": crash_result["pending_round"],
        },
        "recovery": asdict(recovery),
        "replay": {
            "verified_rounds": report.verified_rounds,
            "rerun_rounds": report.rerun_rounds,
            "digests": report.digests,
        },
        "snapshot": str(snapshot),
    }
