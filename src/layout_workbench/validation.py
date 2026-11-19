from __future__ import annotations

import copy
import tempfile
import time
from pathlib import Path
from typing import Any

from .cache import CacheStore
from .canonical import atomic_write_json
from .commands import CommandState, initial_command_state
from .compare import first_divergence, semantic_projection
from .config import RecognitionConfig
from .pipeline import Pipeline
from .runner import _check_invalidation, _commit_snapshot, _safe_id, _snapshot_stages
from .snapshot import SnapshotStore, snapshot_cache_audit


def _run_full_reference(
    state: CommandState,
    command: dict[str, Any] | None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="layout-full-reference-") as temporary:
        return Pipeline(
            state.config,
            CacheStore(Path(temporary)),
            respect_faults=False,
        ).run(
            state.ordered_sources(),
            mode="full",
            snapshot_version=0,
            command=command,
        )


def _step_record(
    *,
    step: int,
    command: dict[str, Any] | None,
    incremental_report: dict[str, Any],
    full_report: dict[str, Any],
    invalidation_mismatches: list[dict[str, Any]],
    reports_dir: Path,
    previous_snapshot_version: int,
) -> dict[str, Any]:
    divergence = first_divergence(incremental_report, full_report)
    semantic_equal = semantic_projection(incremental_report) == semantic_projection(full_report)
    command_id = "initial" if command is None else str(command["id"])
    base = f"step_{step:03d}_{_safe_id(command_id)}"
    incremental_path = reports_dir / f"{base}.incremental.json"
    full_path = reports_dir / f"{base}.full.json"
    atomic_write_json(incremental_path, incremental_report)
    atomic_write_json(full_path, full_report)
    return {
        "step": step,
        "command_id": command_id,
        "command_type": "initial" if command is None else command["type"],
        "semantic_equal": semantic_equal,
        "first_divergence": divergence,
        "invalidation_mismatches": invalidation_mismatches,
        "invalidation_plan_ok": not invalidation_mismatches,
        "incremental_report": str(incremental_path),
        "full_report": str(full_path),
        "snapshot_version_before": previous_snapshot_version,
        "incremental_budget": incremental_report.get("budget", {}),
        "full_budget": full_report.get("budget", {}),
        "pending_match_queue": incremental_report.get("pending_match_queue", []),
    }


def validate_command_sequence(
    *,
    data_dir: Path,
    config: RecognitionConfig,
    commands: list[dict[str, Any]],
    base_dir: Path,
    work_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    reports_dir = work_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    inc_state = initial_command_state(data_dir, config, base_dir)
    full_state = initial_command_state(data_dir, config, base_dir)
    inc_cache = CacheStore(work_dir / "cache")
    snapshot_store = SnapshotStore(work_dir / "snapshots", config.config_hash)
    global_version = 0
    steps: list[dict[str, Any]] = []

    incremental = Pipeline(inc_state.config, inc_cache, respect_faults=True).run(
        inc_state.ordered_sources(), mode="incremental", snapshot_version=0, command=None
    )
    full = _run_full_reference(full_state, None)
    mismatch = _step_record(
        step=0,
        command=None,
        incremental_report=incremental,
        full_report=full,
        invalidation_mismatches=[],
        reports_dir=reports_dir,
        previous_snapshot_version=0,
    )
    global_version, commit = _commit_snapshot(
        snapshot_store,
        incremental,
        global_version=global_version,
        command_index=0,
        command_id="initial",
        config=inc_state.config,
    )
    mismatch["snapshot_commit"] = commit
    mismatch["snapshot_version_after"] = global_version
    steps.append(mismatch)

    for step_index, command in enumerate(commands, start=1):
        previous_state = copy.deepcopy(inc_state)
        inc_effect = inc_state.apply(command)
        full_state.apply(command)
        _validate_state_alignment(inc_state, full_state)
        incremental = Pipeline(inc_state.config, inc_cache, respect_faults=True).run(
            inc_state.ordered_sources(),
            mode="incremental",
            snapshot_version=global_version,
            command={"index": step_index, **command, "effect": inc_effect},
        )
        full = _run_full_reference(full_state, {"index": step_index, **command})
        invalidation_mismatches = _check_invalidation(
            previous_state, inc_state, command, incremental
        )
        record = _step_record(
            step=step_index,
            command=command,
            incremental_report=incremental,
            full_report=full,
            invalidation_mismatches=invalidation_mismatches,
            reports_dir=reports_dir,
            previous_snapshot_version=global_version,
        )
        if step_index % inc_state.config.snapshot_commit_interval_commands == 0:
            global_version, commit = _commit_snapshot(
                snapshot_store,
                incremental,
                global_version=global_version,
                command_index=step_index,
                command_id=str(command["id"]),
                config=inc_state.config,
            )
            record["snapshot_commit"] = commit
        record["snapshot_version_after"] = global_version
        steps.append(record)

    snapshot_document = snapshot_store.load()
    snapshot_verification: dict[str, Any]
    if snapshot_document is None:
        snapshot_verification = {"valid": False, "error": "snapshot missing"}
    else:
        try:
            snapshot_verification = snapshot_store.verify(
                expected_config_hash=snapshot_document.get("config_hash")
            )
        except Exception as exc:  # noqa: BLE001 - validation must report, not crash
            snapshot_verification = {"valid": False, "error": str(exc)}
    if snapshot_document is not None:
        snapshot_verification["cache_audit"] = snapshot_cache_audit(
            snapshot_document, inc_cache
        )

    restored = Pipeline(inc_state.config, inc_cache, respect_faults=False).run(
        inc_state.ordered_sources(),
        mode="incremental_restore",
        snapshot_version=global_version,
        command={"index": len(commands), "id": "snapshot_restore"},
    )
    restore_divergence = first_divergence(incremental, restored)
    restore_equal = semantic_projection(incremental) == semantic_projection(restored)
    atomic_write_json(reports_dir / "snapshot_restore.json", restored)

    single_config = inc_state.config.with_updates(thread_pool_workers=1)
    with tempfile.TemporaryDirectory(prefix="layout-single-thread-") as temporary:
        single_thread = Pipeline(
            single_config, CacheStore(Path(temporary)), respect_faults=False
        ).run(
            inc_state.ordered_sources(),
            mode="full_single_thread",
            snapshot_version=0,
            command={"index": len(commands), "id": "single_thread_reference"},
        )
    concurrency_divergence = first_divergence(incremental, single_thread)
    concurrency_equal = semantic_projection(incremental) == semantic_projection(single_thread)
    atomic_write_json(reports_dir / "single_thread_reference.json", single_thread)

    budget_ok = True
    for step in steps:
        budget = step["incremental_budget"]
        if budget.get("consumed", 0) > budget.get("limit", 0):
            budget_ok = False
        if not all(
            item.get("code") == "E_POINT_BUDGET_PENDING"
            for item in step.get("pending_match_queue", [])
        ):
            budget_ok = False

    first_overall = next(
        (
            {
                "step": step["step"],
                "command_id": step["command_id"],
                **step["first_divergence"],
            }
            for step in steps
            if step["first_divergence"] is not None
        ),
        None,
    )
    passed = (
        all(step["semantic_equal"] for step in steps)
        and all(step["invalidation_plan_ok"] for step in steps)
        and bool(snapshot_verification.get("valid"))
        and bool(snapshot_verification.get("cache_audit", {}).get("valid"))
        and restore_equal
        and concurrency_equal
        and budget_ok
    )
    report = {
        "schema": "layout-validation-report-v1",
        "passed": passed,
        "data_dir": str(data_dir),
        "command_count": len(commands),
        "step_count": len(steps),
        "final_config_hash": inc_state.config.config_hash,
        "final_snapshot_version": global_version,
        "semantic_equivalence": {
            "all_steps_equal": all(step["semantic_equal"] for step in steps),
            "compared_fields": [
                "image order",
                "failure list",
                "suppressed lines",
                "line candidate order and boxes",
                "line scores and scales",
                "segmentation fragments and records",
                "line text",
                "character boxes, text, scores and candidate rankings",
                "pending queue order and cache keys",
                "budget limit, scheduled consumption, remaining and pending count",
            ],
            "run_local_fields_excluded": [
                "timings",
                "cache hit/miss/state metadata",
                "snapshot version",
            ],
            "cache_reuse_note": "The cold full reference necessarily recomputes what the incremental run reuses; reuse/recompute granularity is checked separately by the dependency-plan assertions below.",
        },
        "invalidation": {
            "all_steps_match_dependency_plan": all(
                step["invalidation_plan_ok"] for step in steps
            ),
            "steps": [
                {
                    "step": step["step"],
                    "command_id": step["command_id"],
                    "ok": step["invalidation_plan_ok"],
                    "mismatches": step["invalidation_mismatches"],
                }
                for step in steps
            ],
        },
        "first_overall_divergence": first_overall,
        "snapshot_verification": snapshot_verification,
        "snapshot_restore": {
            "semantic_equal": restore_equal,
            "first_divergence": restore_divergence,
            "snapshot_version": global_version,
        },
        "concurrency_equivalence": {
            "incremental_workers": inc_state.config.thread_pool_workers,
            "single_thread_workers": 1,
            "semantic_equal": concurrency_equal,
            "first_divergence": concurrency_divergence,
        },
        "point_budget": {
            "within_limit_and_pending_codes_valid": budget_ok,
            "steps": [
                {
                    "step": step["step"],
                    "limit": step["incremental_budget"].get("limit"),
                    "consumed": step["incremental_budget"].get("consumed"),
                    "pending_count": step["incremental_budget"].get("pending_count"),
                }
                for step in steps
            ],
        },
        "steps": steps,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    atomic_write_json(work_dir / "validation.json", report)
    atomic_write_json(reports_dir / "validation.json", report)
    return report


def _validate_state_alignment(incremental: CommandState, full: CommandState) -> None:
    if incremental.config != full.config:
        raise RuntimeError("incremental/full config alignment failed")
    if sorted(incremental.sources) != sorted(full.sources):
        raise RuntimeError("incremental/full image alignment failed")
    for name in incremental.sources:
        inc = incremental.sources[name]
        ref = full.sources[name]
        if inc.png_path.resolve() != ref.png_path.resolve():
            raise RuntimeError(f"incremental/full PNG alignment failed for {name}")
        if inc.truth_path.resolve() != ref.truth_path.resolve():
            raise RuntimeError(f"incremental/full truth alignment failed for {name}")
