"""Application services tying the engine, ledger, snapshots and recovery."""

from __future__ import annotations

import os
from typing import Any, Mapping

from .deps import propagate_invalidation
from .engine import (
    apply_incremental,
    compare_runs,
    recompute_full_numeric,
    run_initial,
)
from .errors import OOMError, RecoveryRequired, ReplayError
from .ledger import Ledger
from .models import DiagnosticRun, LayerResult, Plan, SampleSet
from .snapshots import SnapshotInfo, SnapshotStore


def _callbacks_for_ledger(
    ledger: Ledger,
    session_id: str,
    expected_layers: tuple[str, ...],
    crash_after: int | None,
) -> tuple[Any, dict[str, int]]:
    index_by_layer = {layer_id: index for index, layer_id in enumerate(expected_layers)}
    state = {"committed": 0}

    def callback(layer_id: str, result: LayerResult, batches: Any, progress: dict[str, Any]) -> None:
        ledger.commit_layer(session_id, index_by_layer[layer_id], result, batches)
        state["committed"] += 1
        if crash_after is not None and state["committed"] >= crash_after:
            os._exit(137)

    return callback, state


def run_full_session(
    plan: Plan,
    samples: SampleSet,
    ledger: Ledger,
    *,
    snapshot_store: SnapshotStore | None = None,
    session_id: str | None = None,
    crash_after: int | None = None,
    save_snapshot: bool = False,
    snapshot_label: str = "full-run",
) -> dict[str, Any]:
    if crash_after is not None and crash_after < 0:
        raise ValueError("crash_after must be non-negative")
    session = ledger.start_session(plan, samples, session_id=session_id, mode="full")
    expected = ledger.expected_layers(session)
    if crash_after == 0:
        os._exit(137)
    callback, state = _callbacks_for_ledger(ledger, session, expected, crash_after)
    try:
        run = run_initial(plan, samples, progress_callback=callback)
    except OOMError as error:
        ledger.event("E_OOM", error.to_dict(), session_id=session)
        raise
    ledger.record_invalidation(session, run.invalidation, {})
    ledger.complete_session(session, run)
    snapshot_info: SnapshotInfo | None = None
    if save_snapshot and snapshot_store is not None:
        snapshot_info = snapshot_store.save(
            run, label=snapshot_label, ledger=ledger
        )
    return {
        "session_id": session,
        "committed_layers": state["committed"],
        "run": run,
        "snapshot": None if snapshot_info is None else snapshot_info.to_dict(),
    }


def run_incremental_session(
    plan: Plan,
    samples: SampleSet,
    base_run: DiagnosticRun,
    changed_sources: Mapping[str, str],
    ledger: Ledger,
    *,
    snapshot_store: SnapshotStore | None = None,
    session_id: str | None = None,
    crash_after: int | None = None,
    save_snapshot: bool = False,
    snapshot_label: str = "incremental-run",
) -> dict[str, Any]:
    if crash_after is not None and crash_after < 0:
        raise ValueError("crash_after must be non-negative")
    invalidated, trace = propagate_invalidation(plan, base_run.plan, changed_sources)
    session = ledger.start_session(
        plan,
        samples,
        session_id=session_id,
        mode="incremental",
        changed_sources=changed_sources,
        invalidated=invalidated,
        base_run=base_run,
    )
    expected = ledger.expected_layers(session)
    if crash_after == 0:
        os._exit(137)
    callback, state = _callbacks_for_ledger(ledger, session, expected, crash_after)
    try:
        run, _ = apply_incremental(
            plan,
            samples,
            base_run,
            changed_sources,
            progress_callback=callback,
        )
    except OOMError as error:
        ledger.event("E_OOM", error.to_dict(), session_id=session)
        raise
    ledger.record_invalidation(session, trace, changed_sources)
    ledger.complete_session(session, run)
    snapshot_info: SnapshotInfo | None = None
    if save_snapshot and snapshot_store is not None:
        snapshot_info = snapshot_store.save(
            run, label=snapshot_label, ledger=ledger
        )
    return {
        "session_id": session,
        "committed_layers": state["committed"],
        "invalidated": list(invalidated),
        "run": run,
        "snapshot": None if snapshot_info is None else snapshot_info.to_dict(),
    }


def _reference_run_for_session(ledger: Ledger, session_id: str) -> tuple[DiagnosticRun, dict[str, Any]]:
    plan, samples = ledger.session_plan_samples(session_id)
    base = ledger.session_base_run(session_id)
    changed = ledger.session_changed_sources(session_id)
    collected: dict[str, tuple[LayerResult, Any]] = {}

    def collect(layer_id: str, result: LayerResult, batches: Any, progress: dict[str, Any]) -> None:
        collected[layer_id] = (result, batches)

    if base is None:
        run = run_initial(plan, samples, progress_callback=collect)
    else:
        run, _ = apply_incremental(
            plan,
            samples,
            base,
            changed,
            progress_callback=collect,
        )
    return run, collected


def recover_session(ledger: Ledger, session_id: str) -> dict[str, Any]:
    return ledger.recover_session(session_id)


def resume_session(
    ledger: Ledger,
    session_id: str,
    *,
    snapshot_store: SnapshotStore | None = None,
    save_snapshot: bool = False,
    crash_after: int | None = None,
) -> dict[str, Any]:
    scan = ledger.recover_session(session_id)
    if scan["status"] == "complete":
        return {
            "ok": True,
            "session_id": session_id,
            "status": "complete",
            "recovery_scan": scan,
            "exit_report_available": True,
            "message": "session was already complete",
        }
    reference, collected = _reference_run_for_session(ledger, session_id)
    replay_check = ledger.verify_session_commits(session_id, reference)
    expected = ledger.expected_layers(session_id)
    committed = set(ledger.committed_layer_ids(session_id))
    missing = [layer_id for layer_id in expected if layer_id not in committed]
    ledger.mark_session_running(session_id)
    committed_now = 0
    for layer_index, layer_id in enumerate(expected):
        if layer_id in committed:
            continue
        result, batches = collected[layer_id]
        ledger.commit_layer(session_id, layer_index, result, batches)
        committed_now += 1
        if crash_after is not None and committed_now >= crash_after:
            os._exit(137)
    ledger.record_invalidation(session_id, reference.invalidation)
    ledger.complete_session(session_id, reference)
    snapshot_info: SnapshotInfo | None = None
    if save_snapshot and snapshot_store is not None:
        snapshot_info = snapshot_store.save(
            reference, label="recovered", ledger=ledger
        )
    return {
        "ok": True,
        "session_id": session_id,
        "status": "complete",
        "recovery_scan": scan,
        "replay_verification": replay_check,
        "rerun_layers": missing,
        "committed_now": committed_now,
        "exit_report_available": True,
        "run": reference,
        "snapshot": None if snapshot_info is None else snapshot_info.to_dict(),
    }


def replay_session(ledger: Ledger, session_id: str) -> dict[str, Any]:
    scan = ledger.recover_session(session_id)
    if scan["unknown_layers"]:
        raise RecoveryRequired(
            "session has unknown uncommitted layer batches; rerun before replay",
            session_id=session_id,
            unknown_layers=scan["unknown_layers"],
            exit_report_available=False,
            action_required="resume_or_rerun",
        )
    reference, _ = _reference_run_for_session(ledger, session_id)
    verification = ledger.verify_session_commits(session_id, reference)
    session = ledger.session(session_id)
    current = ledger.current_meta()
    if (
        current.get("plan_version") == session["plan_version"]
        and current.get("plan_digest") == session["plan_digest"]
        and current.get("content_digest") != reference.content_digest
    ):
        raise ReplayError(
            "current ledger content digest differs from replay",
            layer_id="<ledger>",
            field="content_digest",
            recorded=current.get("content_digest"),
            replayed=reference.content_digest,
            basis={"session_id": session_id, "plan_version": session["plan_version"]},
        )
    verification["content_digest"] = reference.content_digest
    verification["exit_report_available"] = True
    return verification


def check_incremental(
    plan: Plan,
    samples: SampleSet,
    base_run: DiagnosticRun,
    changed_sources: Mapping[str, str],
) -> dict[str, Any]:
    incremental, invalidated = apply_incremental(
        plan, samples, base_run, changed_sources
    )
    full = recompute_full_numeric(
        plan,
        samples,
        base_run,
        invalidated,
        invalidation_trace=incremental.invalidation,
    )
    comparison = compare_runs(incremental, full)
    comparison["invalidated_layers"] = list(invalidated)
    comparison["incremental_run"] = incremental
    comparison["full_run"] = full
    return comparison
