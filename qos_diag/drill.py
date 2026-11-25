from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json
from .diagnosis import diagnose
from .ledger import Ledger
from .model import bundle_digest, load_bundle
from .runtime import (
    ConcurrencyGovernor,
    ShaperClient,
    ShaperProcessManager,
    build_call_payload,
    kill_stale_pid_files,
    monotonic_ms,
)
from .service import commit_edit_revision, recover_into_ledger, run_full_into_ledger
from .samplegen import generate_all_samples
from .verify import verify_ledger


def clean_generated_artifacts(workspace: str | Path) -> dict[str, Any]:
    root = Path(workspace).resolve()
    artifacts = root / "artifacts"
    run_dir = artifacts / "run"
    killed_pids = kill_stale_pid_files(run_dir)
    removed_files = 0
    removed_dirs = 0
    for relative in ("ledger", "snapshots", "reports"):
        directory = artifacts / relative
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.iterdir():
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
                removed_dirs += 1
            else:
                path.unlink(missing_ok=True)
                removed_files += 1
    if run_dir.exists():
        shutil.rmtree(run_dir)
        removed_dirs += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    for relative in ("ledger", "snapshots", "reports"):
        (artifacts / relative).mkdir(parents=True, exist_ok=True)
    return {"killed_pids": killed_pids, "removed_files": removed_files, "removed_dirs": removed_dirs}


def _bundle(spec_dir: Path) -> dict[str, Any]:
    bundle = load_bundle(spec_dir)
    bundle["spec_digest"] = bundle_digest(bundle)
    return bundle


def _summary_governor(outcome: dict[str, Any]) -> dict[str, Any]:
    governor = outcome.get("governor", {})
    return {
        "max_global_inflight_seen": governor.get("max_global_inflight_seen"),
        "global_limit": governor.get("global_limit"),
        "max_pipeline_inflight_seen": governor.get("max_pipeline_inflight_seen"),
        "pipeline_limit": governor.get("pipeline_limit"),
        "queue_entries": governor.get("queue_entries"),
        "max_wait_ms": governor.get("max_wait_ms"),
    }


def _full_and_edit_scenario(
    *,
    workspace: Path,
    ledger_relative: str,
    old_dir: Path,
    new_dir: Path,
    run_name: str,
    batch_size: int = 12,
) -> dict[str, Any]:
    ledger_path = workspace / ledger_relative
    old_bundle = _bundle(old_dir)
    new_bundle = _bundle(new_dir)
    ledger = Ledger.open(ledger_path, spec_digest=old_bundle["spec_digest"])
    manager = ShaperProcessManager(old_bundle, workspace / "artifacts" / "run" / f"{run_name}-base")
    try:
        manager.start_all()
        base_run = run_full_into_ledger(old_bundle, ledger, manager, batch_size=batch_size)
    finally:
        manager.stop_all()
    if not base_run["ok"]:
        ledger.close()
        return {"ok": False, "stage": "base_run", "errors": base_run.get("errors")}

    edit_manager = ShaperProcessManager(new_bundle, workspace / "artifacts" / "run" / f"{run_name}-edit")
    try:
        edit_manager.start_all()
        edit = commit_edit_revision(
            old_bundle,
            new_bundle,
            ledger,
            edit_manager,
            edit_manifest={"scenario": run_name},
        )
    finally:
        edit_manager.stop_all()
    verification = verify_ledger(new_dir, ledger_path)
    ledger.close()
    if not edit["ok"]:
        return {"ok": False, "stage": "edit", "edit": edit}
    return {
        "ok": bool(verification["ok"]),
        "ledger": str(ledger_path),
        "revision": edit["revision"],
        "affected_event_seqs": edit["closure"]["affected_event_seqs"],
        "affected_windows": edit["closure"]["affected_windows"],
        "unchanged_check": {
            "checked_cells": edit["unchanged_check"]["checked_cells"],
            "unchanged_cells": edit["unchanged_check"]["unchanged_cells"],
            "unchanged_events": edit["unchanged_check"]["unchanged_events"],
            "all_unaffected_byte_identical": edit["unchanged_check"]["all_unaffected_byte_identical"],
        },
        "edit_governor": _summary_governor(edit),
        "verification": verification,
    }


def _timeout_probe(workspace: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    manager = ShaperProcessManager(bundle, workspace / "artifacts" / "run" / "timeout", server_delay_ms=100)
    shaper = sorted(bundle["shapers"], key=lambda item: item["shaper_id"])[0]
    manager.start_one(shaper)
    try:
        diagnosis = diagnose(bundle)
        payload = None
        for result in diagnosis["events"]:
            for record in result["stages"]:
                if record["evaluated"] and record["shaper_id"] == shaper["shaper_id"]:
                    payload = build_call_payload(bundle, result, record)
                    break
            if payload:
                break
        client = ShaperClient(shaper["host"], shaper["port"], 30)
        start = monotonic_ms()
        timed_out = False
        message = ""
        try:
            client.judge(payload or {"type": "ping"})
        except Exception as exc:
            timed_out = getattr(exc, "field", None) == "socket_timeout"
            message = str(exc)
        elapsed = monotonic_ms() - start
        return {
            "ok": timed_out and elapsed >= 25,
            "shaper_id": shaper["shaper_id"],
            "timeout_ms": 30,
            "elapsed_ms": elapsed,
            "real_socket_timeout": timed_out,
            "message": message,
        }
    finally:
        manager.stop_all()


def _concurrency_probe(workspace: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    manager = ShaperProcessManager(bundle, workspace / "artifacts" / "run" / "concurrency", server_delay_ms=20)
    manager.start_all()
    try:
        diagnosis = diagnose(bundle)
        governor = ConcurrencyGovernor(global_limit=4, pipeline_limit=2)
        output = execute_with_governor(bundle, diagnosis["events"][:10], governor)
        snap = output["governor"]
        ok = (
            output["ok"]
            and snap["max_global_inflight_seen"] <= 4
            and all(value <= 2 for value in snap["max_pipeline_inflight_seen"].values())
        )
        return {
            "ok": ok,
            "task_count": output["task_count"],
            "max_global_inflight_seen": snap["max_global_inflight_seen"],
            "max_pipeline_inflight_seen": snap["max_pipeline_inflight_seen"],
            "limits_respected": ok,
        }
    finally:
        manager.stop_all()


def execute_with_governor(bundle: dict[str, Any], results: list[dict[str, Any]], governor: ConcurrencyGovernor) -> dict[str, Any]:
    from .runtime import execute_stage_calls

    return execute_stage_calls(bundle, results, governor=governor)


def run_acceptance_drill(workspace: str | Path = ".") -> dict[str, Any]:
    root = Path(workspace).resolve()
    samples = generate_all_samples(root)
    clean = clean_generated_artifacts(root)
    base_dir = root / "samples" / "base"
    base_bundle = _bundle(base_dir)

    ledger_path = root / "artifacts" / "ledger" / "qos.db"
    ledger = Ledger.open(ledger_path, spec_digest=base_bundle["spec_digest"])
    manager = ShaperProcessManager(base_bundle, root / "artifacts" / "run" / "main")
    kill_shaper_id = "sh-02-relay"
    try:
        manager.start_all()
        interrupted = run_full_into_ledger(
            base_bundle,
            ledger,
            manager,
            batch_size=6,
            kill_shaper_before_batch=(1, kill_shaper_id),
        )
        if interrupted["ok"]:
            raise RuntimeError("crash drill expected the killed shaper call batch to fail")
        manager.restart(kill_shaper_id)
        recovery = recover_into_ledger(base_bundle, ledger, manager)
    finally:
        manager.stop_all()

    base_verification = verify_ledger(base_dir, ledger_path)
    param_bundle = _bundle(root / "samples" / "edited_param")
    param_manager = ShaperProcessManager(param_bundle, root / "artifacts" / "run" / "param-edit")
    try:
        param_manager.start_all()
        param_edit = commit_edit_revision(
            base_bundle,
            param_bundle,
            ledger,
            param_manager,
            edit_manifest={"edit_type": "shaper_rate", "target": kill_shaper_id},
        )
    finally:
        param_manager.stop_all()
    param_verification = verify_ledger(root / "samples" / "edited_param", ledger_path)
    ledger.close()

    atomic_write_json(root / "artifacts" / "snapshots" / "invalidation_param.json", param_edit["closure"])
    atomic_write_json(root / "artifacts" / "snapshots" / "base_path_selection.json", diagnose(base_bundle)["path_selection"])

    arrival = _full_and_edit_scenario(
        workspace=root,
        ledger_relative="artifacts/ledger/arrival-edit.db",
        old_dir=base_dir,
        new_dir=root / "samples" / "edited_arrival",
        run_name="arrival",
        batch_size=12,
    )
    burst = _full_and_edit_scenario(
        workspace=root,
        ledger_relative="artifacts/ledger/burst-edit.db",
        old_dir=base_dir,
        new_dir=root / "samples" / "edited_burst",
        run_name="burst",
        batch_size=12,
    )

    replacement_path = root / "artifacts" / "ledger" / "replacement.db"
    replacement_ledger = Ledger.open(replacement_path, spec_digest=base_bundle["spec_digest"])
    replacement_ledger.close()
    conn = sqlite3.connect(replacement_path)
    conn.execute("UPDATE meta SET value='999' WHERE key='parameter_revision'")
    conn.commit()
    conn.close()
    replaced = Ledger.open(replacement_path)
    replacement = {"ok": bool(replaced.replaced), "replaced": replaced.replaced, "reason": replaced.replacement_reason}
    replaced.close()

    timeout_probe = _timeout_probe(root, base_bundle)
    concurrency_probe = _concurrency_probe(root, base_bundle)

    main_ok = (
        recovery["ok"]
        and recovery["sealed_prefix_byte_identical"]
        and base_verification["ok"]
        and param_edit["ok"]
        and param_edit["unchanged_check"]["all_unaffected_byte_identical"]
        and param_verification["ok"]
    )
    report = {
        "schema": "qos-diag/acceptance-report@1",
        "ok": bool(
            main_ok
            and arrival["ok"]
            and burst["ok"]
            and replacement["ok"]
            and timeout_probe["ok"]
            and concurrency_probe["ok"]
        ),
        "clean": clean,
        "samples": {key: str(value) for key, value in samples.items()},
        "main_recovery_and_param_edit": {
            "ok": main_ok,
            "interrupted_batch_id": interrupted.get("failed_batch_id"),
            "interrupted_error_count": len(interrupted.get("errors", [])),
            "interrupted_governor": _summary_governor(interrupted),
            "recovery": {
                "ok": recovery["ok"],
                "unknown_batches": recovery["unknown_batches"],
                "sealed_seq_before": recovery["sealed_seq_before"],
                "rerun_event_count": recovery["rerun_event_count"],
                "sealed_prefix_byte_identical": recovery["sealed_prefix_byte_identical"],
                "recovery_batches": recovery["recovery_batches"],
            },
            "base_verification": base_verification,
            "param_edit": {
                "ok": param_edit["ok"],
                "revision": param_edit.get("revision"),
                "affected_event_count": len(param_edit["closure"]["affected_event_seqs"]),
                "affected_cell_count": len(param_edit["closure"]["affected_cells"]),
                "affected_window_count": len(param_edit["closure"]["affected_windows"]),
                "unchanged": param_edit["unchanged_check"],
                "governor": _summary_governor(param_edit),
                "verification": param_verification,
            },
        },
        "arrival_edit": arrival,
        "burst_edit": burst,
        "version_digest_replacement": replacement,
        "real_timeout_probe": timeout_probe,
        "concurrency_probe": concurrency_probe,
    }
    return report
