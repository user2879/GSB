"""Repeatable local acceptance scenario and artifact writer."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .errors import DiagnosticError
from .ledger import Ledger
from .lease import LeasePool
from .runtime import InferenceRuntime, recover_and_resume, compare_stored_result
from .schema import load_plan, memory_profile, parse_plan
from .snapshot import build_snapshot, load_snapshot, save_snapshot
from .util import Clock, atomic_write_bytes, atomic_write_json


class DemoClock(Clock):
    def __init__(self, start_ms: int = 1_000) -> None:
        self.now = start_ms

    def monotonic_ms(self) -> int:
        return self.now

    def advance(self, milliseconds: int) -> None:
        self.now += milliseconds


def _remove_sqlite_files(path: Path) -> None:
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass


def _fresh_snapshot_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for candidate in path.glob("snapshot-v*.json"):
        candidate.unlink()


def _numeric_summary(layer_summaries: dict[str, Any]) -> dict[str, Any]:
    return {
        layer_id: {
            "float_digest": item["float_digest"],
            "quant_digest": item["quant_digest"],
            "combined_digest": item["combined_digest"],
            "max_abs_error": item["errors"]["max_abs_error"],
            "mean_abs_error": item["errors"]["mean_abs_error"],
            "sum_abs_error": item["errors"]["sum_abs_error"],
            "error_bound": item["error_bound"],
        }
        for layer_id, item in sorted(layer_summaries.items())
    }


def _run_in_temp(plan: Any, samples: int, batch_size: int, workers: int) -> tuple[dict[str, Any], Any]:
    with tempfile.TemporaryDirectory(prefix="mixedmem-demo-") as temporary:
        ledger = Ledger(Path(temporary) / "ledger.db")
        runtime = InferenceRuntime(
            plan, samples, batch_size=batch_size, ledger=ledger
        )
        report = runtime.run(worker_count=workers)
        ledger.close()
        return report, runtime


def _check(checks: list[dict[str, Any]], name: str, ok: bool, detail: Any = None) -> None:
    item: dict[str, Any] = {"name": name, "ok": bool(ok)}
    if detail is not None:
        item["detail"] = detail
    checks.append(item)


def _lease_semantics() -> dict[str, Any]:
    clock = DemoClock()
    pool = LeasePool(
        block_size_bytes=1,
        memory_budget_bytes=4,
        clock=clock,
    )
    first = pool.acquire("batch-a", "layer-a", 2, ttl_ms=10, plan_version=1)
    released_once = pool.release(first.lease_id, batch_id="batch-a", layer_id="layer-a")
    released_twice = pool.release(first.lease_id, batch_id="batch-a", layer_id="layer-a")
    second = pool.acquire("batch-b", "layer-b", 2, ttl_ms=10, plan_version=1)
    reused_after_release = first.block_range == second.block_range
    pool.release(second.lease_id, batch_id="batch-b", layer_id="layer-b")
    third = pool.acquire("batch-c", "layer-c", 3, ttl_ms=10, plan_version=1)
    clock.advance(10)
    expired = pool.expire_due()
    fourth = pool.acquire("batch-d", "layer-d", 3, ttl_ms=10, plan_version=1)
    reused_after_expiry = third.block_range == fourth.block_range
    ownership_error = None
    try:
        pool.release(fourth.lease_id, batch_id="wrong-owner", layer_id="layer-d")
    except DiagnosticError as exc:
        ownership_error = exc.to_dict()
    # A real monotonic clock, rather than the injected demo clock, is also
    # exercised so expiry is not only a fake-clock property.
    real_pool = LeasePool(block_size_bytes=1, memory_budget_bytes=2)
    real_lease = real_pool.acquire("real", "layer", 1, ttl_ms=1, plan_version=1)
    time.sleep(0.015)
    real_expired = real_pool.expire_due()
    return {
        "ok": (
            released_once == released_twice
            and reused_after_release
            and reused_after_expiry
            and [lease.lease_id for lease in expired] == [third.lease_id]
            and ownership_error is not None
            and ownership_error["code"] == "E_LEASE"
            and [lease.lease_id for lease in real_expired] == [real_lease.lease_id]
        ),
        "idempotent_release": released_once == released_twice,
        "released_range_reused": reused_after_release,
        "expired_range_reused": reused_after_expiry,
        "expired_leases": [lease.to_dict() for lease in expired],
        "ownership_mismatch": ownership_error,
        "real_clock_expiry": [lease.to_dict() for lease in real_expired],
        "pool_invariants": pool.verify_invariants(),
    }


def _budget_wait_scenario(raw_plan: dict[str, Any], workers: int) -> dict[str, Any]:
    raw = json.loads(json.dumps(raw_plan))
    profile = memory_profile(parse_plan(raw), int(raw["batch_size_max"]))
    raw["memory_budget_bytes"] = max(
        item["blocks"] for item in profile.values()
    ) * int(raw["block_size_bytes"])
    # The budget above must hold the whole layer bundle, not only one layer.
    raw["memory_budget_bytes"] = sum(item["blocks"] for item in profile.values()) * int(raw["block_size_bytes"])
    constrained = parse_plan(raw)
    report, _ = _run_in_temp(constrained, 8, 4, workers)
    waits = [int(batch["queue_wait_ms"] or 0) for batch in report["schedule"]["batches"]]
    return {
        "ok": (
            report["ok"]
            and report["schedule"]["invariants"]["ok"]
            and report["schedule"]["pool"]["used_blocks"] == 0
            and max(waits or [0]) > 0
        ),
        "budget_blocks": constrained.block_count,
        "queue_waits_ms": waits,
        "max_queue_wait_ms": max(waits or [0]),
        "workers": sorted(
            {batch["worker_name"] for batch in report["schedule"]["batches"] if batch["worker_name"]}
        ),
        "invariants": report["schedule"]["invariants"],
    }


def run_demo(workspace: Path, plan_path: Path, *, workers: int = 2) -> dict[str, Any]:
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    db_path = workspace / "ledger.db"
    snapshot_dir = workspace / "snapshots"
    _remove_sqlite_files(db_path)
    _fresh_snapshot_dir(snapshot_dir)
    checks: list[dict[str, Any]] = []

    plan_path = plan_path.resolve()
    plan = load_plan(plan_path)
    raw_plan = plan.to_dict()
    from .samples import SampleFactory

    first_factory = SampleFactory(plan, 8)
    second_factory = SampleFactory(plan, 8)
    deterministic_samples = first_factory.summary() == second_factory.summary()
    _check(
        checks,
        "deterministic_samples",
        deterministic_samples,
        {
            "dataset_digest": first_factory.dataset_digest(),
            "inputs": first_factory.summary()["input_matrices"],
            "weights": first_factory.summary()["weight_matrices"],
        },
    )

    ledger = Ledger(db_path)
    runtime = InferenceRuntime(plan, 8, batch_size=4, ledger=ledger)
    full_report = runtime.run(worker_count=workers)
    initial_summaries = runtime.layer_summaries()
    initial_versions = dict(runtime.layer_versions)
    integrity = ledger.verify_integrity()
    _check(
        checks,
        "scheduled_dual_path_run",
        full_report["ok"]
        and not full_report["schedule"]["failed_batches"]
        and full_report["schedule"]["invariants"]["ok"]
        and integrity["ok"],
        {
            "worker_count": workers,
            "batches": full_report["schedule"]["batches"],
            "layer_summaries": initial_summaries,
            "attribution": full_report["attribution"],
            "ledger_integrity": integrity,
        },
    )

    batch_comparisons: dict[str, Any] = {}
    batch_reports: dict[str, Any] = {}
    for batch_size in (1, 4):
        report, _ = _run_in_temp(plan, 8, batch_size, workers)
        batch_reports[str(batch_size)] = report
        batch_comparisons[str(batch_size)] = _numeric_summary(report["layer_summaries"])
    batch_size_invariant = batch_comparisons["1"] == batch_comparisons["4"]
    attribution_invariant = (
        batch_reports["1"]["attribution"] == batch_reports["4"]["attribution"]
    )
    _check(
        checks,
        "batch_size_and_order_invariance",
        batch_size_invariant and attribution_invariant,
        {
            "numeric_digests_equal": batch_size_invariant,
            "attribution_equal": attribution_invariant,
            "batch_size_4_summaries": batch_comparisons["4"],
            "attribution": batch_reports["4"]["attribution"],
        },
    )

    budget_wait = _budget_wait_scenario(raw_plan, workers)
    _check(checks, "real_budget_queue_wait", budget_wait["ok"], budget_wait)

    lease_semantics = _lease_semantics()
    _check(checks, "lease_expiry_and_return", lease_semantics["ok"], lease_semantics)

    held = runtime.hold_version_leases(runtime.batches[0].batch_id)
    cursor_before = ledger.recovery_view()["cursor"]
    snapshot_v1_payload = runtime.snapshot_payload(cursor=cursor_before)
    snapshot_v1 = save_snapshot(snapshot_dir, snapshot_v1_payload)
    _check(
        checks,
        "snapshot_v1_persisted",
        Path(snapshot_v1).exists(),
        {
            "path": snapshot_v1,
            "content_digest": snapshot_v1_payload["content_digest"],
            "processed_batch_cursor": cursor_before,
            "held_leases": [lease.to_dict() for lease in held],
        },
    )

    change = runtime.apply_change("hidden", {"scale": 256})
    after_summaries = runtime.layer_summaries()
    after_versions = dict(runtime.layer_versions)
    unchanged = {
        layer_id: {
            "float_digest_same": initial_summaries[layer_id]["float_digest"] == after_summaries[layer_id]["float_digest"],
            "quant_digest_same": initial_summaries[layer_id]["quant_digest"] == after_summaries[layer_id]["quant_digest"],
            "combined_digest_same": initial_summaries[layer_id]["combined_digest"] == after_summaries[layer_id]["combined_digest"],
            "lease_ranges_same": initial_summaries[layer_id]["lease_range"] == after_summaries[layer_id]["lease_range"],
            "layer_version_same": initial_versions[layer_id] == after_versions[layer_id],
        }
        for layer_id in change["unaffected"]
    }
    incremental_verification = runtime.verify_incremental()
    lease_status_rows = ledger.load_leases(plan.version)
    revoked_rows = [row for row in lease_status_rows if row["status"] == "revoked"]
    _check(
        checks,
        "invalidation_subset_and_lease_replacement",
        (
            change["invalidated"] == ["hidden", "compress", "head"]
            and change["unaffected"] == ["embed"]
            and all(all(item.values()) for item in unchanged.values())
            and len(change["lease_replacement"]["revoked"]) == len(held)
            and len(revoked_rows) == len(held)
            and incremental_verification["ok"]
        ),
        {
            "change": change,
            "unchanged": unchanged,
            "verification": incremental_verification,
            "revoked_rows_in_sqlite": revoked_rows,
        },
    )

    snapshot_v2_payload = runtime.snapshot_payload(cursor=ledger.recovery_view()["cursor"])
    snapshot_v2 = save_snapshot(snapshot_dir, snapshot_v2_payload)
    restored = load_snapshot(
        snapshot_v1,
        current_plan=runtime.plan,
        current_dataset_digest=runtime.factory.dataset_digest(),
        current_ledger_digest=ledger.content_digest(),
        current_lease_records=runtime.current_lease_ledger,
    )
    _check(
        checks,
        "snapshot_whole_state_replacement",
        restored.replaced_whole_state
        and restored.plan.version == 1
        and bool(restored.differences)
        and any(item["item"].endswith("hidden.scale") for item in restored.differences),
        {
            "snapshot_v2": snapshot_v2,
            "restored": restored.to_dict(),
        },
    )

    bad_snapshot_dir = workspace / "bad-snapshots"
    bad_snapshot_dir.mkdir(parents=True, exist_ok=True)
    bad_path = bad_snapshot_dir / "snapshot-bad-format.json"
    bad_payload = dict(snapshot_v1_payload)
    bad_payload["format_version"] = 999
    atomic_write_json(bad_path, bad_payload)
    snapshot_error = None
    try:
        load_snapshot(bad_path, current_plan=runtime.plan)
    except DiagnosticError as exc:
        snapshot_error = exc.to_dict()
    version_mismatch_plan = replace(plan, quantization_scale_version=2)
    version_mismatch_payload = build_snapshot(
        version_mismatch_plan,
        sample_count=8,
        dataset_digest=runtime.factory.dataset_digest(),
        lease_records=[],
        layer_summaries={},
        processed_batch_cursor=None,
        ledger_content_digest=ledger.content_digest(),
    )
    version_snapshot = save_snapshot(bad_snapshot_dir, version_mismatch_payload)
    version_error = None
    try:
        load_snapshot(version_snapshot, current_plan=runtime.plan)
    except DiagnosticError as exc:
        version_error = exc.to_dict()
    _check(
        checks,
        "snapshot_version_rejection",
        snapshot_error is not None
        and snapshot_error["code"] == "E_SNAPSHOT"
        and version_error is not None
        and version_error["code"] == "E_SNAPSHOT",
        {"format_error": snapshot_error, "version_error": version_error},
    )
    ledger.close()

    crash_dir = workspace / "crash"
    crash_dir.mkdir(parents=True, exist_ok=True)
    crash_db = crash_dir / "ledger.db"
    crash_snapshots = crash_dir / "snapshots"
    _remove_sqlite_files(crash_db)
    crash_snapshots.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "mixedmem",
        "run",
        "--plan",
        str(plan_path),
        "--db",
        str(crash_db),
        "--snapshots",
        str(crash_snapshots),
        "--samples",
        "8",
        "--batch-size",
        "4",
        "--workers",
        "1",
        "--crash-after-committed-layers",
        "5",
    ]
    process = subprocess.run(
        command,
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
        timeout=60,
    )
    crash_ledger = Ledger(crash_db)
    crash_integrity = crash_ledger.verify_integrity()
    crash_pre = crash_ledger.recovery_view()
    recovery = recover_and_resume(
        plan=plan,
        sample_count=8,
        batch_size=4,
        ledger=crash_ledger,
        worker_count=workers,
    )
    crash_integrity_after = crash_ledger.verify_integrity()
    final_counts = crash_ledger.table_counts()
    crash_ledger.close()
    _check(
        checks,
        "forced_termination_replay",
        (
            process.returncode == 91
            and crash_integrity["ok"]
            and crash_integrity_after["ok"]
            and recovery["uncommitted_exit_report_available"] is False
            and len(recovery["unknown_batches"]) >= 1
            and len(recovery["unknown_leases"]) >= 1
            and recovery["replay_verification"]["ok"]
            and recovery["complete_after_resume"]
        ),
        {
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "pre_recovery": crash_pre,
            "recovery": recovery,
            "integrity_after": crash_integrity_after,
            "table_counts": final_counts,
        },
    )

    ok = all(item["ok"] for item in checks)
    final_plan_path = workspace / "plan-final.json"
    atomic_write_json(final_plan_path, runtime.plan.to_dict())
    result = {
        "ok": ok,
        "workspace": str(workspace),
        "plan_path": str(plan_path),
        "plan_version": runtime.plan.version,
        "checks": checks,
        "summary": {
            "dataset_digest": first_factory.dataset_digest(),
            "initial_layer_summaries": initial_summaries,
            "final_layer_summaries": after_summaries,
            "invalidated": change["invalidated"],
            "unaffected": change["unaffected"],
            "revoked_lease_count": len(change["lease_replacement"]["revoked"]),
            "new_lease_count": len(change["lease_replacement"]["new_leases"]),
            "batch_size_invariant": batch_size_invariant,
            "attribution_invariant": attribution_invariant,
            "max_queue_wait_ms": budget_wait["max_queue_wait_ms"],
            "snapshot_differences": list(restored.differences),
            "recovery_unknown_batches": recovery["unknown_batches"],
            "recovery_resumed_batches": recovery["resumed_batches"],
        },
    }
    atomic_write_json(workspace / "report.json", result)
    lines = [
        "mixedmem local acceptance report",
        f"overall: {'PASS' if ok else 'FAIL'}",
        f"plan_version: {runtime.plan.version}",
        f"dataset_digest: {first_factory.dataset_digest()}",
        "",
    ]
    for item in checks:
        lines.append(f"[{'PASS' if item['ok'] else 'FAIL'}] {item['name']}")
    lines.extend(
        [
            "",
            f"invalidated: {', '.join(change['invalidated'])}",
            f"unaffected: {', '.join(change['unaffected'])}",
            f"revoked leases: {len(change['lease_replacement']['revoked'])}",
            f"new version leases: {len(change['lease_replacement']['new_leases'])}",
            f"batch-size invariant: {batch_size_invariant}",
            f"attribution invariant: {attribution_invariant}",
            f"max observed budget queue wait: {budget_wait['max_queue_wait_ms']} ms",
            f"forced termination return code: {process.returncode}",
            f"recovery unknown batches: {len(recovery['unknown_batches'])}",
            f"recovery resumed batches: {', '.join(recovery['resumed_batches'])}",
            f"snapshot v1: {snapshot_v1}",
            f"snapshot v2: {snapshot_v2}",
            f"final plan declaration: {final_plan_path}",
        ]
    )
    atomic_write_bytes(workspace / "summary.txt", ("\n".join(lines) + "\n").encode("utf-8"))
    return result
