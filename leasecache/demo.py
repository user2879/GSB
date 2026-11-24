from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .convergence import check_restart_convergence
from .engine import RunOptions, compare_business_snapshots, run_engine
from .util import atomic_write_json, canonical_sha256, mono_ms
from .verifier import verify_ledger


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE = PROJECT_ROOT / "samples" / "demo_spec.json"
SAMPLE_EDITED = PROJECT_ROOT / "samples" / "demo_spec_edited.json"


def _compare(
    name: str,
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    comparison = compare_business_snapshots(
        left["business_snapshot"], right["business_snapshot"]
    )
    comparison["name"] = name
    comparison["ok"] = comparison["equal"]
    return comparison


def run_acceptance(root: Path) -> dict[str, Any]:
    started = mono_ms()
    root = root.resolve()
    run_dir = root / f"run-{time.time_ns() // 1_000_000}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []
    paths: dict[str, str] = {}
    options_fast = RunOptions(lease_ms=100, wait_timeout_ms=6000)
    options_edit = RunOptions(lease_ms=100, wait_timeout_ms=6000)

    # Baseline + reuse in the same ledger.
    incremental_db = run_dir / "incremental.sqlite"
    incremental_ws = run_dir / "incremental-workspace"
    baseline = run_engine(SAMPLE, incremental_db, incremental_ws, options=options_fast)
    reuse = run_engine(SAMPLE, incremental_db, incremental_ws, options=options_fast)
    checks.append(_compare("baseline_vs_reuse", baseline, reuse))
    checks.append(
        {
            "name": "reuse_performed_no_network_requests",
            "ok": not reuse["incremental"]["recomputed_keys"]
            and not reuse["scheduling"]["requests"],
            "recomputed_keys": reuse["incremental"]["recomputed_keys"],
        }
    )

    # Edit a reference, continue incrementally, then compare against a fresh DB
    # full rebuild of the edited specification.
    edited_incremental = run_engine(
        SAMPLE_EDITED, incremental_db, incremental_ws, options=options_edit
    )
    full_db = run_dir / "full.sqlite"
    full_ws = run_dir / "full-workspace"
    full_edited = run_engine(SAMPLE_EDITED, full_db, full_ws, options=options_edit)
    checks.append(_compare("incremental_edited_vs_full_edited", edited_incremental, full_edited))
    checks.append(
        {
            "name": "incremental_only_rebuilt_affected_closure",
            "ok": set(edited_incremental["incremental"]["recomputed_keys"])
            == {"obj/lib", "pkg/bundle"},
            "recomputed_keys": edited_incremental["incremental"]["recomputed_keys"],
        }
    )

    # Force the master process to terminate after two committed grants.  This
    # uses os._exit in a child process: no finally block, clean marker or report
    # can run.  The resumed run may only trust committed SQLite rows.
    recovery_db = run_dir / "recovery.sqlite"
    recovery_ws = run_dir / "recovery-workspace"
    recovery_report_path = recovery_db.with_suffix(".report.json")
    interrupt_command = [
        sys.executable,
        "-m",
        "leasecache",
        "run",
        "--spec",
        str(SAMPLE_EDITED),
        "--db",
        str(recovery_db),
        "--workspace",
        str(recovery_ws),
        "--lease-ms",
        "120",
        "--wait-timeout-ms",
        "6000",
        "--interrupt-after-grants",
        "2",
    ]
    environment = dict(os.environ)
    prior = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(PROJECT_ROOT) + (
        os.pathsep + prior if prior else ""
    )
    interrupted = subprocess.run(
        interrupt_command,
        cwd=str(PROJECT_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    report_absent_immediately = not recovery_report_path.exists()
    checks.append(
        {
            "name": "forced_termination_has_no_exit_report",
            "ok": interrupted.returncode == 86 and report_absent_immediately,
            "returncode": interrupted.returncode,
            "report_absent": report_absent_immediately,
            "stderr_tail": interrupted.stderr[-500:],
        }
    )
    resumed = run_engine(
        SAMPLE_EDITED, recovery_db, recovery_ws, options=options_edit
    )
    checks.append(
        {
            "name": "recovery_marked_open_leases_unknown",
            "ok": bool((resumed.get("recovery") or {}).get("recovered")),
            "recovery": resumed.get("recovery"),
        }
    )
    checks.append(_compare("recovered_vs_full_edited", resumed, full_edited))

    # Independent ledger verification for every final ledger.
    verification: dict[str, Any] = {}
    for name, spec, db in (
        ("incremental", SAMPLE_EDITED, incremental_db),
        ("full", SAMPLE_EDITED, full_db),
        ("recovered", SAMPLE_EDITED, recovery_db),
    ):
        verification[name] = verify_ledger(spec, db)
    checks.append(
        {
            "name": "independent_verification",
            "ok": all(item["status"] == "ok" for item in verification.values()),
            "results": verification,
        }
    )

    # Explicit node restart convergence using a changed seed.  The first sync
    # detects changed digests; every later actual restart regenerates the same
    # copies and reports no further changes.
    changed_spec_raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    for node in changed_spec_raw["nodes"]:
        if node["id"] == "node-a":
            node["random_seed"] = 909
    changed_spec = run_dir / "seed-changed.json"
    atomic_write_json(changed_spec, changed_spec_raw)
    seed_db = run_dir / "seed.sqlite"
    seed_ws = run_dir / "seed-workspace"
    seed_run = run_engine(changed_spec, seed_db, seed_ws, options=options_edit)
    convergence = check_restart_convergence(
        SAMPLE,
        seed_db,
        seed_ws,
        report_path=run_dir / "convergence.json",
    )
    checks.append(
        {
            "name": "node_restart_digest_convergence",
            "ok": bool(convergence["stable_affected_keys"])
            and all(
                not item["post_restart_affected_keys"]
                for item in convergence["restart_results"]
            ),
            "stable_affected_keys": convergence["stable_affected_keys"],
            "restart_results": convergence["restart_results"],
        }
    )

    paths.update(
        {
            "run_dir": str(run_dir),
            "baseline_report": str(incremental_db.with_suffix(".report.json")),
            "incremental_db": str(incremental_db),
            "full_db": str(full_db),
            "recovery_db": str(recovery_db),
            "seed_db": str(seed_db),
            "convergence_report": str(run_dir / "convergence.json"),
        }
    )
    ok = all(check["ok"] for check in checks)
    summary = {
        "ok": ok,
        "format_version": 1,
        "started_mono_ms": started,
        "elapsed_mono_ms": mono_ms() - started,
        "sample_sha256": canonical_sha256(
            json.loads(SAMPLE.read_text(encoding="utf-8"))
        ),
        "checks": checks,
        "verification": verification,
        "paths": paths,
        "actual_results": {
            "baseline_recomputed": baseline["incremental"]["recomputed_keys"],
            "reuse_recomputed": reuse["incremental"]["recomputed_keys"],
            "edited_incremental_recomputed": edited_incremental["incremental"][
                "recomputed_keys"
            ],
            "recovery_unknown_leases": (resumed.get("recovery") or {}).get(
                "recovered", []
            ),
            "queue_waits_in_baseline": len(baseline["scheduling"]["queue_waits"]),
            "peak_global_inflight": baseline["scheduling"]["peak_global_inflight"],
            "seed_run_reusable": seed_run["incremental"]["reusable_keys"],
            "convergence_affected_keys": convergence["affected_keys"],
        },
    }
    atomic_write_json(run_dir / "acceptance.json", summary)
    atomic_write_json(root / "latest.json", summary)
    summary["paths"]["acceptance_report"] = str(run_dir / "acceptance.json")
    return summary
