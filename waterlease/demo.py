from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .errors import DivergenceError
from .evolution import build_crash_spec, build_v2_spec
from .invalidation import plan_invalidation
from .ledger import Ledger, compare_deterministic_exports
from .process_manager import ProcessManager, PROJECT_ROOT
from .scheduler import Scheduler
from .spec import load_spec
from .water import FixedPointMeter, attribute_shard, summarize_attribution


ROOT = PROJECT_ROOT
SPEC_DIR = ROOT / "spec"
ARTIFACTS = ROOT / "artifacts"
LEDGER_DIR = ROOT / "ledger"
V2_DIR = ARTIFACTS / "spec-v2"
CRASH_SPEC_DIR = ARTIFACTS / "spec-crash"
DEMO_DB = LEDGER_DIR / "demo.sqlite3"
FULL_DB = LEDGER_DIR / "full-rebuild-v2.sqlite3"
CRASH_DB = LEDGER_DIR / "crash-recovery.sqlite3"
REPORT_PATH = ARTIFACTS / "demo-report.json"


def reset_database(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        Path(str(path) + suffix).unlink(missing_ok=True)


def scale_matrix(spec: Any) -> dict[str, Any]:
    order = [shard.shard_id for shard in sorted(spec.shards, key=lambda item: item.shard_id)]
    rank = {shard_id: index for index, shard_id in enumerate(order)}
    rows = []
    for exp in range(4):
        meter = FixedPointMeter(exp)
        records = []
        for shard in spec.shards:
            node = {node.node_id: node for node in spec.nodes}[shard.node_id]
            capacity = spec.capacity(shard.capacity_version)
            records.append(
                attribute_shard(
                    meter,
                    shard_id=shard.shard_id,
                    topo_rank=rank[shard.shard_id],
                    total_shards=len(spec.shards),
                    resource_units=shard.resource_units,
                    capacity_budget_units=capacity.budget_units,
                    node_max_leases=node.max_leases,
                )
            )
        summary = summarize_attribution(records)
        rows.append(
            {
                "scale_exp": exp,
                "scale": 10**exp,
                "dominant_segment": summary["dominant_segment"],
                "segment_share_basis_points": summary["segment_share_basis_points"],
                "global_bound": summary["global_budget_aggregation_error_upper_bound"],
            }
        )
    return {"allowed_scale_exponents": [0, 1, 2, 3], "matrix": rows}


def run_crash_demo() -> dict[str, Any]:
    reset_database(CRASH_DB)
    crash_path, crash_spec = build_crash_spec(SPEC_DIR, CRASH_SPEC_DIR, delay_ms=500)
    command = [
        sys.executable,
        "-m",
        "waterlease.cli",
        "--spec-dir",
        str(crash_path),
        "--ledger",
        str(CRASH_DB),
        "run",
        "--recover",
        "--crash-after-inflight",
        "2",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    started = time.monotonic()
    child = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = child.communicate(timeout=12)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(child.pid, signal.SIGKILL)
        stdout, stderr = child.communicate(timeout=5)
    finally:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    time.sleep(0.25)
    recovery_ledger = Ledger(CRASH_DB, crash_spec)
    try:
        recovery = recovery_ledger.recover()
        if len(recovery["reclaimed_reservations"]) < 2:
            raise RuntimeError(
                f"expected at least two active reservations before kill, got {recovery['reclaimed_reservations']}"
            )
        with ProcessManager(crash_spec.nodes, startup_timeout_s=10) as processes:
            scheduler = Scheduler(crash_spec, recovery_ledger, processes)
            rerun = scheduler.run()
        replay = recovery_ledger.assert_replay_ok()
    finally:
        recovery_ledger.close()

    return {
        "command": " ".join(command),
        "child_returncode": child.returncode,
        "child_was_timeout_killed": timed_out,
        "elapsed_s_before_child_reap": round(time.monotonic() - started, 3),
        "stdout_tail": stdout[-1200:],
        "stderr_tail": stderr[-1200:],
        "recovery": recovery,
        "rerun_summary": {
            "completed_total": rerun["completed_total"],
            "completed_in_run": rerun["completed_in_run"],
            "frozen_skipped": rerun["frozen_skipped"],
            "peak_global_units": rerun["peak_global_units"],
        },
        "replay": replay,
        "budget_reclaimed_first": True,
    }


def main() -> int:
    ARTIFACTS.mkdir(exist_ok=True)
    LEDGER_DIR.mkdir(exist_ok=True)
    reset_database(DEMO_DB)
    reset_database(FULL_DB)

    v1 = load_spec(SPEC_DIR)
    report: dict[str, Any] = {"stage": "starting", "v1_spec_digest": v1.spec_digest}

    v1_ledger = Ledger(DEMO_DB, v1)
    try:
        with ProcessManager(v1.nodes) as processes:
            first = Scheduler(v1, v1_ledger, processes).run()
            first_replay = v1_ledger.assert_replay_ok()
            before_bytes = v1_ledger.current_payload_bytes()
        report["first_run"] = {
            "completed_total": first["completed_total"],
            "frozen_skipped": first["frozen_skipped"],
            "peak_global_units": first["peak_global_units"],
            "peak_node_shards": first["peak_node_shards"],
            "global_attribution": first["global_water_attribution"],
            "queue_records": first["queue_records"],
            "dispatch_trace": first["dispatch_trace"],
            "replay": first_replay,
        }
    finally:
        v1_ledger.close()

    v2_path, v2 = build_v2_spec(SPEC_DIR, V2_DIR)
    plan = plan_invalidation(v1, v2)
    report["invalidation"] = plan
    report["scale_matrix"] = scale_matrix(v2)

    v2_ledger = Ledger(DEMO_DB, v2)
    snapshot_report = v2_ledger.snapshot_report
    try:
        with ProcessManager(v2.nodes) as processes:
            incremental = Scheduler(
                v2,
                v2_ledger,
                processes,
                run_ids=set(plan["invalid_set"]),
                force_recompute=set(plan["invalid_set"]),
            ).run()
            after_bytes = v2_ledger.current_payload_bytes()

            byte_preservation = {}
            for shard_id in plan["unaffected_set"]:
                if shard_id in before_bytes:
                    byte_preservation[shard_id] = {
                        "identical": before_bytes[shard_id] == after_bytes.get(shard_id),
                        "bytes": len(before_bytes[shard_id]),
                    }
            all_bytes_preserved = all(item["identical"] for item in byte_preservation.values())
            if not all_bytes_preserved:
                raise DivergenceError(
                    "unaffected ledger payload changed",
                    path="$.current_outcomes",
                    details=byte_preservation,
                )

            reset_database(FULL_DB)
            full_ledger = Ledger(FULL_DB, v2)
            try:
                full_run = Scheduler(v2, full_ledger, processes).run()
                expected = v2_ledger.export_deterministic_outcomes()
                actual = full_ledger.export_deterministic_outcomes()
                consistency = compare_deterministic_exports(expected, actual)
                full_replay = full_ledger.assert_replay_ok()
            finally:
                full_ledger.close()

        incremental_replay = v2_ledger.assert_replay_ok()
        report["incremental_run"] = {
            "completed_in_run": incremental["completed_in_run"],
            "frozen_skipped": incremental["frozen_skipped"],
            "peak_global_units": incremental["peak_global_units"],
            "queue_records": incremental["queue_records"],
            "replay": incremental_replay,
        }
        report["snapshot_replacement"] = {
            key: value
            for key, value in snapshot_report.items()
            if key not in ("differences",)
        }
        report["snapshot_difference_count"] = len(snapshot_report.get("differences", []))
        report["byte_preservation"] = byte_preservation
        report["all_unaffected_bytes_identical"] = all_bytes_preserved
        report["full_rebuild_consistency"] = consistency
        report["full_rebuild_run"] = {
            "completed_total": full_run["completed_total"],
            "peak_global_units": full_run["peak_global_units"],
            "replay": full_replay,
        }
    finally:
        v2_ledger.close()

    report["crash_recovery"] = run_crash_demo()
    report["ok"] = bool(
        report["first_run"]["replay"]["ok"]
        and report["all_unaffected_bytes_identical"]
        and report["full_rebuild_consistency"]["ok"]
        and report["incremental_run"]["replay"]["ok"]
        and report["crash_recovery"]["replay"]["ok"]
    )
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"first_run", "invalidation"}}, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    print(f"\nFull report: {REPORT_PATH}")
    if not report["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
