"""Repeatable end-to-end demonstration and fault-injection driver.

Normal sequence (default):
    1. cold-start full fan-out
    2. tightened SLO + changed concurrency, re-run (deterministic SLO failure)
    3. crash scenario: force-kill mid-session, resume from committed records,
       then deterministic replay verification

Fault injections (input/parameter only; source code is never modified):
    --inject corrupt-scene   -> E_SCENE
    --inject occupied-port   -> E_PORT
    --inject bad-snapshot    -> E_SNAPSHOT

All artifacts land under run/demo/<scenario>/ with JSONL structured logs,
SQLite ledgers, atomic snapshots and stable-order summary reports.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fanout_ledger.config import load_scene
from fanout_ledger.engine import render_report, run_session
from fanout_ledger.errors import LedgerError, PortError, SnapshotError
from fanout_ledger.ledger import Ledger
from fanout_ledger.logging_util import StructuredLogger
from fanout_ledger.mock_service import start_all, stop_all
from fanout_ledger.recovery import resume_session, verify_replay
from fanout_ledger.snapshot import load_snapshot, verify_compatibility, write_snapshot

DEMO_ROOT = ROOT / "run" / "demo"
SAMPLES = ROOT / "samples"
PYTHON = sys.executable


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _mock_config(scene) -> list[dict]:
    return [
        {
            "node_id": n.node_id, "failure_rate": n.failure_rate,
            "slow_rate": n.slow_rate, "process_ms": n.process_ms,
            "slow_extra_ms": n.slow_extra_ms, "seed": n.seed,
        }
        for n in scene.nodes
    ]


def scenario_one(logger: StructuredLogger) -> int:
    logger.info("demo_scenario_start", scenario=1, name="cold_start")
    run_dir = DEMO_ROOT / "01_cold_start"
    _reset_dir(run_dir)
    scene = load_scene(SAMPLES / "scene_cold_start.json")
    handles = start_all(scene.nodes)
    try:
        result = run_session(scene, run_dir, session_id="cold-0001", logger=logger)
    finally:
        stop_all(handles)
    report = render_report(result, scene)
    (run_dir / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    assert result.judgment["passed"], "cold start must pass its SLO"
    return 0


def scenario_two(logger: StructuredLogger) -> int:
    logger.info("demo_scenario_start", scenario=2, name="retune")
    run_dir = DEMO_ROOT / "02_retune"
    _reset_dir(run_dir)
    scene = load_scene(SAMPLES / "scene_retune.json")
    handles = start_all(scene.nodes)
    try:
        result = run_session(scene, run_dir, session_id="retune-0002", logger=logger)
    finally:
        stop_all(handles)
    report = render_report(result, scene)
    (run_dir / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    assert not result.judgment["passed"], "retune scenario must fail the tightened SLO"
    reason_types = {r["type"] for r in result.judgment["reasons"]}
    assert "success_rate" in reason_types and "p95_latency" in reason_types
    assert result.retry_exhaustions, "E_RETRY markers expected"
    return 0


def scenario_three(logger: StructuredLogger) -> int:
    logger.info("demo_scenario_start", scenario=3, name="crash_recovery")
    run_dir = DEMO_ROOT / "03_crash_recovery"
    _reset_dir(run_dir)
    scene = load_scene(SAMPLES / "scene_crash_recovery.json")
    db_path = run_dir / "ledger.sqlite3"
    snapshot_dir = run_dir / "snapshots"

    # --- phase A: child process runs the session, we kill it mid-flight ---
    env = dict(os.environ)
    child = subprocess.Popen(
        [PYTHON, "-m", "fanout_ledger", "run",
         str(SAMPLES / "scene_crash_recovery.json"),
         "--run-dir", str(run_dir), "--session-id", "crash-0003",
         "--snapshot-every", "2"],
        cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    killed = False
    deadline = time.time() + 30
    while time.time() < deadline:
        snapshot_files = sorted(snapshot_dir.glob("snapshot-*.json")) if snapshot_dir.exists() else []
        if db_path.exists() and snapshot_files:
            # a periodic snapshot is durably on disk and more work remains;
            # kill now so some in-flight/uncommitted deliveries are interrupted
            try:
                ledger = Ledger(db_path, "crash-0003")
                committed = len(ledger.fetch_attempts())
                ledger.close()
            except Exception:
                committed = 0
            expected = scene.task.deliveries_per_node * len(scene.nodes)
            if committed >= 2 and committed < expected and not killed:
                child.kill()  # SIGKILL: forced termination, no graceful cleanup
                child.wait(timeout=10)
                killed = True
                logger.info("child_force_killed", committed_attempts=committed,
                            snapshot=snapshot_files[-1].name)
                break
        time.sleep(0.05)
    if not killed:
        child.wait(timeout=10)
        raise RuntimeError("crash injection window was missed")
    assert child.returncode != 0

    # --- phase B: resume using only committed records ---
    from fanout_ledger.snapshot import cleanup_staging_files
    removed_tmp = cleanup_staging_files(snapshot_dir)
    logger.info("staging_files_cleaned", removed=removed_tmp)
    handles = start_all(scene.nodes)
    try:
        snapshots = sorted(snapshot_dir.glob("snapshot-*.json"))
        snapshot = load_snapshot(snapshots[-1])
        ledger = Ledger(db_path, "crash-0003")
        resume = resume_session(scene, ledger, snapshot, snapshot_dir)
        unknown_events = ledger.fetch_recovery_events()
        from fanout_ledger.metrics import aggregate, judge_slo
        from fanout_ledger.scheduler import (FanoutScheduler,
                                             trajectory_from_ledger)
        records = ledger.fetch_attempts()
        metrics = aggregate(records, scene)
        judgment = judge_slo(metrics, scene)
        # rebuild full state from committed ledger rows; never rely on the
        # empty in-memory counters of a freshly started recovery process
        trajectory = trajectory_from_ledger(records)
        node_attempts_used: dict[str, int] = {}
        node_budget: dict[str, int] = {}
        for record in records:
            node_attempts_used[record.node_id] =                 node_attempts_used.get(record.node_id, 0) + 1
            node_budget[record.node_id] =                 node_budget.get(record.node_id, 0) + record.budget_consumed_ms
        global_budget = sum(node_budget.values())
        final_snapshot = FanoutScheduler(scene, ledger).snapshot(
            "recovered", metrics=metrics, judgment=judgment,
            trajectory=trajectory, node_attempts_used=node_attempts_used,
            budget_global_ms=global_budget, budget_per_node_ms=node_budget)
        final_path = write_snapshot(final_snapshot, snapshot_dir,
                                    name="snapshot-crash-0003-recovered.json")
        ledger.close()
        assert unknown_events, "unknown/in-flight attempts must be recorded on resume"
        logger.info("session_resumed",
                    rerun_jobs=resume["rerun_jobs"],
                    unknown_events=[dict(r) for r in unknown_events],
                    recovered_snapshot=str(final_path))

        # --- phase C: deterministic replay against the recovered snapshot ---
        recovered = load_snapshot(final_path)
        verify_compatibility(recovered, scene.scene_version,
                             scene.payload_digest, _mock_config(scene))
        replay_outcome = verify_replay(scene, recovered)
        logger.info("replay_verified", **replay_outcome)
    finally:
        stop_all(handles)

    lines = [
        "=" * 72,
        "CRASH/RECOVERY/REPLAY REPORT session=crash-0003",
        f"unknown_events_marked={len(unknown_events)}",
        f"rerun_jobs={resume['rerun_jobs']}",
        f"replay_matched_attempts={replay_outcome['matched_attempts']}",
        f"replay_p95_jitter_ms={replay_outcome['p95_jitter_ms']}",
        f"slo_passed={judgment['passed']}",
        f"recovered_snapshot={final_path}",
        "=" * 72,
    ]
    report = "\n".join(lines)
    (run_dir / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    return 0


# ------------------------------------------------------------- fault injection
def inject_corrupt_scene(logger: StructuredLogger) -> int:
    run_dir = DEMO_ROOT / "inject_corrupt_scene"
    _reset_dir(run_dir)
    bad_path = run_dir / "broken_scene.json"
    bad_path.write_text('{"scene_version": "x", "nodes": [', encoding="utf-8")
    try:
        load_scene(bad_path)
    except LedgerError as exc:
        logger.error("injection_corrupt_scene", **exc.to_dict())
        print(json.dumps(exc.to_dict(), indent=2, ensure_ascii=False))
        assert exc.code == "E_SCENE"
        assert not (run_dir / "ledger.sqlite3").exists(), "no half-written ledger"
        return 0
    raise AssertionError("E_SCENE was not raised")


def inject_occupied_port(logger: StructuredLogger) -> int:
    run_dir = DEMO_ROOT / "inject_occupied_port"
    _reset_dir(run_dir)
    scene = load_scene(SAMPLES / "scene_cold_start.json")
    victim = scene.nodes[0]
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    holder.bind((victim.host, victim.port))
    holder.listen(1)
    try:
        try:
            start_all(scene.nodes)
        except PortError as exc:
            logger.error("injection_occupied_port", **exc.to_dict())
            print(json.dumps(exc.to_dict(), indent=2, ensure_ascii=False))
            assert exc.code == "E_PORT"
            assert not (run_dir / "ledger.sqlite3").exists(), "no half-written ledger"
            return 0
        raise AssertionError("E_PORT was not raised")
    finally:
        holder.close()


def inject_bad_snapshot(logger: StructuredLogger) -> int:
    run_dir = DEMO_ROOT / "inject_bad_snapshot"
    _reset_dir(run_dir)
    scene = load_scene(SAMPLES / "scene_cold_start.json")
    handles = start_all(scene.nodes)
    try:
        result = run_session(scene, run_dir, session_id="snap-0004", logger=logger)
    finally:
        stop_all(handles)
    target = result.final_snapshot
    envelope = json.loads(target.read_text(encoding="utf-8"))
    envelope["body"]["budget_global_ms"] += 99999  # tamper, digest now stale
    target.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    try:
        load_snapshot(target)
    except SnapshotError as exc:
        logger.error("injection_bad_snapshot", **exc.to_dict())
        print(json.dumps(exc.to_dict(), indent=2, ensure_ascii=False))
        assert exc.code == "E_SNAPSHOT"
        # ledger remains internally consistent; only the tampered snapshot is rejected
        ledger = Ledger(run_dir / "ledger.sqlite3", "snap-0004")
        assert ledger.fetch_attempts(), "committed ledger rows must remain intact"
        ledger.close()
        return 0
    raise AssertionError("E_SNAPSHOT was not raised")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inject", choices=["corrupt-scene", "occupied-port",
                                             "bad-snapshot"], default=None)
    args = parser.parse_args()
    DEMO_ROOT.mkdir(parents=True, exist_ok=True)
    logger = StructuredLogger(DEMO_ROOT / "demo.jsonl")

    if args.inject == "corrupt-scene":
        return inject_corrupt_scene(logger)
    if args.inject == "occupied-port":
        return inject_occupied_port(logger)
    if args.inject == "bad-snapshot":
        return inject_bad_snapshot(logger)

    rc = 0
    for step in (scenario_one, scenario_two, scenario_three):
        rc = step(logger) or rc
    logger.info("demo_sequence_complete", return_code=rc)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
