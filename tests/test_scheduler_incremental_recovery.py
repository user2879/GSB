from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

from waterlease.demo import reset_database
from waterlease.errors import DivergenceError
from waterlease.evolution import build_v2_spec
from waterlease.invalidation import plan_invalidation
from waterlease.ledger import Ledger, compare_deterministic_exports
from waterlease.process_manager import ProcessManager
from waterlease.scheduler import Scheduler
from tests.support import ROOT, write_test_spec


def test_parallel_scheduler_dependencies_and_ledger(tmp_path) -> None:
    spec_dir, spec = write_test_spec(tmp_path, delay_ms=8, budget=6)
    db = tmp_path / "ledger.sqlite3"
    ledger = Ledger(db, spec)
    try:
        with ProcessManager(spec.nodes, startup_timeout_s=10) as processes:
            report = Scheduler(spec, ledger, processes).run()
        assert report["completed_total"] == 6
        assert report["frozen_skipped"] == ["s7_frozen"]
        assert report["replay"]["ok"]
        assert report["peak_global_units"] <= 6

        dispatch_events = [item for item in report["dispatch_trace"] if item["event"] == "dispatch"]
        assert len(dispatch_events) == 6
        for event in dispatch_events:
            assert event["detail"]["active_global_units"] <= 6

        join = ledger.current_outcome("s5_join_b")
        assert join is not None
        assert {item["shard_id"] for item in join["summary"]["upstream_digests"]} == {
            "s3_dep_a",
            "s4_dep_c",
        }
        assert ledger.active_reservations() == []
        frozen_reservations = ledger.conn.execute(
            "SELECT COUNT(*) FROM reservations WHERE shard_id = 's7_frozen'"
        ).fetchone()[0]
        assert frozen_reservations == 0
    finally:
        ledger.close()


def test_incremental_invalidation_preserves_unaffected_bytes(tmp_path) -> None:
    spec_dir, v1 = write_test_spec(tmp_path, delay_ms=4, budget=6)
    db = tmp_path / "incremental.sqlite3"
    first_ledger = Ledger(db, v1)
    try:
        with ProcessManager(v1.nodes, startup_timeout_s=10) as processes:
            Scheduler(v1, first_ledger, processes).run()
            before = first_ledger.current_payload_bytes()
        first_ledger.close()

        v2_dir, v2 = build_v2_spec(spec_dir, tmp_path / "spec-v2")
        plan = plan_invalidation(v1, v2)
        assert plan["invalid_set"] == [
            "s2_raw_b",
            "s4_dep_c",
            "s5_join_b",
            "s6_independent_c",
            "s7_frozen",
        ]
        assert plan["unaffected_set"] == ["s1_raw_a", "s3_dep_a"]

        v2_ledger = Ledger(db, v2)
        try:
            with ProcessManager(v2.nodes, startup_timeout_s=10) as processes:
                report = Scheduler(
                    v2,
                    v2_ledger,
                    processes,
                    run_ids=set(plan["invalid_set"]),
                    force_recompute=set(plan["invalid_set"]),
                ).run()
            after = v2_ledger.current_payload_bytes()
            assert report["completed_in_run"] == [
                "s2_raw_b",
                "s4_dep_c",
                "s5_join_b",
                "s6_independent_c",
            ]
            for shard_id in plan["unaffected_set"]:
                assert before[shard_id] == after[shard_id]
            assert v2_ledger.snapshot_report["action"] == "replaced"
            assert v2_ledger.snapshot_report["backup_path"]
        finally:
            v2_ledger.close()
    finally:
        if first_ledger.conn:
            first_ledger.close()


def test_blank_rebuild_comparison_locates_divergence() -> None:
    expected = {"a": {"b": 1, "c": [1, 2]}}
    actual = {"a": {"b": 2, "c": [1, 2]}}
    with pytest.raises(DivergenceError) as exc:
        compare_deterministic_exports(expected, actual)
    assert exc.value.code == "E_DIVERGE"
    assert exc.value.path == "$.outcomes.a.b"


def test_sigkill_recovery_reclaims_and_replays(tmp_path) -> None:
    spec_dir, spec = write_test_spec(tmp_path, delay_ms=300, budget=6)
    db = tmp_path / "crash.sqlite3"
    reset_database(db)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "-m",
        "waterlease.cli",
        "--spec-dir",
        str(spec_dir),
        "--ledger",
        str(db),
        "run",
        "--recover",
        "--crash-after-inflight",
        "2",
    ]
    child = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = child.communicate(timeout=12)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        stdout, stderr = child.communicate(timeout=5)
    finally:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    assert child.returncode == -9, (stdout, stderr)
    time.sleep(0.2)

    ledger = Ledger(db, spec)
    try:
        recovery = ledger.recover()
        assert len(recovery["reclaimed_reservations"]) == 2
        assert set(recovery["unknown_shards"]) == {"s1_raw_a", "s2_raw_b"}
        assert recovery["replay"]["ok"]
        with ProcessManager(spec.nodes, startup_timeout_s=10) as processes:
            report = Scheduler(spec, ledger, processes).run()
        assert report["completed_total"] == 6
        assert ledger.assert_replay_ok()["ok"]
    finally:
        ledger.close()
