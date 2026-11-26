import subprocess
import sys
import threading
from pathlib import Path

from chaindiag.scheduler import BudgetScheduler, Task
from chaindiag.snapshot import SnapshotStore


def test_snapshot_commit_recovery_and_config_hash_change(tmp_path):
    store = SnapshotStore(tmp_path / "snap")
    report = {"status": "ok", "scheduler": {"pending": [], "consumed_budget": 3}}
    manifest = store.commit_report(report, config_hash="h1", exchange_version="e2")
    assert manifest["version"] == 1

    decision = store.recover(config_hash="h1", exchange_version="e2")
    assert decision.resume is True
    assert decision.reason == "committed_snapshot_usable"

    changed = store.recover(config_hash="h2", exchange_version="e2")
    assert changed.resume is False
    assert changed.reason == "config_hash_mismatch_rebuild"


def test_real_sigkill_leaves_committed_snapshot_recoverable(tmp_path):
    root = tmp_path / "kill-snap"
    store = SnapshotStore(root)
    store.commit_report({"status": "ok"}, config_hash="hash", exchange_version="e2")

    result = subprocess.run(
        [sys.executable, "-c",
         "from chaindiag.snapshot import simulate_hard_kill; simulate_hard_kill(%r)" % str(root)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == -9
    assert list(root.glob(".staging-*"))

    decision = SnapshotStore(root).recover(config_hash="hash", exchange_version="e2")
    assert decision.resume is True
    assert not list(root.glob(".staging-*"))


def test_thread_pool_never_exceeds_configured_concurrency():
    barrier = threading.Barrier(4)
    active = 0
    observed = 0
    lock = threading.Lock()

    def work():
        nonlocal active, observed
        with lock:
            active += 1
            observed = max(observed, active)
        try:
            barrier.wait(timeout=2)
            return True
        finally:
            with lock:
                active -= 1

    tasks = [Task(task_id=f"t{i}", ordinal=i, op_seq=0, token_seq=i, work=work) for i in range(8)]
    results, report = BudgetScheduler(point_budget=100, operator_quota=100, max_workers=4).run(tasks)
    assert len(results) == 8
    assert report.max_observed_workers == 4
    assert observed <= 4
