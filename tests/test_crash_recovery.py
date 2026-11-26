import os
import signal
import subprocess
import sys

from edge_cache.ledger import Ledger
from edge_cache.snapshot import SnapshotStore, build_diagnostic_report

HELPER = os.path.join(os.path.dirname(__file__), "support", "crash_runner.py")


def test_sigkill_recovery_uses_only_committed_snapshots(sample_config_dir, tmp_workdir):
    work_dir = os.path.join(tmp_workdir, "crashed")
    proc = subprocess.Popen(
        [sys.executable, HELPER, sample_config_dir, work_dir, "0.8"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        proc.wait(timeout=4.0)
        killed = False
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
        killed = True
    assert killed
    assert proc.returncode == -signal.SIGKILL

    snapshot_store = SnapshotStore(os.path.join(work_dir, "snapshots"))
    torn = snapshot_store.cleanup_temp()
    assert isinstance(torn, list)
    recovery = snapshot_store.load_current()
    assert recovery["status"] in ("recovered", "partial")
    if recovery["status"] == "partial":
        assert recovery["missing_stages"]
        assert recovery["manifest"]["cursor"]["unfinished_shards"]

    ledger = Ledger(os.path.join(work_dir, "ledger"))
    report = build_diagnostic_report(
        recovery, ledger,
        format_counts={"c1": 0, "c2": 0, "c3": 0},
        node_advances={},
        budget_consumed=recovery.get("manifest", {})
        .get("cursor", {})
        .get("budget_consumed", 0),
    )
    assert report["recovery_status"] in ("recovered", "partial")
    assert isinstance(report["unfinished_shards"], list)
    assert isinstance(report["missing_stages"], list)
    assert recovery["manifest"]["global_version"] >= 1
    assert recovery["manifest"]["config_hash"]
