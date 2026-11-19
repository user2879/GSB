from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from rsseg.edits import copy_sample_set

ROOT = Path(__file__).resolve().parents[1]


def test_sigterm_worker_finishes_current_commit_then_reports(sample_set, tmp_path):
    sources = tmp_path / "sources"
    work = tmp_path / "work"
    copy_sample_set(sample_set, sources)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "signal_demo.py"),
         "--sample-dir", str(sources), "--work-dir", str(work),
         "--signal-before-start"],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert proc.returncode == 0
    lines = proc.stdout.strip().splitlines()
    assert lines[-1] == "signaled"
    # No snapshot is published when no tile completed.
    assert lines[0] == "None"


def test_sigterm_during_normal_run_leaves_only_complete_snapshots(sample_set, tmp_path):
    sources = tmp_path / "sources"
    work = tmp_path / "work"
    copy_sample_set(sample_set, sources)
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "signal_demo.py"),
         "--sample-dir", str(sources), "--work-dir", str(work)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    import time

    deadline = time.time() + 10
    while time.time() < deadline and not (work / "cache").exists():
        if proc.poll() is not None:
            break
        time.sleep(0.01)
    proc.terminate()
    stdout, stderr = proc.communicate(timeout=30)
    # If the signal landed after handler installation the worker exits 0 with
    # a signaled marker; during interpreter startup the OS default may apply.
    assert proc.returncode in (0, -15)
    tmp_leftovers = list((work / "snapshots").glob("snapshot.tmp.*"))
    assert not tmp_leftovers
    if proc.returncode == 0:
        assert "signaled" in stdout or "normal" in stdout
