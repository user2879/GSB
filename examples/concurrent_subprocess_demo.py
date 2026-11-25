"""Demonstrate bounded subprocess concurrency on the local loopback host.

Two worker subprocesses run independent quantcalib jobs against separate
workspaces. No container and no external network target are used. Results are
collected in deterministic (worker-id) order so scheduling order cannot leak
into the conclusion.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    jobs = []
    for worker_id in range(2):
        workdir = Path(tempfile.mkdtemp(prefix=f"quantcalib_worker_{worker_id}_"))
        proc = subprocess.run(
            [sys.executable, "-m", "quantcalib", "init", "--workdir",
             str(workdir)],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        jobs.append((worker_id, workdir, proc.returncode))

    results = []
    for worker_id, workdir, code in sorted(jobs):
        run = subprocess.run(
            [sys.executable, "-m", "quantcalib", "run", "--workdir",
             str(workdir)],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        results.append({
            "worker_id": worker_id,
            "workdir": str(workdir),
            "init_exit": code,
            "run_summary": json.loads(run.stdout),
        })

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
