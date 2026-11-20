"""Direct entry point: python demo.py [workdir]."""
from __future__ import annotations

import sys

from sched_diag.demo import run_demo


if __name__ == "__main__":
    workdir = sys.argv[1] if len(sys.argv) > 1 else "artifacts/demo-state"
    run_demo(workdir)
