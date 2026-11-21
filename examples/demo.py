#!/usr/bin/env python3
"""Run the single-stage, three-stage and snapshot-replay demo scenarios."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qosfdiag.demo import format_demo_summary, run_demo  # noqa: E402


if __name__ == "__main__":
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts" / "demo"
    print(format_demo_summary(run_demo(output)))
