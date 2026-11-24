#!/usr/bin/env python3
"""Run the complete probelease demonstration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probelease.demo import run_demo


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/results/demo")
    report = run_demo(root)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
