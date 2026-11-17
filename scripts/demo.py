#!/usr/bin/env python3
"""Windows-friendly demo entry point; run from any current directory."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exp_tracker.demo import demo_json  # noqa: E402


def main() -> None:
    workspace = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "demo_workspace"
    print(demo_json(workspace))
    print("DEMO_OK")


if __name__ == "__main__":
    main()

