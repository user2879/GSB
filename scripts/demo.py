#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from colpage_recovery.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["demo", *sys.argv[1:]]))
