"""Workspace launcher that prefers the locally staged NumPy 2 runtime."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
for candidate in (ROOT / "src", ROOT / ".deps"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from gsb_ledger.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

