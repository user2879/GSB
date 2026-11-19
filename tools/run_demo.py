"""Windows-friendly wrapper: python tools\\run_demo.py --all-anomalies."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_ledger.cli import main  # noqa: E402


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if not arguments:
        arguments = ["--all-anomalies"]
    raise SystemExit(main(["demo", *arguments]))
