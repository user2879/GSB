"""Windows-friendly example entry point.

Run from the repository root:
    python examples\\run_demo.py --out output
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vgcover.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["demo", *sys.argv[1:]]))
