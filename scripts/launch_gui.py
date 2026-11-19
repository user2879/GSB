from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shardrep.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["gui", "--spec", str(ROOT / "samples" / "spec.json"), *sys.argv[1:]]))

