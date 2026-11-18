from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mixedprec_diag.cli import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "demo",
                "--config-dir",
                str(ROOT / "configs"),
                "--report-dir",
                str(ROOT / "reports"),
                "--snapshot-dir",
                str(ROOT / "snapshots"),
                *sys.argv[1:],
            ]
        )
    )
