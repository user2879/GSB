from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joinlab.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "demo",
                "--workload",
                str(ROOT / "samples" / "demo_workload.json"),
                "--db",
                str(ROOT / "artifacts" / "demo_ledger.sqlite"),
                "--out",
                str(ROOT / "artifacts" / "demo_report.json"),
                "--reset",
                *sys.argv[1:],
            ]
        )
    )
