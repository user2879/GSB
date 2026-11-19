"""Windows-friendly Tkinter launcher for the bundled divergence sample."""

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
        arguments = [
            "--spec",
            str(ROOT / "samples" / "divergence" / "spec.json"),
            "--sample",
            "divergence",
            "--runtime",
            str(ROOT / "runtime" / "gui"),
            "--ledger",
            str(ROOT / "runtime" / "gui" / "ledger.sqlite"),
            "--log",
            str(ROOT / "runtime" / "gui" / "structured.jsonl"),
        ]
    raise SystemExit(main(["gui", *arguments]))
