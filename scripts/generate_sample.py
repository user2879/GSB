#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cross_coding.cli import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "generate-data",
                "--features",
                str(ROOT / "config" / "features.yaml"),
                "--out",
                str(ROOT / "data" / "feature_table.json"),
            ]
        )
    )
