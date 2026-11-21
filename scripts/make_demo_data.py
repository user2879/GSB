#!/usr/bin/env python3
"""Regenerate the checked-in deterministic demo declarations."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlpipe_workbench.demo_data import write_demo_inputs  # noqa: E402


def main() -> None:
    data_path, pipeline_path = write_demo_inputs(ROOT / "data")
    print(f"data={data_path}")
    print(f"pipeline={pipeline_path}")


if __name__ == "__main__":
    main()
