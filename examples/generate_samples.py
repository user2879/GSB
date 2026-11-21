#!/usr/bin/env python3
"""Generate all bundled local JSON samples without external downloads."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qosfdiag.samples import write_sample_files  # noqa: E402


if __name__ == "__main__":
    for path in write_sample_files(ROOT / "examples" / "data"):
        print(path)
