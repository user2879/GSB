#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gsb.fixtures import write_fixture_tree


if __name__ == "__main__":
    paths = write_fixture_tree(ROOT)
    for key, value in paths.items():
        print(f"{key}: {value.relative_to(ROOT)}")
