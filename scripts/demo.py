#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cdn_edge.cli import main  # noqa: E402

raise SystemExit(main(["demo", *sys.argv[1:]]))
