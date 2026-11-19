#!/usr/bin/env python3
"""Launch the Tkinter console (requires a desktop + tkinter installation)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from block_snap.ui import TkUnavailable, launch  # noqa: E402

DEFAULT_WORKSPACE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workdata")

if __name__ == "__main__":
    workspace = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WORKSPACE
    try:
        launch(workspace)
    except TkUnavailable as exc:
        print(f"[ui unavailable] {exc}")
        raise SystemExit(3)
