#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audio_fingerprint.demo import create_demo_workspace


if __name__ == "__main__":
    workspace = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "sample_workspace")
    print(create_demo_workspace(workspace))
