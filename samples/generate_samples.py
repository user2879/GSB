#!/usr/bin/env python3
"""Regenerate deterministic local samples without external services."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mmfm.samplegen import generate_samples  # noqa: E402


if __name__ == "__main__":
    result = generate_samples(ROOT)
    print(json.dumps({"ok": True, "generated": result}, ensure_ascii=False, indent=2))
