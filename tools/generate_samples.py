#!/usr/bin/env python3
"""Generate deterministic local samples. No network or external services are used."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vecflow.samples import generate_samples  # noqa: E402


def main() -> int:
    result = generate_samples(ROOT / "samples")
    print(json.dumps({"ok": True, **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
