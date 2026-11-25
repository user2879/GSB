#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge_ledger.samplegen import generate_workspace


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic local samples")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    summary = generate_workspace(args.root.resolve())
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
