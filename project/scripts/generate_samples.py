#!/usr/bin/env python3
"""Generate all deterministic samples under the repository root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arbitration_workbench.generator import generate_samples  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--seed", type=int, default=20261121)
    args = parser.parse_args()
    manifest = generate_samples(args.root, seed=args.seed)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

