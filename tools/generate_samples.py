#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cm_fusion.sample_data import generate_samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic local JSONL samples")
    parser.add_argument("--out", default="samples")
    parser.add_argument("--seed", type=int, default=20261120)
    args = parser.parse_args()
    result = generate_samples(args.out, args.seed)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

