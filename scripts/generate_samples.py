#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fixed_lab.samplegen import generate_samples, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=18)
    parser.add_argument("--seed", type=int, default=396)
    parser.add_argument("--out", default="data/samples.jsonl")
    args = parser.parse_args()
    path = write_jsonl(generate_samples(args.count, args.seed), ROOT / args.out if not Path(args.out).is_absolute() else args.out)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
