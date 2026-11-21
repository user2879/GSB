#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from candidate_exchange.generator import generate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the deterministic local sample dataset")
    parser.add_argument("--out", default="data/sample")
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = generate_dataset(Path(args.out), seed=args.seed, force=args.force)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
