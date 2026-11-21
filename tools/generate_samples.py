#!/usr/bin/env python3
"""Generate a deterministic JSON sample set for local, offline validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memdiag.util import atomic_write_json, generate_sample_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--output", default="data/samples.json")
    args = parser.parse_args()
    payload = generate_sample_payload(seed=args.seed, count=args.count, dim=args.dim)
    atomic_write_json(args.output, payload)
    first = payload["samples"][0]
    last = payload["samples"][-1]
    print(
        f"wrote {args.output}: seed={args.seed} count={args.count} dim={args.dim} "
        f"first={first['id']} last={last['id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
