"""Generate deterministic local samples and injected-invalid fixtures.

Usage:
    python generate_samples.py --seed 20261124 --out samples
"""
from __future__ import annotations

import argparse
import json

from alert_pipeline import samplegen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--out", default="samples")
    args = parser.parse_args()
    paths = samplegen.generate_all(args.out, seed=args.seed)
    injected = samplegen.generate_injected(args.out)
    print(json.dumps({"samples": paths,
                      "injected_dir": injected["injected_dir"]},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
