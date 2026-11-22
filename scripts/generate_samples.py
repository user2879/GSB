#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stemscope.samplegen import DEFAULT_SEED, generate_samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic StemScope sample data")
    parser.add_argument("--out", default="sample_data", help="output directory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()
    manifest = generate_samples(args.out, seed=args.seed, force=not args.keep_existing)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
