#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmwin.generate import DEFAULT_SEED, generate_samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic multimodal samples")
    parser.add_argument("--samples", default=str(ROOT / "samples"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--keep", action="store_true", help="do not remove the sample directory first")
    args = parser.parse_args()
    summary = generate_samples(args.samples, seed=args.seed, clean=not args.keep)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
