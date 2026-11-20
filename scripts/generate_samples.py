#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamparse.samplegen import DEFAULT_SEED, write_sample


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic local streamparse samples")
    parser.add_argument("--out", default="sample_data")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    metadata = write_sample(args.out, args.seed)
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
