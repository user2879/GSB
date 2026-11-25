#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence_diag.sample import DEFAULT_SEED, generate_sample  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic local sample")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    result = generate_sample(args.root, args.seed)
    print(json_dumps(result))
    return 0


def json_dumps(value: object) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
