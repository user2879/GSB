#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from search_proto.generator import generate_samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic local samples")
    parser.add_argument("--out-dir", type=Path, default=Path("samples"))
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--documents", type=int, default=24)
    parser.add_argument("--queries", type=int, default=8)
    args = parser.parse_args()
    manifest = generate_samples(
        args.out_dir,
        seed=args.seed,
        document_count=args.documents,
        query_count=args.queries,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
