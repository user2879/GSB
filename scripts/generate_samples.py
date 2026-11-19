#!/usr/bin/env python3
"""Generate the deterministic local sample dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_forensics.samples import generate_sample_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="sample_data")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = generate_sample_dataset(args.output, force=args.force)
    print(
        json.dumps(
            {
                "status": "ok",
                "output": args.output,
                "seed": manifest["seed"],
                "occurrences": len(manifest["occurrences"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
