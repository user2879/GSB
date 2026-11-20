#!/usr/bin/env python3
"""Convenience wrapper for offline sample generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aedwb.samplegen import generate_sample


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--injected", action="store_true")
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()
    path = generate_sample(args.out, injected=args.injected, clean=not args.no_clean)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
