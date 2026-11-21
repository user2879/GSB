#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from candidate_exchange.validation import run_validation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete local acceptance suite")
    parser.add_argument("--data", default="data/sample")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()
    report = run_validation(Path(args.data), Path(args.out), seed=args.seed, keep_work=args.keep_work)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    return 0 if report["summary"]["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
