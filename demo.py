#!/usr/bin/env python3
"""Run the deterministic local demonstration."""

from __future__ import annotations

import argparse
import json
import sys

from txnmsg.demo import run_demo


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the txnmsg deterministic demo")
    parser.add_argument("--workdir", default="demo_workspace")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    result = run_demo(args.workdir)
    if args.quiet:
        print(f"DEMO {result['status']} summary={result['summary_path']}")
    else:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
