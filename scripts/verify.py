#!/usr/bin/env python3
"""Run the full deterministic acceptance suite and write artifacts/validation.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gsb.acceptance import run_acceptance  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GSB end-to-end acceptance verification")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--out", default="artifacts/validation.json")
    args = parser.parse_args(argv)
    report = run_acceptance(args.root, output_path=args.out)
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
