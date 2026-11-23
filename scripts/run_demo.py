#!/usr/bin/env python3
"""Run the repeatable end-to-end demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from near_dup.demo import run_demo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="samples")
    parser.add_argument("--run-dir", default="runs/demo")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()
    report = run_demo(sample_dir=args.samples, run_dir=args.run_dir, report_dir=args.report_dir)
    print(
        json.dumps(
            {
                "incremental_vs_full_rebuild": report["incremental_vs_full_rebuild"],
                "concurrency_equal": report["concurrency"]["equal"],
                "recovery_equal": report["snapshot_recovery"]["resumed_equals_full"],
                "reports": [f"{args.report_dir}/diagnostic_report.json", f"{args.report_dir}/verification.md"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

