"""Repeatable Windows-friendly entry point for the full acceptance sequence."""

from __future__ import annotations

import argparse
from pathlib import Path

from spectral_diag.demo import run_fixed_sequence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="demo_workspace")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    report = run_fixed_sequence(Path(args.workspace), reset=args.reset)
    print(f"status: {report['status']}")
    print(f"workspace: {report['workspace']}")
    print(f"report: {report['report_path']}")
    print(f"verification: {report['markdown_path']}")
    for name, passed in report["checks"].items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
