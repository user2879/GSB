from __future__ import annotations

import argparse
import json
from pathlib import Path

from .demo import run_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spmm-diag",
        description="Deterministic CSR SpMM incremental diagnostics workbench",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run the deterministic acceptance scenario")
    demo.add_argument("--seed", type=int, default=20261123)
    demo.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    demo.add_argument("--snapshot-dir", type=Path, default=Path("artifacts/snapshots"))
    demo.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        report = run_demo(
            seed=args.seed,
            output_dir=args.output_dir,
            snapshot_dir=args.snapshot_dir,
        )
        if not args.quiet:
            print(
                json.dumps(
                    {
                        "report": str(args.output_dir / "demo_report.json"),
                        "markdown": str(args.output_dir / "diagnostic_report.md"),
                        "final_c": str(args.output_dir / "final_c.npy"),
                        "all_acceptance_checks_passed": report["all_acceptance_checks_passed"],
                        "assertions": report["assertions"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        return 0 if report["all_acceptance_checks_passed"] else 1
    return 2

