"""Command line entry point."""

from __future__ import annotations

import argparse
import sys

from .demo import run_demo, write_report
from .errors import GlyphError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="glyphforge", description="Deterministic outline geometry demo")
    subparsers = parser.add_subparsers(dest="command")
    demo_parser = subparsers.add_parser("demo", help="run the fixed end-to-end sequence")
    demo_parser.add_argument("--seed", type=int, default=20261119)
    demo_parser.add_argument("--report", default=None, help="optional output JSON path")
    demo_parser.add_argument(
        "--sensitivity-sizes",
        default="12,24,48",
        help="comma-separated pixel sizes used by sensitivity diagnostics",
    )
    args = parser.parse_args(argv)
    if args.command != "demo":
        parser.print_help(sys.stderr)
        return 2
    try:
        sizes = tuple(int(part) for part in args.sensitivity_sizes.split(",") if part)
        report = run_demo(args.seed, sensitivity_sizes=sizes)
        from .demo import canonical_report

        data = canonical_report(report)
        if args.report:
            write_report(report, args.report)
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
        return 0
    except (GlyphError, ValueError) as exc:
        if isinstance(exc, GlyphError):
            sys.stderr.write(f"error: {exc.code.value}: {exc.message}\n")
        else:
            sys.stderr.write(f"error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
