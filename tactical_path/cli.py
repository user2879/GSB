"""Command-line interface for generation, experiments and recovery."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .canonical import canonical_bytes
from .config import load_config
from .demo import run_demo
from .errors import ErrorCode, TacticalError
from .mapgen import generate_all
from .persistence import StateStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tactical-path",
        description="Deterministic tactical grid routing experiments",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generator = subparsers.add_parser("generate", help="generate all deterministic tiers")
    generator.add_argument("--config", required=True)
    generator.add_argument("--output", required=True)

    demo = subparsers.add_parser("demo", help="run the full reproducible experiment")
    demo.add_argument("--config", required=True)
    demo.add_argument("--workspace", required=True)

    recover = subparsers.add_parser("recover", help="recover a committed state store")
    recover.add_argument("--workspace", required=True)

    codes = subparsers.add_parser("error-codes", help="print business error codes")
    codes.set_defaults(show_codes=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            cfg = load_config(args.config)
            report = generate_all(
                seed=cfg.seed,
                tiers=cfg.tiers,
                mobility_rules=cfg.mobility,
                output_root=args.output,
            )
            sys.stdout.buffer.write(canonical_bytes(report))
            return 0
        if args.command == "demo":
            report = run_demo(args.config, args.workspace)
            sys.stdout.buffer.write(canonical_bytes(report))
            return 0
        if args.command == "recover":
            store = StateStore(args.workspace)
            report = store.recover()
            sys.stdout.buffer.write(canonical_bytes(report))
            return 0
        if getattr(args, "show_codes", False):
            sys.stdout.buffer.write(
                canonical_bytes({"error_codes": [code.value for code in ErrorCode]})
            )
            return 0
        parser.error("unknown command")
        return 2
    except TacticalError as exc:
        sys.stderr.buffer.write(
            canonical_bytes({"status": "error", "error": exc.to_dict()})
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

