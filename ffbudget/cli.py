"""Command line entry point for the feature-freshness budget component."""
from __future__ import annotations

import argparse
from pathlib import Path

from .engine import full_replay, run_pipeline
from .fingerprint import canonical_json


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(payload))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ffbudget",
        description="feature freshness budget and backfill idempotency")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the four-stage pipeline")
    run.add_argument("--config", required=True)
    run.add_argument("--batches", required=True)
    run.add_argument("--requests", required=True)
    run.add_argument("--state", required=True)
    run.add_argument("--mode", choices=["full", "auto"], default="auto")
    run.add_argument("--concurrency", type=int, default=None)
    run.add_argument("--out", required=True)

    sub.add_parser("gen-samples", help="regenerate deterministic samples")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "gen-samples":
        from tools.generate_samples import main as gen_main
        gen_main()
        return 0

    if args.mode == "full":
        canonical, _, _ = full_replay(
            args.config, args.batches, args.requests, args.state,
            concurrency_override=args.concurrency)
    else:
        canonical, _, _ = run_pipeline(
            args.config, args.batches, args.requests, args.state,
            mode="auto", concurrency_override=args.concurrency)
    _write(Path(args.out), canonical)
    print(f"wrote canonical result to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
