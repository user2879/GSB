"""Command line entry point for pipeline runs and exchange conversion."""
from __future__ import annotations

import argparse
import json

from .config import PipelineConfig
from .convert import (conversion_pair_v1_to_v2, conversion_pair_v2_to_v1,
                      read_sidecar, restore_v1_with_sidecar)
from .errors import FailureLog
from .pipeline import load_exchange, run_pipeline
from .v2 import write_v2


def _run(args) -> None:
    config = PipelineConfig.load(args.config)
    report = run_pipeline(
        exchange_path=args.exchange, version=args.version,
        source_wav=args.source, config=config,
        snapshot_dir=args.snapshot, output_wav=args.output,
        full_rebuild=args.full_rebuild, max_workers=args.workers)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _convert(args) -> None:
    config = PipelineConfig.load(args.config)
    failures = FailureLog()
    segments = load_exchange(args.input, args.from_version, config, failures)
    if failures.to_list():
        print(json.dumps(failures.to_list(), indent=2, ensure_ascii=False))
    if args.from_version == "v1":
        upgraded = conversion_pair_v1_to_v2(segments, config, args.output)
        print(f"upgraded {len(upgraded)} segments -> {args.output}")
    else:
        sidecar = conversion_pair_v2_to_v1(
            segments, config, args.output, args.sidecar)
        print(f"downgraded {len(sidecar['segments'])} segments; "
              f"sidecar -> {args.sidecar}")


def _restore(args) -> None:
    config = PipelineConfig.load(args.config)
    failures = FailureLog()
    segments = load_exchange(args.v1, "v1", config, failures)
    restored = restore_v1_with_sidecar(segments, read_sidecar(args.sidecar))
    write_v2(args.output, restored)
    print(f"restored {len(restored)} segments -> {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic VC pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the pipeline")
    run.add_argument("--exchange", required=True)
    run.add_argument("--version", choices=("v1", "v2"), required=True)
    run.add_argument("--source", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--snapshot", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--full-rebuild", action="store_true")
    run.add_argument("--workers", type=int, default=None)
    run.set_defaults(func=_run)

    conv = sub.add_parser("convert", help="convert between v1 and v2")
    conv.add_argument("--input", required=True)
    conv.add_argument("--from-version", choices=("v1", "v2"), required=True)
    conv.add_argument("--output", required=True)
    conv.add_argument("--config", required=True)
    conv.add_argument("--sidecar", default="sidecar.json")
    conv.set_defaults(func=_convert)

    rest = sub.add_parser("restore", help="restore v2 from v1 + sidecar")
    rest.add_argument("--v1", required=True)
    rest.add_argument("--sidecar", required=True)
    rest.add_argument("--output", required=True)
    rest.add_argument("--config", required=True)
    rest.set_defaults(func=_restore)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
