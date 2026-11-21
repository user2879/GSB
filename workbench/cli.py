"""Command line interface for the verification workbench."""

import argparse
import json
from pathlib import Path

from .config import Config
from .edits import apply_edits
from .inject import run_injections
from .pipeline import run_pipeline
from .samplegen import build_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(prog="workbench")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("gen-samples", help="generate deterministic local dataset")
    p_gen.add_argument("--config", required=True)
    p_gen.add_argument("--data", required=True)

    p_run = sub.add_parser("run", help="run verification pipeline")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--data", required=True)
    p_run.add_argument("--out", required=True)
    p_run.add_argument("--cache", default=None)
    p_run.add_argument("--snapshots", default=None)
    p_run.add_argument("--incremental", action="store_true")
    p_run.add_argument("--concurrency", type=int, default=None)
    p_run.add_argument("--verify-serial", action="store_true")

    p_edit = sub.add_parser("apply-edits", help="apply edit sequence to a data copy")
    p_edit.add_argument("--config", required=True)
    p_edit.add_argument("--data", required=True)
    p_edit.add_argument("--edits", required=True)

    p_inj = sub.add_parser("inject-run", help="run with fault injections")
    p_inj.add_argument("--config", required=True)
    p_inj.add_argument("--data", required=True)
    p_inj.add_argument("--injections", required=True)
    p_inj.add_argument("--out", required=True)

    args = parser.parse_args(argv)

    if args.command == "gen-samples":
        config = Config.load(args.config)
        samples = build_dataset(config, args.data)
        print(json.dumps({"samples": samples}, ensure_ascii=False))
        return 0

    if args.command == "run":
        report = run_pipeline(
            args.config,
            args.data,
            args.out,
            cache_path=args.cache,
            snapshot_dir=args.snapshots,
            incremental=args.incremental,
            concurrency=args.concurrency,
            verify_serial=args.verify_serial,
        )
        summary = {
            "samples": len(report["samples"]),
            "ledger_entries": len(report["ledger"]),
            "arbitration_records": len(report["arbitration"]),
            "deadlocks": len(report["deadlocks"]),
            "failures": len(report["failures"]),
            "serial_consistent": report["concurrency"]["serial_consistent"],
            "stale": report["snapshot"]["stale"],
        }
        print(json.dumps(summary, ensure_ascii=False))
        return 0

    if args.command == "apply-edits":
        count = apply_edits(args.edits, args.data, args.config)
        print(json.dumps({"applied": count}, ensure_ascii=False))
        return 0

    if args.command == "inject-run":
        report = run_injections(args.config, args.data, args.injections, args.out)
        print(json.dumps({"injections": report["injections"]}, ensure_ascii=False))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
