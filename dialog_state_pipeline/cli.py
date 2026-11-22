"""Command line entry point.

Examples
--------
python -m dialog_state_pipeline.cli run \
    --events samples/events.jsonl --fsm samples/fsm.yaml \
    --config samples/pipeline_config.json --report out/report.json
python -m dialog_state_pipeline.cli run ... --injection samples/injection_config.json
python -m dialog_state_pipeline.cli run ... --reset --mode full
"""

import argparse
import json

from .pipeline import run_from_files
from .sample_data import write_sample_dir


def build_parser():
    parser = argparse.ArgumentParser(description="Deterministic dialogue state pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("gen-samples", help="generate deterministic local samples")
    gen.add_argument("--out", default="samples")

    run = sub.add_parser("run", help="run the pipeline")
    run.add_argument("--events", required=True)
    run.add_argument("--fsm", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--report", required=True)
    run.add_argument("--injection", default=None)
    run.add_argument("--mode", default="incremental",
                     choices=["incremental", "full"])
    run.add_argument("--reset", action="store_true",
                     help="discard any existing snapshot before running")
    run.add_argument("--refill-budget", action="store_true",
                     help="treat global_budget as a fresh allocation on resume")
    run.add_argument("--stop-after-stage", default=None,
                     choices=["load", "advance", "settle"])
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "gen-samples":
        path = write_sample_dir(args.out)
        print(f"samples written to {path}")
        return 0
    report = run_from_files(
        args.events, args.fsm, args.config, args.report,
        injection_path=args.injection, mode=args.mode,
        stop_after_stage=args.stop_after_stage, reset=args.reset,
        refill_budget=args.refill_budget)
    if report is not None:
        print(json.dumps({
            "report": args.report,
            "mode": report["mode"],
            "completed_batches": report["completed_batches"],
            "failures": len(report["failures"]),
            "report_fingerprint": report["report_fingerprint"],
        }, indent=2))
    else:
        print(f"stopped after stage {args.stop_after_stage}; snapshot committed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
