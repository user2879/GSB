"""Command line interface for mlimgedit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .injection import load_injections
from .plan import EditPlan
from .runner import (
    comparable_state,
    full_run,
    rebuild_final_state,
    run_plan,
    write_report,
)
from .snapshot import SnapshotStore


def _print_summary(report: dict) -> None:
    counts = report["counts"]
    print(
        "batch={batch} samples={samples} reuse={reuse} recompute={recompute} "
        "pending={pending} failures={failures}".format(
            batch=report["batch_index"],
            samples=counts["samples"],
            reuse=counts["reuse"],
            recompute=counts["recompute"],
            pending=counts["pending"],
            failures=counts["failures"],
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlimgedit", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="one full pass over a sample directory")
    run.add_argument("--config", required=True)
    run.add_argument("--samples", required=True)
    run.add_argument("--store", required=True)
    run.add_argument("--report", required=True)
    run.add_argument("--injections")

    plan = sub.add_parser("plan", help="replay an incremental edit plan")
    plan.add_argument("--plan", required=True)
    plan.add_argument("--workdir", required=True)
    plan.add_argument("--store", required=True)
    plan.add_argument("--reports-dir", required=True)
    plan.add_argument("--injections")

    verify = sub.add_parser("verify", help="incremental vs full rebuild field comparison")
    verify.add_argument("--plan", required=True)
    verify.add_argument("--workdir", required=True)
    verify.add_argument("--store", required=True)
    verify.add_argument("--reports-dir", required=True)
    verify.add_argument("--injections")

    sub.add_parser("inspect", help="print the committed snapshot manifest").add_argument(
        "--store", required=True
    )
    return parser


def _cmd_run(args) -> int:
    from .runner import run_from_paths

    report = run_from_paths(
        args.config, args.samples, args.store, args.report, args.injections
    )
    _print_summary(report)
    return 0


def _cmd_plan(args) -> int:
    edit_plan = EditPlan.load(args.plan)
    injections = load_injections(args.injections)
    reports = run_plan(
        edit_plan,
        args.workdir,
        args.store,
        injections=injections,
        reports_dir=args.reports_dir,
    )
    for report in reports:
        write_report(Path(args.reports_dir) / f"report_batch_{report['batch_index']:02d}.json", report)
        _print_summary(report)
    return 0


def _cmd_verify(args) -> int:
    edit_plan = EditPlan.load(args.plan)
    injections = load_injections(args.injections)
    reports = run_plan(
        edit_plan,
        args.workdir,
        args.store,
        injections=injections,
        reports_dir=args.reports_dir,
    )
    final_config, final_samples = rebuild_final_state(edit_plan, Path(args.workdir))
    full_report = full_run(
        final_config,
        final_samples,
        Path(args.workdir) / "full_store",
        injections=injections,
    )
    write_report(Path(args.reports_dir) / "report_full_rebuild.json", full_report)
    final_incremental = reports[-1]
    left = comparable_state(final_incremental)
    right = comparable_state(full_report)
    equal = left == right
    result = {
        "equal": equal,
        "incremental_report": str(Path(args.reports_dir) / "report_batch_final.json"),
        "full_report": str(Path(args.reports_dir) / "report_full_rebuild.json"),
        "batches": len(reports),
    }
    write_report(Path(args.reports_dir) / "verification.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if equal else 2


def _cmd_inspect(args) -> int:
    store = SnapshotStore(args.store)
    print(json.dumps(store.load_manifest(), sort_keys=True, indent=2))
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "plan":
        return _cmd_plan(args)
    if args.command == "verify":
        return _cmd_verify(args)
    if args.command == "inspect":
        return _cmd_inspect(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
