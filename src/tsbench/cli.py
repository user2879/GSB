"""Command line entry point for the deterministic time-series workbench."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys

# Pin BLAS to one thread: mathematical results must not depend on scheduling.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

from .config import load_config
from .errors import E_CONFIG_INVALID, ConfigError, sort_failures
from .runner import GracefulExit, WorkbenchRunner
from .updates import load_update_plan
from .util import read_json


class SignalGate:
    """Raises after the current stage snapshot has been fully replaced."""

    def __init__(self) -> None:
        self.requested = False
        self.signal_name = ""
        signal.signal(signal.SIGTERM, self._handle)
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, signum, _frame) -> None:
        self.requested = True
        self.signal_name = signal.Signals(signum).name

    def checkpoint(self) -> None:
        if self.requested:
            raise GracefulExit(self.signal_name)


def _print(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def build_runner(config_path: str, output_dir: str) -> WorkbenchRunner:
    config = load_config(config_path)
    runner = WorkbenchRunner(config, output_dir)
    runner.initialize_series()
    return runner


def install_signal_gate() -> SignalGate:
    return SignalGate()


def cmd_run(args: argparse.Namespace) -> int:
    runner = build_runner(args.config, args.output)
    recovery = runner.load_recovery()
    batches = load_update_plan(args.updates)
    gate = SignalGate()
    try:
        reports = runner.apply_batches(
            batches, from_batch=runner.last_batch, signal_gate=gate
        )
    except GracefulExit as exit_event:
        _print(
            {
                "status": "graceful_exit",
                "signal": exit_event.signal_name,
                "exit_code": 0,
                "recovery_point": runner.state.recovery_point(),
            }
        )
        return 0

    last = reports[-1] if reports else None
    _print(
        {
            "status": "ok",
            "mode": "incremental",
            "recovery": recovery,
            "batches_processed": [report["batch_id"] for report in reports],
            "final_global_version": last["global_version"] if last else 0,
            "reuse_counts": [
                {
                    "batch_id": report["batch_id"],
                    "reused": len(report["reuse_list"]),
                    "recomputed": len(report["recompute_list"]),
                }
                for report in reports
            ],
            "failures": sort_failures(runner.failure_records),
        }
    )
    return 0


def cmd_full(args: argparse.Namespace) -> int:
    runner = build_runner(args.config, args.output)
    runner.load_recovery()
    report = runner.run_once(batch_id=0, changed_fragments={}, changed_indices={}, operations={})
    _print(
        {
            "status": "ok",
            "mode": "full",
            "global_version": report["global_version"],
            "series": [r["series_id"] for r in report["series_results"]],
            "failures": report["failures"],
        }
    )
    return 0


def cmd_full_final(args: argparse.Namespace) -> int:
    runner = build_runner(args.config, args.output)
    runner.load_recovery()
    batches = load_update_plan(args.updates)
    runner.materialize_final_state(batches)
    gate = install_signal_gate()
    try:
        report = runner.run_once(
            batch_id=9999,
            changed_fragments={},
            changed_indices={},
            operations={},
            signal_gate=gate,
        )
    except GracefulExit as exit_event:
        _print(
            {
                "status": "graceful_exit",
                "signal": exit_event.signal_name,
                "exit_code": 0,
                "recovery_point": runner.state.recovery_point(),
            }
        )
        return 0
    _print(
        {
            "status": "ok",
            "mode": "full-final",
            "global_version": report["global_version"],
            "data_fingerprint": report["data_fingerprint"],
            "series": [r["series_id"] for r in report["series_results"]],
            "failures": report["failures"],
        }
    )
    return 0


COMPARED_FIELDS = (
    "data_fingerprint",
    "preprocess",
    "features",
    "models",
    "backtest",
)


def compare_reports(full_report: dict, incremental_report: dict) -> dict:
    full_by_id = {item["series_id"]: item for item in full_report["series_results"]}
    inc_by_id = {item["series_id"]: item for item in incremental_report["series_results"]}
    mismatches: list[str] = []
    for series_id in sorted(full_by_id):
        if series_id not in inc_by_id:
            mismatches.append(f"{series_id}: missing in incremental report")
            continue
        left, right = full_by_id[series_id], inc_by_id[series_id]
        for field in COMPARED_FIELDS:
            if left.get(field) != right.get(field):
                mismatches.append(f"{series_id}: field '{field}' differs")
        # Explicit two-step coverage check.
        for fold_left, fold_right in zip(
            left["backtest"]["folds"], right["backtest"]["folds"]
        ):
            if len(fold_left["predictions"]) != 2 or len(fold_right["predictions"]) != 2:
                mismatches.append(
                    f"{series_id}:{fold_left['fold_id']} forecast_steps coverage != 2"
                )
    if full_report["failures"] != incremental_report["failures"]:
        mismatches.append("failure list ordering/content differs")
    return {
        "identical": not mismatches,
        "series_compared": sorted(full_by_id),
        "forecast_steps_covered": 2,
        "mismatches": mismatches,
    }


def cmd_verify(args: argparse.Namespace) -> int:
    full_report = read_json(args.full_report)
    incremental_report = read_json(args.incremental_report)
    result = compare_reports(full_report, incremental_report)
    _print({"status": "ok" if result["identical"] else "mismatch", **result})
    return 0 if result["identical"] else 3


def cmd_recovery(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    runner = WorkbenchRunner(config, args.output)
    _print(runner.state.recovery_point())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tsbench")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="apply update batches incrementally")
    run_p.add_argument("--config", required=True)
    run_p.add_argument("--updates", required=True)
    run_p.add_argument("--output", required=True)
    run_p.set_defaults(func=cmd_run)

    full_p = sub.add_parser("full", help="one-shot full rebuild from a blank state")
    full_p.add_argument("--config", required=True)
    full_p.add_argument("--output", required=True)
    full_p.set_defaults(func=cmd_full)

    final_p = sub.add_parser(
        "full-final", help="apply all batches then run one full rebuild"
    )
    final_p.add_argument("--config", required=True)
    final_p.add_argument("--updates", required=True)
    final_p.add_argument("--output", required=True)
    final_p.set_defaults(func=cmd_full_final)

    verify_p = sub.add_parser("verify", help="compare full vs incremental reports")
    verify_p.add_argument("--full-report", required=True)
    verify_p.add_argument("--incremental-report", required=True)
    verify_p.set_defaults(func=cmd_verify)

    rec_p = sub.add_parser("recovery", help="print the persisted recovery point")
    rec_p.add_argument("--config", required=True)
    rec_p.add_argument("--output", required=True)
    rec_p.set_defaults(func=cmd_recovery)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        _print(
            {
                "status": "failed",
                "code": exc.code,
                "message": exc.message,
            }
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
