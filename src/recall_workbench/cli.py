"""Command line interface for sample generation, validation, and experiments."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path
from typing import Sequence

from .errors import ConfigMismatch, InputNotFound, SnapshotInvalid, WorkbenchError
from .report import build_report, write_reports
from .runner import ExperimentRunner
from .samplegen import DEFAULT_SEED, generate_samples
from .snapshot import SnapshotStore, atomic_write_json
from .validation import discover_inputs, load_config, load_dataset


def _print_json(value: object, *, stream: object | None = None) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), file=stream or sys.stdout)


def _config_failure_exit(failures: list[object], report_dir: Path | None) -> int:
    payload = {
        "code": "E_CONFIG_INVALID",
        "failures": [failure.to_dict() for failure in failures],  # type: ignore[attr-defined]
    }
    if report_dir is not None:
        atomic_write_json(report_dir.resolve() / "failures.json", payload)
    _print_json(payload, stream=sys.stderr)
    return 2


def command_generate(args: argparse.Namespace) -> int:
    manifest = generate_samples(Path(args.output), seed=args.seed)
    _print_json({"status": "generated", "output": str(Path(args.output).resolve()), "manifest": manifest})
    return 0


def command_validate(args: argparse.Namespace) -> int:
    input_dir = Path(args.input).resolve()
    config_path = discover_inputs(input_dir)["config"]
    config, config_failures = load_config(config_path)
    if config is None:
        return _config_failure_exit(config_failures, None)
    dataset = load_dataset(input_dir, config)
    _print_json(
        {
            "status": "valid" if not dataset.failures else "valid_with_record_failures",
            "config_hash": config.config_hash,
            "counts": {
                "interactions_valid": len(dataset.interactions),
                "catalog_valid": len(dataset.items),
                "users": len(dataset.user_ids),
                "truth_users": len(dataset.truth),
                "failures": len(dataset.failures),
            },
            "failures": [failure.to_dict() for failure in dataset.failures],
        }
    )
    return 0


def _install_signal_handlers(stop_event: threading.Event) -> dict[int, object]:
    previous: dict[int, object] = {}

    def request_stop(signum: int, frame: object) -> None:
        if stop_event.is_set():
            return
        stop_event.set()
        print(
            f"[recall-workbench] received signal {signum}; committing the current stage and exiting with code 0",
            file=sys.stderr,
        )

    for name in ("SIGINT", "SIGTERM"):
        value = getattr(signal, name, None)
        if value is None:
            continue
        try:
            previous[value] = signal.getsignal(value)
            signal.signal(value, request_stop)
        except (OSError, ValueError):
            pass
    return previous


def _restore_signal_handlers(previous: dict[int, object]) -> None:
    for signum, handler in previous.items():
        try:
            signal.signal(signum, handler)
        except (OSError, ValueError):
            pass


def command_run(args: argparse.Namespace) -> int:
    input_dir = Path(args.input).resolve()
    run_dir = Path(args.run_dir).resolve()
    report_dir = Path(args.report_dir).resolve() if args.report_dir else run_dir / "report"
    stop_event = threading.Event()
    previous_handlers = _install_signal_handlers(stop_event)
    try:
        config_path = discover_inputs(input_dir)["config"]
        config, config_failures = load_config(config_path)
        if config is None:
            return _config_failure_exit(config_failures, report_dir)
        dataset = load_dataset(input_dir, config)
        snapshot = SnapshotStore(run_dir, config, args.version)
        runner = ExperimentRunner(dataset, config, snapshot, serial=args.serial, stop_event=stop_event)
        outcome = runner.run()
        response: dict[str, object] = {
            "status": outcome.status,
            "exit_code": outcome.exit_code,
            "run_dir": str(run_dir),
            "config_hash": config.config_hash,
            "execution": outcome.stats,
            "stages": outcome.stats.get("stages", {}),
        }
        if outcome.status == "completed":
            report = build_report(
                dataset,
                config,
                snapshot,
                outcome,
                execution_mode="serial" if args.serial else "thread_pool",
            )
            paths = write_reports(report, report_dir)
            response["report"] = paths
            response["result_digest"] = report["result_digest"]
            response["evaluation"] = report["evaluation"]["summary"]
        else:
            response["resume_command"] = (
                f"python run.py run --input {input_dir} --run-dir {run_dir} --version {args.version}"
            )
        _print_json(response)
        return outcome.exit_code
    except ConfigMismatch as exc:
        _print_json(exc.to_dict(), stream=sys.stderr)
        return 2
    except (SnapshotInvalid, InputNotFound) as exc:
        _print_json(exc.to_dict(), stream=sys.stderr)
        return 2
    finally:
        _restore_signal_handlers(previous_handlers)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recall-workbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic local sample data")
    generate.add_argument("--output", default="data", help="output sample directory")
    generate.add_argument("--seed", type=int, default=DEFAULT_SEED, help="fixed sample seed")
    generate.set_defaults(handler=command_generate)

    validate = subparsers.add_parser("validate", help="validate local inputs")
    validate.add_argument("--input", default="data", help="sample directory")
    validate.set_defaults(handler=command_validate)

    run = subparsers.add_parser("run", help="run recall, fusion, diagnostics, and evaluation")
    run.add_argument("--input", default="data", help="sample directory")
    run.add_argument("--run-dir", default="runs/experiment", help="snapshot directory")
    run.add_argument("--version", type=int, default=1, help="global run version used for resume checks")
    run.add_argument("--serial", action="store_true", help="force serial execution for equivalence verification")
    run.add_argument("--report-dir", default=None, help="report output directory (defaults to <run-dir>/report)")
    run.set_defaults(handler=command_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except WorkbenchError as exc:
        _print_json(exc.to_dict(), stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
