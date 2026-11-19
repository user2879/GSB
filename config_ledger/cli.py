"""Command line entry points for the local diagnostics console."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .demo import run_all_demos, run_demo_case
from .errors import ReplayError, SnapshotError, SpecValidationError
from .service import DiagnosticService
from .spec import load_spec
from .structured_log import StructuredLogger


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _error_payload(code: str, exc: Exception) -> dict[str, Any]:
    payload = {
        "status": "ERROR",
        "code": code,
        "message": str(exc),
    }
    details = getattr(exc, "details", None)
    if details:
        payload["details"] = details
    first_diff = getattr(exc, "first_diff", None)
    if first_diff is not None:
        payload["first_diff"] = first_diff
    return payload


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        spec = load_spec(args.spec)
    except SpecValidationError as exc:
        logger = StructuredLogger(args.log)
        logger.error(
            "E_SPEC",
            str(exc),
            source=args.spec,
            issues=[issue.as_dict() for issue in exc.issues],
        )
        _print_json(_error_payload("E_SPEC", exc))
        return 2
    _print_json(
        {
            "status": "OK",
            "name": spec.name,
            "parameter_version": spec.parameter_version,
            "fingerprint": spec.fingerprint,
            "nodes": [node.as_dict() for node in spec.nodes],
            "keys": {key: spec.keys[key].as_dict() for key in spec.sorted_keys},
        }
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        service = DiagnosticService(args.spec, ledger_path=args.ledger, log_path=args.log)
    except SpecValidationError as exc:
        _print_json(_error_payload("E_SPEC", exc))
        return 2
    except SnapshotError as exc:
        _print_json(_error_payload("E_SNAPSHOT", exc))
        return 3
    try:
        report = service.run_round()
        if args.replay_after:
            try:
                report["replay"] = service.replay()
            except ReplayError as exc:
                report["replay"] = _error_payload("E_REPLAY", exc)
                _print_json(report)
                return 4
        _print_json(report)
        return 0
    finally:
        service.close()


def _cmd_replay(args: argparse.Namespace) -> int:
    try:
        service = DiagnosticService(args.spec, ledger_path=args.ledger, log_path=args.log)
    except SpecValidationError as exc:
        _print_json(_error_payload("E_SPEC", exc))
        return 2
    except SnapshotError as exc:
        _print_json(_error_payload("E_SNAPSHOT", exc))
        return 3
    try:
        report = service.replay()
        _print_json(report)
        return 0
    except ReplayError as exc:
        _print_json(_error_payload("E_REPLAY", exc))
        return 4
    finally:
        service.close()


def _cmd_demo(args: argparse.Namespace) -> int:
    runtime = Path(args.runtime)
    if args.all_anomalies:
        results = run_all_demos(runtime)
    else:
        results = [
            run_demo_case(
                sample=args.sample,
                inject=args.inject,
                runtime_dir=runtime,
            )
        ]
    _print_json({"status": "OK", "runtime": str(runtime), "results": results})
    return 0


def _cmd_gui(args: argparse.Namespace) -> int:
    from .ui import run_ui

    try:
        return run_ui(
            spec_path=args.spec,
            ledger_path=args.ledger,
            log_path=args.log,
            sample=args.sample,
            runtime_dir=args.runtime,
        )
    except SpecValidationError as exc:
        _print_json(_error_payload("E_SPEC", exc))
        return 2
    except SnapshotError as exc:
        _print_json(_error_payload("E_SNAPSHOT", exc))
        return 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="config-ledger",
        description="Local Windows configuration-effect ledger and observation merger",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a JSON declaration")
    validate.add_argument("--spec", required=True)
    validate.add_argument("--log", default="runtime/logs/validate.jsonl")
    validate.set_defaults(func=_cmd_validate)

    run = subparsers.add_parser("run", help="pull observations and commit one round")
    run.add_argument("--spec", required=True)
    run.add_argument("--ledger", default="runtime/ledger.sqlite")
    run.add_argument("--log", default="runtime/logs/structured.jsonl")
    run.add_argument("--replay-after", action="store_true")
    run.set_defaults(func=_cmd_run)

    replay = subparsers.add_parser("replay", help="replay the current committed round")
    replay.add_argument("--spec", required=True)
    replay.add_argument("--ledger", default="runtime/ledger.sqlite")
    replay.add_argument("--log", default="runtime/logs/structured.jsonl")
    replay.set_defaults(func=_cmd_replay)

    demo = subparsers.add_parser(
        "demo", help="start deterministic loopback simulators and run end-to-end cases"
    )
    demo.add_argument("--runtime", default="runtime/demo")
    demo.add_argument(
        "--sample",
        choices=("divergence", "missing", "string_expected"),
        default="divergence",
    )
    demo.add_argument(
        "--inject",
        choices=("none", "invalid_json", "port_unreachable", "snapshot_mismatch"),
        default="none",
    )
    demo.add_argument(
        "--all-anomalies",
        action="store_true",
        help="run baseline, missing-key, invalid-JSON, unreachable-port and digest-mismatch cases",
    )
    demo.set_defaults(func=_cmd_demo)

    gui = subparsers.add_parser("gui", help="open the Tkinter diagnostics console")
    gui.add_argument("--spec", required=True)
    gui.add_argument("--ledger", default="runtime/ledger.sqlite")
    gui.add_argument("--log", default="runtime/logs/structured.jsonl")
    gui.add_argument("--sample", choices=("divergence", "missing", "string_expected"))
    gui.add_argument("--runtime", default="runtime/gui")
    gui.set_defaults(func=_cmd_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
