from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .demo import injection_report, run_end_to_end_demo
from .errors import DiagnosticError
from .loader import load_bundle
from .service import DiagnosticService
from .viewmodel import build_state_view_model, render_plain_text_summary


def print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="colpage-diag",
        description="Columnar page WAL crash-recovery diagnostics",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate pages/WAL/budget JSON")
    validate.add_argument("--samples", default="samples")
    validate.add_argument(
        "--inject",
        choices=("commit-conflict", "range-overflow"),
        help="optional targeted WAL fault injection",
    )

    run = subparsers.add_parser("run", help="replay WAL and commit ledger batches")
    run.add_argument("--samples", default="samples")
    run.add_argument("--state-dir", default="state")
    run.add_argument("--reset", action="store_true")
    run.add_argument("--crash-before-commit", type=int, default=None)
    run.add_argument("--json", dest="json_path", default=None)

    recover = subparsers.add_parser("recover", help="mark unknown batches and verify replay")
    recover.add_argument("--samples", default="samples")
    recover.add_argument("--state-dir", default="state")

    demo = subparsers.add_parser("demo", help="run repeatable crash/recovery demonstration")
    demo.add_argument("--samples", default="samples")
    demo.add_argument("--state-dir", default="state/demo")
    demo.add_argument("--report", default="artifacts/demo_report.json")
    demo.add_argument(
        "--inject",
        choices=("commit-conflict", "range-overflow", "snapshot-mismatch"),
        default=None,
    )

    diagnose = subparsers.add_parser("diagnose", help="inject one explicit diagnostic fault")
    diagnose.add_argument(
        "--inject",
        required=True,
        choices=("commit-conflict", "range-overflow", "snapshot-mismatch"),
    )
    diagnose.add_argument("--samples", default="samples")
    diagnose.add_argument("--state-dir", default="state/injection")

    gui = subparsers.add_parser("gui", help="launch Tkinter diagnostic interface")
    gui.add_argument("--state-dir", default="state/demo")
    gui.add_argument("--report", default=None)

    view = subparsers.add_parser("view", help="print stable text view of SQLite state")
    view.add_argument("--state-dir", default="state/demo")
    view.add_argument("--json", action="store_true")
    return parser


def _validate(args: argparse.Namespace) -> int:
    if args.inject:
        report = injection_report(args.inject, args.samples)
        print_json(report)
        return 2 if not report.get("ok") else 0
    bundle = load_bundle(args.samples)
    print_json(
        {
            "ok": True,
            "code": "OK",
            "pages": len(bundle.pages),
            "wal_entries": len(bundle.entries),
            "transactions": sorted({entry.tx_id for entry in bundle.entries}),
            "budget": bundle.budget.__dict__,
        }
    )
    return 0


def _run(args: argparse.Namespace) -> int:
    with DiagnosticService(state_dir=args.state_dir, sample_dir=args.samples) as service:
        service.initialize(reset=args.reset)
        result = service.replay_and_commit(crash_before_round=args.crash_before_commit)
        view_model = build_state_view_model(args.state_dir)
    payload = {"ok": True, "code": "OK", "run": result, "view": view_model}
    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print_json({"ok": True, "code": "OK", "run": result})
    return 0


def _recover(args: argparse.Namespace) -> int:
    with DiagnosticService(state_dir=args.state_dir, sample_dir=args.samples) as service:
        service.initialize(reset=False)
        report = service.recover_and_verify()
    print_json({"ok": True, "code": "OK", "recovery": report})
    return 0


def _demo(args: argparse.Namespace) -> int:
    if args.inject:
        state_dir = Path(args.state_dir).parent / f"inject-{args.inject}"
        report = injection_report(args.inject, args.samples, state_dir)
        print_json(report)
        return 2 if not report.get("ok") else 0
    report = run_end_to_end_demo(
        args.samples,
        args.state_dir,
        report_path=args.report,
    )
    summary = {
        "ok": True,
        "report_path": args.report,
        "determinism": report["determinism"]["status"],
        "force_kill_returncode": report["force_kill"]["returncode"],
        "unknown_batches": report["recovery"]["unknown_batches"],
        "recovery_status": report["recovery"]["status"],
        "spec_generation": report["spec_evolution"]["to_generation"],
        "current_pages": len(report["state_view"]["pages"]),
    }
    print_json(summary)
    return 0


def _diagnose(args: argparse.Namespace) -> int:
    report = injection_report(args.inject, args.samples, args.state_dir)
    print_json(report)
    return 2 if not report.get("ok") else 0


def _gui(args: argparse.Namespace) -> int:
    from .gui import launch_gui

    return launch_gui(state_dir=args.state_dir, report_path=args.report)


def _view(args: argparse.Namespace) -> int:
    model = build_state_view_model(args.state_dir)
    if args.json:
        print_json(model)
    else:
        print(render_plain_text_summary(model))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            return _validate(args)
        if args.command == "run":
            return _run(args)
        if args.command == "recover":
            return _recover(args)
        if args.command == "demo":
            return _demo(args)
        if args.command == "diagnose":
            return _diagnose(args)
        if args.command == "gui":
            return _gui(args)
        if args.command == "view":
            return _view(args)
        parser.error(f"unknown command {args.command}")
        return 2
    except DiagnosticError as error:
        print_json({"ok": False, **error.to_dict()})
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
