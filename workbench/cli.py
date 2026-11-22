"""Command-line interface for the local scaling workbench."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .engine import ScalingEngine
from .errors import ERROR_CODES, WorkbenchError
from .faults import (
    inject_declaration_corruption,
    inject_length_mismatch,
    inject_snapshot_mismatch,
)
from .gui import build_view_model
from .latency import LatencyRecorder
from .ledger import Ledger
from .loader import check_series_lengths, load_declaration
from .logging_utils import JsonlLogger
from .preview import PreviewCache
from .sample_data import write_samples

DEFAULT_DB = os.path.join("data", "workbench.db")
DEFAULT_LOG = os.path.join("logs", "workbench.jsonl")
DEFAULT_SAMPLE_DIR = "samples"


def _error_code(exc: WorkbenchError) -> int:
    try:
        return ERROR_CODES.index(exc.code) + 2
    except ValueError:
        return 19


def _build_context(args: argparse.Namespace):
    os.makedirs(os.path.dirname(os.path.abspath(args.db)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    logger = JsonlLogger(args.log)
    ledger = Ledger(args.db)
    recorder = LatencyRecorder(
        budget_ms=getattr(args, "latency_budget", 50) or 50,
        fixed_point_scale=1000,
        logger=logger,
    )  # declaration-specific budget is refreshed when a spec is loaded
    preview = PreviewCache()
    engine = ScalingEngine(ledger, logger, recorder, preview)
    return logger, ledger, recorder, preview, engine


def _print(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def cmd_init_samples(args: argparse.Namespace) -> int:
    paths = write_samples(args.sample_dir)
    _print({"written": paths})
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    logger, ledger, recorder, preview, engine = _build_context(args)
    spec = load_declaration(args.declaration)
    recorder.budget_ms = spec.latency_budget_ms
    version = engine.register_declaration(spec)
    report = engine.run_pipeline(spec, version)
    ledger.verify_snapshots(version)
    preview.verify_against_authority(ledger.load_decisions(version))
    _print({
        "version": report.version,
        "minutes": report.minutes,
        "selection": report.selection,
        "selection_reason": report.selection_reason,
        "threshold_boundary": report.threshold_boundary,
        "committed": report.committed,
        "latency_overall": recorder.aggregate().as_dict(),
        "over_budget": [sample.as_dict() for sample in recorder.over_budget_samples()],
    })
    ledger.close()
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    logger, ledger, recorder, preview, engine = _build_context(args)
    spec = load_declaration(args.declaration)
    recorder.budget_ms = spec.latency_budget_ms
    version = args.version or ledger.active_version() or 1
    ledger.verify_snapshots(version)
    engine.verify_replay(spec, version)
    authoritative = engine.authoritative_recompute(spec, version)
    for decision in authoritative:
        if preview.contains(version, decision.minute):
            preview.populate(decision)
    _print({"replayed_minutes": len(authoritative), "version": version,
            "replay_latency": recorder.aggregate("replay").as_dict()})
    ledger.close()
    return 0


def cmd_recover(args: argparse.Namespace) -> int:
    logger, ledger, recorder, preview, engine = _build_context(args)
    spec = load_declaration(args.declaration)
    recorder.budget_ms = spec.latency_budget_ms
    version = args.version or ledger.active_version() or 1
    state = engine.recovery_state(spec, version)
    if state["unknown_minutes"]:
        report = engine.run_pipeline(spec, version)
        state["reran_minutes"] = report.unknown_minutes
    ledger.verify_snapshots(version)
    _print(state)
    ledger.close()
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    logger, ledger, recorder, preview, engine = _build_context(args)
    version = args.version or ledger.active_version()
    if version is None:
        raise WorkbenchError("no committed version; run the pipeline first",
                             details={"field": "version"})
    view = build_view_model(ledger, version, recorder)
    if args.headless:
        _print({
            "decision_rows": len(view.decision_rows()),
            "curve_points": len(view.budget_curve()),
            "trim_reasons": view.trim_reason_rows(),
            "snapshots": view.snapshot_rows(),
            "stale_versions": view.stale_versions,
            "latency": view.latency_overall.as_dict(),
        })
        ledger.close()
        return 0
    from .gui import launch

    launch(ledger, version, recorder)
    ledger.close()
    return 0


def cmd_fault(args: argparse.Namespace) -> int:
    logger, ledger, recorder, preview, engine = _build_context(args)
    if args.kind == "schema":
        inject_declaration_corruption(args.declaration, logger)
    elif args.kind == "length":
        if not args.declaration:
            raise WorkbenchError("length injection requires a declaration path",
                                 details={"field": "declaration"})
        spec = load_declaration(args.declaration)
        tampered = inject_length_mismatch(spec, logger)
        check_series_lengths(tampered)
    elif args.kind == "snapshot":
        version = args.version or ledger.active_version() or 1
        inject_snapshot_mismatch(ledger, version, logger)
    ledger.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scaling-workbench")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("--latency-budget", type=int, default=50)
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init-samples")
    init_parser.add_argument("--sample-dir", default=DEFAULT_SAMPLE_DIR)
    init_parser.set_defaults(func=cmd_init_samples)

    run_parser = sub.add_parser("run")
    run_parser.add_argument("declaration")
    run_parser.set_defaults(func=cmd_run)

    replay_parser = sub.add_parser("replay")
    replay_parser.add_argument("declaration")
    replay_parser.add_argument("--version", type=int)
    replay_parser.set_defaults(func=cmd_replay)

    recover_parser = sub.add_parser("recover")
    recover_parser.add_argument("declaration")
    recover_parser.add_argument("--version", type=int)
    recover_parser.set_defaults(func=cmd_recover)

    gui_parser = sub.add_parser("gui")
    gui_parser.add_argument("--version", type=int)
    gui_parser.add_argument("--headless", action="store_true")
    gui_parser.set_defaults(func=cmd_gui)

    fault_parser = sub.add_parser("fault")
    fault_parser.add_argument("kind", choices=("schema", "length", "snapshot"))
    fault_parser.add_argument("declaration", nargs="?", default="")
    fault_parser.add_argument("--version", type=int)
    fault_parser.set_defaults(func=cmd_fault)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except WorkbenchError as exc:
        payload = exc.as_dict()
        print(json.dumps(payload, sort_keys=True, ensure_ascii=False), file=sys.stderr)
        return _error_code(exc)


if __name__ == "__main__":
    raise SystemExit(main())
