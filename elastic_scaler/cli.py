"""Command-line entry point for local, offline experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .engine import ForcedKill, ScalingEngine
from .errors import ScalerError
from .fault_injection import (
    inject_budget_violation,
    inject_load_gap,
    inject_snapshot_version_conflict,
)
from .gui import ScalerWorkbench
from .spec import load_spec, parse_spec

DEFAULT_DB = "data/scaler.sqlite3"
DEFAULT_LOG = "data/events.jsonl"


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def _engine(args: argparse.Namespace) -> ScalingEngine:
    return ScalingEngine(args.db, args.log, run_id=args.run_id)


def _load_raw(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cmd_init(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    with _engine(args) as engine:
        version, created = engine.initialize(spec)
        engine.logger.log("cli_init", spec=args.spec, spec_version=version, created=created)
        _print({"spec_version": version, "created": created, "digest": spec.digest,
                "run_id": engine.run_id, "horizon": spec.horizon})
    return 0


def cmd_advance(args: argparse.Namespace) -> int:
    with _engine(args) as engine:
        if engine.ledger.get_run(engine.run_id) is None:
            if not args.spec:
                raise ScalerError(
                    f"run {engine.run_id!r} does not exist; pass --spec to initialize it",
                    fields=[f"/runs/{engine.run_id}"],
                )
            engine.initialize(load_spec(args.spec))
        else:
            engine.recover()
        if args.simulate_kill is not None:
            try:
                engine.advance_all(simulate_kill_at=args.simulate_kill)
            except ForcedKill:
                _print({"status": "FORCED_KILL_SIMULATED",
                        "uncommitted_minute": args.simulate_kill,
                        "note": "no exit report exists; recover and re-run the minute"})
                return 0
        else:
            decisions = engine.advance_all()
            _print({"advanced_minutes": [d.minute for d in decisions],
                    "next_minute": engine.next_minute})
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    with _engine(args) as engine:
        engine.recover()
        decision = engine.fast_query(args.minute)
        _print(decision.to_dict())
    return 0


def cmd_authoritative(args: argparse.Namespace) -> int:
    with _engine(args) as engine:
        engine.recover()
        decision = engine.authoritative_query(args.minute)
        _print(decision.to_dict())
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    with _engine(args) as engine:
        recovery = engine.recover()
        replay = engine.verify_replay()
        engine.verify_fast_vs_authoritative()
        _print({"recovery": recovery.__dict__, "replay": replay.__dict__,
                "fast_path": "MATCHES_AUTHORITATIVE"})
    return 0


def cmd_replace_spec(args: argparse.Namespace) -> int:
    new_spec = load_spec(args.spec)
    with _engine(args) as engine:
        engine.recover()
        report = engine.replace_spec(new_spec)
        engine.advance_all()
        _print({"replacement": {
            "old_version": report.old_version,
            "new_version": report.new_version,
            "created": report.created,
            "affected_services": list(report.affected_services),
            "affected_minute_range": list(report.affected_minute_range),
            "parameter_diffs": list(report.parameter_diffs),
        }, "new_run_id": engine.run_id})
    return 0


def cmd_versions(args: argparse.Namespace) -> int:
    with _engine(args) as engine:
        _print({"versions": engine.list_versions(),
                "snapshots": engine.ledger.list_snapshots()})
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    with _engine(args) as engine:
        engine.recover()
        engine.save_snapshot(args.snapshot_id, args.minute, note=args.note)
        _print({"saved": args.snapshot_id, "run_id": engine.run_id,
                "spec_version": engine.spec_version, "target_minute": args.minute})
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    with _engine(args) as engine:
        engine.recover()
        report = engine.restore_snapshot(args.snapshot_id)
        engine.verify_replay()
        _print({"restored": args.snapshot_id, "recovery": report.__dict__})
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    with _engine(args) as engine:
        engine.recover()
        workbench = ScalerWorkbench(engine)
        if workbench.headless:
            view = workbench.refresh()
            _print({"headless": True, "view": view,
                    "latency": workbench.latency_report()})
        else:
            workbench.run()
    return 0


def cmd_inject(args: argparse.Namespace) -> int:
    raw = _load_raw(args.spec)
    with _engine(args) as engine:
        if args.fault == "load-gap":
            mutated = inject_load_gap(raw, args.service, args.minute)
            parse_spec(mutated)  # expected to raise E_LOAD first
        elif args.fault == "bad-budget":
            mutated = inject_budget_violation(raw, args.bad_budget)
            parse_spec(mutated)  # expected to raise E_BUDGET first
        elif args.fault == "version-conflict":
            spec = load_spec(args.spec)
            version, _ = engine.initialize(spec)
            if engine.ledger.get_run(engine.run_id) is not None and not engine.ledger.committed_minutes(engine.run_id):
                engine.advance_minute(0)
            engine.save_snapshot("good-snapshot", 0)
            inject_snapshot_version_conflict(
                engine.ledger, "conflict-snapshot", engine.run_id,
                version + 1, 0,
            )
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local deterministic elastic scaler")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("--run-id", default="run-current")
    sub = parser.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser("init")
    init_p.add_argument("--spec", required=True)
    init_p.set_defaults(func=cmd_init)

    advance_p = sub.add_parser("advance")
    advance_p.add_argument("--spec", default=None,
                           help="cluster spec used when the run does not exist yet")
    advance_p.add_argument("--simulate-kill", type=int, default=None)
    advance_p.set_defaults(func=cmd_advance)

    query_p = sub.add_parser("query")
    query_p.add_argument("--minute", type=int, required=True)
    query_p.set_defaults(func=cmd_query)

    auth_p = sub.add_parser("authoritative")
    auth_p.add_argument("--minute", type=int, required=True)
    auth_p.set_defaults(func=cmd_authoritative)

    verify_p = sub.add_parser("verify")
    verify_p.set_defaults(func=cmd_verify)

    replace_p = sub.add_parser("replace-spec")
    replace_p.add_argument("--spec", required=True)
    replace_p.set_defaults(func=cmd_replace_spec)

    versions_p = sub.add_parser("versions")
    versions_p.set_defaults(func=cmd_versions)

    snap_p = sub.add_parser("snapshot")
    snap_p.add_argument("--snapshot-id", required=True)
    snap_p.add_argument("--minute", type=int, required=True)
    snap_p.add_argument("--note", default="")
    snap_p.set_defaults(func=cmd_snapshot)

    restore_p = sub.add_parser("restore")
    restore_p.add_argument("--snapshot-id", required=True)
    restore_p.set_defaults(func=cmd_restore)

    gui_p = sub.add_parser("gui")
    gui_p.set_defaults(func=cmd_gui)

    inject_p = sub.add_parser("inject")
    inject_p.add_argument("fault", choices=("load-gap", "bad-budget", "version-conflict"))
    inject_p.add_argument("--spec", required=True)
    inject_p.add_argument("--service", default="asr")
    inject_p.add_argument("--minute", type=int, default=4)
    inject_p.add_argument("--bad-budget", type=int, default=0)
    inject_p.set_defaults(func=cmd_inject)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ScalerError as exc:
        logger = StructuredLoggerShim(args.log)
        logger.log_error(exc)
        payload = exc.to_dict()
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
              file=sys.stderr)
        return 2
    except ForcedKill:
        print(json.dumps({"code": "E_INTERNAL", "message": "unexpected kill path"}),
              file=sys.stderr)
        return 3


class StructuredLoggerShim:
    def __init__(self, path: str) -> None:
        from .logging_util import StructuredLogger
        self._logger = StructuredLogger(path)

    def log_error(self, exc: ScalerError) -> None:
        self._logger.log_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())
