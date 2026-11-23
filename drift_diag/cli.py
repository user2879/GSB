"""Command line entry point and deterministic exception injection."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from .canonical import load_json_strict
from .config import EvaluationConfig
from .declarations import load_declaration, normalize_declaration
from .drift import aggregate_all, compare_field
from .errors import DiagnosticError
from .events import apply_event, load_events
from .ledger import LedgerStore
from .models import FIELD_INDEX, FIELD_ORDER
from .propagation import apply_change_set, parse_change_set
from .replay import verify_replay
from .session import run_session
from .ui import launch_gui, ui_environment


EXIT_CODES = {
    "E_FORMAT": 2,
    "E_EVENT": 3,
    "E_REF": 4,
    "E_SNAPSHOT": 5,
    "E_REPLAY": 6,
    "E_UI": 7,
}


def _parse_tolerance(value: str) -> tuple[int, int]:
    try:
        numerator_text, denominator_text = value.split("/", 1)
        numerator = int(numerator_text)
        denominator = int(denominator_text)
    except (ValueError, AttributeError) as exc:
        raise DiagnosticError(
            "E_FORMAT", "tolerance must use NUM/DEN syntax", field="tolerance"
        ) from exc
    if numerator < 0 or denominator <= 0:
        raise DiagnosticError(
            "E_FORMAT",
            "tolerance must be non-negative with a positive denominator",
            field="tolerance",
        )
    return numerator, denominator


def _config_from_args(args: argparse.Namespace) -> EvaluationConfig:
    numerator, denominator = _parse_tolerance(args.tolerance)
    return EvaluationConfig(
        scale_exp=args.scale_exp,
        scale_version=args.scale_version,
        tolerance_num=numerator,
        tolerance_den=denominator,
        global_limit=args.global_limit,
        per_workload_quota=args.per_workload_quota,
        task_delay_ms=args.task_delay_ms,
        batch_events=args.batch_events,
    ).validate()


def _model_config_from_args(args: argparse.Namespace) -> EvaluationConfig:
    numerator, denominator = _parse_tolerance(args.tolerance)
    return EvaluationConfig(
        scale_exp=args.scale_exp,
        scale_version=args.scale_version,
        tolerance_num=numerator,
        tolerance_den=denominator,
    ).validate()


def _inject_declaration_corrupt() -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as handle:
        handle.write('{"format_version": 3, "workloads": [')
        path = handle.name
    try:
        load_declaration(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _inject_unknown_event(declaration: str, defaults_version: str, events_path: str) -> None:
    raw_declaration = load_declaration(declaration)
    target = normalize_declaration(raw_declaration.raw, defaults_version=defaults_version)
    lines = Path(events_path).read_text(encoding="utf-8").splitlines()
    changed = False
    for index, line in enumerate(lines):
        if line.strip():
            event = json.loads(line)
            event["field"] = "__injected_unknown_field__"
            lines[index] = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
            changed = True
            break
    if not changed:
        lines.append(
            '{"seq":1,"workload":"__injected_unknown_workload__","field":"replicas","value":1,"at_ms":0}'
        )
    with tempfile.NamedTemporaryFile("w", suffix=".ndjson", encoding="utf-8", delete=False) as handle:
        handle.write("\n".join(lines) + "\n")
        path = handle.name
    try:
        load_events(path, target)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _inject_ledger_digest(args: argparse.Namespace, config: EvaluationConfig) -> None:
    db_path = Path(args.db)
    if not db_path.exists():
        run_session(
            declaration_path=args.declaration,
            events_path=args.events,
            db_path=args.db,
            defaults_version=args.defaults_version,
            config=config,
        )
    with LedgerStore(args.db) as ledger:
        row = ledger.conn.execute(
            "SELECT batch_id, workload, field FROM decisions ORDER BY batch_id, workload, field LIMIT 1"
        ).fetchone()
        if row is None:
            raise DiagnosticError(
                "E_SNAPSHOT", "no ledger decision exists to inject", field="decisions"
            )
        ledger.conn.execute(
            "UPDATE decisions SET reason = 'injected_digest_mismatch' "
            "WHERE batch_id = ? AND workload = ? AND field = ?",
            (row["batch_id"], row["workload"], row["field"]),
        )
        ledger.verify_integrity()


def _write_output(result: dict, output: str | None) -> None:
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output:
        Path(output).write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)


def _add_evaluation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--declaration", required=True, help="JSON declaration file")
    parser.add_argument("--events", required=True, help="NDJSON observation event stream")
    parser.add_argument("--db", required=True, help="SQLite ledger path")
    parser.add_argument("--defaults-version", default="defaults-v1")
    parser.add_argument("--scale-exp", type=int, default=3)
    parser.add_argument("--scale-version", default="fixed-point-v1")
    parser.add_argument("--tolerance", default="0/1", help="convergence tolerance NUM/DEN")
    parser.add_argument("--global-limit", type=int, default=3)
    parser.add_argument("--per-workload-quota", type=int, default=1)
    parser.add_argument("--task-delay-ms", type=float, default=0.0)
    parser.add_argument("--batch-events", type=int, default=4)
    parser.add_argument("--output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drift-diag",
        description="Local declarative configuration drift convergence diagnostics",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="load, normalize, judge, and persist batches")
    _add_evaluation_args(run_parser)
    run_parser.add_argument(
        "--inject",
        choices=("declaration-corrupt", "unknown-event-field", "ledger-digest"),
    )
    run_parser.add_argument(
        "--crash-after-attempt",
        action="store_true",
        help="commit an inflight marker then terminate before the batch transaction",
    )

    verify_parser = subparsers.add_parser("verify", help="replay all committed batches")
    _add_evaluation_args(verify_parser)

    inspect_parser = subparsers.add_parser("inspect", help="show snapshot and ledger summary")
    inspect_parser.add_argument("--db", required=True)
    inspect_parser.add_argument("--output")

    gui_parser = subparsers.add_parser("gui", help="open the read-only Tkinter diagnostic view")
    gui_parser.add_argument("--db", required=True)

    propagate_parser = subparsers.add_parser(
        "propagate", help="apply a change set and show partial recomputation effects"
    )
    propagate_parser.add_argument("--declaration", required=True)
    propagate_parser.add_argument("--events", required=True)
    propagate_parser.add_argument("--change-set", required=True)
    propagate_parser.add_argument("--defaults-version", default="defaults-v1")
    propagate_parser.add_argument("--scale-exp", type=int, default=3)
    propagate_parser.add_argument("--scale-version", default="fixed-point-v1")
    propagate_parser.add_argument("--tolerance", default="0/1", help="convergence tolerance NUM/DEN")
    propagate_parser.add_argument("--output")

    subparsers.add_parser("check", help="show local dependency availability")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            result = {
                "python": sys.version.split()[0],
                "sqlite3_available": True,
                **ui_environment(),
            }
            _write_output(result, None)
            return 0

        if args.command == "inspect":
            with LedgerStore(args.db, read_only=True) as ledger:
                result = {"ok": True, **ledger.ledger_summary()}
                latest = ledger.latest_batch()
                result["latest_batch"] = (
                    None
                    if latest is None
                    else {
                        "batch_id": latest["batch_id"],
                        "batch_key": latest["batch_key"],
                        "event_cursor": latest["event_cursor"],
                        "content_digest": latest["content_digest"],
                        "decisions": latest["decisions"],
                        "workload_metrics": latest["workload_metrics"],
                    }
                )
            _write_output(result, args.output)
            return 0

        if args.command == "gui":
            launch_gui(args.db)
            return 0

        if args.command == "propagate":
            config = _model_config_from_args(args)
            raw_declaration = load_declaration(args.declaration)
            target = normalize_declaration(
                raw_declaration.raw, defaults_version=args.defaults_version
            )
            events = load_events(args.events, target)
            actual_state: dict[str, dict[str, object]] = {}
            for event in events:
                apply_event(actual_state, event)
            base_decisions = sorted(
                (
                    compare_field(
                        target.workloads[workload.name],
                        field,
                        actual_state,
                        scale_exp=config.scale_exp,
                    )
                    for workload in target.ordered_workloads()
                    for field in FIELD_ORDER
                ),
                key=lambda decision: (decision["workload"], FIELD_INDEX[decision["field"]]),
            )
            change_raw = load_json_strict(args.change_set)
            propagation = apply_change_set(
                target,
                actual_state,
                base_decisions,
                config,
                parse_change_set(change_raw),
            )
            result = {
                "ok": True,
                "base_target_digest": target.target_digest,
                "base_overall_converged": aggregate_all(
                    target, base_decisions, config
                )["overall_converged"],
                "change_set": change_raw,
                "propagation": propagation.report,
                "target": propagation.target.to_dict(),
                "decisions": propagation.decisions,
                "workload_metrics": propagation.workload_results,
            }
            _write_output(result, args.output)
            return 0

        config = _config_from_args(args)

        if getattr(args, "inject", None) == "declaration-corrupt":
            _inject_declaration_corrupt()
        elif getattr(args, "inject", None) == "unknown-event-field":
            _inject_unknown_event(args.declaration, args.defaults_version, args.events)
        elif getattr(args, "inject", None) == "ledger-digest":
            _inject_ledger_digest(args, config)

        if args.command == "run":
            session = run_session(
                declaration_path=args.declaration,
                events_path=args.events,
                db_path=args.db,
                defaults_version=args.defaults_version,
                config=config,
                crash_after_attempt=args.crash_after_attempt,
            )
            _write_output(session.to_dict(), args.output)
            return 0

        if args.command == "verify":
            raw_declaration = load_declaration(args.declaration)
            target = normalize_declaration(
                raw_declaration.raw, defaults_version=args.defaults_version
            )
            events = load_events(args.events, target)
            with LedgerStore(args.db, read_only=True) as ledger:
                replay = verify_replay(ledger, target, events, config)
            _write_output({"ok": True, **replay.to_dict()}, args.output)
            return 0

        parser.error("unknown command")
        return 2
    except DiagnosticError as exc:
        sys.stderr.write(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        return EXIT_CODES.get(exc.code, 1)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
