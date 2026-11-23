"""Command line interface for validation, dispatch, recovery, and Tk UI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .batching import build_plan
from .engine import DiagnosticEngine, FaultInjection
from .errors import BusDiagnosticError, ReplayError, SpecError
from .logging_utils import StructuredLogger
from .spec import load_dataset


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPEC = ROOT / "samples" / "spec.json"
DEFAULT_EVENTS = ROOT / "samples" / "events.ndjson"
DEFAULT_DB = ROOT / "artifacts" / "ledger.sqlite"
DEFAULT_LOG = ROOT / "artifacts" / "bus.jsonl"


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _error_payload(error: BusDiagnosticError) -> dict:
    return {"ok": False, "error": error.as_dict()}


def _load(args) -> tuple:
    dataset = load_dataset(args.spec, args.events)
    plan = build_plan(dataset.declaration, dataset.events)
    return dataset, plan


def _reset_database(path: str) -> None:
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()


def _parse_faults(args) -> dict[str, FaultInjection]:
    faults: dict[str, FaultInjection] = {}
    for item in args.processing_delay or []:
        subscriber_id, separator, raw_ms = item.partition("=")
        if not separator or not raw_ms.isdigit():
            raise SpecError(
                "processing delay must use SUBSCRIBER=MILLISECONDS",
                field="--processing-delay",
            )
        current = faults.get(subscriber_id, FaultInjection())
        faults[subscriber_id] = FaultInjection(
            processing_delay_ms=int(raw_ms), omit_ack_field=current.omit_ack_field
        )
    choices = {"subscriber_id", "batch_id", "event_ids", "processing_ms"}
    for item in args.omit_ack or []:
        subscriber_id, separator, field = item.partition("=")
        if not separator or field not in choices:
            raise SpecError(
                "omit-ack must use SUBSCRIBER=FIELD where FIELD is one of "
                + ", ".join(sorted(choices)),
                field="--omit-ack",
            )
        current = faults.get(subscriber_id, FaultInjection())
        faults[subscriber_id] = FaultInjection(
            processing_delay_ms=current.processing_delay_ms, omit_ack_field=field
        )
    return faults


def command_validate(args) -> int:
    try:
        dataset, plan = _load(args)
    except BusDiagnosticError as exc:
        _print_json(_error_payload(exc))
        return 1
    _print_json(
        {
            "ok": True,
            "declaration_version": dataset.declaration.declaration_version,
            "subscriber_count": len(dataset.declaration.subscribers),
            "channel_count": len(dataset.declaration.channels),
            "event_count": len(dataset.events),
            "batch_count": len(plan.batches),
            "plan_digest": plan.digest(),
            "content_digest": dataset.content_digest,
            "subscriptions": {
                channel_id: list(subscriber_ids)
                for channel_id, subscriber_ids in sorted(plan.subscribers_by_channel.items())
            },
        }
    )
    return 0


def command_plan(args) -> int:
    try:
        _, plan = _load(args)
    except BusDiagnosticError as exc:
        _print_json(_error_payload(exc))
        return 1
    _print_json(
        {
            "ok": True,
            "plan_digest": plan.digest(),
            "batches": [
                {
                    "ordinal": batch.ordinal,
                    "batch_id": batch.batch_id,
                    "channel_id": batch.channel_id,
                    "topic": batch.topic,
                    "subscriber_id": batch.subscriber_id,
                    "batch_index": batch.batch_index,
                    "event_ids": list(batch.event_ids),
                    "content_digest": batch.content_digest,
                }
                for batch in plan.batches
            ],
        }
    )
    return 0


def _run(args, *, recovery_only: bool) -> int:
    try:
        if args.fresh:
            _reset_database(args.db)
        elif recovery_only and not Path(args.db).exists():
            raise BusDiagnosticError(
                "E_LEDGER", f"recovery database does not exist: {args.db}"
            )
        dataset, _ = _load(args)
        faults = _parse_faults(args)
        logger = StructuredLogger(args.log)
        engine = DiagnosticEngine(
            dataset,
            db_path=args.db,
            log_path=args.log,
            worker_log_path=args.worker_log,
            crash_after=args.crash_after,
            fault_injections=faults,
            logger=logger,
        )
        try:
            report = engine.run(resume=True, verify=not args.no_verify)
            _print_json({"ok": not report.replay_mismatch, "report": report.to_dict()})
            if report.replay_mismatch:
                return 3
            if report.failed_batches or report.unknown_batches:
                return 2
            return 0
        finally:
            engine.close()
    except ReplayError as exc:
        _print_json(_error_payload(exc))
        return 3
    except BusDiagnosticError as exc:
        _print_json(_error_payload(exc))
        return 2


def command_run(args) -> int:
    return _run(args, recovery_only=False)


def command_recover(args) -> int:
    return _run(args, recovery_only=True)


def command_ui(args) -> int:
    from .ui import run_ui

    return run_ui(
        spec_path=args.spec,
        events_path=args.events,
        db_path=args.db,
        log_path=args.log,
    )


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spec", default=str(DEFAULT_SPEC), help="declaration JSON path")
    parser.add_argument("--events", default=str(DEFAULT_EVENTS), help="NDJSON event stream path")


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    _add_data_args(parser)
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite ledger path")
    parser.add_argument("--log", default=str(DEFAULT_LOG), help="structured JSONL log path")
    parser.add_argument("--worker-log", default=None, help="subscriber process JSONL log path")
    parser.add_argument("--fresh", action="store_true", help="delete the existing ledger first")
    parser.add_argument("--no-verify", action="store_true", help="skip final ledger replay check")
    parser.add_argument(
        "--crash-after",
        type=int,
        default=None,
        help="test-only: force-terminate the process after N completed batches",
    )
    parser.add_argument(
        "--processing-delay",
        action="append",
        default=[],
        metavar="SUBSCRIBER=MS",
        help="test-only deterministic service delay",
    )
    parser.add_argument(
        "--omit-ack",
        action="append",
        default=[],
        metavar="SUBSCRIBER=FIELD",
        help="test-only omission of an acknowledgement field",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eventbus-diag",
        description="Local deterministic event-bus integration diagnostic bench",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate declarations and events")
    _add_data_args(validate)
    validate.set_defaults(handler=command_validate)

    plan = subparsers.add_parser("plan", help="print the deterministic batch plan")
    _add_data_args(plan)
    plan.set_defaults(handler=command_plan)

    run = subparsers.add_parser("run", help="dispatch and persist a diagnostic run")
    _add_runtime_args(run)
    run.set_defaults(handler=command_run)

    recover = subparsers.add_parser("recover", help="verify and resume the latest ledger run")
    _add_runtime_args(recover)
    recover.set_defaults(handler=command_recover)

    ui = subparsers.add_parser("ui", help="open the Tkinter diagnostic workbench")
    _add_data_args(ui)
    ui.add_argument("--db", default=str(DEFAULT_DB))
    ui.add_argument("--log", default=str(DEFAULT_LOG))
    ui.set_defaults(handler=command_ui)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except SpecError as exc:
        _print_json(_error_payload(exc))
        return 1
    except BusDiagnosticError as exc:
        _print_json(_error_payload(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
