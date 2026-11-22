"""Command-line entry point for diagnostics, anomaly injection, and replay checks."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from .errors import WorkbenchError
from .logging import append_structured_log
from .snapshot import load_snapshot, restore_snapshot, save_snapshot
from .session import WorkbenchSession
from .spec import load_injection_catalog, mutate_declaration_file


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _error_payload(exc: WorkbenchError, phase: str) -> dict[str, Any]:
    payload = {"phase": phase, "error": exc.to_dict()}
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quota-workbench",
        description="Local deterministic multi-tenant quota preemption diagnostics.",
    )
    parser.add_argument("--spec", default="samples/spec.json")
    parser.add_argument("--events", default="samples/normal.ndjson")
    parser.add_argument("--db", default="demo_output/workbench.sqlite3")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--verify-replay", action="store_true")
    parser.add_argument("--save-snapshot")
    parser.add_argument("--snapshot")
    parser.add_argument("--restore-snapshot")
    parser.add_argument("--restore-db")
    parser.add_argument("--current-spec")
    parser.add_argument("--inject")
    parser.add_argument("--injection-file", default="samples/injections.json")
    parser.add_argument("--log", default="demo_output/structured.ndjson")
    parser.add_argument("--ui", action="store_true")
    parser.add_argument("--crash-at-event", type=int, help=argparse.SUPPRESS)
    return parser


def _handle_injection(args: argparse.Namespace) -> int:
    catalog = load_injection_catalog(args.injection_file)
    injection = catalog.get(args.inject)
    if injection is None:
        raise WorkbenchError(
            "E_SPEC",
            f"unknown injection id {args.inject!r}",
            object_type="injection",
            object_id=args.inject,
            field="id",
            details={"available": sorted(catalog)},
        )
    mutation = injection.get("mutation") or {}
    operation = mutation.get("op")
    try:
        if operation == "snapshot_digest_mismatch":
            if not args.snapshot:
                raise WorkbenchError(
                    "E_SNAPSHOT",
                    "snapshot-digest injection requires --snapshot",
                    object_type="snapshot",
                    field="snapshot",
                )
            load_snapshot(args.snapshot, force_digest_mismatch=True)
        else:
            mutate_declaration_file(args.spec, mutation)
        raise AssertionError("injection did not fail validation")
    except WorkbenchError as exc:
        payload = {
            "phase": "INJECTION",
            "injection_id": args.inject,
            "declared_code": injection.get("code"),
            "error": exc.to_dict(),
        }
        append_structured_log(args.log, payload)
        _print(payload)
        return exc.exit_code


def _result_summary(result: Any) -> dict[str, Any]:
    record = result.record
    return {
        "event_id": record["event_id"],
        "terminal": record["terminal"],
        "reclaimed_units": record["terminal"]["reclaimed_units"],
        "preemptions": [
            {
                "tenant_id": item["tenant_id"],
                "workload_id": item["workload_id"],
                "reclaimed_units": item["reclaimed_units"],
                "tenant_usage_before_units": item["tenant_usage_before_units"],
                "tenant_usage_after_units": item["tenant_usage_after_units"],
            }
            for item in record["preemptions"]
        ],
        "gap": record["gap"],
    }


def run(args: argparse.Namespace) -> int:
    if args.inject:
        return _handle_injection(args)
    if args.crash_at_event is not None:
        if args.crash_at_event < 1:
            raise WorkbenchError(
                "E_SPEC",
                "crash-at-event must be positive",
                object_type="cli",
                field="crash_at_event",
            )
    session: WorkbenchSession | None = None
    restore_report: dict[str, Any] | None = None
    try:
        if args.restore_snapshot:
            if not args.restore_db:
                raise WorkbenchError(
                    "E_SNAPSHOT",
                    "--restore-snapshot requires a fresh --restore-db path",
                    object_type="cli",
                    field="restore_db",
                )
            restored = restore_snapshot(
                args.restore_snapshot,
                args.events,
                args.restore_db,
                current_declaration_path=args.current_spec or args.spec,
            )
            session = restored.session
            restore_report = restored.change_report
        else:
            session = WorkbenchSession.open(args.spec, args.events, args.db)

        if args.ui:
            from .ui import launch_ui

            launch_ui(session)
            return 0

        results: list[dict[str, Any]] = []
        crash_triggered = False
        if args.crash_at_event is not None:
            while not session.completed:
                if session.events[session.cursor].event_id == args.crash_at_event:
                    session.run_next(crash_after_prepare=True)
                    crash_triggered = True
                    break
                result = session.run_next()
                if result is None:
                    break
                results.append(_result_summary(result))
        elif args.run_all or args.max_events is not None:
            limit = args.max_events
            while not session.completed and (limit is None or len(results) < limit):
                result = session.run_next()
                if result is None:
                    break
                results.append(_result_summary(result))
                if result.unsatisfied:
                    break

        replay_report = session.verify_replay() if args.verify_replay else None
        snapshot_report = None
        if args.save_snapshot:
            snapshot = save_snapshot(session, args.save_snapshot)
            snapshot_report = {
                "path": str(snapshot.path),
                "cursor": snapshot.ledger_cursor,
                "head_digest": snapshot.ledger_head_digest,
                "content_digest": snapshot.content_digest,
                "readonly": True,
            }
        payload = {
            "status": "UNSATISFIED" if any(item["terminal"]["code"] == "E_UNSATISFIED" for item in results) else "OK",
            "declaration_versions": session.declaration_versions(),
            "cursor": session.cursor,
            "event_count": len(session.events),
            "completed": session.completed,
            "recovery": session.recovery_report(),
            "restore_report": restore_report,
            "results": results,
            "replay": replay_report,
            "snapshot": snapshot_report,
            "final_matrix": session.current_matrix(),
            "ledger_events": session.ledger.event_rows(),
            "ledger_head_digest": session.ledger.head_digest,
        }
        _print(payload)
        if crash_triggered:
            return 137
        if any(item["terminal"]["code"] == "E_UNSATISFIED" for item in results):
            return 13
        return 0
    finally:
        if session is not None:
            session.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except WorkbenchError as exc:
        payload = _error_payload(exc, "RUNTIME")
        try:
            append_structured_log(args.log, payload)
        except OSError:
            pass
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return exc.exit_code
    except KeyboardInterrupt:
        print(json.dumps({"phase": "RUNTIME", "error": {"code": "E_INTERNAL", "message": "interrupted"}}))
        return 130


if __name__ == "__main__":
    sys.exit(main())
