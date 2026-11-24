"""Command line interface for validation, runs, replay, and recovery."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

from .declarations import DeclarationSet, load_declarations
from .engine import LeaseOrchestrator
from .errors import ProbeLeaseError
from .ledger import Ledger, restore_from_snapshot
from .logging import StructuredLogger
from .replay import compare_splice_ledgers, replay_verify


def _common_declaration_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sessions", required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--ack-policy", required=True)
    parser.add_argument("--log", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="probelease",
        description="Deterministic loopback UDP lease ledger",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate declarations")
    _common_declaration_args(validate)
    validate.add_argument(
        "--inject",
        choices=("none", "corrupt-session", "ack-version", "dependency-cycle"),
        default="none",
    )

    run = subparsers.add_parser("run", help="run probe rounds")
    _common_declaration_args(run)
    run.add_argument("--ledger", required=True)
    run.add_argument("--session", default=None)
    run.add_argument("--cycles", type=int, default=1)
    run.add_argument("--tick", action="store_true", help="honor declared poll periods")
    run.add_argument("--snapshot", default=None)
    run.add_argument("--snapshot-after-cycle", type=int, default=None)
    run.add_argument("--crash-before-commit-cycle", type=int, default=None)

    replay = subparsers.add_parser("replay", help="replay committed rounds over UDP")
    _common_declaration_args(replay)
    replay.add_argument("--ledger", required=True)
    replay.add_argument("--session", default=None)

    splice = subparsers.add_parser("splice", help="compare logical ledgers byte-for-byte")
    _common_declaration_args(splice)
    splice.add_argument("--left", required=True)
    splice.add_argument("--right", required=True)
    splice.add_argument("--session", action="append", default=None)
    splice.add_argument("--start-cycle", type=int, default=None)
    splice.add_argument("--end-cycle", type=int, default=None)

    restore = subparsers.add_parser("restore", help="wholesale restore from snapshot")
    _common_declaration_args(restore)
    restore.add_argument("--ledger", required=True)
    restore.add_argument("--snapshot", required=True)

    leases = subparsers.add_parser("leases", help="print current lease state")
    _common_declaration_args(leases)
    leases.add_argument("--ledger", required=True)

    demo = subparsers.add_parser("demo", help="run the complete repeatable demo")
    demo.add_argument("--root", default="artifacts/results/demo")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = StructuredLogger(getattr(args, "log", None))
    try:
        if args.command == "demo":
            from .demo import run_demo

            report = run_demo(Path(args.root))
            _print(report)
            return 0
        declarations = _load_with_injection(args, logger)
        if args.command == "validate":
            report = {
                "code": "OK",
                "format_version": declarations.format_version,
                "session_set_version": declarations.session_set_version,
                "probe_parameter_version": declarations.probe_parameter_version,
                "ack_policy_version": declarations.ack_policy_version,
                "content_digest": declarations.content_digest,
                "sessions": [session.session_id for session in declarations.sessions],
                "inject": args.inject,
            }
        elif args.command == "run":
            report = _run(args, declarations, logger)
        elif args.command == "replay":
            report = replay_verify(
                args.ledger,
                declarations,
                session_id=args.session,
                logger=logger,
            ).to_dict()
            report["code"] = "OK"
        elif args.command == "splice":
            left_path = Path(args.left)
            right_path = Path(args.right)
            if not left_path.exists() or not right_path.exists():
                raise ProbeLeaseError(
                    "E_SPLICE",
                    "both ledger paths must already exist",
                    "splice.paths",
                    {"left_exists": left_path.exists(), "right_exists": right_path.exists()},
                )
            with Ledger(left_path, declarations, logger=logger, recover_incomplete=False) as left:
                with Ledger(right_path, declarations, logger=logger, recover_incomplete=False) as right:
                    splice_report = compare_splice_ledgers(
                        left,
                        right,
                        session_ids=args.session,
                        start_cycle=args.start_cycle,
                        end_cycle=args.end_cycle,
                    )
            report = splice_report.to_dict()
            report["code"] = "OK"
        elif args.command == "restore":
            restore_report = restore_from_snapshot(
                snapshot_path=args.snapshot,
                db_path=args.ledger,
                declarations=declarations,
                logger=logger,
            )
            report = restore_report.to_dict()
            report["code"] = "OK"
        elif args.command == "leases":
            with Ledger(
                args.ledger, declarations, logger=logger, recover_incomplete=False
            ) as ledger:
                report = {"code": "OK", "sessions": ledger.all_sessions()}
        else:
            raise AssertionError(f"unhandled command {args.command}")
        logger.emit("cli.completed", command=args.command, report=report)
        _print(report)
        return 0
    except ProbeLeaseError as exc:
        payload = exc.to_dict()
        logger.emit("cli.error", code=exc.code, error=payload)
        _print({"event": "cli.error", **payload})
        return 2
    except Exception as exc:  # defensive CLI boundary
        payload = {
            "code": "E_INTERNAL",
            "message": str(exc),
            "details": {"type": type(exc).__name__},
        }
        logger.emit("cli.error", code="E_INTERNAL", error=payload)
        _print({"event": "cli.error", **payload})
        return 3


def _run(
    args: argparse.Namespace, declarations: DeclarationSet, logger: StructuredLogger
) -> dict[str, Any]:
    snapshot_after = args.snapshot_after_cycle
    if snapshot_after is not None and not args.snapshot:
        raise ProbeLeaseError(
            "E_SESSION",
            "--snapshot-after-cycle requires --snapshot",
            "cli.snapshot",
        )
    orchestrator = LeaseOrchestrator(declarations, args.ledger, logger=logger)
    results: list[dict[str, Any]] = []
    snapshot_report: dict[str, Any] | None = None
    try:
        orchestrator.start_probes()
        sessions = (
            [declarations.session(args.session)]
            if args.session is not None
            else orchestrator.topological_sessions()
        )
        next_tick: dict[str, int] = {}
        for _ in range(args.cycles):
            for session in sessions:
                state = orchestrator.ledger.get_session(session.session_id)
                if state["status"] != "ACTIVE":
                    continue
                if args.tick and session.session_id in next_tick:
                    delay = next_tick[session.session_id] - time.monotonic_ns()
                    if delay > 0:
                        time.sleep(delay / 1_000_000_000)
                target_cycle = int(state["cycle_cursor"]) + 1
                crashed = (
                    args.crash_before_commit_cycle is not None
                    and target_cycle == args.crash_before_commit_cycle
                )
                result = orchestrator.execute_round(
                    session.session_id, crash_before_commit=crashed
                )
                results.append(result.to_dict())
                next_tick[session.session_id] = (
                    time.monotonic_ns() + session.poll_period_ms * 1_000_000
                )
                if snapshot_after is not None and result.cycle == snapshot_after:
                    snapshot_report = orchestrator.save_snapshot(args.snapshot)
        if args.snapshot and snapshot_report is None:
            snapshot_report = orchestrator.save_snapshot(args.snapshot)
        return {
            "code": "OK",
            "ledger": str(Path(args.ledger)),
            "recovery_report": orchestrator.recovery_report.to_dict(),
            "rounds": results,
            "leases": orchestrator.query_leases(),
            "snapshot": snapshot_report,
            "forced_termination_exit_report": None,
        }
    finally:
        orchestrator.close()


def _load_with_injection(
    args: argparse.Namespace, logger: StructuredLogger
) -> DeclarationSet:
    inject = getattr(args, "inject", "none")
    if inject == "none":
        return load_declarations(args.sessions, args.probes, args.ack_policy)
    with tempfile.TemporaryDirectory(prefix="probelease-inject-") as temp_dir:
        sessions_path = Path(temp_dir) / "sessions.json"
        if inject == "corrupt-session":
            sessions_path.write_text('{"format_version": 1, "sessions": [', encoding="utf-8")
        else:
            document = json.loads(Path(args.sessions).read_text(encoding="utf-8"))
            if inject == "ack-version":
                document["ack_policy_version"] = int(document.get("ack_policy_version", 0)) + 1
            elif inject == "dependency-cycle":
                document = _make_dependency_cycle(document)
            else:
                raise ValueError(f"unknown injection {inject}")
            sessions_path.write_text(
                json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        logger.emit("fault.injected", inject=inject)
        return load_declarations(sessions_path, args.probes, args.ack_policy)


def _make_dependency_cycle(document: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(document)
    sessions = result.get("sessions", [])
    by_id = {session.get("session_id"): session for session in sessions}
    for session in sessions:
        dependencies = session.get("depends_on", [])
        if dependencies:
            target = dependencies[0]
            target_session = by_id.get(target)
            if target_session is not None:
                target_deps = target_session.setdefault("depends_on", [])
                if session["session_id"] not in target_deps:
                    target_deps.append(session["session_id"])
                return result
    if sessions:
        sessions[0].setdefault("depends_on", []).append(sessions[0]["session_id"])
    return result


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())

