"""Command-line interface for validation, live scheduling, replay and diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .errors import WorkbenchError
from .gui.view import format_text_view
from .nodes import NodeSupervisor, SocketDispatcher
from .replay import ReplayVerifier
from .scheduler import Scheduler
from .spec import load_spec
from .storage import Ledger
from .util import StructuredLog, canonical_json, ensure_directory


def _db_path(args: argparse.Namespace) -> Path:
    if getattr(args, "db", None):
        return Path(args.db)
    return Path(args.data_dir) / "ledger.sqlite3"


def _print(value: Any, json_output: bool, human: str | None = None) -> None:
    if json_output:
        print(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2))
    elif human is not None:
        print(human, end="" if human.endswith("\n") else "\n")
    else:
        print(canonical_json(value))


def cmd_validate(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    value = {
        "ok": True,
        "spec_version": spec.spec_version,
        "spec_digest": spec.digest,
        "budget_digest": spec.budget_digest,
        "current_version": spec.current_version,
        "nodes": [node.node_id for node in spec.nodes],
        "tasks": [task.task_id for task in spec.tasks],
    }
    _print(value, args.json, f"SPEC_OK: spec_version={spec.spec_version} digest={spec.digest}\n")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    data_dir = ensure_directory(args.data_dir)
    ensure_directory(data_dir / "logs")
    ensure_directory(data_dir / "snapshots")
    logger = StructuredLog(data_dir / "logs" / "workbench.jsonl")
    logger.write("spec_validated", spec_digest=spec.digest, spec_version=spec.spec_version)
    ledger = Ledger(_db_path(args))
    try:
        activation = ledger.activate_spec(spec)
        snapshots: list[dict[str, Any]] = []

        def snapshot_hook(state: dict[str, Any], label: str) -> None:
            created = ledger.create_snapshot(activation.session_id, state, label)
            snapshots.append(created)
            logger.write("snapshot_created", **created)

        supervisor = NodeSupervisor(spec, data_dir, logger)
        try:
            supervisor.start_all()
            scheduler = Scheduler(
                spec,
                ledger,
                activation.session_id,
                SocketDispatcher(spec, logger),
                recovered_state=activation.state,
                logger=logger,
                snapshot_hook=snapshot_hook,
            )
            summary = scheduler.run()
        finally:
            supervisor.stop_all()
        final_snapshot = ledger.create_snapshot(activation.session_id, ledger.latest_state(activation.session_id) or {}, "session-final")
        value = {
            **summary.to_dict(),
            "activation": activation.action,
            "stale_summary": activation.stale_summary,
            "snapshots": snapshots + [final_snapshot],
        }
        logger.write(
            "run_completed",
            session_id=activation.session_id,
            deterministic_digest=summary.deterministic_digest,
        )
        human = (
            f"session={activation.session_id} activation={activation.action}\n"
            f"tasks={len(summary.tasks)} preemptions={len(summary.preemptions)} "
            f"receipts={len(summary.receipts)} deterministic_digest={summary.deterministic_digest}\n"
            + "untrust_ranking="
            + canonical_json([(row['node_id'], row['score']) for row in summary.node_untrust])
            + "\n"
        )
        _print(value, args.json, human)
        return 0
    finally:
        ledger.close()


def cmd_replay(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    data_dir = ensure_directory(args.data_dir)
    ledger = Ledger(_db_path(args))
    try:
        session_id = args.session or ledger.current_session_id()
        if session_id is None:
            raise WorkbenchError("E_REPLAY", "session", "no current session to replay")
        snapshot_id = args.snapshot
        if snapshot_id in (None, "latest", "session-start"):
            snapshots = ledger.list_snapshots(session_id)
            if snapshot_id == "session-start":
                matches = [item for item in snapshots if item["label"] == "session-start"]
                snapshot_id = matches[0]["snapshot_id"] if matches else None
            elif snapshots:
                snapshot_id = snapshots[-1]["snapshot_id"]
            else:
                snapshot_id = None
        dispatcher = None
        supervisor = None
        if not args.offline:
            logger = StructuredLog(data_dir / "logs" / "workbench.jsonl")
            supervisor = NodeSupervisor(spec, data_dir, logger)
            dispatcher = SocketDispatcher(spec, logger)
            supervisor.start_all()
        try:
            verifier = ReplayVerifier(
                spec,
                ledger,
                session_id,
                snapshot_id=snapshot_id,
                dispatcher=dispatcher,
                redispatch=not args.offline,
            )
            summary = verifier.verify()
        finally:
            if supervisor is not None:
                supervisor.stop_all()
        value = summary.to_dict()
        human = (
            f"REPLAY_OK session={session_id} snapshot={snapshot_id} checks={summary.checks} "
            f"redispatched={summary.redispatch_count} remaining={summary.remaining_task_ids}\n"
            f"result_digest={summary.result_digest}\n"
        )
        _print(value, args.json, human)
        return 0
    finally:
        ledger.close()


def cmd_diagnose(args: argparse.Namespace) -> int:
    ledger = Ledger(_db_path(args))
    try:
        view = ledger.diagnostic_view(args.session)
        _print(view, args.json, format_text_view(view))
        return 0
    finally:
        ledger.close()


def cmd_gui(args: argparse.Namespace) -> int:
    try:
        from .gui import launch
        launch(str(_db_path(args)), args.session)
    except ImportError as exc:
        raise WorkbenchError("E_TK", "tkinter", f"Tkinter is unavailable: {exc}") from exc
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    ledger = Ledger(_db_path(args))
    try:
        activation = ledger.activate_spec(spec)
        state = ledger.latest_state(activation.session_id)
        if state is None:
            raise WorkbenchError("E_SNAPSHOT", "state", "session has no committed scheduler state")
        value = ledger.create_snapshot(activation.session_id, state, args.label)
        _print(value, args.json, f"snapshot={value['snapshot_id']} digest={value['content_digest']}\n")
        return 0
    finally:
        ledger.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m modeldash", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="strictly validate a JSON specification")
    validate.add_argument("--spec", required=True)
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(func=cmd_validate)

    run = sub.add_parser("run", help="start local nodes and execute one scheduling session")
    run.add_argument("--spec", required=True)
    run.add_argument("--data-dir", default="runtime")
    run.add_argument("--db")
    run.add_argument("--json", action="store_true")
    run.set_defaults(func=cmd_run)

    replay = sub.add_parser("replay", help="replay ledger state and verify live outputs")
    replay.add_argument("--spec", required=True)
    replay.add_argument("--data-dir", default="runtime")
    replay.add_argument("--db")
    replay.add_argument("--session")
    replay.add_argument("--snapshot", help="snapshot id, latest, session-start, or omit")
    replay.add_argument("--offline", action="store_true", help="verify ledger arithmetic without redispatch")
    replay.add_argument("--json", action="store_true")
    replay.set_defaults(func=cmd_replay)

    diagnose = sub.add_parser("diagnose", help="print the read-only diagnostic view")
    diagnose.add_argument("--data-dir", default="runtime")
    diagnose.add_argument("--db")
    diagnose.add_argument("--session")
    diagnose.add_argument("--json", action="store_true")
    diagnose.set_defaults(func=cmd_diagnose)

    gui = sub.add_parser("gui", help="open the Tkinter diagnostic workbench")
    gui.add_argument("--data-dir", default="runtime")
    gui.add_argument("--db")
    gui.add_argument("--session")
    gui.set_defaults(func=cmd_gui)

    snapshot = sub.add_parser("snapshot", help="create a snapshot from the latest committed state")
    snapshot.add_argument("--spec", required=True)
    snapshot.add_argument("--data-dir", default="runtime")
    snapshot.add_argument("--db")
    snapshot.add_argument("--label", default="manual")
    snapshot.add_argument("--json", action="store_true")
    snapshot.set_defaults(func=cmd_snapshot)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except WorkbenchError as exc:
        print(canonical_json(exc.to_dict()), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(canonical_json({"ok": False, "code": "E_INTERRUPTED", "field": "process", "detail": "interrupted"}), file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
