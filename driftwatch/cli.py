from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import load_inputs
from .engine import DriftEngine, recover_ledger, replay_ledger
from .errors import DriftError
from .ledger import Ledger
from .models import agents_version
from .processes import port_is_free


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _error_exit(exc: DriftError) -> int:
    _print(exc.to_dict())
    return 4 if exc.code == "E_REPLAY" else 3 if exc.code in {"E_PORT", "E_TIMEOUT", "E_RECORD"} else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="driftwatch", description="Local deployment drift merge and diagnosis")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate snapshot, agent, and criteria declarations")
    validate.add_argument("--snapshots", required=True)
    validate.add_argument("--agents", required=True)
    validate.add_argument("--criteria", default=None)
    validate.add_argument("--check-ports", action="store_true")

    run = sub.add_parser("run", help="collect observations, merge authorities, persist, and diagnose")
    run.add_argument("--snapshots", required=True)
    run.add_argument("--agents", required=True)
    run.add_argument("--criteria", default=None)
    run.add_argument("--db", required=True)
    run.add_argument("--verify-replay", action="store_true")
    run.add_argument("--no-cleanup-stale", action="store_true")
    run.add_argument("--strict", action="store_true", help="return 1 when any rejected or sparse row exists")

    status = sub.add_parser("status", help="show persisted generation and batch status")
    status.add_argument("--db", required=True)

    show = sub.add_parser("show", help="show the active merged and drift view")
    show.add_argument("--db", required=True)
    show.add_argument("--drift-only", action="store_true")

    recover = sub.add_parser("recover", help="recover, replay, and recollect using declarations stored in the ledger")
    recover.add_argument("--db", required=True)
    recover.add_argument("--no-cleanup-stale", action="store_true")

    replay = sub.add_parser("replay", help="replay committed observations from the active ledger generation")
    replay.add_argument("--db", required=True)
    replay.add_argument("--no-cleanup-stale", action="store_true")

    gui = sub.add_parser("gui", help="open the optional Tk ledger viewer")
    gui.add_argument("--db", required=True)
    return parser


def command_validate(args: argparse.Namespace) -> int:
    chain, agents, criteria = load_inputs(args.snapshots, args.agents, args.criteria)
    payload: dict[str, Any] = {
        "ok": True,
        "snapshot_chain_version": chain.version(),
        "agent_set_version": agents_version(agents),
        "scale_version": criteria.scale_version(),
        "snapshots": [snapshot.seq for snapshot in chain.ordered()],
        "agents": [agent.agent_id for agent in agents],
    }
    if args.check_ports:
        occupied = [
            {"agent_id": agent.agent_id, "port": agent.port}
            for agent in agents
            if not port_is_free(agent.host, agent.port)
        ]
        payload["ports_free"] = not occupied
        payload["occupied"] = occupied
        if occupied:
            raise DriftError.port("one or more loopback ports are already occupied", occupied=occupied)
    _print(payload)
    return 0


def command_run(args: argparse.Namespace) -> int:
    engine = DriftEngine.from_files(args.snapshots, args.agents, args.criteria, args.db)
    try:
        report = engine.run(
            verify_replay=args.verify_replay,
            cleanup_stale=not args.no_cleanup_stale,
        )
        payload = report.to_dict()
        _print(payload)
        if args.strict and (
            payload["view_summary"]["rejected"] or payload["view_summary"]["unknown"]
        ):
            return 1
        return 0
    finally:
        engine.close()


def command_status(args: argparse.Namespace) -> int:
    with Ledger(args.db) as ledger:
        _print(ledger.status())
    return 0


def command_show(args: argparse.Namespace) -> int:
    with Ledger(args.db) as ledger:
        view = ledger.active_view()
        if args.drift_only:
            view = {"generation": view["generation"], "drift_view": view["drift_view"]}
        _print(view)
    return 0


def command_recover(args: argparse.Namespace) -> int:
    _print(recover_ledger(args.db, cleanup_stale=not args.no_cleanup_stale))
    return 0


def command_replay(args: argparse.Namespace) -> int:
    _print(replay_ledger(args.db, cleanup_stale=not args.no_cleanup_stale))
    return 0


def command_gui(args: argparse.Namespace) -> int:
    from .gui import main as gui_main

    return gui_main(args.db)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            return command_validate(args)
        if args.command == "run":
            return command_run(args)
        if args.command == "status":
            return command_status(args)
        if args.command == "show":
            return command_show(args)
        if args.command == "recover":
            return command_recover(args)
        if args.command == "replay":
            return command_replay(args)
        if args.command == "gui":
            return command_gui(args)
        parser.error("unknown command")
        return 2
    except DriftError as exc:
        return _error_exit(exc)
    except KeyboardInterrupt:
        _print({"ok": False, "code": "E_INTERRUPTED", "message": "interrupted"})
        return 130
    except Exception as exc:
        _print({"ok": False, "code": "E_INTERNAL", "message": str(exc)})
        return 5
