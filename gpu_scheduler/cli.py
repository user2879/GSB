"""Command line entry point for the GPU quota and shard scheduler demo.

All outputs stay inside the chosen workspace directory.  Commands:

* ``run``        - deterministic run with ledger and snapshot persistence
* ``inject``     - run one of the four required fault injections
* ``crash``      - force-terminate the process before a round commits
* ``recover``    - restart from committed rounds and verify replay equivalence
* ``replace``    - demonstrate whole-state snapshot replacement on version drift
* ``stats``      - print fixed-point statistics and error bounds
* ``gui``        - open the Tkinter diagnostic view (or console fallback)
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List

from .diagnostics import _TK_AVAILABLE, launch_gui, render_console
from .engine import SimulationEngine
from .errors import SchedulerError
from .injector import (
    INJECTION_KINDS,
    run_injected_validation,
    tamper_ledger,
)
from .ledger import Ledger
from .loader import load_declaration
from .logging_utils import StructuredLogger
from .scheduler import SchedulingSession, resume_session
from . import replay as replay_mod

SPEC_DIR = os.path.join("samples", "specs")
WORK_DIR = "workdata"


def _paths(args) -> tuple:
    base = args.specs or SPEC_DIR
    return (os.path.join(base, "session.json"),
            os.path.join(base, "pools.json"),
            os.path.join(base, "tasks.json"))


def _workspace_paths(args) -> dict:
    root = args.workspace or WORK_DIR
    return {
        "root": root,
        "ledger": os.path.join(root, "ledger.sqlite"),
        "snapshots": os.path.join(root, "snapshots"),
        "backup": os.path.join(root, "backup"),
        "logs": os.path.join(root, "logs", "events.jsonl"),
        "injected": os.path.join(root, "injected"),
    }


def _load(args):
    session_path, pools_path, tasks_path = _paths(args)
    return load_declaration(session_path, pools_path, tasks_path)


def cmd_run(args) -> int:
    logger = StructuredLogger(_workspace_paths(args)["logs"])
    paths = _workspace_paths(args)
    declaration = _load(args)
    _reset_paths(paths)
    session = SchedulingSession(
        declaration, paths["ledger"], paths["snapshots"], paths["backup"])
    result = session.run()
    session.close()
    logger.info("RUN_COMPLETE", "deterministic scheduling finished",
                rounds=result["rounds"], ledger_digest=result["ledger_digest"])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_inject(args) -> int:
    logger = StructuredLogger(_workspace_paths(args)["logs"])
    session_path, pools_path, tasks_path = _paths(args)
    try:
        if args.kind == "ledger":
            return _inject_ledger(args, logger)
        code, detail = run_injected_validation(
            args.kind, session_path, pools_path, tasks_path,
            _workspace_paths(args)["injected"])
        logger.error(code, f"injected fault {args.kind} rejected", detail=detail)
        print(json.dumps(detail, indent=2, ensure_ascii=False))
        return 2
    except SchedulerError as exc:
        logger.error(exc.code, exc.message, detail=exc.to_dict())
        print(json.dumps(exc.to_dict(), indent=2, ensure_ascii=False))
        return 2


def _inject_ledger(args, logger: StructuredLogger) -> int:
    paths = _workspace_paths(args)
    _reset_paths(paths)
    declaration = _load(args)
    session = SchedulingSession(
        declaration, paths["ledger"], paths["snapshots"], paths["backup"])
    session.run()
    session.close()
    tamper_ledger(paths["ledger"])
    try:
        ledger = Ledger(paths["ledger"])
        ledger.verify()
        ledger.close()
        raise AssertionError("ledger tampering was not detected")
    except SchedulerError as exc:
        logger.error(exc.code, "injected ledger tampering detected",
                     detail=exc.to_dict())
        print(json.dumps(exc.to_dict(), indent=2, ensure_ascii=False))
        return 2


def cmd_crash(args) -> int:
    logger = StructuredLogger(_workspace_paths(args)["logs"])
    paths = _workspace_paths(args)
    _reset_paths(paths)
    declaration = _load(args)
    session = SchedulingSession(
        declaration, paths["ledger"], paths["snapshots"], paths["backup"])
    result = session.run(crash_after_round=args.after_round)
    session.close()
    logger.error("FORCED_TERMINATION",
                 "process killed before round commit", **result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_recover(args) -> int:
    logger = StructuredLogger(_workspace_paths(args)["logs"])
    paths = _workspace_paths(args)
    declaration = _load(args)
    reference = replay_mod.reference_run(_load(args))
    result = resume_session(
        declaration, paths["ledger"], paths["snapshots"], paths["backup"],
        reference_rounds=reference.rt.rounds)
    logger.info("RECOVERY_COMPLETE", "resumed and replayed from ledger",
                **result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_replace(args) -> int:
    logger = StructuredLogger(_workspace_paths(args)["logs"])
    paths = _workspace_paths(args)
    declaration = _load(args)
    # First create a committed snapshot from the current declaration.
    _reset_paths(paths)
    session = SchedulingSession(
        declaration, paths["ledger"], paths["snapshots"], paths["backup"])
    session.run()
    session.close()
    # Mutate the declaration version by changing a non-semantic sample value in
    # a copied spec directory, then resume: the snapshot must be replaced.
    drifted_dir = os.path.join(paths["injected"], "drifted_specs")
    _write_drifted_specs(_paths(args), drifted_dir)
    args.specs = drifted_dir
    drifted = _load(args)
    result = resume_session(
        drifted, paths["ledger"], paths["snapshots"], paths["backup"])
    logger.info("SNAPSHOT_REPLACED",
                "version drift triggered whole-state replacement", **result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_stats(args) -> int:
    declaration = _load(args)
    engine = SimulationEngine(declaration)
    engine.run_until()
    print(render_console(engine, _workspace_paths(args)["snapshots"]))
    return 0


def cmd_gui(args) -> int:
    declaration = _load(args)
    engine = SimulationEngine(declaration)
    engine.run_until()
    if _TK_AVAILABLE:
        launch_gui(engine, _workspace_paths(args)["snapshots"])
        return 0
    print(render_console(engine, _workspace_paths(args)["snapshots"]))
    print("\nTkinter unavailable in this interpreter; console view rendered.")
    return 0


def _reset_paths(paths: dict) -> None:
    for path in (paths["ledger"],):
        if os.path.exists(path):
            os.remove(path)
    if os.path.isdir(paths["snapshots"]):
        import shutil
        shutil.rmtree(paths["snapshots"])
    os.makedirs(paths["snapshots"], exist_ok=True)
    os.makedirs(paths["backup"], exist_ok=True)


def _write_drifted_specs(origin: tuple, target_dir: str) -> None:
    import shutil
    os.makedirs(target_dir, exist_ok=True)
    for src in origin:
        shutil.copy(src, os.path.join(target_dir, os.path.basename(src)))
    tasks_path = os.path.join(target_dir, "tasks.json")
    tasks = json.load(open(tasks_path, encoding="utf-8"))
    tasks["tasks"][0]["estimated_ms"] += 1  # changes task version hash
    with open(tasks_path, "w", encoding="utf-8") as handle:
        json.dump(tasks, handle, indent=2, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPU quota and shard scheduling domain component demo")
    parser.add_argument("--specs", help="directory of session/pools/tasks JSON")
    parser.add_argument("--workspace", help="local workspace output directory")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run").set_defaults(func=cmd_run)
    inject = sub.add_parser("inject")
    inject.add_argument("kind", choices=INJECTION_KINDS)
    inject.set_defaults(func=cmd_inject)

    crash = sub.add_parser("crash")
    crash.add_argument("--after-round", type=int, default=2)
    crash.set_defaults(func=cmd_crash)

    sub.add_parser("recover").set_defaults(func=cmd_recover)
    sub.add_parser("replace").set_defaults(func=cmd_replace)
    sub.add_parser("stats").set_defaults(func=cmd_stats)
    sub.add_parser("gui").set_defaults(func=cmd_gui)
    return parser


def main(argv: List[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SchedulerError as exc:
        print(json.dumps(exc.to_dict(), indent=2, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
