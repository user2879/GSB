"""Command line interface for the local drill workbench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .errors import BusinessError
from .workbench import Workbench


ERROR_EXIT_CODES = {
    "E_PLAN": 2,
    "E_CYCLE": 3,
    "E_SNAPSHOT": 4,
    "E_REPLAY": 5,
    "E_LEDGER": 6,
    "E_STAGE": 7,
}


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--db", default="data/session.db")
    parser.add_argument("--snapshots-dir", default="data/snapshots")
    parser.add_argument("--log", default="data/logs/drill.jsonl")
    return parser


def _workbench(args: argparse.Namespace) -> Workbench:
    return Workbench(args.db, args.snapshots_dir, args.log)


def _parse_injection(value: str):
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise BusinessError(
            "E_PLAN",
            "注入事件格式必须为 event_id,type,offset_ms,stage_seq",
            field_path="--inject",
        )
    try:
        offset_ms = int(parts[2])
        stage_seq = int(parts[3])
    except ValueError as exc:
        raise BusinessError(
            "E_PLAN",
            "注入事件 offset_ms 与 stage_seq 必须是整数",
            field_path="--inject",
        ) from exc
    return {
        "event_id": parts[0],
        "type": parts[1],
        "offset_ms": offset_ms,
        "stage_seq": stage_seq,
    }


def command_validate(args: argparse.Namespace) -> dict[str, Any]:
    workbench = _workbench(args)
    plan = workbench.load_plan_file(args.plan)
    return {
        "ok": True,
        "plan_id": plan.plan_id,
        "version": plan.version,
        "stage_count": len(plan.stages),
        "event_count": len(plan.events),
        "digest": plan.digest,
    }


def command_init(args: argparse.Namespace) -> dict[str, Any]:
    workbench = _workbench(args)
    view = workbench.create_session(args.session, args.plan)
    return {
        "ok": True,
        "session_id": args.session,
        "plan_version": view.plan.version,
        "generation": view.session.generation,
        "stage_count": len(view.stages),
        "event_count": len(view.events),
    }


def command_run(args: argparse.Namespace) -> dict[str, Any]:
    workbench = _workbench(args)
    if not workbench.repository.session_exists(args.session):
        workbench.create_session(args.session, args.plan)
    if args.refresh_plan:
        workbench.publish_plan(args.session, args.refresh_plan)
    for raw_event in args.inject:
        workbench.inject_event(args.session, _parse_injection(raw_event))
    summary = workbench.run_session(
        args.session,
        abort_after_stage=args.abort_after_stage,
        crash_after_stage=args.crash_after_stage,
        hard_crash=args.hard_crash,
    )
    return {"ok": True, **summary}


def command_inject(args: argparse.Namespace) -> dict[str, Any]:
    workbench = _workbench(args)
    runtime = workbench.inject_event(
        args.session,
        {
            "event_id": args.event_id,
            "type": args.type,
            "offset_ms": args.offset_ms,
            "stage_seq": args.stage_seq,
        },
    )
    return {"ok": True, "session_id": args.session, "event": runtime.to_dict()}


def command_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    workbench = _workbench(args)
    return {"ok": True, **workbench.save_snapshot(args.session)}


def command_restore(args: argparse.Namespace) -> dict[str, Any]:
    workbench = _workbench(args)
    return {"ok": True, **workbench.restore_snapshot(args.session, args.snapshot)}


def command_replay(args: argparse.Namespace) -> dict[str, Any]:
    workbench = _workbench(args)
    report = workbench.replay_snapshot(args.session, args.snapshot)
    return {"ok": True, **report.to_dict()}


def command_recover(args: argparse.Namespace) -> dict[str, Any]:
    workbench = _workbench(args)
    return {"ok": True, **workbench.recover(args.session).to_dict()}


def command_resume(args: argparse.Namespace) -> dict[str, Any]:
    workbench = _workbench(args)
    return {"ok": True, **workbench.resume(args.session)}


def command_render(args: argparse.Namespace) -> dict[str, Any]:
    workbench = _workbench(args)
    return {"ok": True, **workbench.render_model(args.session)}


def command_snapshots(args: argparse.Namespace) -> dict[str, Any]:
    workbench = _workbench(args)
    return {"ok": True, "snapshots": workbench.list_snapshots(args.session)}


def command_abort(args: argparse.Namespace) -> dict[str, Any]:
    workbench = _workbench(args)
    summary = workbench.new_engine(args.session).abort()
    return {"ok": True, **summary}


def build_parser() -> argparse.ArgumentParser:
    common = _common_parser()
    parser = argparse.ArgumentParser(
        prog="python -m emergency_drill.cli",
        description="本地确定性应急预案演练工作台",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", parents=[common])
    validate.add_argument("--plan", default="samples/plan_valid.json")
    validate.set_defaults(handler=command_validate)

    init = subparsers.add_parser("init", parents=[common])
    init.add_argument("--session", required=True)
    init.add_argument("--plan", default="samples/plan_valid.json")
    init.set_defaults(handler=command_init)

    run = subparsers.add_parser("run", parents=[common])
    run.add_argument("--session", required=True)
    run.add_argument("--plan", default="samples/plan_valid.json")
    run.add_argument("--refresh-plan")
    run.add_argument("--inject", action="append", default=[])
    run.add_argument("--abort-after-stage", type=int)
    run.add_argument("--crash-after-stage", type=int)
    run.add_argument("--hard-crash", action="store_true")
    run.set_defaults(handler=command_run)

    inject = subparsers.add_parser("inject", parents=[common])
    inject.add_argument("--session", required=True)
    inject.add_argument("--event-id", required=True)
    inject.add_argument("--type", required=True)
    inject.add_argument("--offset-ms", required=True, type=int)
    inject.add_argument("--stage-seq", required=True, type=int)
    inject.set_defaults(handler=command_inject)

    snapshot = subparsers.add_parser("snapshot", parents=[common])
    snapshot.add_argument("--session", required=True)
    snapshot.set_defaults(handler=command_snapshot)

    restore = subparsers.add_parser("restore", parents=[common])
    restore.add_argument("--session", required=True)
    restore.add_argument("--snapshot", required=True)
    restore.set_defaults(handler=command_restore)

    replay = subparsers.add_parser("replay", parents=[common])
    replay.add_argument("--session", required=True)
    replay.add_argument("--snapshot", required=True)
    replay.set_defaults(handler=command_replay)

    recover = subparsers.add_parser("recover", parents=[common])
    recover.add_argument("--session", required=True)
    recover.set_defaults(handler=command_recover)

    resume = subparsers.add_parser("resume", parents=[common])
    resume.add_argument("--session", required=True)
    resume.set_defaults(handler=command_resume)

    render = subparsers.add_parser("render", parents=[common])
    render.add_argument("--session", required=True)
    render.set_defaults(handler=command_render)

    snapshots = subparsers.add_parser("snapshots", parents=[common])
    snapshots.add_argument("--session")
    snapshots.set_defaults(handler=command_snapshots)

    abort = subparsers.add_parser("abort", parents=[common])
    abort.add_argument("--session", required=True)
    abort.set_defaults(handler=command_abort)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    workbench: Workbench | None = None
    try:
        workbench = _workbench(args)
        result = args.handler(args)
        _print(result)
        return 0
    except BusinessError as exc:
        if workbench is None:
            try:
                workbench = _workbench(args)
            except Exception:
                workbench = None
        if workbench is not None:
            workbench.log_error(exc, event="cli_error", command=args.command)
        _print({"ok": False, **exc.to_dict()})
        return ERROR_EXIT_CODES.get(exc.code, 1)
    except (OSError, ValueError) as exc:
        _print({"ok": False, "code": "E_IO", "message": str(exc), "stage_seq": None})
        return 1


if __name__ == "__main__":
    sys.exit(main())
