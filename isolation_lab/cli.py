from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .demo import render_result, run_demo
from .errors import BusinessError
from .models import SchedulerParams, canonical_json
from .scheduler import Engine, compare_run_outcomes
from .scripts import load_script
from .snapshot import replay_snapshot, restore_snapshot, save_snapshot
from .storage import SnapshotStore, Storage


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _add_json_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")


def _scheduler_params(args: argparse.Namespace) -> SchedulerParams:
    return SchedulerParams(
        global_concurrency_limit=args.global_concurrency,
        clock_quantum_ms=args.clock_quantum,
    )


def _snapshot_path(args: argparse.Namespace, db_path: str) -> str:
    if getattr(args, "snapshots", None):
        return args.snapshots
    parent = Path(db_path).parent
    return str(parent / "snapshots.sqlite")


def _render_upgrade(script: Any) -> list[str]:
    lines = [
        f"valid=true source_version={script.source_version} normalized_version={script.normalized_version}",
        f"script_id={script.script_id} digest={script.semantic_digest}",
        f"sessions={len(script.sessions)}",
    ]
    for session in script.sessions:
        lines.append(
            f"  {session.session_id} isolation={session.isolation_level} "
            f"budget={session.lock_wait_budget_ms}ms concurrency={session.concurrency} "
            f"operation_timeout={session.operation_timeout_ms}ms "
            f"lock_order={','.join(session.lock_order)} operations={len(session.operations)}"
        )
    if script.upgrade_changes:
        lines.append("upgrade_changes:")
        for change in script.upgrade_changes:
            lines.append(f"  {change}")
    return lines


def command_validate(args: argparse.Namespace) -> int:
    script = load_script(args.script)
    payload = {
        "valid": True,
        "source_version": script.source_version,
        "normalized_version": script.normalized_version,
        "script_id": script.script_id,
        "script_digest": script.semantic_digest,
        "upgrade_changes": list(script.upgrade_changes),
        "normalized_script": script.to_dict(include_source=True),
    }
    print(_json_text(payload) if args.json else "\n".join(_render_upgrade(script)))
    return 0


def command_upgrade(args: argparse.Namespace) -> int:
    script = load_script(args.script)
    text = _json_text(script.to_dict(include_source=True)) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        if not args.json:
            print(f"upgraded={args.output}")
            print(f"source_version={script.source_version} normalized_version={script.normalized_version}")
    else:
        print(text, end="")
    return 0


def command_run(args: argparse.Namespace) -> int:
    script = load_script(args.script)
    params = _scheduler_params(args)
    storage = Storage(args.db)
    result = Engine(storage, params).run(script)
    snapshot_id = args.snapshot_id
    if snapshot_id:
        snapshot_path = _snapshot_path(args, args.db)
        with SnapshotStore(snapshot_path) as store:
            save_snapshot(store, snapshot_id, script, result)
    payload = {
        "summary": result.summary_dict(),
        "snapshot_id": snapshot_id,
        "snapshot_path": _snapshot_path(args, args.db) if snapshot_id else None,
    }
    if args.json:
        print(_json_text(payload))
    else:
        lines = render_result(result)
        if snapshot_id:
            lines.append(f"  snapshot={snapshot_id} store={_snapshot_path(args, args.db)}")
        print("\n".join(lines))
    storage.close()
    return 0


def command_ledger(args: argparse.Namespace) -> int:
    with Storage(args.db) as storage:
        records = storage.read_ledger(run_id=args.run_id)
    if args.json:
        print(_json_text([record.to_dict() for record in records]))
        return 0
    source_versions = sorted({record.source_version for record in records})
    lines = [
        f"ledger_records={len(records)} "
        f"source_versions={','.join(str(version) for version in source_versions) or '-'} "
        f"normalized_versions={','.join(str(version) for version in sorted({record.normalized_version for record in records})) or '-'}"
    ]
    for record in records:
        lines.append(
            f"  event={record.event_id} run={record.run_id} seq={record.trace_event_seq} "
            f"script={record.script_id} source_version={record.source_version} "
            f"normalized_version={record.normalized_version} "
            f"session={record.session_id or '-'} op={record.op_no if record.op_no is not None else '-'} "
            f"type={record.event_type} virtual_ms={record.virtual_ms}"
        )
    print("\n".join(lines))
    return 0


def command_snapshot_list(args: argparse.Namespace) -> int:
    with SnapshotStore(args.snapshots) as store:
        ids = store.list_ids()
    if args.json:
        print(_json_text({"snapshot_ids": ids}))
    else:
        print("snapshots=" + (",".join(ids) if ids else "-"))
    return 0


def command_snapshot_replay(args: argparse.Namespace) -> int:
    with SnapshotStore(args.snapshots) as store:
        report, replay = replay_snapshot(store, args.snapshot_id, raise_on_error=False)
    payload = report.to_dict()
    if not report.reproducible:
        payload["code"] = "E_REPLAY"
    if args.json:
        print(_json_text(payload))
    else:
        print(
            f"snapshot={report.snapshot_id} reproducible={'true' if report.reproducible else 'false'} "
            f"data_version={replay.global_data_version}"
        )
        if not report.reproducible:
            print(
                f"E_REPLAY: {report.message}; session={report.session_id or '-'} "
                f"op={report.op_no if report.op_no is not None else '-'} field={report.field or '-'} "
                f"expected={_json_text(report.expected)} actual={_json_text(report.actual)}",
                file=sys.stderr,
            )
    return 0 if report.reproducible else 3


def command_snapshot_restore(args: argparse.Namespace) -> int:
    with SnapshotStore(args.snapshots) as store, Storage(args.db) as target:
        report = restore_snapshot(store, args.snapshot_id, target_storage=target)
    payload = report.to_dict()
    if args.json:
        print(_json_text(payload))
    else:
        print(
            "restore="
            f"snapshot:{report.snapshot_id},mode:{report.mode},"
            f"script_digest_match:{report.script_digest_match},"
            f"param_digest_match:{report.param_digest_match},"
            f"parameter_version_match:{report.parameter_version_match},"
            f"data_version:{report.data_version},"
            f"changed_sessions:{','.join(report.changed_sessions) or '-'},"
            f"changed_params:{','.join(report.changed_parameter_items) or '-'},"
            f"cleared_lock_entries:{report.cleared_lock_entries},"
            f"trace_entries:{report.trace_entries}"
        )
    return 0


def command_equivalence(args: argparse.Namespace) -> int:
    params = _scheduler_params(args)
    left_script = load_script(args.left)
    right_script = load_script(args.right)
    left_storage = Storage(":memory:")
    right_storage = Storage(":memory:")
    left = Engine(left_storage, params).run(left_script)
    right = Engine(right_storage, params).run(right_script)
    differences = compare_run_outcomes(left, right)
    payload = {
        "equivalent": not differences,
        "left": {
            "script_id": left.script_id,
            "source_version": left.source_version,
            "run_id": left.run_id,
            "data_version": left.global_data_version,
            "final_rows": left.final_rows,
        },
        "right": {
            "script_id": right.script_id,
            "source_version": right.source_version,
            "run_id": right.run_id,
            "data_version": right.global_data_version,
            "final_rows": right.final_rows,
        },
        "differences": differences,
    }
    if args.json:
        print(_json_text(payload))
    else:
        print(
            f"equivalent={'true' if not differences else 'false'} "
            f"left_source={left.source_version} right_source={right.source_version} "
            f"data_version={left.global_data_version}/{right.global_data_version}"
        )
        for difference in differences:
            print(
                f"  diff field={difference['field']} "
                f"left={canonical_json(difference['left'])} "
                f"right={canonical_json(difference['right'])}"
            )
    left_storage.close()
    right_storage.close()
    return 0 if not differences else 4


def command_demo(args: argparse.Namespace) -> int:
    report = run_demo(args.workdir)
    if args.json:
        print(_json_text({"report": report, "workdir": args.workdir}))
    else:
        print(report, end="")
    return 0


def command_gui(args: argparse.Namespace) -> int:
    from .gui import launch_gui

    launch_gui(db_path=args.db, snapshots_path=args.snapshots)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="isolation-lab",
        description="Deterministic SQLite row-lock and transaction-isolation lab",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="strictly validate and normalize a script")
    validate.add_argument("--script", required=True)
    _add_json_flag(validate)
    validate.set_defaults(func=command_validate)

    upgrade = subparsers.add_parser("upgrade", help="write a normalized v3 script with source metadata")
    upgrade.add_argument("--script", required=True)
    upgrade.add_argument("--output")
    _add_json_flag(upgrade)
    upgrade.set_defaults(func=command_upgrade)

    run = subparsers.add_parser("run", help="schedule a script against a SQLite file")
    run.add_argument("--script", required=True)
    run.add_argument("--db", required=True)
    run.add_argument("--snapshots")
    run.add_argument("--snapshot-id")
    run.add_argument("--global-concurrency", type=int, default=2)
    run.add_argument("--clock-quantum", type=int, default=1)
    _add_json_flag(run)
    run.set_defaults(func=command_run)

    ledger = subparsers.add_parser("ledger", help="read the cross-version event ledger")
    ledger.add_argument("--db", required=True)
    ledger.add_argument("--run-id")
    _add_json_flag(ledger)
    ledger.set_defaults(func=command_ledger)

    snapshot_list = subparsers.add_parser("snapshot-list", help="list snapshots in a snapshot SQLite file")
    snapshot_list.add_argument("--snapshots", required=True)
    _add_json_flag(snapshot_list)
    snapshot_list.set_defaults(func=command_snapshot_list)

    snapshot_replay = subparsers.add_parser("snapshot-replay", help="replay and compare a snapshot")
    snapshot_replay.add_argument("--snapshots", required=True)
    snapshot_replay.add_argument("--snapshot-id", required=True)
    _add_json_flag(snapshot_replay)
    snapshot_replay.set_defaults(func=command_snapshot_replay)

    snapshot_restore = subparsers.add_parser("snapshot-restore", help="replace target state from a snapshot")
    snapshot_restore.add_argument("--snapshots", required=True)
    snapshot_restore.add_argument("--snapshot-id", required=True)
    snapshot_restore.add_argument("--db", required=True)
    _add_json_flag(snapshot_restore)
    snapshot_restore.set_defaults(func=command_snapshot_restore)

    equivalence = subparsers.add_parser("equivalence", help="compare final outcomes of two scripts")
    equivalence.add_argument("--left", required=True)
    equivalence.add_argument("--right", required=True)
    equivalence.add_argument("--global-concurrency", type=int, default=2)
    equivalence.add_argument("--clock-quantum", type=int, default=1)
    _add_json_flag(equivalence)
    equivalence.set_defaults(func=command_equivalence)

    demo = subparsers.add_parser("demo", help="run the four deterministic demonstration scenarios")
    demo.add_argument("--workdir", default=str(Path("runtime") / "demo"))
    _add_json_flag(demo)
    demo.set_defaults(func=command_demo)

    gui = subparsers.add_parser("gui", help="open the Tkinter workbench")
    gui.add_argument("--db", default=str(Path("runtime") / "lab.sqlite"))
    gui.add_argument("--snapshots", default=str(Path("runtime") / "snapshots.sqlite"))
    gui.set_defaults(func=command_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except BusinessError as exc:
        if getattr(args, "json", False):
            print(_json_text({"error": exc.to_dict()}), file=sys.stderr)
        else:
            print(str(exc), file=sys.stderr)
        return 2 if exc.code in ("E_SCRIPT", "E_SNAPSHOT") else 3
    except (OSError, ValueError) as exc:
        if getattr(args, "json", False):
            print(_json_text({"error": {"code": "E_SCRIPT", "message": str(exc)}}), file=sys.stderr)
        else:
            print(f"E_SCRIPT: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
