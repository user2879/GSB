from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .canonical import pretty_json
from .config import load_requests, load_revision_document, load_workspace
from .engine import Engine
from .errors import MeshError
from .models import normalize_revision
from .replay import replay_snapshot


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _print_error(exc: MeshError) -> int:
    _print_json(exc.as_dict())
    return 2


def _normalize_from_args(args) -> tuple:
    workspace = load_workspace(args.workspace)
    document = load_revision_document(
        args.revision if getattr(args, "revision", None) else workspace.revision_path
    )
    revision = normalize_revision(
        document,
        workspace.services,
        workspace_id=workspace.workspace_id,
        defaults=workspace.defaults,
    )
    return workspace, revision


def command_normalize(args) -> int:
    _, revision = _normalize_from_args(args)
    payload = {
        "ok": True,
        "revision_version": revision.revision_version,
        "digest": revision.digest,
        "canonical_text": revision.canonical_text,
        "dependency_graph": revision.dependency_graph.as_dict(),
        "rules": [rule.as_dict() for rule in revision.rules],
    }
    if args.json:
        _print_json(payload)
    else:
        print(f"REVISION {revision.revision_version}")
        print(f"DIGEST {revision.digest}")
        print(f"CANONICAL {revision.canonical_text}")
        for rule in revision.rules:
            print(
                "RULE "
                f"{rule.rule_id} format=v{rule.format_version} "
                f"prefix={rule.prefix} target={rule.target_service} "
                f"timeout_ms={rule.timeout_ms} retries={rule.retries} "
                f"circuit_threshold={rule.circuit_threshold} "
                f"cooldown_ms={rule.cooldown_ms} "
                f"weights={json.dumps(dict(rule.instance_weights), sort_keys=True)} "
                f"depends_on={json.dumps([item.label() for item in rule.depends_on])}"
            )
        for source, target, kind in revision.dependency_graph.edges:
            print(f"EDGE {source} -[{kind}]-> {target}")
    return 0


def command_validate(args) -> int:
    workspace = load_workspace(args.workspace)
    with Engine(workspace) as engine:
        prepared = engine.prepare_revision(
            load_revision_document(args.revision) if args.revision else None
        )
    payload = {
        "ok": True,
        "code": "OK",
        "revision_version": prepared.revision.revision_version,
        "digest": prepared.revision.digest,
        "dependency_graph_version": prepared.revision.dependency_graph.version,
        "change": prepared.change.as_dict(),
    }
    _print_json(payload)
    return 0


def command_run(args) -> int:
    workspace = load_workspace(args.workspace)
    document = (
        load_revision_document(args.revision) if args.revision else None
    )
    revision_for_id = document or load_revision_document(workspace.revision_path)
    revision = normalize_revision(
        revision_for_id,
        workspace.services,
        workspace_id=workspace.workspace_id,
        defaults=workspace.defaults,
    )
    round_id = args.round_id or (
        f"round-r{revision.revision_version}-{time.strftime('%Y%m%d%H%M%S')}"
    )
    requests = load_requests(args.requests) if args.requests else None
    with Engine(workspace) as engine:
        summary = engine.run_round(
            round_id=round_id,
            snapshot_id=args.snapshot_id,
            revision_document=document,
            requests=requests,
            crash_after_completed=args.crash_after,
        )
    if args.json:
        _print_json({"ok": True, "summary": summary.as_dict()})
    else:
        print(f"ROUND {summary.round_id} revision={summary.revision_version}")
        print(
            "OUTCOMES "
            f"success={summary.success} failure={summary.failure} "
            f"timeout={summary.timeout} circuit_open={summary.circuit_open} "
            f"unknown={summary.unknown}"
        )
        print(
            "CONCURRENCY "
            f"max_global={summary.max_global_active} "
            f"max_service={json.dumps(summary.max_service_active, sort_keys=True)}"
        )
        if summary.change is not None:
            print(
                "CHANGE "
                + json.dumps(summary.change.as_dict(), sort_keys=True, ensure_ascii=False)
            )
        print(f"SNAPSHOT {summary.snapshot_path}")
        for result in summary.results:
            print(
                "REQUEST "
                f"{result.request_index:03d} {result.request_id} "
                f"rule={result.rule_id or '-'} outcome={result.outcome} "
                f"reason={result.reason} attempts={result.attempt_count} "
                f"instances={','.join(result.instance_sequence) or '-'} "
                f"budget_ms={result.budget_consumed_ms}"
            )
    return 0


def command_recover(args) -> int:
    workspace = load_workspace(args.workspace)
    with Engine(workspace) as engine:
        report = engine.recover_snapshot(args.snapshot)
    if args.json:
        _print_json({"ok": True, "recovery": report.as_dict()})
    else:
        print(f"RECOVERY {report.snapshot_id} {report.consistency}")
        print(
            f"REVISION snapshot={report.revision_version} "
            f"current={report.current_revision_version}"
        )
        print(
            "REPLACED "
            f"revision={report.revision_replaced} "
            f"dependency_graph={report.dependency_graph_replaced} "
            f"ledger={report.ledger_replaced}"
        )
        print(f"UNKNOWN {','.join(report.unknown_request_ids) or '-'}")
        print(f"RERUN {','.join(report.rerun_required) or '-'}")
        for basis in report.basis:
            print(f"BASIS {basis}")
    return 0


def command_replay(args) -> int:
    workspace = load_workspace(args.workspace)
    current_revision = None
    if args.revision:
        document = load_revision_document(args.revision)
        current_revision = normalize_revision(
            document,
            workspace.services,
            workspace_id=workspace.workspace_id,
            defaults=workspace.defaults,
        )
    with Engine(workspace) as engine:
        report = replay_snapshot(
            engine, args.snapshot, current_revision=current_revision
        )
    payload = {"ok": True, "replay": report.as_dict()}
    if args.json:
        _print_json(payload)
    else:
        print(f"REPLAY {report.snapshot_id} {report.consistency}")
        print(
            f"COMPARED requests={report.requests_compared} "
            f"ledger_requests={report.ledger_requests_compared} "
            f"ledger_attempts={report.ledger_attempts_compared}"
        )
        print(
            "EXCLUDED_FIELDS "
            + ",".join(report.clock_fields_excluded)
        )
    return 0


def command_snapshots(args) -> int:
    workspace = load_workspace(args.workspace)
    rows = Engine(workspace).snapshots.list()
    if args.json:
        _print_json({"ok": True, "snapshots": list(rows)})
    else:
        for row in rows:
            print(
                "SNAPSHOT "
                f"{row['snapshot_id']} status={row['status']} "
                f"revision={row['revision_version']} "
                f"completed={row['completed']} in_progress={row['in_progress']} "
                f"file={row['file']}"
            )
    return 0


def command_view(args) -> int:
    workspace = load_workspace(args.workspace)
    with Engine(workspace) as engine:
        view = engine.ledger.current_view()
        rounds = engine.ledger.list_rounds()
    if view is None:
        payload = {"ok": True, "current": None, "rounds": list(rounds)}
    else:
        payload = {"ok": True, "current": view, "rounds": list(rounds)}
    if args.json:
        _print_json(payload)
    else:
        if view is None:
            print("CURRENT -")
        else:
            round_row = view["round"]
            print(
                f"CURRENT {round_row['round_id']} "
                f"revision={round_row['revision_version']} stale={round_row['stale']}"
            )
            for row in view["requests"]:
                print(
                    f"REQUEST {row['request_id']} outcome={row['outcome']} "
                    f"attempts={row['attempt_count']} unknown={row['unknown']}"
                )
        for row in rounds:
            print(
                f"HISTORY {row['round_id']} revision={row['revision_version']} "
                f"stale={row['stale']} status={row['status']}"
            )
    return 0


def command_ui(args) -> int:
    from .ui import run_ui

    return run_ui(Path(args.workspace))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meshdiag",
        description="Local Windows service-mesh routing diagnostic workbench",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize = subparsers.add_parser("normalize", help="归一规则并输出摘要")
    normalize.add_argument("--workspace", required=True)
    normalize.add_argument("--revision")
    normalize.add_argument("--json", action="store_true")
    normalize.set_defaults(func=command_normalize)

    validate = subparsers.add_parser("validate", help="校验引用与依赖图")
    validate.add_argument("--workspace", required=True)
    validate.add_argument("--revision")
    validate.set_defaults(func=command_validate)

    run = subparsers.add_parser("run", help="执行一轮治理")
    run.add_argument("--workspace", required=True)
    run.add_argument("--revision")
    run.add_argument("--requests")
    run.add_argument("--round-id")
    run.add_argument("--snapshot-id")
    run.add_argument("--crash-after", type=int, default=None, help="仅用于故障恢复演示")
    run.add_argument("--json", action="store_true")
    run.set_defaults(func=command_run)

    recover = subparsers.add_parser("recover", help="从原子快照恢复")
    recover.add_argument("--workspace", required=True)
    recover.add_argument("--snapshot", required=True)
    recover.add_argument("--json", action="store_true")
    recover.set_defaults(func=command_recover)

    replay = subparsers.add_parser("replay", help="重放并逐项校验账本")
    replay.add_argument("--workspace", required=True)
    replay.add_argument("--snapshot", required=True)
    replay.add_argument("--revision")
    replay.add_argument("--json", action="store_true")
    replay.set_defaults(func=command_replay)

    snapshots = subparsers.add_parser("snapshots", help="列出快照")
    snapshots.add_argument("--workspace", required=True)
    snapshots.add_argument("--json", action="store_true")
    snapshots.set_defaults(func=command_snapshots)

    view = subparsers.add_parser("view", help="查看当前账本视图")
    view.add_argument("--workspace", required=True)
    view.add_argument("--json", action="store_true")
    view.set_defaults(func=command_view)

    ui = subparsers.add_parser("ui", help="启动 Tkinter 诊断工作台")
    ui.add_argument("--workspace", required=True)
    ui.set_defaults(func=command_ui)

    demo = subparsers.add_parser("demo", help="运行完整本地演示")
    demo.set_defaults(func=lambda args: __import__("demo").main())
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except MeshError as exc:
        return _print_error(exc)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        if getattr(args, "json", False):
            _print_json(
                {
                    "ok": False,
                    "code": "E_INTERNAL",
                    "message": str(exc),
                    "errors": [],
                }
            )
        else:
            print(f"E_INTERNAL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
