from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .errors import AppError
from .planner import compile_plan, plan_bytes
from .service import (
    execute_run,
    preview_ledger,
    recover_ledger,
    replay_restored_snapshot,
    restore_snapshot,
    save_ledger_snapshot,
    verify_run,
)
from .validation import load_spec


def _print(value: Any, pretty: bool = True) -> None:
    if pretty:
        print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(canonical_json(value))


def _cmd_validate(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    plan = compile_plan(spec)
    _print(
        {
            "ok": True,
            "clusters": len(spec.clusters),
            "revisions": len(spec.revisions),
            "tasks": len(spec.tasks),
            "shards": len(plan["shards"]),
            "plan_digest": plan["plan_digest"],
        }
    )
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    plan = compile_plan(spec)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_bytes(plan_bytes(plan) + b"\n")
    result = {
        "ok": True,
        "out": str(args.out),
        "plan_digest": plan["plan_digest"],
        "shards": len(plan["shards"]),
        "shard_ids": [shard["shard_id"] for shard in plan["shards"]],
    }
    _print(result)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    plan = compile_plan(spec)
    result = execute_run(
        spec,
        plan,
        args.ledger,
        snapshot_path=args.snapshot,
        handshake_timeout_ms=args.handshake_timeout_ms,
        crash_after_commits=args.crash_after_commits,
    )
    if args.json:
        _print(result)
    else:
        _print(
            {
                "ok": result["counts"]["failed"] == 0 and result["counts"]["blocked"] == 0,
                "run_id": result["run_id"],
                "plan_digest": result["plan_digest"],
                "counts": result["counts"],
                "peak": result["peak"],
                "limits": result["limits"],
                "replacement": result["replacement"],
                "stale_shards_after": result["stale_shards_after"],
                "ledger": result["ledger"],
                "snapshot": result["snapshot"],
            }
        )
        print("shard_id                         outcome   wait_ms start_ms finish_ms elapsed_ms")
        for trace in result["trace"]:
            print(
                f"{trace['shard_id']:<32} {trace['outcome']:<9} "
                f"{trace['wait_ended_ms'] - trace['wait_started_ms']:>7} "
                f"{trace['started_ms'] or 0:>8} {trace['finished_ms'] or 0:>9} "
                f"{trace['elapsed_ms'] if trace['elapsed_ms'] is not None else '-':>10}"
            )
    return 0 if result["counts"]["failed"] == 0 and result["counts"]["blocked"] == 0 else 4


def _cmd_preview(args: argparse.Namespace) -> int:
    _print(preview_ledger(args.ledger))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    result = verify_run(
        args.ledger,
        use_nodes=not args.no_nodes,
        handshake_timeout_ms=args.handshake_timeout_ms,
    )
    _print(result)
    return 0


def _cmd_recover(args: argparse.Namespace) -> int:
    _print(recover_ledger(args.ledger, run_id=args.run_id))
    return 0


def _cmd_snapshot(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    _print(save_ledger_snapshot(spec, args.ledger, args.out))
    return 0


def _cmd_restore(args: argparse.Namespace) -> int:
    snapshot = restore_snapshot(args.snapshot)
    _print(
        {
            "ok": True,
            "restore": snapshot["restore"],
            "versions": snapshot["versions"],
            "run_id": snapshot["run"]["run_id"],
            "content_digest": snapshot["content_digest"],
        }
    )
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    _print(
        replay_restored_snapshot(
            args.snapshot,
            handshake_timeout_ms=args.handshake_timeout_ms,
        )
    )
    return 0


def _cmd_gui(args: argparse.Namespace) -> int:
    try:
        from .gui import launch_gui

        launch_gui(args.spec, args.ledger, args.snapshot)
    except ImportError as exc:
        raise AppError(
            "E_TK",
            f"当前 Python 无法导入 Tkinter: {exc}",
            "tkinter",
            "python",
            {"hint": "Windows 官方 Python 3.11+ 安装包默认包含 Tcl/Tk"},
        ) from None
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from .demo import run_demo

    run_demo(Path(args.out))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m configdist", description="多集群配置分发领域应用")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="强校验清单并编译计划摘要")
    validate.add_argument("--spec", required=True)
    validate.set_defaults(func=_cmd_validate)

    plan = sub.add_parser("plan", help="编译稳定分发计划")
    plan.add_argument("--spec", required=True)
    plan.add_argument("--out", default="artifacts/plan.json")
    plan.set_defaults(func=_cmd_plan)

    run = sub.add_parser("run", help="启动节点并执行三重预算调度")
    run.add_argument("--spec", required=True)
    run.add_argument("--ledger", default="artifacts/ledger.sqlite3")
    run.add_argument("--snapshot")
    run.add_argument("--handshake-timeout-ms", type=int, default=3000)
    run.add_argument("--json", action="store_true")
    run.add_argument(
        "--crash-after-commits",
        type=int,
        default=0,
        help="仅供强制终止恢复验收：提交 N 个分片后 os._exit(17)",
    )
    run.set_defaults(func=_cmd_run)

    preview = sub.add_parser("preview", help="从账本缓存显示当前结论")
    preview.add_argument("--ledger", default="artifacts/ledger.sqlite3")
    preview.set_defaults(func=_cmd_preview)

    verify = sub.add_parser("verify", help="逐分片权威重算并比对缓存")
    verify.add_argument("--ledger", default="artifacts/ledger.sqlite3")
    verify.add_argument("--no-nodes", action="store_true")
    verify.add_argument("--handshake-timeout-ms", type=int, default=3000)
    verify.set_defaults(func=_cmd_verify)

    recover = sub.add_parser("recover", help="从完整提交记录恢复，未知分片要求重跑")
    recover.add_argument("--ledger", default="artifacts/ledger.sqlite3")
    recover.add_argument("--run-id")
    recover.set_defaults(func=_cmd_recover)

    snapshot = sub.add_parser("snapshot", help="保存会话快照")
    snapshot.add_argument("--spec", required=True)
    snapshot.add_argument("--ledger", default="artifacts/ledger.sqlite3")
    snapshot.add_argument("--out", default="artifacts/session.snapshot.json")
    snapshot.set_defaults(func=_cmd_snapshot)

    restore = sub.add_parser("restore", help="识别并整体替换恢复快照")
    restore.add_argument("--snapshot", required=True)
    restore.set_defaults(func=_cmd_restore)

    replay = sub.add_parser("replay", help="恢复后重新下发已完成分片")
    replay.add_argument("--snapshot", required=True)
    replay.add_argument("--handshake-timeout-ms", type=int, default=3000)
    replay.set_defaults(func=_cmd_replay)

    gui = sub.add_parser("gui", help="启动 Tkinter 编辑器")
    gui.add_argument("--spec", default="sample/spec.json")
    gui.add_argument("--ledger", default="artifacts/gui-ledger.sqlite3")
    gui.add_argument("--snapshot", default="artifacts/gui-snapshot.json")
    gui.set_defaults(func=_cmd_gui)

    demo = sub.add_parser("demo", help="运行完整本地验收演示")
    demo.add_argument("--out", default="artifacts/demo/acceptance.txt")
    demo.set_defaults(func=_cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except AppError as exc:
        _print(exc.to_dict())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
