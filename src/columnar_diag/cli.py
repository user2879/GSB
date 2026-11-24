from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .demo import run_demo
from .errors import AppError
from .models import BudgetConfig
from .sample import build_sample_spec, write_sample_files
from .snapshot import restore_snapshot, save_snapshot
from .spec import load_spec
from .verifier import authoritative_replay
from .workspace import Workspace


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _budget_from_args(args: argparse.Namespace, fallback: BudgetConfig | None = None) -> BudgetConfig:
    base = fallback or BudgetConfig()
    return BudgetConfig(
        global_parallelism=(
            args.parallelism
            if getattr(args, "parallelism", None) is not None
            else base.global_parallelism
        ),
        column_memory_budget_bytes=(
            args.memory_budget
            if getattr(args, "memory_budget", None) is not None
            else base.column_memory_budget_bytes
        ),
        segment_concurrency=(
            args.segment_quota
            if getattr(args, "segment_quota", None) is not None
            else base.segment_concurrency
        ),
    )


def _add_budget_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--parallelism", type=int, help="全局并行度上限")
    parser.add_argument("--memory-budget", type=int, help="每列解码内存预算（字节）")
    parser.add_argument("--segment-quota", type=int, help="每段并发任务配额")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="columnar-diag",
        description="Windows 本地列式存储压缩与扫描诊断应用",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample", help="生成本地确定性样本 JSON")
    sample.add_argument("--output", default="samples")

    init = subparsers.add_parser("init", help="创建段文件与账本工作区")
    init.add_argument("--workspace", required=True)
    init.add_argument("--spec", help="规范 JSON；省略则使用内置样本")
    init.add_argument("--overwrite", action="store_true")
    _add_budget_args(init)

    scan = subparsers.add_parser("scan", help="从账本游标继续块级扫描")
    scan.add_argument("--workspace", required=True)
    scan.add_argument("--batch-size", type=int, default=5)
    scan.add_argument("--max-tasks", type=int)
    scan.add_argument("--delay-ms", type=float, default=0.0)
    scan.add_argument("--no-partial-tail", action="store_true")
    _add_budget_args(scan)

    update = subparsers.add_parser("update", help="应用增量变更集")
    update.add_argument("--workspace", required=True)
    update.add_argument("--changeset", required=True)
    update.add_argument("--delay-ms", type=float, default=0.0)

    budget = subparsers.add_parser("budget", help="修改预算参数并整体推进版本")
    budget.add_argument("--workspace", required=True)
    budget.add_argument("--parallelism", type=int, required=True)
    budget.add_argument("--memory-budget", type=int, required=True)
    budget.add_argument("--segment-quota", type=int, required=True)
    budget.add_argument("--batch-size", type=int, default=5)
    budget.add_argument("--delay-ms", type=float, default=0.0)

    encoding = subparsers.add_parser("encoding-policy", help="修改列编码选择策略")
    encoding.add_argument("--workspace", required=True)
    encoding.add_argument("--column", required=True)
    encoding.add_argument("--encoding", choices=("raw", "dictionary", "rle"), required=True)
    encoding.add_argument("--compressed", choices=("yes", "no"), required=True)
    encoding.add_argument("--delay-ms", type=float, default=0.0)

    verify = subparsers.add_parser("verify", help="从段文件权威重放并校验账本")
    verify.add_argument("--workspace", required=True)

    snapshot = subparsers.add_parser("snapshot", help="保存会话快照")
    snapshot.add_argument("--workspace", required=True)
    snapshot.add_argument("--output", required=True)

    restore = subparsers.add_parser("restore", help="整体替换恢复会话快照")
    restore.add_argument("--workspace", required=True)
    restore.add_argument("--snapshot", required=True)

    replay = subparsers.add_parser("replay", help="恢复后逐块重放校验")
    replay.add_argument("--workspace", required=True)

    demo = subparsers.add_parser("demo", help="运行完整本地验收")
    demo.add_argument("--output", default="artifacts")

    subparsers.add_parser("gui", help="启动 Tkinter 桌面界面")
    return parser


def _load_changeset(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("changes"), list):
        return payload["changes"]
    raise AppError("E_SPEC", "变更集必须为数组或包含 changes 数组", field="changeset")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "sample":
            _print(write_sample_files(args.output))
            return 0
        if args.command == "demo":
            report = run_demo(args.output)
            _print(
                {
                    "all_checks_passed": report["all_checks_passed"],
                    "checks": report["checks"],
                    "report": str(Path(args.output) / "acceptance_report.json"),
                    "markdown": str(Path(args.output) / "ACCEPTANCE.md"),
                }
            )
            return 0 if report["all_checks_passed"] else 1
        if args.command == "gui":
            try:
                from .gui import launch_gui

                launch_gui()
            except ImportError as error:
                raise AppError(
                    "E_TK",
                    "Tkinter 不可用；请使用官方 Windows CPython 并启用 tcl/tk",
                    field="tkinter",
                    details={"error": str(error)},
                ) from error
            except Exception as error:
                if type(error).__name__ == "TclError":
                    raise AppError(
                        "E_TK",
                        "无法启动 Tk 窗口；请检查本地图形会话",
                        field="tkinter",
                        details={"error": str(error)},
                    ) from error
                raise
            return 0
        if args.command == "init":
            spec = load_spec(args.spec) if args.spec else build_sample_spec()
            workspace = Workspace.create(
                args.workspace,
                spec,
                budget=_budget_from_args(args, BudgetConfig(**{
                    "global_parallelism": 4,
                    "column_memory_budget_bytes": 128,
                    "segment_concurrency": 3,
                })),
                overwrite=args.overwrite,
            )
            try:
                _print(
                    {
                        "workspace": str(workspace.root),
                        "segment_digest": workspace.image.file_digest,
                        "plan_digest": workspace.bundle.digest,
                        "task_count": len(workspace.bundle.tasks),
                        "budget": workspace.budget.to_dict(),
                        "versions": workspace.versions,
                    }
                )
            finally:
                workspace.close()
            return 0

        workspace = Workspace(args.workspace)
        try:
            if args.command == "scan":
                budget = _budget_from_args(args, workspace.budget)
                if budget != workspace.budget:
                    result = workspace.change_budget(
                        budget,
                        batch_size=args.batch_size,
                        delay_ms=args.delay_ms,
                    )
                else:
                    result = workspace.scan_pending(
                        batch_size=args.batch_size,
                        max_tasks=args.max_tasks,
                        delay_ms=args.delay_ms,
                        commit_partial_tail=not args.no_partial_tail,
                    )
                _print(result)
            elif args.command == "update":
                _print(
                    workspace.apply_changeset(
                        _load_changeset(args.changeset),
                        budget=workspace.budget,
                        delay_ms=args.delay_ms,
                    )
                )
            elif args.command == "budget":
                _print(
                    workspace.change_budget(
                        _budget_from_args(args, workspace.budget),
                        batch_size=args.batch_size,
                        delay_ms=args.delay_ms,
                    )
                )
            elif args.command == "encoding-policy":
                _print(
                    workspace.change_encoding_policy(
                        column=args.column,
                        encoding=args.encoding,
                        compressed=args.compressed == "yes",
                        delay_ms=args.delay_ms,
                    )
                )
            elif args.command == "verify":
                _print(authoritative_replay(workspace, error_code="E_DIVERGE"))
            elif args.command == "replay":
                _print(authoritative_replay(workspace, error_code="E_REPLAY"))
            elif args.command == "snapshot":
                _print(save_snapshot(workspace, args.output))
            elif args.command == "restore":
                _print(restore_snapshot(workspace, args.snapshot))
            else:
                raise AppError("E_CLI", f"未知命令: {args.command}", field="command")
        finally:
            workspace.close()
        return 0
    except AppError as error:
        print(json.dumps(error.to_dict(), ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(
            json.dumps(
                {"code": "E_IO", "message": str(error)},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
