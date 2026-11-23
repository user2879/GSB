from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .engine import RunOptions, Workbench
from .errors import DiagnosticError, SpecError
from .events import EventLogger
from .models import load_spec


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spec", required=True, help="本地上传声明 JSON")
    parser.add_argument("--workdir", default=".ossdiag-work", help="运行目录（默认 .ossdiag-work）")
    parser.add_argument("--ledger", help="覆盖 SQLite 账本路径")
    parser.add_argument("--data-dir", help="覆盖分片数据目录")
    parser.add_argument("--log", help="覆盖结构化 JSONL 日志路径")
    parser.add_argument("--backup-dir", help="覆盖只读备份目录")


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    return {
        "workdir": workdir,
        "ledger_path": Path(args.ledger) if args.ledger else workdir / "ledger.sqlite",
        "data_dir": Path(args.data_dir) if args.data_dir else workdir / "data",
        "log_path": Path(args.log) if args.log else workdir / "events.jsonl",
        "backup_dir": Path(args.backup_dir) if args.backup_dir else workdir / "ledger_backups",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ossdiag",
        description="Windows 本地对象存储分片上传诊断工作台",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="执行诊断会话")
    _add_common(run_parser)
    run_parser.add_argument("--inject-lease-overflow", action="store_true", help="注入租约越界申请（E_LEASE）")
    run_parser.add_argument("--inject-digest-mismatch", action="store_true", help="注入分片摘要不符（E_DIGEST）")
    run_parser.add_argument("--inject-snapshot-conflict", action="store_true", help="注入账本版本冲突（E_SNAPSHOT）")
    run_parser.add_argument("--target-object", help="摘要异常的目标对象键")
    run_parser.add_argument("--target-shard", type=int, help="摘要异常的目标分片序号")
    run_parser.add_argument(
        "--crash-after-batches",
        type=int,
        help="仅用于恢复演示：写完指定批次数后 os._exit(137)，不会生成退出报告",
    )
    run_parser.add_argument("--json-out", help="运行摘要 JSON 路径（默认 workdir/run_summary.json）")
    run_parser.add_argument("--json", action="store_true", help="向 stdout 输出完整 JSON")
    run_parser.add_argument("--quiet", action="store_true")

    gui_parser = subparsers.add_parser("gui", help="打开 Tkinter 诊断工作台")
    _add_common(gui_parser)
    gui_parser.add_argument("--auto-run", action="store_true", help="窗口打开后自动运行一次")

    inspect_parser = subparsers.add_parser("inspect", help="打开账本并输出当前诊断快照 JSON（可能执行恢复/替换）")
    _add_common(inspect_parser)
    inspect_parser.add_argument("--json-out", help="快照 JSON 输出路径")

    return parser


def _workbench(args: argparse.Namespace, paths: dict[str, Path] | None = None) -> Workbench:
    resolved = paths or _paths(args)
    session = load_spec(args.spec)
    return Workbench(
        session,
        ledger_path=resolved["ledger_path"],
        data_dir=resolved["data_dir"],
        log_path=resolved["log_path"],
        backup_dir=resolved["backup_dir"],
    )


def _print_concise(summary: dict[str, Any]) -> None:
    print(f"ok={summary['ok']}  ledger_format={summary['ledger']['format_version']}  "
          f"ledger_digest={str(summary['ledger']['ledger_digest'])[:16]}…")
    print(f"throughput={summary['throughput']['used_bytes']}/{summary['throughput']['limit_bytes']} bytes  "
          f"peak_leases={summary['leases']['peak_global']}/{summary['leases']['global_limit']}")
    for item in summary["objects"]:
        fixed = item["fixed_stats"]
        print(
            f"- {item['object_key']}: {item['status']} path={item['selected_path']} "
            f"boundary={item['boundary_case']} completion={fixed['completion_display']} "
            f"digest={item['digest_status']} unknown_shards={item['unknown_shards']}"
        )
    for item in summary["ranking"]:
        print(
            f"  rank#{item['rank']} {item['object_key']} {item['completion_display']} "
            f"error_affected={item['ordering_affected_by_error_bound']}"
        )
    if summary["path_verification"]:
        print("path_consistency=" + ",".join(
            f"{item['object_key']}:{'PASS' if item['consistent'] else 'FAIL'}"
            for item in summary["path_verification"]
        ))
    if summary["errors"]:
        print("errors=" + ",".join(
            f"{item['code']}@{item.get('object_key')}#{item.get('shard_index')}" for item in summary["errors"]
        ))
    if summary["reconcile"].get("replaced"):
        print(
            f"ledger_replaced snapshot={summary['reconcile']['snapshot_id']} "
            f"backup={summary['reconcile']['backup_path']}"
        )
    print("note=" + summary["note"])


def _exit_code(summary: dict[str, Any]) -> int:
    codes = {item.get("code") for item in summary.get("errors", [])}
    if codes.intersection({"E_LEASE", "E_DIGEST", "E_SNAPSHOT"}):
        return 3
    if not summary.get("ok"):
        return 5
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths: dict[str, Path] | None = None
    try:
        paths = _paths(args)
        session = load_spec(args.spec)
    except SpecError as exc:
        if paths is not None:
            logger = EventLogger(paths["log_path"])
            logger.business_error(exc)
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"无法准备运行目录: {exc}", file=sys.stderr)
        return 1

    try:
        workbench = Workbench(
            session,
            ledger_path=paths["ledger_path"],
            data_dir=paths["data_dir"],
            log_path=paths["log_path"],
            backup_dir=paths["backup_dir"],
        )
    except Exception as exc:
        print(f"无法打开账本: {exc}", file=sys.stderr)
        return 1

    if args.command == "gui":
        from .gui import launch_gui

        try:
            launch_gui(workbench, auto_run=args.auto_run)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            workbench.close()
            return 1
        finally:
            workbench.close()
        return 0

    if args.command == "inspect":
        try:
            reconcile = workbench.initialize()
            dashboard = workbench.build_dashboard()
            dashboard["reconcile"] = reconcile.as_dict()
            target = Path(args.json_out) if args.json_out else paths["workdir"] / "inspect.json"
            target.write_text(json.dumps(dashboard, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(dashboard, ensure_ascii=False, indent=2))
            return 0
        finally:
            workbench.close()

    options = RunOptions(
        inject_lease_overflow=args.inject_lease_overflow,
        inject_digest_mismatch=args.inject_digest_mismatch,
        inject_snapshot_conflict=args.inject_snapshot_conflict,
        target_object_key=args.target_object,
        target_shard_index=args.target_shard,
        crash_after_batches=args.crash_after_batches,
    )
    try:
        summary = workbench.run(options)
        payload = summary.as_dict()
        output = Path(args.json_out) if args.json_out else paths["workdir"] / "run_summary.json"
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        elif not args.quiet:
            _print_concise(payload)
        return _exit_code(payload)
    except DiagnosticError as exc:
        EventLogger(paths["log_path"]).business_error(exc)
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2), file=sys.stderr)
        return 3 if exc.code in {"E_LEASE", "E_DIGEST", "E_SNAPSHOT"} else 1
    except KeyboardInterrupt:
        print("已中断；本次进程不生成强制终止退出报告，请重新运行以依据已提交账本恢复。", file=sys.stderr)
        return 130
    finally:
        workbench.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
