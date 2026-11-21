"""Command-line entry points for the diagnostic workbench."""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import threading
from pathlib import Path
from typing import Any

from .config import WorkbenchConfig
from .errors import ConfigInvalid
from .loader import read_json_records
from .pipeline import Workbench
from .util import atomic_write_json, canonical_json, content_fingerprint


class SignalState:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.signum: int | None = None

    def handler(self, signum: int, _frame: Any) -> None:
        self.signum = signum
        self.event.set()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="news-diag",
                                     description="中文新闻摘要与重写诊断工作台")
    subparsers = parser.add_subparsers(dest="subcommand")

    run_parser = subparsers.add_parser("run", help="增量运行或从空白重建")
    _add_shared_paths(run_parser)
    run_parser.add_argument("--snapshot-dir", default=".workbench/snapshots")
    run_parser.add_argument("--injections", default=None)
    run_parser.add_argument("--report", default="artifacts/report.json")
    run_parser.add_argument("--output", default="artifacts/output.json")
    run_parser.add_argument("--rebuild", action="store_true", help="忽略快照缓存并从空白重建")
    run_parser.add_argument("--clean-snapshots", action="store_true", help="运行前删除快照目录")
    run_parser.add_argument("--apply-through", type=int, default=None,
                            help="仅应用命令流前 N 条命令")
    run_parser.add_argument("--expect-version", type=int, default=None,
                            help="要求当前快照版本号等于指定值，否则 E_SNAPSHOT_STALE")
    run_parser.add_argument("--stop-after-docs", type=int, default=None,
                            help="模拟可处理退出信号：完成 N 个文档提交后以 0 退出")
    run_parser.add_argument("--doc-delay-ms", type=float, default=0.0,
                            help="每个文档任务开始前延迟，用于真实信号测试")
    run_parser.add_argument("--json-stdout", action="store_true")

    verify_parser = subparsers.add_parser("verify", help="逐步比较增量与空白全量结果")
    _add_shared_paths(verify_parser)
    verify_parser.add_argument("--work-dir", default="artifacts/verify_work")
    verify_parser.add_argument("--report", default="artifacts/equivalence_report.json")
    verify_parser.add_argument("--output", default="artifacts/verified_output.json")
    return parser


def _add_shared_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--docs", default="samples/news.jsonl")
    parser.add_argument("--config", default="samples/config.yaml")
    parser.add_argument("--commands", default="samples/commands.jsonl")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.subcommand is None:
        args = parser.parse_args(["run", *(argv or [])])
    if args.subcommand == "verify":
        return _verify(args)
    return _run(args)


def _run(args: argparse.Namespace) -> int:
    signal_state = SignalState()
    previous_handlers: dict[int, Any] = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, signal_state.handler)
    try:
        try:
            config = WorkbenchConfig.load(args.config)
        except ConfigInvalid as exc:
            _write_fatal(args, exc)
            return 2
        try:
            workbench = Workbench(
                docs_path=args.docs,
                config=config,
                commands_path=args.commands,
                snapshot_dir=args.snapshot_dir,
                injections_path=args.injections,
            )
            run = workbench.run(
                rebuild=args.rebuild,
                apply_through=args.apply_through,
                expect_version=args.expect_version,
                stop_event=signal_state.event,
                stop_after_docs=args.stop_after_docs,
                document_delay_seconds=max(0.0, args.doc_delay_ms) / 1000.0,
                clean_snapshots=args.clean_snapshots,
            )
        except Exception as exc:
            error = ConfigInvalid(f"工作台运行失败: {exc}", stage="runtime",
                                  details={"exception_type": type(exc).__name__})
            _write_fatal(args, error)
            return 2

        report = run.report
        report["exit_code"] = 0
        report["handled_signal"] = signal_state.signum
        report["simulated_signal"] = args.stop_after_docs is not None
        if args.stop_after_docs is not None:
            report["stop_reason"] = "simulated_graceful_stop"
        elif signal_state.signum is not None:
            report["stop_reason"] = "handled_exit_signal"
        else:
            report["stop_reason"] = "completed"
        report["inputs"] = {
            "docs": str(args.docs),
            "config": str(args.config),
            "commands": str(args.commands),
            "injections": str(args.injections) if args.injections else None,
        }
        atomic_write_json(Path(args.output), run.canonical)
        atomic_write_json(Path(args.report), report)
        if args.json_stdout:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(_summary_line(report, args.report, args.output))
        return 0
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _summary_line(report: dict[str, Any], report_path: str, output_path: str) -> str:
    stats = report.get("summary", {})
    return (
        f"mode={report.get('mode')} snapshot=v{report.get('snapshot_version')} "
        f"docs={stats.get('document_count', 0)} failures={report.get('failure_count', 0)} "
        f"reused={report.get('reuse_counts', {}).get('reused', 0)} "
        f"interrupted={str(report.get('interrupted', False)).lower()} "
        f"report={report_path} output={output_path}"
    )


def _write_fatal(args: argparse.Namespace, error: ConfigInvalid) -> None:
    failure = error.to_failure()
    report = {
        "schema_version": 1,
        "mode": "fatal",
        "exit_code": 2,
        "summary": {
            "document_count": 0,
            "failed_document_count": 0,
            "summary_sentence_count": 0,
            "replacement_count": 0,
            "truncation_count": 0,
            "summary_character_count": 0,
        },
        "failures": [failure],
        "failure_count": 1,
        "snapshot_version": 0,
        "stage_timings_ms": {},
        "invalidation_reuse": [],
        "reuse_counts": {"checked": 0, "reused": 0, "recomputed": 0, "failed": 0},
        "injections": [],
        "canonical_digest": content_fingerprint({"fatal": failure}),
    }
    try:
        atomic_write_json(Path(args.report), report)
        atomic_write_json(Path(args.output), {
            "schema_version": 1,
            "documents": [],
            "summary": report["summary"],
            "failures": [failure],
            "stage_entries": [],
            "invalidation_reuse": [],
            "cache_keys": {},
        })
    except OSError:
        pass
    print(canonical_json({"exit_code": 2, "failure": failure}))


def _verify(args: argparse.Namespace) -> int:
    config = WorkbenchConfig.load(args.config)
    command_records = read_json_records(args.commands)
    total_steps = len(command_records)
    work_root = Path(args.work_dir)
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    incremental_snapshot = work_root / "incremental_snapshot"
    full_snapshot = work_root / "full_snapshot"
    steps: list[dict[str, Any]] = []
    all_equal = True
    final_incremental = None

    for step in range(total_steps + 1):
        incremental = Workbench(args.docs, config, args.commands, incremental_snapshot).run(
            apply_through=step,
        )
        full = Workbench(args.docs, config, args.commands, full_snapshot).run(
            rebuild=True,
            apply_through=step,
        )
        equal = canonical_json(incremental.canonical) == canonical_json(full.canonical)
        if not equal:
            all_equal = False
        final_incremental = incremental
        steps.append({
            "step": step,
            "commands_applied": step,
            "command": command_records[step - 1].value if step > 0 else None,
            "equal": equal,
            "incremental_digest": incremental.report["canonical_digest"],
            "full_digest": full.report["canonical_digest"],
            "incremental_reuse_counts": incremental.report["reuse_counts"],
            "full_reuse_counts": full.report["reuse_counts"],
            "incremental_snapshot_version": incremental.report["snapshot_version"],
            "full_snapshot_version": full.report["snapshot_version"],
            "failure_count": incremental.report["failure_count"],
        })
        if not equal:
            break

    report = {
        "schema_version": 1,
        "all_equal": all_equal,
        "command_count": total_steps,
        "steps_verified": len(steps),
        "comparison_projection": "complete output.json canonical projection "
                                 "(documents, per-sentence scores, replacements, failures, cache keys, "
                                 "stage entries/invalidation reuse key order)",
        "steps": steps,
    }
    atomic_write_json(Path(args.report), report)
    if final_incremental is not None:
        atomic_write_json(Path(args.output), final_incremental.canonical)
    print(json.dumps({
        "all_equal": all_equal,
        "steps_verified": len(steps),
        "report": str(args.report),
    }, ensure_ascii=False))
    return 0 if all_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
