from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .errors import GSBError
from .generation import corrupt_sample_file
from .logging_store import StructuredLogger
from .model import load_scenario, parse_scenario
from .session import Session, restore_snapshot_into
from .util import atomic_write_text, canonical_json


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gsb-join-lab",
        description="本地等值联接规模估算、采样归因与快照诊断实验台",
    )
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="加载场景、生成样本并完成双路径估算")
    run.add_argument("--spec", required=True, help="场景 JSON 路径")
    run.add_argument("--workdir", default=".gsb-work", help="NPZ/日志工作目录")
    run.add_argument("--report", help="完整 JSON 报告输出路径")
    run.add_argument("--log-db", help="结构化 SQLite 日志路径")
    run.add_argument("--log-jsonl", help="附加 JSONL 日志路径")
    run.add_argument("--snapshot-out", help="分析完成后保存快照")
    run.add_argument("--restore-snapshot", help="先恢复该快照，再按快照参数运行")
    run.add_argument("--evolve-buckets", type=int, help="参数演进：新分桶数")
    run.add_argument("--evolve-sample-size", type=int, help="参数演进：新抽样规模")
    run.add_argument("--evolve-sample-seed", type=int, help="参数演进：新抽样种子")
    run.add_argument(
        "--evolve-int-rule",
        choices=["left_closed_right_open", "left_open_right_closed"],
        help="参数演进：整数键边界规则",
    )
    run.add_argument(
        "--evolve-float-rule",
        choices=["left_closed_right_open", "left_open_right_closed"],
        help="参数演进：浮点键边界规则",
    )
    run.add_argument("--evolve-top-k", type=int, help="参数演进：归因 top_k")
    run.add_argument("--inject-sample-corruption", action="store_true", help="注入 NPZ 文件损坏")
    run.add_argument("--inject-missing-version", action="store_true", help="注入参数版本缺失")
    run.add_argument("--inject-buckets-zero", action="store_true", help="注入分桶数为零")
    run.add_argument("--strict-estimate", action="store_true", help="E_EST 时返回非零退出码")
    run.add_argument("--quiet", action="store_true", help="只输出最终摘要")

    restore = subparsers.add_parser("restore", help="恢复快照并逐项重算比较")
    restore.add_argument("--snapshot", required=True, help="快照 JSON 路径")
    restore.add_argument("--workdir", default=".gsb-work", help="NPZ/日志工作目录")
    restore.add_argument("--current-spec", help="当前会话场景，用于比较并整体替换")
    restore.add_argument("--report", help="恢复与重放报告输出路径")
    restore.add_argument("--log-db", help="结构化 SQLite 日志路径")
    restore.add_argument("--log-jsonl", help="附加 JSONL 日志路径")

    logs = subparsers.add_parser("logs", help="查看结构化日志")
    logs.add_argument("--log-db", required=True, help="SQLite 日志路径")
    logs.add_argument("--limit", type=int, default=30, help="最多显示条数")

    subparsers.add_parser("gui", help="启动 Tkinter 本地操作台")
    return parser


def _logger(workdir: Path, args: argparse.Namespace) -> StructuredLogger:
    database = Path(args.log_db) if getattr(args, "log_db", None) else workdir / "logs" / "events.sqlite3"
    jsonl = Path(args.log_jsonl) if getattr(args, "log_jsonl", None) else workdir / "logs" / "events.jsonl"
    return StructuredLogger(database, jsonl)


def _load_spec_with_injection(path: str, args: argparse.Namespace):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return load_scenario(path)
    if not isinstance(payload, dict):
        return parse_scenario(payload, source=path)
    changed = False
    join = payload.get("join")
    if args.inject_missing_version and isinstance(join, dict):
        join.pop("parameter_version", None)
        changed = True
    if args.inject_buckets_zero and isinstance(join, dict):
        join["buckets"] = 0
        changed = True
    if changed:
        return parse_scenario(payload, source=path)
    return parse_scenario(payload, source=path)


def _console_summary(session: Session, report: dict[str, Any], *, quiet: bool = False) -> None:
    decision = report["decision"]
    estimates = decision["estimates"]
    interval = decision["interval"]
    print(f"场景: {report['scenario']}  参数版本: {report['parameters']['parameter_version']}")
    truth = report["truth"]
    print(f"过滤后行数: left={truth['left_filtered_rows']} right={truth['right_filtered_rows']}")
    print(
        "估算: histogram={:.6f} sampling={:.6f} truth={} interval=[{}, {}]".format(
            estimates["histogram"],
            estimates["sampling"],
            estimates["truth"],
            interval["lower"],
            interval["upper"],
        )
    )
    print(
        "判定: {}  选择路径: {}  区间内: histogram={} sampling={}".format(
            decision["outcome"],
            decision["selected_path"],
            decision["histogram"]["within_interval"],
            decision["sampling"]["within_interval"],
        )
    )
    attribution = report["attribution"]["top_k"]
    if attribution:
        first = attribution[0]
        print(
            "首要差异桶: {}  source={}  diff={:.6f}  basis={}".format(
                first["pair_id"],
                first["source"],
                first["estimate_difference"],
                first["basis"],
            )
        )
    if not quiet:
        print(f"分析摘要: {report['analysis_digest']}")
        print(f"样本目录: {session.workdir / 'samples'}")


def _write_report(path: str | None, payload: dict[str, Any]) -> None:
    if path:
        atomic_write_text(path, canonical_json(payload, pretty=True))


def _run(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    logger = _logger(workdir, args)
    try:
        session: Session
        restored_replay: dict[str, Any] | None = None
        transition: dict[str, Any] | None = None
        if args.restore_snapshot:
            current = None
            if args.spec:
                current = Session.from_spec_file(args.spec, workdir, logger=logger)
            session, restore_transition = restore_snapshot_into(
                args.restore_snapshot,
                workdir,
                current=current,
                logger=logger,
            )
            restored_replay = {
                "snapshot": args.restore_snapshot,
                "transition": restore_transition,
                "analysis_digest": session.current.analysis_digest if session.current else None,
            }
        else:
            spec = _load_spec_with_injection(args.spec, args)
            session = Session(spec, workdir, logger=logger)
        logger.log(
            "INFO",
            "OK",
            "scenario_loaded",
            {
                "scenario": session.spec.name,
                "spec_digest": session.spec.digest,
                "parameter_version": session.parameter_version,
                "inject_sample_corruption": bool(args.inject_sample_corruption),
                "inject_missing_version": bool(args.inject_missing_version),
                "inject_buckets_zero": bool(args.inject_buckets_zero),
            },
        )
        if args.inject_sample_corruption:
            target = session.samples[sorted(session.samples)[0]].path
            details = corrupt_sample_file(target)
            logger.log("ERROR", "E_SAMPLE", "sample_corruption_injected", details)
            session.verify_sample_files()
            raise GSBError("E_SAMPLE", "样本损坏注入未被检出", details)
        if args.restore_snapshot and session.current is None:
            session.run()
        if not args.restore_snapshot:
            session.run()
        update_kwargs: dict[str, Any] = {}
        if args.evolve_buckets is not None:
            update_kwargs["buckets"] = args.evolve_buckets
        if args.evolve_sample_size is not None:
            update_kwargs["sample_size"] = args.evolve_sample_size
        if args.evolve_sample_seed is not None:
            update_kwargs["sample_seed"] = args.evolve_sample_seed
        if args.evolve_top_k is not None:
            update_kwargs["top_k"] = args.evolve_top_k
        boundary_updates: dict[str, str] = {}
        if args.evolve_int_rule:
            boundary_updates["int"] = args.evolve_int_rule
        if args.evolve_float_rule:
            boundary_updates["float"] = args.evolve_float_rule
        if boundary_updates:
            update_kwargs["boundary_rules"] = boundary_updates
        if update_kwargs:
            evolution = session.evolve(**update_kwargs)
            transition = evolution.transition
        if session.current is None:
            session.run()
        report = session.current_report()
        if restored_replay is not None:
            report = dict(report)
            report["restored_replay"] = restored_replay
        if transition is not None:
            report = dict(report)
            report["evolution"] = transition
        if args.snapshot_out:
            snapshot_meta = session.save_snapshot(args.snapshot_out)
            report = dict(report)
            report["snapshot"] = snapshot_meta
        report_path = args.report
        _write_report(report_path, report)
        _console_summary(session, report, quiet=args.quiet)
        if report_path:
            print(f"报告: {report_path}")
        if args.strict_estimate and report["decision"]["outcome"] == "E_EST":
            return 7
        return 0
    except GSBError as error:
        logger.error(error, "run_failed")
        _write_report(args.report, error.to_dict())
        print(json.dumps(error.to_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return error.exit_code
    except Exception as error:
        wrapped = GSBError("E_SPEC", "未处理异常", {"reason": str(error), "type": type(error).__name__})
        logger.error(wrapped, "run_failed")
        _write_report(args.report, wrapped.to_dict())
        print(json.dumps(wrapped.to_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return wrapped.exit_code


def _restore(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    logger = _logger(workdir, args)
    try:
        current = None
        if args.current_spec:
            current = Session.from_spec_file(args.current_spec, workdir, logger=logger)
        session, transition = restore_snapshot_into(
            args.snapshot,
            workdir,
            current=current,
            logger=logger,
        )
        report = dict(session.current_report())
        report["restore"] = transition
        _write_report(args.report, report)
        print(f"恢复完成: version={session.parameter_version} status={report['decision']['outcome']}")
        compared = transition.get("compared", transition.get("replay", {}).get("compared", []))
        print(f"重放比较项: {', '.join(compared)}")
        if args.report:
            print(f"报告: {args.report}")
        return 0
    except GSBError as error:
        logger.error(error, "restore_failed")
        _write_report(args.report, error.to_dict())
        print(json.dumps(error.to_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return error.exit_code
    except Exception as error:
        wrapped = GSBError("E_REPLAY", "恢复过程未处理异常", {"reason": str(error), "type": type(error).__name__})
        logger.error(wrapped, "restore_failed")
        _write_report(args.report, wrapped.to_dict())
        print(json.dumps(wrapped.to_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return wrapped.exit_code


def _show_logs(args: argparse.Namespace) -> int:
    logger = StructuredLogger(args.log_db, None)
    events = logger.read_events(args.limit)
    print(canonical_json(events, pretty=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "gui":
        from .gui import main as gui_main

        return gui_main()
    if args.command == "run":
        return _run(args)
    if args.command == "restore":
        return _restore(args)
    if args.command == "logs":
        return _show_logs(args)
    parser.error(f"unknown command: {args.command}")
    return 2
