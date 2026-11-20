"""ShardBench 命令行入口。

子命令：
* ``run``       运行一个路由轮次并提交账本/快照；
* ``replay``    从指定已提交轮次独立重放并逐项比对；
* ``compare``   增量结果与全量重算逐项比较；
* ``inject``    结构化异常注入（corrupt_config/ring_conflict/version_mismatch）；
* ``gui``       启动 Tkinter 诊断界面；
* ``recover``   检查未提交轮次并标记需要重跑。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from .config import load_config
from .engine import Engine, compile_all, full_recompute
from .errors import ShardBenchError
from .invalidation import compare_recomputations
from .injection import (
    trigger_config_injection,
    trigger_ring_injection,
    trigger_version_injection,
)
from .logging_utils import StructuredLogger
from .replay import replay_round


def _default_paths(workspace: str) -> dict[str, str]:
    base = os.path.join(workspace, "shardbench", "samples")
    data = os.path.join(workspace, "shardbench", "data")
    return {
        "config": os.path.join(base, "config.json"),
        "units": os.path.join(base, "units.json"),
        "ledger": os.path.join(data, "ledger.sqlite3"),
        "snapshot": os.path.join(data, "snapshot.json"),
        "log": os.path.join(data, "events.jsonl"),
    }


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _resolve(args: argparse.Namespace) -> dict[str, str]:
    workspace = args.workspace or os.getcwd()
    paths = _default_paths(workspace)
    for key in ("config", "units", "ledger", "snapshot", "log"):
        value = getattr(args, key, None)
        if value:
            paths[key] = value
    return paths


def cmd_run(args: argparse.Namespace) -> int:
    paths = _resolve(args)
    with StructuredLogger(paths["log"]) as logger:
        engine = Engine(
            paths["config"],
            paths["units"],
            paths["ledger"],
            paths["snapshot"],
            logger=logger,
        )
        try:
            result = engine.run_round(
                changed_tables=args.changed_table or None,
                fault_overrides=_parse_faults(args.fault),
            )
        finally:
            engine.close()
    _print_json(
        {
            "round_id": result.round_id,
            "versions": result.versions.as_dict(),
            "invalidation": result.invalidation,
            "replacement": result.replacement,
            "scheduler_verification": result.scheduler_verification,
            "outcomes": [
                {"unit_id": trace["unit_id"], "outcome_code": trace["outcome_code"]}
                for trace in result.traces
            ],
            "cache_digest": result.cache_digest,
            "history_digest": result.history_digest,
        }
    )
    return 0


def _parse_faults(raw: list[str] | None) -> dict[str, dict[str, bool]]:
    overrides: dict[str, dict[str, bool]] = {}
    for item in raw or []:
        if "=" not in item:
            continue
        unit_id, fault = item.split("=", 1)
        overrides.setdefault(unit_id, {})[fault] = True
    return overrides


def cmd_replay(args: argparse.Namespace) -> int:
    paths = _resolve(args)
    result = replay_round(paths["ledger"], args.round_id)
    _print_json(result)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    paths = _resolve(args)
    config = load_config(paths["config"], paths["units"])
    incremental = compile_all(config)
    full = full_recompute(config)
    differences = compare_recomputations(incremental, full)
    _print_json(
        {
            "units_compared": len(config.units),
            "identical": not differences,
            "differences": differences,
        }
    )
    return 0 if not differences else 1


def cmd_inject(args: argparse.Namespace) -> int:
    paths = _resolve(args)
    with StructuredLogger(paths["log"]) as logger:
        with open(paths["config"], "r", encoding="utf-8") as handle:
            config_payload = json.load(handle)
        with open(paths["units"], "r", encoding="utf-8") as handle:
            units_payload = json.load(handle)
        if args.kind == "corrupt_config":
            error = trigger_config_injection(config_payload, units_payload)
        elif args.kind == "ring_conflict":
            error = trigger_ring_injection(config_payload, units_payload)
        else:
            error = trigger_version_injection(paths["snapshot"])
        logger.log("injection_reported", **error.to_dict())
        _print_json(error.to_dict())
    return 0


def cmd_recover(args: argparse.Namespace) -> int:
    paths = _resolve(args)
    from .ledger import Ledger

    ledger = Ledger(paths["ledger"])
    try:
        unknown = ledger.unknown_rounds()
        ledger.mark_unknown_for_rerun(unknown)
        last_committed = ledger.last_committed_round_id()
    finally:
        ledger.close()
    _print_json(
        {
            "uncommitted_rounds": unknown,
            "action": "marked_requires_rerun",
            "last_committed_cursor": last_committed,
            "note": "仅依据已完整提交轮次恢复，未提交轮次不生成退出报告",
        }
    )
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    paths = _resolve(args)
    try:
        from .gui import launch_gui
    except ImportError as error:  # pragma: no cover - 环境差异
        print(
            "当前环境缺少 tkinter，无法启动图形界面。"
            "Windows 官方 Python 安装包默认包含 tkinter。\n"
            f"底层原因: {error}",
            file=sys.stderr,
        )
        return 2
    return launch_gui(paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shardbench", description=__doc__)
    parser.add_argument("--workspace", help="工作区根目录，默认当前目录")
    parser.add_argument("--config", help="分片配置 JSON 路径")
    parser.add_argument("--units", help="路由单元 JSON 路径")
    parser.add_argument("--ledger", help="SQLite 账本路径")
    parser.add_argument("--snapshot", help="会话快照路径")
    parser.add_argument("--log", help="结构化 JSONL 日志路径")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="运行一个路由轮次")
    run.add_argument("--changed-table", action="append", help="标记发生规则变更的逻辑表")
    run.add_argument(
        "--fault",
        action="append",
        help="注入探查故障，格式 unit_id=force_config_missing|force_dependency_not_ready",
    )
    run.set_defaults(func=cmd_run)

    replay = sub.add_parser("replay", help="从已提交轮次重放校验")
    replay.add_argument("round_id", type=int)
    replay.set_defaults(func=cmd_replay)

    compare = sub.add_parser("compare", help="增量与全量重算逐项比较")
    compare.set_defaults(func=cmd_compare)

    inject = sub.add_parser("inject", help="结构化异常注入")
    inject.add_argument(
        "kind",
        choices=["corrupt_config", "ring_conflict", "version_mismatch"],
    )
    inject.set_defaults(func=cmd_inject)

    recover = sub.add_parser("recover", help="强制终止后恢复检查")
    recover.set_defaults(func=cmd_recover)

    gui = sub.add_parser("gui", help="启动 Tkinter 诊断界面")
    gui.set_defaults(func=cmd_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ShardBenchError as error:
        _print_json(error.to_dict())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
