"""命令行入口：生成工作区、增量运行、全量重建与恢复判定。"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .engine import Engine
from .generator import generate_workspace
from .state import read_json, write_json_atomic


def _load_commands(workspace: str) -> list[dict]:
    return read_json(os.path.join(workspace, "commands.json"), [])


def cmd_generate(args) -> int:
    info = generate_workspace(args.workspace, sample_count=args.samples)
    print(f"已生成 {args.samples} 个基础样本、2 个近重复样本与 3 个非法样本到 {args.workspace}")
    print(f"命令流共 {len(info['commands'])} 条 -> {os.path.join(args.workspace, 'commands.json')}")
    return 0


def cmd_run(args) -> int:
    commands = _load_commands(args.workspace)
    engine = Engine(args.workspace, rebuild=False)
    resume_ok, resume_reason = engine.store.can_resume(engine.config.config_hash())
    print(f"[恢复判定] {resume_reason}")
    reports = engine.execute_stream(commands)
    out_path = os.path.join(args.workspace, "reports_incremental.json")
    write_json_atomic(out_path, reports)
    print(f"[增量运行] {len(reports)} 份阶段报告 -> {out_path}")
    return 0


def cmd_rebuild(args) -> int:
    commands = _load_commands(args.workspace)
    # 全量重建以命令流声明的最终样本集合与最终配置为准
    final_ids = _final_sample_ids(args.workspace, commands)
    engine = Engine(args.workspace, rebuild=True)
    engine.apply_final_config(commands)
    report = engine.execute_rebuild(final_ids)
    out_path = os.path.join(args.workspace, "reports_rebuild.json")
    write_json_atomic(out_path, report)
    print(f"[全量重建] 最终样本 {len(final_ids)} 个 -> {out_path}")
    return 0


def _final_sample_ids(workspace: str, commands: list[dict]) -> list[str]:
    active: set[str] = set()
    for command in commands:
        cmd = command.get("cmd")
        sid = command.get("sample_id")
        if cmd == "add":
            active.add(sid)
        elif cmd == "replace":
            active.add(sid)
        elif cmd == "delete":
            active.discard(sid)
    # 仅保留磁盘上真实存在的样本
    sample_dir = os.path.join(workspace, "samples")
    return sorted(sid for sid in active if os.path.exists(os.path.join(sample_dir, f"{sid}.png")))


def cmd_resume(args) -> int:
    engine = Engine(args.workspace, rebuild=False)
    ok, reason = engine.store.can_resume(engine.config.config_hash())
    print(json.dumps({"can_resume": ok, "reason": reason}, ensure_ascii=False, indent=2))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="跨模态哈希签名与近重复簇账本")
    sub = parser.add_subparsers(dest="action", required=True)

    gen = sub.add_parser("generate", help="生成确定性样本工作区")
    gen.add_argument("workspace")
    gen.add_argument("--samples", type=int, default=10)
    gen.set_defaults(func=cmd_generate)

    run = sub.add_parser("run", help="按命令流增量运行并持久化")
    run.add_argument("workspace")
    run.set_defaults(func=cmd_run)

    rebuild = sub.add_parser("rebuild", help="从空白全量重建")
    rebuild.add_argument("workspace")
    rebuild.set_defaults(func=cmd_rebuild)

    resume = sub.add_parser("resume", help="快照恢复判定")
    resume.add_argument("workspace")
    resume.set_defaults(func=cmd_resume)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
