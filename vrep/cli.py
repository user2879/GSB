"""命令行入口。

子命令：
- generate-samples : 生成确定性合成图像、真值、配置、编辑命令流与注入配置
- run               : 执行编辑命令流（增量模式或空白全量模式），输出账本与诊断
- query             : 对最新索引执行一次相似度查询
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading

from .engine import Engine
from .injection import InjectionManager
from .jsonio import read_json, write_json
from .samplegen import generate_all


def _build_stop_event() -> threading.Event:
    stop_event = threading.Event()

    def handler(signum, frame):  # pragma: no cover - 信号路径由集成测试覆盖
        stop_event.set()

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handler)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, handler)
    return stop_event


def cmd_generate(args) -> int:
    summary = generate_all(args.samples, image_size=args.image_size,
                           count=args.count, seed=args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cmd_run(args) -> int:
    injections = InjectionManager(args.injections) if args.injections else InjectionManager(None)
    engine = Engine(args.workspace, injection_manager=injections,
                    stop_event=_build_stop_event())

    recovery = None
    if args.recover:
        recovery = engine.recover()
        if recovery is not None:
            print(f"[recover] 从快照 v{recovery['global_version']} 恢复: "
                  f"已完成图像 {len(recovery['completed_images'])} 张")

    commands = read_json(args.commands)["commands"]
    sample_dir = args.samples
    config_path = args.config

    if args.mode == "full":
        # 空白全量：先把全部编辑施加到工作库，再用最终配置一次性构建。
        engine.seed_library(sample_dir)
        for command in commands:
            if command["op"] != "config":
                engine.apply_command(command, sample_dir)
        config, config_failure = engine.load_config(config_path)
        if config_failure is not None:
            write_json(os.path.join(args.workspace, "diagnostics.json"),
                       injections.diagnostics())
            print(f"[fatal] {config_failure.code}: {config_failure.detail}")
            return 2
        batch_no = 1
        engine.run_batch(config, batch_no=batch_no)
        # 全量构建在同一最终状态上自动排空预算待办；
        # 若某批次无任何待办进展（例如图像持续失败），停止排空避免空转。
        while engine.ledger["runtime"]["pending"]:
            before = len(engine.ledger["runtime"]["pending"])
            batch_no += 1
            engine.run_batch(config, batch_no=batch_no)
            after = len(engine.ledger["runtime"]["pending"])
            if after >= before:
                break
    else:
        # 增量模式：逐条命令，每条命令后运行一个批次。
        if not os.path.exists(engine.manifest_path):
            engine.seed_library(sample_dir)
        config, config_failure = engine.load_config(config_path)
        if config_failure is not None:
            print(f"[fatal] {config_failure.code}: {config_failure.detail}")
            write_json(os.path.join(args.workspace, "diagnostics.json"),
                       injections.diagnostics())
            return 2
        for batch_no, command in enumerate(commands, start=1):
            if command["op"] == "config":
                candidate = command.get("config_path", config_path)
                if not os.path.isabs(candidate) and not os.path.exists(candidate):
                    candidate = os.path.join(sample_dir, candidate)
                config_path = candidate
                updated, failure = engine.load_config(
                    config_path, allow_quant_injection=True
                )
                if failure is not None:
                    print(f"[warn] 批次 {batch_no} 配置非法，保留上一版配置: "
                          f"{failure.code} {failure.detail}")
                    # 仅注入触发的配置故障进入失败清单与诊断报告；
                    # 普通坏配置按“拒绝该次编辑、保留上一版配置”处理。
                    if failure.injection_id is not None:
                        engine.record_config_failure(failure, batch_no)
                else:
                    config = updated
                engine.apply_command(command, sample_dir)
            else:
                engine.apply_command(command, sample_dir)
            if engine.stop_event.is_set():
                print(f"[signal] 批次 {batch_no} 收到退出信号，完成当前提交后停止")
                engine.run_batch(config, batch_no=batch_no)
                write_json(os.path.join(args.workspace, "diagnostics.json"),
                           injections.diagnostics())
                return 0
            engine.run_batch(config, batch_no=batch_no)

        # 命令流结束后继续消耗预算，直到待办队列清空或无进展（失败图像）。
        while engine.ledger["runtime"]["pending"]:
            before = len(engine.ledger["runtime"]["pending"])
            batch_no += 1
            engine.run_batch(config, batch_no=batch_no)
            after = len(engine.ledger["runtime"]["pending"])
            if after >= before:
                break

        # 命令流结束后继续消耗预算，直到待办队列清空或无进展（失败图像）。
        while engine.ledger["runtime"]["pending"]:
            before = len(engine.ledger["runtime"]["pending"])
            batch_no += 1
            engine.run_batch(config, batch_no=batch_no)
            after = len(engine.ledger["runtime"]["pending"])
            if after >= before:
                break

    diagnostics = injections.diagnostics()
    write_json(os.path.join(args.workspace, "diagnostics.json"), diagnostics)
    ledger = engine.ledger
    print(f"[done] 全局版本 v{ledger['global_version']}，"
          f"索引图像 {len(ledger['index']['images'])} 张，"
          f"失败 {len(ledger['failures'])} 条，"
          f"待办 {len(ledger['runtime']['pending'])} 项，"
          f"注入命中 {diagnostics['hit_count']}/{diagnostics['injection_count']}")
    if recovery is not None:
        print(f"[recover] 恢复点: v{recovery['global_version']}")
    return 0


def cmd_query(args) -> int:
    engine = Engine(args.workspace)
    engine.recover()
    ledger = engine.ledger
    if args.image not in ledger["images"]:
        print(f"图像 {args.image} 不在库中", file=sys.stderr)
        return 2
    embedding = ledger["images"][args.image]["embedding"]
    if embedding is None:
        print(f"图像 {args.image} 尚无可用嵌入", file=sys.stderr)
        return 2
    entries = [
        {"image": name, "vector": payload["embedding"]["vector"]}
        for name, payload in ledger["images"].items()
        if payload["embedding"] is not None and name != args.image
    ]
    from .index import query

    result = query(embedding["vector"], entries, args.threshold
                   if args.threshold is not None else ledger["index"]["threshold"])
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vrep", description="确定性视觉表征增量编辑应用")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-samples", help="生成确定性合成样本")
    gen.add_argument("--samples", required=True)
    gen.add_argument("--image-size", type=int, default=96)
    gen.add_argument("--count", type=int, default=12)
    gen.add_argument("--seed", type=int, default=20261121)
    gen.set_defaults(func=cmd_generate)

    run = sub.add_parser("run", help="执行编辑命令流")
    run.add_argument("--samples", required=True)
    run.add_argument("--workspace", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--commands", required=True)
    run.add_argument("--injections")
    run.add_argument("--mode", choices=("incremental", "full"), default="incremental")
    run.add_argument("--recover", action="store_true")
    run.set_defaults(func=cmd_run)

    qry = sub.add_parser("query", help="相似度查询")
    qry.add_argument("--workspace", required=True)
    qry.add_argument("--image", required=True)
    qry.add_argument("--threshold", type=float)
    qry.set_defaults(func=cmd_query)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
