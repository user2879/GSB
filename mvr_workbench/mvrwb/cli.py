"""命令行入口：生成样本 / 应用编辑流 / 执行重建并写出 JSON 诊断报告。"""

from __future__ import annotations

import argparse
import os
import sys

from .editor import apply_edit_stream
from .pipeline import ReconstructionPipeline, install_signal_handlers
from .sampler import generate_sample
from .util import write_json_atomic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="多视图确定性三维重建工作台")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="生成固定种子本地样本")
    gen.add_argument("--output", required=True)
    gen.add_argument("--views", type=int, default=6)

    run = sub.add_parser("run", help="执行一次重建（可携带编辑流与注入配置）")
    run.add_argument("--input", required=True, help="源样本目录")
    run.add_argument("--work", required=True, help="编辑工作副本目录")
    run.add_argument("--snapshot", required=True)
    run.add_argument("--report", required=True)
    run.add_argument("--edits", default=None)
    run.add_argument("--injections", default=None)

    args = parser.parse_args(argv)
    if args.command == "generate":
        generate_sample(args.output, view_count=args.views)
        return 0

    apply_edit_stream(args.input, args.work, args.edits)
    stop_flag = {"stop": False}
    install_signal_handlers(stop_flag)
    pipeline = ReconstructionPipeline(
        args.work, args.snapshot, injection_config=args.injections
    )
    pipeline.stop_flag = stop_flag
    outcome = pipeline.run()
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    write_json_atomic(args.report, outcome.report)
    return outcome.exit_code


if __name__ == "__main__":
    sys.exit(main())
