"""命令行入口（无 GUI 也可完整运行；GUI 为可选诊断视图）。"""
from __future__ import annotations

import argparse
import os
import sys

from .demo import CRASH_ARG, run_sequence, run_with_crash_recovery, write_reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="syncbench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_demo = sub.add_parser("demo", help="运行固定序列演示")
    p_demo.add_argument("--workspace", default="workspace")
    p_demo.add_argument("--crash", action="store_true",
                        help="参考切换处强制终止（恢复演示用）")
    p_demo.add_argument(CRASH_ARG, action="store_true",
                        help=argparse.SUPPRESS)

    p_gui = sub.add_parser("gui", help="启动 PySide6 诊断视图")
    p_gui.add_argument("--workspace", default="workspace")

    args = parser.parse_args(argv)
    ws = os.path.abspath(args.workspace)

    if args.cmd == "demo":
        if getattr(args, CRASH_ARG.lstrip("-").replace("-", "_"), False):
            run_sequence(ws, crash=True)  # 内部硬退出
            return 137
        if args.crash:
            report = run_with_crash_recovery(ws)
        else:
            report = run_sequence(ws, crash=False)
        report_path, hashes_path = write_reports(ws, report)
        print(f"demo report : {report_path}")
        print(f"node hashes : {hashes_path}")
        print(f"reference   : ch{report['final_reference_channel']}")
        print(f"error codes : {report['all_error_codes']}")
        return 0

    if args.cmd == "gui":
        from .gui import launch_gui
        return launch_gui(ws)

    return 2


if __name__ == "__main__":
    sys.exit(main())
