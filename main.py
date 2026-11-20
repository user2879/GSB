#!/usr/bin/env python3
"""推理请求归属账本互操作工具 - 命令行入口."""

import argparse
import os
import sys

from ledger_tool import app


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ledger-tool")
    parser.add_argument("--workspace", default=os.getcwd(),
                        help="工作区目录(默认当前目录)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    run_p = sub.add_parser("run")
    run_p.add_argument("--max-upgrades", type=int, default=None,
                       help="本次运行最多升级的条数(用于模拟中断)")
    sub.add_parser("verify")
    inj_p = sub.add_parser("inject")
    inj_p.add_argument("kind", choices=["digest", "chain", "checkpoint"])
    sub.add_parser("acceptance")
    sub.add_parser("gui")
    args = parser.parse_args(argv)
    ws = os.path.abspath(args.workspace)
    if args.cmd == "prepare":
        app.cmd_prepare(ws)
        return 0
    if args.cmd == "run":
        return app.cmd_run(ws, max_upgrades=args.max_upgrades)
    if args.cmd == "verify":
        return app.cmd_verify(ws)
    if args.cmd == "inject":
        return app.cmd_inject(ws, args.kind)
    if args.cmd == "acceptance":
        return app.cmd_acceptance(ws)
    if args.cmd == "gui":
        import gui
        return gui.main(ws)
    return 1


if __name__ == "__main__":
    sys.exit(main())
