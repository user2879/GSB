"""命令行入口（Windows / Linux 通用）。"""

from __future__ import annotations

import argparse
import json
import sys

from .errors import PipelineError
from .injection import HardExit, InjectionConfig
from .lifecycle import hard_exit, run_worker
from .validator import validate_workspace


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="score_pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    worker_p = sub.add_parser("worker", help="工作进程：合成并增量提交")
    worker_p.add_argument("--workspace", required=True)
    worker_p.add_argument("--injection", required=True)
    worker_p.add_argument("--max-op", type=int, default=None)

    validate_p = sub.add_parser("validate", help="独立验证器：只消费文件复核时间轴")
    validate_p.add_argument("--workspace", required=True)

    args = parser.parse_args(argv)
    if args.command == "worker":
        injection = InjectionConfig.load(args.injection)
        try:
            run_worker(args.workspace, injection, max_op=args.max_op)
        except HardExit as exc:
            hard_exit(exc.point, exc.exit_code)
        except PipelineError as exc:
            sys.stderr.write(json.dumps(exc.to_record(), sort_keys=True) + "\n")
            return 2
        return 0
    if args.command == "validate":
        report = validate_workspace(args.workspace)
        print(json.dumps(report, sort_keys=True, indent=2))
        return 0 if report["ok"] else 3
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
