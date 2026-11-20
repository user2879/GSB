"""命令行入口：run（增量/续跑）、full（全量重建）、verify（一致性诊断）。"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .errors import E_CONFIG_MISMATCH, E_CONFIG_INVALID
from .runner import run_workbench


def _print_summary(result) -> int:
    payload = {
        "business_code": result.business_code,
        "recovery_batch_no": result.recovery_batch_no,
        "failure_count": len(result.failures),
        "failures": [failure.to_dict() for failure in result.failures],
    }
    if result.report is not None:
        payload["totals"] = result.report["totals"]
        payload["snapshot"] = result.report["snapshot"]
        payload["shadow_match"] = result.report["diagnostics"].get("shadow_match")
        diffs = result.report["diagnostics"].get("field_differences", [])
        payload["field_difference_count"] = len(diffs)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if result.business_code in (E_CONFIG_MISMATCH, E_CONFIG_INVALID):
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sft-diag",
        description="确定性指令微调数据构造诊断工作台",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="增量运行（支持快照续跑）")
    run_parser.add_argument("--sample-dir", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--up-to-batch", type=int, default=None)
    run_parser.add_argument("--no-shadow", action="store_true")

    full_parser = sub.add_parser("full", help="使用全新输出目录做一次全量构造")
    full_parser.add_argument("--sample-dir", required=True)
    full_parser.add_argument("--output-dir", required=True)
    full_parser.add_argument("--up-to-batch", type=int, default=None)

    verify_parser = sub.add_parser(
        "verify", help="运行增量并强制影子全量对照，输出差异诊断"
    )
    verify_parser.add_argument("--sample-dir", required=True)
    verify_parser.add_argument("--output-dir", required=True)
    verify_parser.add_argument("--up-to-batch", type=int, default=None)

    args = parser.parse_args(argv)
    if args.command == "full":
        if os.path.exists(os.path.join(args.output_dir, "snapshot", "manifest.json")):
            print(
                json.dumps(
                    {
                        "business_code": E_CONFIG_MISMATCH,
                        "message": "output directory already contains a snapshot; "
                        "use an empty directory for full rebuild",
                    },
                    ensure_ascii=False,
                )
            )
            return 2
        result = run_workbench(
            args.sample_dir,
            args.output_dir,
            up_to_batch=args.up_to_batch,
            shadow_check=False,
        )
    elif args.command == "verify":
        result = run_workbench(
            args.sample_dir,
            args.output_dir,
            up_to_batch=args.up_to_batch,
            shadow_check=True,
        )
    else:
        result = run_workbench(
            args.sample_dir,
            args.output_dir,
            up_to_batch=args.up_to_batch,
            shadow_check=not args.no_shadow,
        )
    return _print_summary(result)


if __name__ == "__main__":
    sys.exit(main())
