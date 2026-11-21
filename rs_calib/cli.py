"""本地命令行入口：generate / run / compare。"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .runner import compare_reports, run
from .samplegen import generate_sample_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rs-calib", description="多波段辐射定标与跨传感器回归对齐组件")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="固定种子生成样本目录")
    gen.add_argument("--out", required=True)
    gen.add_argument("--seed", type=int, default=20261122)

    runner_p = sub.add_parser("run", help="执行增量或全量运行")
    runner_p.add_argument("--sample-dir", required=True)
    runner_p.add_argument("--output-dir", required=True)
    runner_p.add_argument("--mode", choices=["incremental", "full"], default="incremental")
    runner_p.add_argument("--injection", default=None)
    runner_p.add_argument("--resume", action="store_true")

    cmp_p = sub.add_parser("compare", help="比较两份诊断报告的逐字段一致性")
    cmp_p.add_argument("report_a")
    cmp_p.add_argument("report_b")

    args = parser.parse_args(argv)
    if args.command == "generate":
        os.makedirs(args.out, exist_ok=True)
        generate_sample_dataset(args.out, seed=args.seed)
        print(f"样本已生成: {args.out}")
        return 0
    if args.command == "run":
        report = run(args.sample_dir, args.output_dir, mode=args.mode,
                     injection_path=args.injection, resume=args.resume)
        print(json.dumps({
            "mode": report["mode"],
            "calib_records": len(report["calibration_records"]),
            "models": len(report["regression_models"]),
            "alignments": len(report["alignment_records"]),
            "failures": len(report["failure_list"]),
            "reconciled": report["budget_ledger"]["reconciled"],
            "fingerprints": report["content_fingerprints"],
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "compare":
        with open(args.report_a, encoding="utf-8") as fh:
            a = json.load(fh)
        with open(args.report_b, encoding="utf-8") as fh:
            b = json.load(fh)
        result = compare_reports(a, b)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["equal"] else 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
