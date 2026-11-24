"""命令行入口：python -m receipt_ledger --config ... --samples ... --commands ..."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .engine import LedgerEngine


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m receipt_ledger",
        description="本地推理回执归并与资源诊断应用",
    )
    parser.add_argument("--config", required=True, help="工作器配置 JSON")
    parser.add_argument("--samples", required=True, help="样本目录")
    parser.add_argument("--commands", required=True, help="命令流 JSON")
    parser.add_argument("--output", default=None, help="输出目录（默认 <samples>/../output）")
    parser.add_argument("--inject", default=None, help="独立异常注入配置 JSON")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="线程池并发度，结果与并发度无关（默认 1）",
    )
    parser.add_argument(
        "--fast-budget-ms",
        type=float,
        default=None,
        help="设置后额外执行快速路径并输出 fast_view.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output or os.path.join(os.path.dirname(args.samples.rstrip("/")), "output")
    os.makedirs(output_dir, exist_ok=True)

    engine = LedgerEngine(
        config_path=args.config,
        samples_dir=args.samples,
        commands_path=args.commands,
        output_dir=output_dir,
        injection_path=args.inject,
        max_workers=args.workers,
    )
    report = engine.run_authoritative()
    report_path = engine.store.save_report("report.json", report)
    manifest_path = engine.store.save_report("snapshot_manifest.json", engine.store.manifest)

    fast_path = None
    if args.fast_budget_ms is not None:
        fast_engine = LedgerEngine(
            config_path=args.config,
            samples_dir=args.samples,
            commands_path=args.commands,
            output_dir=output_dir,
            injection_path=args.inject,
            max_workers=args.workers,
        )
        fast_report = fast_engine.fast_view(args.fast_budget_ms)
        fast_path = fast_engine.store.save_report("fast_view.json", fast_report)

    summary = {
        "report": os.path.basename(report_path),
        "manifest": os.path.basename(manifest_path),
        "fast_view": os.path.basename(fast_path) if fast_path else None,
        "totals": report["totals"],
        "failure_codes": sorted({item["code"] for item in report["failures"]}),
        "injection_hits": report["injection"]["hit_ids"],
        "output_dir": os.path.abspath(output_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
