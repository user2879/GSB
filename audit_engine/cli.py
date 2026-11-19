"""命令行入口：离线审计一批 JSON 清单。"""

from __future__ import annotations

import argparse
import glob
import sys

from .budget import BudgetConfig
from .errors import E_LOCK
from .models import NormalizationPolicy
from .runner import EXIT_FATAL, EXIT_LOCK, run_batch

DEFAULT_POLICIES = [
    NormalizationPolicy(fold_case=True, unicode_nfc=True, resolve_links=True, name="strict"),
    NormalizationPolicy(fold_case=False, unicode_nfc=True, resolve_links=True, name="case_sensitive"),
    NormalizationPolicy(fold_case=True, unicode_nfc=False, resolve_links=True, name="raw_unicode"),
    NormalizationPolicy(fold_case=True, unicode_nfc=True, resolve_links=False, name="no_link_resolve"),
]


def _build_policies(specs: list[str] | None) -> list[NormalizationPolicy]:
    if not specs:
        return DEFAULT_POLICIES
    policies: list[NormalizationPolicy] = []
    for spec in specs:
        name, fold, nfc, links = spec.split(":")
        policies.append(
            NormalizationPolicy(
                fold_case=fold == "1",
                unicode_nfc=nfc == "1",
                resolve_links=links == "1",
                name=name,
            )
        )
    return policies


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="文件系统权限与路径检查审计引擎")
    parser.add_argument("inputs", nargs="+", help="JSON 清单文件或 glob 模式")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-records", type=int, default=100_000)
    parser.add_argument("--backend", choices=("serial", "thread", "process"), default="thread")
    parser.add_argument("--policy", action="append", help="name:fold:nfc:links，可重复")
    args = parser.parse_args(argv)

    paths: list[str] = []
    for pattern in args.inputs:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        print("未匹配到任何清单文件", file=sys.stderr)
        return EXIT_FATAL

    budget = BudgetConfig(
        concurrency=max(1, args.concurrency),
        per_file_timeout_seconds=args.timeout,
        max_records_per_file=args.max_records,
    )
    try:
        _, exit_code = run_batch(
            manifest_paths=paths,
            policies=_build_policies(args.policy),
            budget=budget,
            checkpoint_dir=args.checkpoint_dir,
            report_path=args.report,
            backend=args.backend,
        )
    except Exception as exc:
        if getattr(exc, "code", None) == E_LOCK:
            print(f"E_LOCK: {exc}", file=sys.stderr)
            return EXIT_LOCK
        print(f"致命错误: {exc}", file=sys.stderr)
        return EXIT_FATAL
    print(f"审计完成，退出码 {exit_code}，报告 {args.report}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
