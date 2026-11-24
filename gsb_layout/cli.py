"""Command-line interface for the deterministic vertical-layout toolkit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

from .cache import atomic_write_json
from .config import load_config
from .pipeline import Pipeline
from .sample_generator import generate_samples
from .verification import verify_incremental_equivalence


def _clear_full_state(state_dir: Path) -> None:
    for name in ("cache", "snapshot"):
        path = state_dir / name
        if path.exists():
            shutil.rmtree(path)
    state_dir.mkdir(parents=True, exist_ok=True)


def _cmd_generate(args: argparse.Namespace) -> int:
    result = generate_samples(
        args.samples,
        config_path=args.config,
        overwrite=not args.no_overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _write_report(report: dict[str, Any], path: str | Path) -> None:
    atomic_write_json(path, report)


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    state_dir = Path(args.state)
    state_dir.mkdir(parents=True, exist_ok=True)
    if args.full:
        _clear_full_state(state_dir)
    report = Pipeline(args.samples, config, state_dir).run(full_rebuild=args.full)
    _write_report(report, args.report)
    hard_failures = [
        failure
        for failure in report["failures"]
        if failure["code"] not in {"E_MATCH_BUDGET_PENDING"}
    ]
    print(
        json.dumps(
            {
                "report": str(Path(args.report)),
                "mode": report["run"]["mode"],
                "global_version": report["run"]["global_version"],
                "semantic_digest": report["semantic_digest"],
                "summary": report["summary"],
                "failure_codes": [failure["code"] for failure in report["failures"]],
                "cache": report["cache"]["summary"],
                "budget": {
                    "limit": report["budget"]["limit"],
                    "consumed_points": report["budget"]["consumed_points"],
                    "pending_jobs": report["budget"]["pending_jobs"],
                },
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 1 if hard_failures else 0


def _cmd_verify(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    result = verify_incremental_equivalence(
        args.samples,
        config,
        args.workdir,
        regenerate_samples=args.regenerate_samples,
    )
    _write_report(result, args.report)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["passed"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gsb_layout",
        description="本地确定性竖排古籍版面识别、增量编辑与验收工具",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate-samples", help="生成确定性 PNG 样本、真值与配置")
    gen.add_argument("--samples", default="samples", help="样本输出目录")
    gen.add_argument("--config", default=None, help="配置输出路径，默认 <samples>/config.json")
    gen.add_argument("--no-overwrite", action="store_true", help="存在样本时拒绝覆盖")
    gen.set_defaults(func=_cmd_generate)

    run = subparsers.add_parser("run", help="执行增量或全量版面识别")
    run.add_argument("--samples", default="samples", help="输入样本目录")
    run.add_argument("--config", default="samples/config.json", help="识别配置 JSON")
    run.add_argument("--state", default="state", help="缓存与快照目录")
    run.add_argument("--report", default="reports/report.json", help="JSON 报告输出路径")
    run.add_argument("--full", action="store_true", help="清空缓存与快照后从空白全量执行")
    run.set_defaults(func=_cmd_run)

    verify = subparsers.add_parser("verify", help="验证编辑序列后的增量/全量逐字段一致")
    verify.add_argument("--samples", default="samples", help="输入样本目录")
    verify.add_argument("--config", default="samples/config.json", help="最终识别配置 JSON")
    verify.add_argument("--workdir", default=".verify", help="验证临时工作目录")
    verify.add_argument("--report", default="reports/verification.json", help="验证报告输出路径")
    verify.add_argument(
        "--regenerate-samples",
        action="store_true",
        help="验证前重新生成确定性样本",
    )
    verify.set_defaults(func=_cmd_verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # business errors still yield one stable JSON-ish line
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
