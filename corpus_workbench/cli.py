"""Command line entry point for the deterministic local workbench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .demo_data import generate_demo_workspace
from .pipeline import Workbench


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="corpus-workbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-demo", help="regenerate deterministic demo inputs")
    generate.add_argument("--root", default=".", help="project root (relative paths supported)")

    run = subparsers.add_parser("run", help="run or resume the staged workbench")
    run.add_argument("--data", default="data/corpus")
    run.add_argument("--balance", default="configs/balance.json")
    run.add_argument("--dedup", default="configs/dedup.json")
    run.add_argument("--scheduler", default="configs/scheduler.json")
    run.add_argument("--injection", default="configs/injection.json")
    run.add_argument("--output", default="output")
    run.add_argument("--cache-root", default=None)
    run.add_argument("--run-id", default="demo")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--reset", action="store_true")
    run.add_argument("--expected-version", type=int, default=None)
    run.add_argument("--stop-after-batches", type=int, default=None)
    run.add_argument("--quiet", action="store_true")

    show = subparsers.add_parser("show", help="print the latest report summary")
    show.add_argument("--output", default="output")
    show.add_argument("--run-id", default="demo")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "generate-demo":
        manifest = generate_demo_workspace(Path(args.root))
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "show":
        path = Path(args.output) / "reports" / f"{args.run_id}.json"
        if not path.exists():
            print(f"report not found: {path}", file=sys.stderr)
            return 2
        report = json.loads(path.read_text(encoding="utf-8"))
        summary = {
            "run_id": report.get("run_id"),
            "status": report.get("status"),
            "complete": report.get("complete"),
            "global_version": report.get("global_version"),
            "samples": (report.get("sampling") or {}).get("sample_count"),
            "before_dedup": (report.get("documents") or {}).get("sampled_unique_before_dedup"),
            "after_dedup": (report.get("documents") or {}).get("after_dedup"),
            "duplicate_groups": (report.get("dedup") or {}).get("duplicate_group_count"),
            "consumed_points": (report.get("budget") or {}).get("consumed_points"),
            "pending": len(report.get("pending_queue", [])),
            "failures": report.get("failure_count"),
            "result_digest": report.get("deterministic_result_digest"),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    workbench = Workbench(
        data_path=args.data,
        balance_path=args.balance,
        dedup_path=args.dedup,
        scheduler_path=args.scheduler,
        injection_path=args.injection,
        output_root=args.output,
        run_id=args.run_id,
        cache_root=args.cache_root,
    )
    report = workbench.run(
        resume=args.resume,
        expected_version=args.expected_version,
        stop_after_batches=args.stop_after_batches,
        reset=args.reset,
    )
    if not args.quiet:
        summary = {
            "run_id": report.get("run_id"),
            "status": report.get("status"),
            "complete": report.get("complete"),
            "global_version": report.get("global_version"),
            "samples": (report.get("sampling") or {}).get("sample_count"),
            "before_dedup": (report.get("documents") or {}).get("sampled_unique_before_dedup"),
            "after_dedup": (report.get("documents") or {}).get("after_dedup"),
            "duplicate_groups": (report.get("dedup") or {}).get("duplicate_group_count"),
            "consumed_points": (report.get("budget") or {}).get("consumed_points"),
            "pending": len(report.get("pending_queue", [])),
            "failures": report.get("failure_count"),
            "failure_codes": sorted({item.get("code") for item in report.get("failures", [])}),
            "result_digest": report.get("deterministic_result_digest"),
            "report": f"{Path(args.output) / 'reports' / (args.run_id + '.json')}",
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("status") not in {"config_invalid", "stale", "input_invalid"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
