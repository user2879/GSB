from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .canonical import atomic_write, canonical_json
from .errors import VScanError
from .planner import compile_plan, plan_justification
from .session import Session
from .spec import load_spec
from .store import build_store


def _emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _write_output(value: Any, output: str | None) -> None:
    if output:
        atomic_write(output, (canonical_json(value) + "\n").encode("utf-8"))


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spec", required=True, help="path to the JSON scan specification")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vscan", description="deterministic vectorized columnar scan")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build deterministic column files")
    _add_common(build)
    build.add_argument("--data", required=True, help="column store directory")
    build.add_argument("--output")

    plan = sub.add_parser("plan", help="compile and print a batch execution plan")
    _add_common(plan)
    plan.add_argument("--include-batches", action="store_true")
    plan.add_argument("--output")

    run = sub.add_parser("run", help="execute and test the scan pipeline")
    _add_common(run)
    run.add_argument("--data", required=True)
    run.add_argument("--ledger", required=True)
    run.add_argument("--output")
    run.add_argument("--force-full", action="store_true")
    run.add_argument("--verify", action="store_true")

    verify = sub.add_parser("verify", help="run then compare against a blank full rebuild")
    _add_common(verify)
    verify.add_argument("--data", required=True)
    verify.add_argument("--ledger", required=True)
    verify.add_argument("--output")

    explain = sub.add_parser("explain", help="print pruning/pushdown plan justifications")
    _add_common(explain)
    explain.add_argument("--output")
    return parser


def _run(args: argparse.Namespace, *, verify: bool = False) -> int:
    spec = load_spec(args.spec)
    with Session(spec, args.data, args.ledger) as session:
        report = session.run(force_full=getattr(args, "force_full", False))
        if verify:
            report["full_rebuild_verification"] = session.verify_against_full_rebuild()
        _write_output(report, getattr(args, "output", None))
        _emit(
            {
                "ok": True,
                "command": "verify" if verify else "run",
                "queries": [query["query_id"] for query in report["queries"]],
                "batch_count": len(report["batch_results"]),
                "recomputed_batch_count": report["incremental"]["recomputed_batch_count"],
                "reused_batch_count": report["incremental"]["reused_batch_count"],
                "scheduler": report["scheduler"],
                "ledger": report["ledger"],
                "recovery_unknown": len(report["recovery"]["unknown"]),
                "full_rebuild_verification": report.get("full_rebuild_verification"),
            }
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            spec = load_spec(args.spec)
            state = build_store(spec, args.data)
            result = {
                "store": state.to_dict(),
                "columns": [
                    {
                        "segment_id": segment.id,
                        "column": column.name,
                        "encoding": column.encoding,
                        "non_null_count": column.non_null_count,
                        "column_digest": column.column_digest,
                    }
                    for segment in state.segments
                    for column in segment.columns
                ],
            }
            _write_output(result, args.output)
            _emit(result)
            return 0
        if args.command == "plan":
            spec = load_spec(args.spec)
            execution_plan = compile_plan(spec)
            result = execution_plan.to_dict(include_batches=args.include_batches)
            result["justifications"] = [
                justification
                for query_plan in execution_plan.queries
                for justification in plan_justification(query_plan)
            ]
            _write_output(result, args.output)
            _emit(result)
            return 0
        if args.command == "run":
            return _run(args)
        if args.command == "verify":
            return _run(args, verify=True)
        if args.command == "explain":
            spec = load_spec(args.spec)
            execution_plan = compile_plan(spec)
            result = {
                query_plan.query.id: {
                    "summary": query_plan.summary,
                    "batch_boundaries_digest": query_plan.batch_boundaries_digest,
                    "justification": plan_justification(query_plan),
                }
                for query_plan in execution_plan.queries
            }
            _write_output(result, args.output)
            _emit(result)
            return 0
    except VScanError as exc:
        print(canonical_json(exc.to_dict()), file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
