"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .conversion import convert_file
from .demo import print_demo_summary, run_demo
from .edits import apply_edit_script
from .errors import AQevalError, InterruptedRun
from .formats import read_exchange
from .generator import generate_workspace
from .levels import LevelMap
from .pipeline import EvaluationPipeline


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _workspace(args: argparse.Namespace) -> Path:
    return Path(args.workspace).resolve()


def _query(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    level_map = LevelMap.load(workspace / "level_map.json")
    versions = [args.version] if args.version else [1, 2, 3]
    paths = {
        1: workspace / "eval/v1/results_v1.txt",
        2: workspace / "eval/v2/results_v2.json",
        3: workspace / "eval/v3/results_v3.bin",
    }
    output: dict[str, Any] = {"schema": "aqeval.query.v1", "versions": {}}
    failure_count = 0
    for version in versions:
        outcome = read_exchange(version, paths[version], level_map)
        records = outcome.records
        if args.file:
            records = [record for record in records if record.file_id == args.file]
        output["versions"][str(version)] = {
            "path": str(paths[version]),
            "records": [record.to_dict() for record in records],
            "failures": [failure.to_dict() for failure in outcome.failures],
        }
        failure_count += len(outcome.failures)
    _print_json(output)
    return 1 if failure_count else 0


def _convert(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    level_map = LevelMap.load(workspace / "level_map.json")
    config = json.loads((workspace / "config.json").read_text(encoding="utf-8"))
    fixed_bits = int(config["fixed_decimal_bits"])
    bypass = Path(args.bypass).resolve() if args.bypass else None
    summary = convert_file(
        args.from_version,
        args.to_version,
        Path(args.input).resolve(),
        Path(args.output).resolve(),
        level_map,
        fixed_bits,
        bypass,
    )
    _print_json(summary)
    return 1 if summary["failures"] else 0


def _run(args: argparse.Namespace) -> int:
    try:
        report = EvaluationPipeline(
            _workspace(args),
            rebuild_cache=args.rebuild_cache,
            rebuild_snapshot=args.rebuild_snapshot,
            stop_after_stage=args.stop_after,
            serial_mode=args.serial,
        ).run()
    except InterruptedRun as exc:
        _print_json({"code": exc.code, "message": str(exc), "exit": 75})
        return 75
    summary = {
        "summary": report["summary"],
        "snapshot": report["snapshot"],
        "concurrency": report["concurrency"],
        "outputs": report["outputs"],
        "injections": report["injections"],
        "report": str(_workspace(args) / "reports" / "diagnostic.json"),
    }
    _print_json(summary)
    hard_failures = [
        item
        for item in report["failures"]
        if item.get("code") not in {"E_SNAPSHOT_STALE"}
    ]
    return 1 if hard_failures else 0


def _edit(args: argparse.Namespace) -> int:
    result = apply_edit_script(
        _workspace(args),
        Path(args.script).resolve(),
        run_after_each=not args.no_run_each,
        run_final=args.run_final,
        rebuild=args.rebuild,
    )
    _print_json(result)
    return 0


def _demo(args: argparse.Namespace) -> int:
    result = run_demo(_workspace(args))
    print_demo_summary(result)
    return 0 if result["full_rebuild_comparison"]["equal"] and result["serial_vs_concurrent_comparison"]["equal"] else 1


def _generate(args: argparse.Namespace) -> int:
    result = generate_workspace(_workspace(args), clean=not args.keep)
    _print_json(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aqeval",
        description="Deterministic audio-quality evaluation exchange toolkit",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="generate deterministic samples and fixtures")
    generate.add_argument("--workspace", default="sample_workspace")
    generate.add_argument("--keep", action="store_true", help="keep existing workspace contents")
    generate.set_defaults(func=_generate)

    run = sub.add_parser("run", help="run the staged incremental pipeline")
    run.add_argument("--workspace", default="sample_workspace")
    run.add_argument("--rebuild-cache", action="store_true")
    run.add_argument("--rebuild-snapshot", action="store_true")
    run.add_argument("--stop-after", choices=("parse", "features", "scores", "conversion"))
    run.add_argument("--serial", action="store_true", help="force one worker for verification")
    run.set_defaults(func=_run)

    query = sub.add_parser("query", help="query one or all exchange versions")
    query.add_argument("--workspace", default="sample_workspace")
    query.add_argument("--version", type=int, choices=(1, 2, 3))
    query.add_argument("--file")
    query.set_defaults(func=_query)

    convert = sub.add_parser("convert", help="convert an exchange file and retain bypass loss data")
    convert.add_argument("--workspace", default="sample_workspace")
    convert.add_argument("--from-version", dest="from_version", type=int, choices=(1, 2, 3), required=True)
    convert.add_argument("--to-version", dest="to_version", type=int, choices=(1, 2, 3), required=True)
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--bypass")
    convert.set_defaults(func=_convert)

    edit = sub.add_parser("edit", help="apply a JSONL edit sequence and run incrementally")
    edit.add_argument("--workspace", default="sample_workspace")
    edit.add_argument("--script", required=True)
    edit.add_argument("--no-run-each", action="store_true")
    edit.add_argument("--run-final", action="store_true")
    edit.add_argument("--rebuild", action="store_true")
    edit.set_defaults(func=_edit)

    demo = sub.add_parser("demo", help="run generation, conversion, edits, resume and equivalence checks")
    demo.add_argument("--workspace", default="validation_workspace")
    demo.set_defaults(func=_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except AQevalError as exc:
        print(json.dumps({"code": exc.code, "message": str(exc)}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"code": "E_AQEVAL", "message": str(exc)}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

