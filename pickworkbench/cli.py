"""Command-line entry points."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .canonical import content_hash, read_json, write_json
from .demo import verify_bundle_repeat, write_demo_files
from .engine import execute_case, reproduce_witness
from .errors import WorkbenchError
from .generator import GeneratorConfig, generate_sample_with_config, summarize_scene
from .injection import empty_injection

_DEFAULT_CONFIG = GeneratorConfig()


def _config_from_args(args: argparse.Namespace) -> GeneratorConfig:
    return GeneratorConfig(
        seed=int(args.seed),
        object_count=int(args.object_count),
        ray_count=int(args.ray_count),
        click_count=int(args.click_count),
        drag_count=int(args.drag_count),
        tolerance_px=float(args.tolerance),
        viewport_width=int(args.width),
        viewport_height=int(args.height),
    )


def command_generate(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    scene, queries = generate_sample_with_config(config)
    target = Path(args.out)
    write_json(target / "scene.json", scene)
    write_json(target / "queries.json", queries)
    summary = summarize_scene(scene, queries)
    print(f"objects={summary['object_count']}")
    print(f"bounds_min={summary['bounds']['min']}")
    print(f"bounds_max={summary['bounds']['max']}")
    print(f"queries={summary['query_count']} ({summary['query_counts']})")
    print(f"scene_hash={content_hash(scene)}")
    print(f"queries_hash={content_hash(queries)}")
    return 0


def command_validate(args: argparse.Namespace) -> int:
    scene = read_json(args.scene)
    queries = read_json(args.queries)
    injection = read_json(args.injection) if args.injection else empty_injection("external")
    report = execute_case("external_validation", scene, queries, injection, args.expect or [])
    target = Path(args.out)
    write_json(target, report)
    status = report["status"]
    print(f"status={status}")
    print(f"queries={report['summary']['query_count']} passed={report['summary']['passed']} failed={report['summary']['failed']}")
    if report.get("first_affected"):
        first = report["first_affected"]
        print(f"first_affected={first['query_id']} code={first['error_code']}")
    print(f"report_hash={report['content_hash']}")
    return 0 if status == "pass" or ("E_INJECT_CONFIG" in report.get("expected_error_codes", [])) else 1


def command_witness(args: argparse.Namespace) -> int:
    witness = read_json(args.witness)
    outcome = reproduce_witness(witness)
    print(f"reproduced={str(outcome['reproduced']).lower()}")
    print(f"error_code={outcome['error_code']}")
    print(f"query_id={outcome['query_id']}")
    print(f"witness_hash={witness.get('content_hash')}")
    return 0 if outcome["reproduced"] else 1


def command_demo(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    files, report = verify_bundle_repeat(config)
    write_demo_files(args.out, files)
    print("repeat_check=identical")
    print(f"cases={len(report['cases'])}")
    for case in report["cases"]:
        first = case.get("first_affected")
        first_text = "-" if first is None else f"{first['query_id']}:{first['error_code']}"
        print(
            f"case={case['name']} status={case['status']} queries={case['summary']['query_count']} "
            f"failed={case['summary']['failed']} first={first_text} witness={case['witness_minimal']}"
        )
    print(f"overall_hash={report['overall_hash']}")
    print(f"files={len(files)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pickworkbench",
        description="Deterministic ray/click/drag picking validation workbench.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_generation_options(target: argparse.ArgumentParser) -> None:
        target.add_argument("--seed", type=int, default=_DEFAULT_CONFIG.seed)
        target.add_argument("--object-count", type=int, default=_DEFAULT_CONFIG.object_count)
        target.add_argument("--ray-count", type=int, default=_DEFAULT_CONFIG.ray_count)
        target.add_argument("--click-count", type=int, default=_DEFAULT_CONFIG.click_count)
        target.add_argument("--drag-count", type=int, default=_DEFAULT_CONFIG.drag_count)
        target.add_argument("--tolerance", type=float, default=_DEFAULT_CONFIG.tolerance_px)
        target.add_argument("--width", type=int, default=_DEFAULT_CONFIG.viewport_width)
        target.add_argument("--height", type=int, default=_DEFAULT_CONFIG.viewport_height)

    generate = subparsers.add_parser("generate", help="Generate one deterministic scene/query pair.")
    generate.add_argument("--out", required=True)
    add_generation_options(generate)
    generate.set_defaults(handler=command_generate)

    validate = subparsers.add_parser("validate", help="Run one scene/query/injection batch.")
    validate.add_argument("--scene", required=True)
    validate.add_argument("--queries", required=True)
    validate.add_argument("--injection")
    validate.add_argument("--out", required=True)
    validate.add_argument("--expect", action="append")
    validate.set_defaults(handler=command_validate)

    witness = subparsers.add_parser("witness", help="Re-run a minimal counterexample witness.")
    witness.add_argument("--witness", required=True)
    witness.set_defaults(handler=command_witness)

    demo = subparsers.add_parser("demo", help="Build the fixed, repeatable demo bundle.")
    demo.add_argument("--out", default="artifacts/demo")
    add_generation_options(demo)
    demo.set_defaults(handler=command_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except WorkbenchError as error:
        print(f"{error.code}: {error.message}", file=sys.stderr)
        return 2
