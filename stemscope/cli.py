from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .conversion import convert_package
from .dashboard import generate_dashboard
from .injection import apply_injection_case, load_injection_manifest, run_injection_case
from .package import parse_package
from .report import run_acceptance
from .samplegen import DEFAULT_SEED, generate_samples


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _cmd_generate(args: argparse.Namespace) -> int:
    manifest = generate_samples(args.out, seed=args.seed, force=not args.keep_existing)
    _print_json(manifest)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    report = run_acceptance(
        args.data,
        args.out,
        config_path=args.config,
        interactions_path=args.interactions,
        resume=args.resume,
        expected_global_version=args.expected_version,
        workers_override=args.workers,
        budget_override_ms=args.budget_ms,
        format_override=args.format,
    )
    print(f"report: {Path(args.out).resolve() / 'report.json'}")
    print(f"dashboard: {Path(args.out).resolve() / 'dashboard.html'}")
    print(f"report_id: {report['report_id']}")
    print(f"report_fingerprint: {report['report_fingerprint']}")
    print(
        "acceptance: "
        f"{report['acceptance']['passed_count']}/{report['acceptance']['total_count']} "
        f"({'PASS' if report['acceptance']['passed'] else 'FAIL'})"
    )
    for check in report["acceptance"]["checks"]:
        print(f"  {'PASS' if check['passed'] else 'FAIL'} {check['id']}")
    return 0 if report["acceptance"]["passed"] else 1


def _cmd_parse(args: argparse.Namespace) -> int:
    segment = Path(args.segment)
    clip_id = args.clip or segment.name
    result = parse_package(segment, args.format, clip_id=clip_id)
    payload = {
        "format_version": result.format_version,
        "ok": result.ok,
        "package": result.package.to_package_info() if result.package is not None else None,
        "failures": [error.to_dict() for error in result.failures],
    }
    _print_json(payload)
    return 0 if result.ok else 1


def _cmd_convert(args: argparse.Namespace) -> int:
    segment = Path(args.segment)
    parsed = parse_package(segment, args.source, clip_id=segment.name)
    if not parsed.ok or parsed.package is None:
        _print_json({"ok": False, "failures": [error.to_dict() for error in parsed.failures]})
        return 1
    result = convert_package(parsed.package, args.target, args.out, clip_id=segment.name)
    payload = {
        "ok": result.ok,
        "record": result.record,
        "target_package": result.target_package.to_package_info()
        if result.target_package is not None
        else None,
        "failures": [error.to_dict() for error in result.failures],
    }
    _print_json(payload)
    return 0 if result.ok else 1


def _cmd_inject(args: argparse.Namespace) -> int:
    manifest = load_injection_manifest(args.manifest)
    cases = manifest.get("cases", [])
    selected = cases if args.case == "all" else [case for case in cases if case["id"] == args.case]
    if not selected:
        print(f"unknown injection case {args.case!r}", file=sys.stderr)
        return 2
    results = []
    for case in selected:
        destination = Path(args.out) if len(selected) == 1 else Path(args.out) / str(case["id"])
        apply_injection_case(args.sample_root, destination, case)
        if args.parse:
            results.append(run_injection_case(args.sample_root, destination, case))
        else:
            results.append({"case_id": case["id"], "applied_to": str(destination)})
    _print_json(results)
    return 0 if all(item.get("passed", True) for item in results) else 1


def _cmd_dashboard(args: argparse.Namespace) -> int:
    with open(args.report, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    path = generate_dashboard(report, args.out)
    print(path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stemscope",
        description="Deterministic stem separation rendering and diagnostic workbench",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-samples", help="generate deterministic local samples")
    generate.add_argument("--out", default="sample_data")
    generate.add_argument("--seed", type=int, default=DEFAULT_SEED)
    generate.add_argument("--keep-existing", action="store_true")
    generate.set_defaults(func=_cmd_generate)

    run = subparsers.add_parser("run", help="run the full acceptance workbench and emit a report")
    run.add_argument("--data", default="sample_data")
    run.add_argument("--out", default="artifacts")
    run.add_argument("--config")
    run.add_argument("--interactions")
    run.add_argument("--format", choices=("v1", "v2", "v3"))
    run.add_argument("--workers", type=int)
    run.add_argument("--budget-ms", type=float)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--expected-version")
    run.set_defaults(func=_cmd_run)

    parse = subparsers.add_parser("parse", help="parse one v1/v2/v3 package and print unified track info")
    parse.add_argument("--segment", required=True)
    parse.add_argument("--format", required=True, choices=("v1", "v2", "v3"))
    parse.add_argument("--clip")
    parse.set_defaults(func=_cmd_parse)

    convert = subparsers.add_parser("convert", help="convert one parsed package to another format")
    convert.add_argument("--segment", required=True)
    convert.add_argument("--source", required=True, choices=("v1", "v2", "v3"))
    convert.add_argument("--target", required=True, choices=("v1", "v2", "v3"))
    convert.add_argument("--out", required=True)
    convert.set_defaults(func=_cmd_convert)

    inject = subparsers.add_parser("inject", help="apply a deterministic input-only fault injection")
    inject.add_argument("--sample-root", default="sample_data")
    inject.add_argument("--manifest", default="sample_data/injection_manifest.json")
    inject.add_argument("--case", required=True)
    inject.add_argument("--out", required=True)
    inject.add_argument("--parse", action="store_true")
    inject.set_defaults(func=_cmd_inject)

    dashboard = subparsers.add_parser("dashboard", help="regenerate the HTML dashboard from report JSON")
    dashboard.add_argument("--report", default="artifacts/report.json")
    dashboard.add_argument("--out", default="artifacts/dashboard.html")
    dashboard.set_defaults(func=_cmd_dashboard)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
