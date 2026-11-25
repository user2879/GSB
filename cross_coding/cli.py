from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import invalid_bundle_report, load_bundle
from .reporting import write_generated_table, write_report
from .table import build_table


def command_generate_data(args: argparse.Namespace) -> int:
    feature_path = Path(args.features)
    import yaml

    feature_doc = yaml.safe_load(feature_path.read_text(encoding="utf-8"))
    table = build_table(feature_doc)
    out = Path(args.out)
    write_generated_table(table, out)
    print(json.dumps({"generated": str(out), "csv": str(out.with_suffix(".csv")), "rows": table["row_count"]}, ensure_ascii=False))
    return 0


def command_run(args: argparse.Namespace) -> int:
    bundle, validation = load_bundle(args.features, args.policy)
    if not validation.ok:
        report = invalid_bundle_report(
            validation.errors,
            config_paths={"feature": args.features, "policy": args.policy},
        )
        write_report(report, args.report)
        print(json.dumps({"status": "E_CONFIG_INVALID", "errors": len(validation.errors), "report": args.report}, ensure_ascii=False))
        return 2

    from .pipeline import run_pipeline

    report = run_pipeline(
        bundle,
        args.state,
        parallel=not args.serial,
        command_cursor=args.command_cursor,
    )
    path = write_report(report, args.report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "combinations": len(report["combinations"]),
                "pending": len(report["pending_queue"]),
                "report": str(path),
                "manifest_version": (report.get("manifest") or {}).get("global_version"),
            },
            ensure_ascii=False,
        )
    )
    return 0


def command_inject(args: argparse.Namespace) -> int:
    root = Path(args.root)
    entries = []
    counts: dict[str, int] = {}
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        features = case_dir / "features.yaml"
        policy = case_dir / "policy.yaml"
        if not features.exists() or not policy.exists():
            continue
        bundle, validation = load_bundle(features, policy)
        if bundle is not None:
            entry = {
                "case": case_dir.name,
                "expected": "E_CONFIG_INVALID",
                "unexpected_valid": True,
                "report": None,
            }
        else:
            report = invalid_bundle_report(
                validation.errors,
                injection_case=case_dir.name,
                config_paths={"feature": str(features), "policy": str(policy)},
            )
            report_path = Path(args.report_dir) / f"injection_{case_dir.name}.json"
            write_report(report, report_path)
            entry = {
                "case": case_dir.name,
                "expected": "E_CONFIG_INVALID",
                "unexpected_valid": False,
                "error_count": len(validation.errors),
                "detail_codes": [item.get("detail_code") for item in validation.errors],
                "report": str(report_path),
            }
            for code, count in report["business_code_counts"].items():
                counts[code] = counts.get(code, 0) + count
        entries.append(entry)

    aggregate = {
        "ok": all(not entry.get("unexpected_valid") for entry in entries),
        "status": "injections_verified" if all(not entry.get("unexpected_valid") for entry in entries) else "injection_failed",
        "business_code_counts": counts,
        "cases": entries,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(output), "ok": aggregate["ok"], "cases": len(entries)}, ensure_ascii=False))
    return 0 if aggregate["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cross_coding", description="Deterministic categorical cross-coding ledger")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate-data", help="generate deterministic table JSON/CSV from YAML")
    generate.add_argument("--features", default="config/features.yaml")
    generate.add_argument("--out", default="data/feature_table.json")
    generate.set_defaults(func=command_generate_data)

    run = sub.add_parser("run", help="run enumeration, coding, diagnostics, and adjudication")
    run.add_argument("--features", default="config/features.yaml")
    run.add_argument("--policy", default="config/policy.yaml")
    run.add_argument("--state", default="state")
    run.add_argument("--report", default="reports/report.json")
    run.add_argument("--serial", action="store_true")
    run.add_argument("--command-cursor", type=int, default=None)
    run.set_defaults(func=command_run)

    inject = sub.add_parser("inject", help="validate independent injected failure configurations")
    inject.add_argument("--root", default="config/injections")
    inject.add_argument("--out", default="reports/injection_report.json")
    inject.add_argument("--report-dir", default="reports")
    inject.set_defaults(func=command_inject)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
