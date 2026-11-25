from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .commands import apply_command, load_command_stream
from .config import load_config
from .pipeline import run_pipeline
from .reporting import atomic_write_json, export_judgments
from .samplegen import generate_sample_tree
from .snapshot import SnapshotStore
from .truth import load_truth, validate_truth


def command_generate(args: argparse.Namespace) -> int:
    result = generate_sample_tree(args.root)
    root = Path(args.root)
    config = result["config"]
    truth_by_image, truth_issues = validate_truth(result["truth"], config)
    ledger, _ = run_pipeline(
        str(root),
        config,
        truth_by_image=truth_by_image,
        truth_issues=truth_issues,
        exchange_version="v2",
    )
    export_judgments(ledger, root)
    print(f"generated samples and exchange judgments in {result['samples_dir']}")
    return 0


def command_run(args: argparse.Namespace) -> int:
    input_dir = Path(args.input)
    config, config_issues = load_config(args.config or input_dir / "recognition_config.json")
    if config is None:
        report = {"ok": False, "issues": config_issues}
        atomic_write_json(Path(args.output) / "config_errors.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    truth, truth_issues = load_truth(input_dir / "array_truth.json", config)
    store = SnapshotStore(args.output, workspace_root=args.workspace or Path.cwd())
    previous_state = None
    if not args.fresh:
        previous_state, recovery_issues = store.load()
        if recovery_issues:
            truth_issues = [*truth_issues, *recovery_issues]

    ledger, state = run_pipeline(
        str(input_dir),
        config,
        truth_by_image=truth,
        truth_issues=truth_issues,
        config_issues=[],
        previous_state=previous_state,
        budget=args.budget,
        exchange_version=args.exchange_version,
    )
    store.commit(state)
    atomic_write_json(Path(args.output) / "ledger.json", ledger)
    export_judgments(ledger, args.output)
    print(
        f"components={len(ledger['components'])} failures={len(ledger['failures'])} "
        f"pending={len(ledger['pending'])} output={args.output}"
    )
    return 0 if not ledger["pending"] else 3


def command_verify(args: argparse.Namespace) -> int:
    from .diagnostics import run_verification

    report = run_verification(Path(args.workspace), refresh_samples=not args.keep_samples)
    report_path = Path(args.report)
    atomic_write_json(report_path, report)
    markdown_path = report_path.with_suffix(".md")
    markdown_path.write_text(report["markdown"], encoding="utf-8", newline="\n")
    print(f"verification ok={report['ok']} report={report_path}")
    return 0 if report["ok"] else 1


def command_validate_stream(args: argparse.Namespace) -> int:
    commands, issues = load_command_stream(args.stream)
    result = {
        "ok": not issues,
        "command_count": len(commands),
        "issues": issues,
        "commands": commands,
    }
    atomic_write_json(args.output, result) if args.output else None
    print(json.dumps({"ok": not issues, "command_count": len(commands), "issues": issues}, ensure_ascii=False, indent=2))
    return 0 if not issues else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pv-ir-ledger", description="PV infrared component exchange and diagnostic ledger")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-samples")
    generate.add_argument("--root", default="samples")
    generate.set_defaults(func=command_generate)

    run = subparsers.add_parser("run")
    run.add_argument("--input", default="samples")
    run.add_argument("--config")
    run.add_argument("--output", default="runs/current")
    run.add_argument("--workspace", default=".")
    run.add_argument("--budget", type=int)
    run.add_argument("--exchange-version", choices=["v1", "v2"], default="v1")
    run.add_argument("--fresh", action="store_true")
    run.set_defaults(func=command_run)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--workspace", default=".")
    verify.add_argument("--report", default="reports/diagnostic_report.json")
    verify.add_argument("--keep-samples", action="store_true")
    verify.set_defaults(func=command_verify)

    stream = subparsers.add_parser("validate-stream")
    stream.add_argument("--stream", default="samples/command_stream.json")
    stream.add_argument("--output")
    stream.set_defaults(func=command_validate_stream)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
