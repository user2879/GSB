"""Command-line interface for the deterministic timbre workbench."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from .config import load_config, load_labels
from .engine import Workbench, load_or_initialize_workbench, semantic_projection, write_report_files
from .errors import WorkbenchError
from .synth import generate_demo
from .util import atomic_write_json, canonical_json, sha256_obj


def _load_commands(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    commands = payload.get("commands", payload if isinstance(payload, list) else None)
    if not isinstance(commands, list):
        raise ValueError("command file must contain a commands array")
    return commands


def _json_print(value: Any) -> None:
    sys.stdout.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def command_demo(args: argparse.Namespace) -> int:
    info = generate_demo(Path(args.out))
    _json_print(info)
    return 0


def command_run(args: argparse.Namespace) -> int:
    input_dir = Path(args.input).resolve()
    config = load_config(Path(args.config).resolve() if args.config else None)
    labels = load_labels(Path(args.labels).resolve() if args.labels else None)
    commands_path = Path(args.commands).resolve() if args.commands else None
    commands = _load_commands(commands_path) if commands_path else []
    work_dir = Path(args.work).resolve()
    report_path = Path(args.report).resolve()
    html_path = None if args.no_html else Path(args.html or report_path.with_suffix(".html")).resolve()
    responses_path = Path(args.responses).resolve() if args.responses else report_path.with_name("responses.json")

    workbench, warnings = load_or_initialize_workbench(
        input_dir=input_dir,
        config=config,
        labels=labels,
        commands=commands,
        work_dir=work_dir,
        resume=bool(args.resume),
        expected_global_version=args.expected_version,
        command_base_dir=commands_path.parent if commands_path else input_dir,
    )
    responses = workbench.run_commands(
        commands,
        command_base_dir=commands_path.parent if commands_path else input_dir,
    )
    report = workbench.refresh_report()
    if (
        workbench.snapshot_manager is not None
        and not config.get("runtime", {}).get("snapshot_after_each_command", True)
        and not responses
    ):
        snapshot_info = workbench._commit_snapshot(report)
        report["diagnostics"]["snapshot"] = snapshot_info
        workbench.last_result = report
    write_report_files(report, report_path, html_path)
    atomic_write_json(responses_path, {"schema_version": 1, "responses": responses})

    pending_count = sum(1 for response in responses if response.get("status") == "pending")
    output = {
        "status": "ok",
        "report": str(report_path),
        "html": str(html_path) if html_path else None,
        "responses": str(responses_path),
        "command_count": len(commands),
        "commands_executed_this_invocation": len(responses),
        "pending_first_responses": pending_count,
        "canonical_digest": report.get("canonical_digest"),
        "summary": report.get("summary"),
        "snapshot_warnings": warnings,
    }
    _json_print(output)
    return 0


def command_compare(args: argparse.Namespace) -> int:
    input_dir = Path(args.input).resolve()
    config = load_config(Path(args.config).resolve() if args.config else None)
    labels = load_labels(Path(args.labels).resolve() if args.labels else None)
    commands_path = Path(args.commands).resolve() if args.commands else None
    commands = _load_commands(commands_path) if commands_path else []
    command_base_dir = commands_path.parent if commands_path else input_dir

    with tempfile.TemporaryDirectory(prefix="timbre-compare-incremental-") as temp_a, tempfile.TemporaryDirectory(
        prefix="timbre-compare-full-"
    ) as temp_b:
        incremental = Workbench.from_directory(
            input_dir,
            config=config,
            labels=labels,
            snapshot_manager=None,
            command_base_dir=command_base_dir,
        )
        incremental.run_commands(commands, command_base_dir=command_base_dir)
        incremental_report = incremental.refresh_report()

        full = Workbench.from_directory(
            input_dir,
            config=config,
            labels=labels,
            snapshot_manager=None,
            command_base_dir=command_base_dir,
        )
        full.run_commands(commands, command_base_dir=command_base_dir, force_full=True)
        full_report = full.refresh_report()

    projection_a = semantic_projection(incremental_report)
    projection_b = semantic_projection(full_report)
    equivalent = canonical_json(projection_a) == canonical_json(projection_b)
    differences: list[dict[str, Any]] = []
    if not equivalent:
        for key in sorted(set(projection_a) | set(projection_b)):
            if canonical_json(projection_a.get(key)) != canonical_json(projection_b.get(key)):
                differences.append({"field": key, "incremental": projection_a.get(key), "full": projection_b.get(key)})
    comparison = {
        "schema_version": 1,
        "equivalent": equivalent,
        "incremental_digest": incremental_report.get("canonical_digest"),
        "full_digest": full_report.get("canonical_digest"),
        "semantic_digest": sha256_obj(projection_a),
        "differences": differences,
        "incremental_execution": {
            "cache_events": incremental_report.get("diagnostics", {}).get("cache_events", {}),
        },
        "full_execution": {
            "cache_events": full_report.get("diagnostics", {}).get("cache_events", {}),
        },
    }
    output_path = Path(args.output).resolve() if args.output else None
    if output_path:
        atomic_write_json(output_path, comparison)
    _json_print(comparison)
    return 0 if equivalent else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="timbre-workbench", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="generate the deterministic demo workspace")
    demo.add_argument("--out", default="examples/demo")
    demo.set_defaults(handler=command_demo)

    run = subparsers.add_parser("run", help="execute a command stream incrementally")
    run.add_argument("--input", required=True, help="audio directory")
    run.add_argument("--config", required=True, help="analysis configuration JSON")
    run.add_argument("--labels", help="speaker annotation JSON")
    run.add_argument("--commands", help="command stream JSON")
    run.add_argument("--work", default=".timbre-work", help="snapshot and temporary work root")
    run.add_argument("--report", default="timbre-report.json")
    run.add_argument("--html", help="HTML diagnostics path")
    run.add_argument("--responses", help="per-command response log JSON")
    run.add_argument("--no-html", action="store_true")
    run.add_argument("--resume", action="store_true", help="resume a validated snapshot")
    run.add_argument("--expected-version", type=int, help="require this snapshot global version")
    run.set_defaults(handler=command_run)

    compare = subparsers.add_parser("compare", help="prove incremental/full semantic equality")
    compare.add_argument("--input", required=True)
    compare.add_argument("--config", required=True)
    compare.add_argument("--labels")
    compare.add_argument("--commands")
    compare.add_argument("--output")
    compare.set_defaults(handler=command_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except WorkbenchError as exc:
        sys.stderr.write(canonical_json(exc.as_failure()) + "\n")
        return 2
    except (OSError, ValueError, KeyError) as exc:
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
