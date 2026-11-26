"""Command-line entry point for the GSB local demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from gsb.acceptance import run_acceptance
from gsb.commands import load_command_stream, run_command_stream
from gsb.context import issues_payload, load_project
from gsb.errors import GsbError
from gsb.injections import run_injection_suite
from gsb.pipeline import run_pipeline
from gsb.samples import scan_png_directory
from gsb.snapshot import SnapshotStore
from gsb.synth import generate_project
from gsb.trace import analyze_attribution


def _json_dump(path: str | Path | None, payload: Any) -> None:
    if path is None:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _parse_levels(value: str) -> tuple[str, ...]:
    levels = tuple(item.strip() for item in value.split(",") if item.strip())
    if not levels:
        raise GsbError("E_CONFIG_INVALID", "at least one alternate quantization level is required")
    return levels


def _add_project_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default="samples/dataset.yaml")
    parser.add_argument("--templates", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gsb", description="Ancient-page fixed-point demo")
    parser.add_argument("--root", default=".", help="project root")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate config, templates, dataset and PNGs")
    _add_project_paths(validate)
    validate.add_argument("--out", default=None)

    run = commands.add_parser("run", help="run a complete pipeline")
    _add_project_paths(run)
    run.add_argument("--out", default="artifacts/report.json")

    command_stream = commands.add_parser("commands", help="apply an incremental edit stream")
    _add_project_paths(command_stream)
    command_stream.add_argument("--commands", default="configs/commands.jsonl")
    command_stream.add_argument("--out", default="artifacts/commands-report.json")
    command_stream.add_argument("--log-out", default="artifacts/command-log.json")

    trace = commands.add_parser("trace", help="trace fixed-point errors and flip attribution")
    _add_project_paths(trace)
    trace.add_argument("--levels", default="Q4,Q6,Q10")
    trace.add_argument("--out", default="artifacts/attribution.json")

    inject = commands.add_parser("inject", help="run the numbered isolated fault injections")
    _add_project_paths(inject)
    inject.add_argument("--injections", default="injections/injections.yaml")
    inject.add_argument("--out", default="artifacts/injections.json")

    snapshot = commands.add_parser("snapshot", help="commit a versioned stage snapshot")
    _add_project_paths(snapshot)
    snapshot.add_argument("--commands", default=None)
    snapshot.add_argument("--store", default="snapshots/latest")
    snapshot.add_argument("--out", default="artifacts/snapshot.json")

    verify = commands.add_parser("verify", help="run the full acceptance suite")
    verify.add_argument("--out", default="artifacts/validation.json")

    generate = commands.add_parser("generate", help="regenerate deterministic samples and fixtures")
    generate.add_argument("--out", default=None)
    return parser


def _context_from_args(args: argparse.Namespace):
    return load_project(
        root=args.root,
        config_path=args.config,
        dataset_path=args.dataset,
        templates_path=args.templates,
    )


def _command_validate(args: argparse.Namespace) -> int:
    context = _context_from_args(args)
    png_issues = scan_png_directory(Path(args.root).resolve() / "samples", config=context.config)
    payload = {
        "status": "ok"
        if not context.template_issues and not context.dataset_issues and not png_issues
        else "invalid",
        "config_path": str(context.config_path),
        "dataset_path": str(context.dataset_path),
        "templates_path": str(context.templates_path),
        "template_issues": issues_payload(context.template_issues),
        "dataset_issues": issues_payload(context.dataset_issues),
        "png_issues": issues_payload(png_issues),
        "samples": context.dataset.sample_order,
    }
    _json_dump(args.out, payload)
    _print(payload)
    return 0 if payload["status"] == "ok" else 1


def _command_run(args: argparse.Namespace) -> int:
    context = _context_from_args(args)
    result = run_pipeline(
        context.dataset,
        config=context.config,
        templates=context.require_templates(),
        initial_issues=context.dataset_issues,
    )
    _json_dump(args.out, result.report)
    _print(
        {
            "out": str(Path(args.out).resolve()),
            "samples": len(result.report["samples"]),
            "failures": len(result.report["failures"]),
            "pending": len(result.pending_ids),
            "budget": result.report["budget"],
            "quantization": result.report["quantization"],
        }
    )
    return 0 if not result.report["failures"] else 1


def _command_commands(args: argparse.Namespace) -> int:
    context = _context_from_args(args)
    commands, command_issues = load_command_stream(args.commands)
    baseline = run_pipeline(
        context.dataset,
        config=context.config,
        templates=context.require_templates(),
    )
    result, log, _ = run_command_stream(
        dataset=context.dataset,
        config=context.config,
        templates=context.require_templates(),
        commands=commands,
        cache=baseline.cache,
        prior_result=baseline,
    )
    _json_dump(args.out, result.report)
    _json_dump(args.log_out, log)
    _print(
        {
            "out": str(Path(args.out).resolve()),
            "log_out": str(Path(args.log_out).resolve()),
            "command_issues": issues_payload(command_issues),
            "applied": sum(1 for item in log if item["status"] == "applied"),
            "failed": sum(1 for item in log if item["status"] == "failed"),
            "incremental": result.report["incremental"],
        }
    )
    return 0 if not command_issues and all(item["status"] == "applied" for item in log) else 1


def _command_trace(args: argparse.Namespace) -> int:
    context = _context_from_args(args)
    templates = context.require_templates()
    baseline = run_pipeline(
        context.dataset,
        config=context.config,
        templates=templates,
    )
    payload = analyze_attribution(
        context.dataset,
        config=context.config,
        templates=templates,
        base_result=baseline,
        alternate_levels=_parse_levels(args.levels),
    )
    _json_dump(args.out, payload)
    _print(
        {
            "out": str(Path(args.out).resolve()),
            "reference": payload["reference"],
            "flip_count": payload["flip_count"],
            "alternate_levels": payload["alternate_levels"],
        }
    )
    return 0


def _command_inject(args: argparse.Namespace) -> int:
    context = _context_from_args(args)
    payload = run_injection_suite(
        root=args.root,
        config_path=context.config_path,
        dataset_path=context.dataset_path,
        templates_path=context.templates_path,
        injections_path=Path(args.root).resolve() / args.injections,
    )
    _json_dump(args.out, payload)
    _print(
        {
            "out": str(Path(args.out).resolve()),
            "all_hit": payload["all_hit"],
            "injections": [
                {"id": item["id"], "hit": item["hit"], "code": item["code"]}
                for item in payload["injections"]
            ],
        }
    )
    return 0 if payload["all_hit"] else 1


def _command_snapshot(args: argparse.Namespace) -> int:
    context = _context_from_args(args)
    templates = context.require_templates()
    completed_cursor = 0
    if args.commands:
        commands, command_issues = load_command_stream(args.commands)
        if command_issues:
            raise GsbError(
                "E_COMMAND_INVALID",
                "command stream failed validation",
                details={"issues": issues_payload(command_issues)},
            )
        baseline = run_pipeline(context.dataset, config=context.config, templates=templates)
        result, log, _ = run_command_stream(
            dataset=context.dataset,
            config=context.config,
            templates=templates,
            commands=commands,
            cache=baseline.cache,
            prior_result=baseline,
        )
        completed_cursor = max(
            (item["seq"] for item in log if item["status"] == "applied"),
            default=0,
        )
    else:
        result = run_pipeline(context.dataset, config=context.config, templates=templates)
    store = SnapshotStore(Path(args.root).resolve() / args.store)
    commit = store.commit(
        result.report,
        completed_command_cursor=completed_cursor,
        consumed_budget=result.report["budget"]["consumed"],
    )
    payload = {
        "out": str(Path(args.out).resolve()),
        "global_version": commit.global_version,
        "snapshot_dir": str(commit.snapshot_dir),
        "committed": commit.committed,
        "completed_command_cursor": commit.manifest["completed_command_cursor"],
        "consumed_budget": commit.manifest["consumed_budget"],
        "config_hash": commit.manifest["config_hash"],
        "template_version": commit.manifest["template_version"],
    }
    _json_dump(args.out, payload)
    _print(payload)
    return 0


def _command_verify(args: argparse.Namespace) -> int:
    payload = run_acceptance(args.root, output_path=args.out)
    _print({"status": payload["status"], "summary": payload["summary"], "out": str(Path(args.out).resolve())})
    return 0 if payload["status"] == "passed" else 1


def _command_generate(args: argparse.Namespace) -> int:
    payload = generate_project(args.root)
    _json_dump(args.out, payload)
    _print(payload)
    return 0


COMMAND_HANDLERS = {
    "validate": _command_validate,
    "run": _command_run,
    "commands": _command_commands,
    "trace": _command_trace,
    "inject": _command_inject,
    "snapshot": _command_snapshot,
    "verify": _command_verify,
    "generate": _command_generate,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return COMMAND_HANDLERS[args.command](args)
    except (GsbError, OSError, ValueError) as exc:
        if isinstance(exc, GsbError):
            payload = {"status": "error", "error": exc.to_dict()}
        else:
            payload = {
                "status": "error",
                "error": {"code": "E_INTERNAL", "message": str(exc)},
            }
        _print(payload)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
