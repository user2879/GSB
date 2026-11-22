from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .config import (
    default_config_dict,
    load_config,
)
from .errors import E_REPLAY, SlowDiagError, source_error
from .generator import check_repeatable_generation, generate_all
from .logging_utils import StructuredLogger, read_jsonl
from .pipeline import make_format_conflict_source, run_pipeline
from .snapshots import create_snapshot, restore_snapshot
from .storage import Ledger


DEFAULT_WORKSPACE = "workbench"


def _workspace(value: str) -> Path:
    return Path(value).resolve(strict=False)


def _config_path(args: argparse.Namespace, workspace: Path) -> Path:
    if getattr(args, "config", None):
        return Path(args.config).resolve(strict=False)
    return workspace / "slow-sources.json"


def _json_output(value: dict[str, Any], pretty: bool) -> None:
    if pretty:
        text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        text = canonical_json(value)
    sys.stdout.write(text + "\n")


def _init_workspace(workspace: Path, force: bool) -> dict[str, Any]:
    workspace.mkdir(parents=True, exist_ok=True)
    for name in ("sources", "ledger", "snapshots", "logs", "reports", "_injection"):
        (workspace / name).mkdir(parents=True, exist_ok=True)
    config_path = workspace / "slow-sources.json"
    wrote_config = False
    if force or not config_path.exists():
        config_path.write_text(
            json.dumps(default_config_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        wrote_config = True
    config = load_config(config_path, workspace)
    for source in config.sources:
        source.resolved_data_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(workspace)
    ledger.initialize()
    output = {
        "status": "initialized",
        "workspace": str(workspace),
        "config": str(config_path),
        "config_written": wrote_config,
        "source_directories": [
            str(source.resolved_data_path.parent) for source in config.sorted_sources()
        ],
        "ledger": str(ledger.path),
        "snapshots": str(workspace / "snapshots"),
        "logs": str(workspace / "logs" / "slowdiag.jsonl"),
    }
    return output


def _summarize_view(
    view: dict[str, Any], view_sha: str | None = None
) -> dict[str, Any]:
    queries = view.get("queries", [])
    slow_queries = [query["query_id"] for query in queries if query["diagnosis"]["slow"]]
    migration_counts = view.get("migration", {}).get("counts", {})
    attribution_counts = view.get("attribution", {}).get("counts", {})
    invalidation = view.get("invalidation", {})
    summary = {
        "business_code": "OK",
        "schema": view.get("schema"),
        "plan_version": view.get("plan_version"),
        "query_count": len(queries),
        "slow_query_count": len(slow_queries),
        "slow_queries": slow_queries,
        "sources": [
            {
                "source_id": item["source_id"],
                "source_type": item["source_type"],
                "format_versions": item["format_versions"],
                "content_sha256": item["content_sha256"],
                "sample_period_ms": item["sample_period_ms"],
            }
            for item in view.get("source_summaries", [])
        ],
        "migration_counts": migration_counts,
        "attribution_counts": attribution_counts,
        "invalidation": {
            "changed_sources": invalidation.get("changed_sources", []),
            "affected_queries": invalidation.get("affected_queries", []),
            "stale_query_count": invalidation.get("stale_query_count", 0),
            "unaffected_byte_identical": invalidation.get(
                "unaffected_byte_identical", True
            ),
            "replacement_mode": invalidation.get("replacement_mode"),
        },
    }
    if view_sha is not None:
        summary["view_sha256"] = view_sha
    return summary


def _write_report(workspace: Path, view: dict[str, Any]) -> Path:
    reports = workspace / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "latest.json"
    temp = path.with_suffix(".json.tmp")
    temp.write_text(canonical_json(view) + "\n", encoding="utf-8")
    temp.replace(path)
    return path


def _prepare_injection(
    args: argparse.Namespace,
    workspace: Path,
    config_path: Path,
    logger: StructuredLogger,
) -> tuple[Any, bool, dict[str, Any]]:
    kind = getattr(args, "inject", "none")
    if kind in (None, "none"):
        return load_config(config_path, workspace), False, {"kind": "none"}
    logger.log("injection_prepared", details={"kind": kind})
    if kind == "source":
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            raw = {"sources": raw}
        sources = raw.get("sources")
        if not isinstance(sources, list) or not sources:
            raise source_error("injection requires at least one source", "$.sources")
        sources[0]["type"] = "injected_unknown_source_type"
        target = workspace / "_injection" / "source-damaged.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        logger.log(
            "injection_input_written",
            details={"kind": kind, "path": str(target), "field": "sources[0].type"},
        )
        return load_config(target, workspace), False, {
            "kind": kind,
            "path": str(target),
        }
    if kind == "format":
        config = load_config(config_path, workspace)
        if not config.sources:
            raise source_error("injection requires at least one source", "$.sources")
        sources = list(config.sources)
        sources[0] = make_format_conflict_source(workspace, sources[0])
        injected = replace(config, sources=tuple(sources))
        logger.log(
            "injection_input_written",
            details={
                "kind": kind,
                "source_id": sources[0].id,
                "path": str(sources[0].resolved_data_path),
            },
        )
        return injected, False, {
            "kind": kind,
            "source_id": sources[0].id,
            "path": str(sources[0].resolved_data_path),
        }
    if kind == "empty":
        config = load_config(config_path, workspace)
        return config, True, {"kind": kind, "parameter": "empty_after_merge=true"}
    raise source_error(f"unsupported injection kind: {kind}", "--inject")


def cmd_init(args: argparse.Namespace, logger: StructuredLogger) -> dict[str, Any]:
    workspace = _workspace(args.workspace)
    result = _init_workspace(workspace, args.force)
    logger.log("workspace_initialized", details=result)
    return result


def cmd_generate(args: argparse.Namespace, logger: StructuredLogger) -> dict[str, Any]:
    workspace = _workspace(args.workspace)
    config_path = _config_path(args, workspace)
    config = load_config(config_path, workspace)
    summaries = generate_all(config)
    repeatability = None
    if args.check_repeat:
        repeatability = check_repeatable_generation(config)
        if not repeatability["repeatable"]:
            raise SlowDiagError(
                "E_FORMAT",
                "deterministic sample generation produced different content digests",
                details=repeatability,
            )
    result = {
        "status": "generated",
        "business_code": "OK",
        "config": str(config_path),
        "sources": summaries,
        "repeatability": repeatability,
    }
    logger.log("samples_generated", details=result)
    return result


def cmd_run(args: argparse.Namespace, logger: StructuredLogger) -> dict[str, Any]:
    workspace = _workspace(args.workspace)
    config_path = _config_path(args, workspace)
    config, empty_after_merge, injection = _prepare_injection(
        args, workspace, config_path, logger
    )
    ledger = Ledger(workspace)
    result = run_pipeline(
        config,
        ledger,
        empty_after_merge=empty_after_merge,
    )
    report_path = _write_report(workspace, result.view)
    output = _summarize_view(result.view, result.view_sha256)
    output.update(
        {
            "status": "completed",
            "report": str(report_path),
            "ledger": str(ledger.path),
            "timing": result.timings,
            "injection": injection,
        }
    )
    logger.log("diagnostic_run_completed", details=output)
    return output


def cmd_snapshot(args: argparse.Namespace, logger: StructuredLogger) -> dict[str, Any]:
    workspace = _workspace(args.workspace)
    config_path = _config_path(args, workspace)
    config = load_config(config_path, workspace)
    ledger = Ledger(workspace)
    current = ledger.get_current_view()
    if current is None:
        raise SlowDiagError(
            E_REPLAY,
            "no current view is available; run the diagnostic pipeline first",
            field="current_view",
        )
    path = create_snapshot(workspace, current["view"], config, args.name)
    result = {
        "status": "snapshot_created",
        "business_code": "OK",
        "snapshot": str(path),
        "plan_version": current["plan_version"],
        "view_sha256": current["view_sha256"],
    }
    logger.log("snapshot_created", details=result)
    return result


def cmd_restore(args: argparse.Namespace, logger: StructuredLogger) -> dict[str, Any]:
    workspace = _workspace(args.workspace)
    ledger = Ledger(workspace)
    result = restore_snapshot(ledger, Path(args.snapshot), workspace)
    logger.log("snapshot_restored", details=result)
    return result


def cmd_show(args: argparse.Namespace, logger: StructuredLogger) -> dict[str, Any]:
    workspace = _workspace(args.workspace)
    ledger = Ledger(workspace)
    current = ledger.get_current_view()
    if current is None:
        raise SlowDiagError(
            E_REPLAY,
            "no current view is available; run the diagnostic pipeline first",
            field="current_view",
        )
    if args.full:
        return {
            "status": "current_view",
            "business_code": "OK",
            "view_sha256": current["view_sha256"],
            "view": current["view"],
        }
    result = _summarize_view(current["view"], current["view_sha256"])
    result["status"] = "current_view"
    logger.log("current_view_read", details={"plan_version": current["plan_version"]})
    return result


def cmd_events(args: argparse.Namespace, logger: StructuredLogger) -> dict[str, Any]:
    workspace = _workspace(args.workspace)
    path = workspace / "logs" / "slowdiag.jsonl"
    events = read_jsonl(path, limit=args.limit)
    return {
        "status": "events",
        "business_code": "OK",
        "log": str(path),
        "events": events,
    }


def cmd_demo(args: argparse.Namespace, logger: StructuredLogger) -> dict[str, Any]:
    workspace = _workspace(args.workspace)
    init_result = _init_workspace(workspace, args.force)
    config_path = _config_path(args, workspace)
    config = load_config(config_path, workspace)
    source_summaries = generate_all(config)
    repeatability = check_repeatable_generation(config)
    if not repeatability["repeatable"]:
        raise SlowDiagError(
            "E_FORMAT",
            "deterministic sample generation is not repeatable",
            details=repeatability,
        )
    ledger = Ledger(workspace)
    pipeline_result = run_pipeline(config, ledger)
    report_path = _write_report(workspace, pipeline_result.view)
    snapshot_path = create_snapshot(
        workspace, pipeline_result.view, config, args.name
    )
    restore_result = restore_snapshot(ledger, snapshot_path, workspace)
    output = _summarize_view(pipeline_result.view, pipeline_result.view_sha256)
    output.update(
        {
            "status": "demo_completed",
            "workspace": str(workspace),
            "config": str(config_path),
            "init": init_result,
            "sources": source_summaries,
            "repeatability": repeatability,
            "report": str(report_path),
            "snapshot": str(snapshot_path),
            "restore": restore_result,
            "timing": pipeline_result.timings,
        }
    )
    logger.log("demo_completed", details=output)
    return output


def cmd_gui(args: argparse.Namespace, logger: StructuredLogger) -> dict[str, Any]:
    workspace = _workspace(args.workspace)
    config_path = _config_path(args, workspace)
    from .gui import run_gui

    logger.log("gui_started", details={"workspace": str(workspace)})
    run_gui(workspace, config_path)
    return {
        "status": "gui_closed",
        "business_code": "OK",
        "workspace": str(workspace),
    }


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    common.add_argument("--config", default=None)
    common.add_argument("--pretty", action="store_true")

    parser = argparse.ArgumentParser(
        prog="slowdiag",
        description="Deterministic multi-source slow-query evidence workbench",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", parents=[common])
    init_parser.add_argument("--force", action="store_true")
    init_parser.set_defaults(handler=cmd_init)

    generate_parser = subparsers.add_parser("generate", parents=[common])
    generate_parser.add_argument("--check-repeat", action="store_true")
    generate_parser.set_defaults(handler=cmd_generate)

    run_parser = subparsers.add_parser("run", parents=[common])
    run_parser.add_argument(
        "--inject",
        choices=("none", "source", "format", "empty"),
        default="none",
        help="inject E_SOURCE, E_FORMAT, or E_EMPTY through input files/parameters",
    )
    run_parser.set_defaults(handler=cmd_run)

    snapshot_parser = subparsers.add_parser("snapshot", parents=[common])
    snapshot_parser.add_argument("--name", default=None)
    snapshot_parser.set_defaults(handler=cmd_snapshot)

    restore_parser = subparsers.add_parser("restore", parents=[common])
    restore_parser.add_argument("--snapshot", required=True)
    restore_parser.set_defaults(handler=cmd_restore)

    show_parser = subparsers.add_parser("show", parents=[common])
    show_parser.add_argument("--full", action="store_true")
    show_parser.set_defaults(handler=cmd_show)

    events_parser = subparsers.add_parser("events", parents=[common])
    events_parser.add_argument("--limit", type=int, default=20)
    events_parser.set_defaults(handler=cmd_events)

    demo_parser = subparsers.add_parser("demo", parents=[common])
    demo_parser.add_argument("--force", action="store_true")
    demo_parser.add_argument("--name", default="demo-session.json")
    demo_parser.set_defaults(handler=cmd_demo)

    gui_parser = subparsers.add_parser("gui", parents=[common])
    gui_parser.set_defaults(handler=cmd_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    workspace = _workspace(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    logger = StructuredLogger(workspace)
    logger.log(
        "command_started",
        details={"command": args.command, "workspace": str(workspace)},
    )
    try:
        result = args.handler(args, logger)
        logger.log(
            "command_completed",
            details={
                "command": args.command,
                "status": result.get("status"),
                "business_code": result.get("business_code", "OK"),
            },
        )
        _json_output(result, args.pretty)
        return 0
    except SlowDiagError as exc:
        logger.log(
            "command_error",
            code=exc.code,
            severity="error",
            message=exc.message,
            details=exc.to_dict(),
        )
        output = {
            "status": "error",
            "business_code": exc.code,
            **exc.to_dict(),
            "log_file": str(logger.path),
        }
        _json_output(output, args.pretty)
        return 3 if exc.code == E_REPLAY else 2
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        logger.log(
            "command_error",
            code="E_INTERNAL",
            severity="critical",
            message=str(exc),
            details={"traceback": traceback.format_exc(limit=12)},
        )
        _json_output(
            {
                "status": "error",
                "business_code": "E_INTERNAL",
                "message": str(exc),
                "log_file": str(logger.path),
            },
            args.pretty,
        )
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
