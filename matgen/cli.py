"""Command-line entry points for sample generation and verified runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .cache import CacheStore
from .diagnostics import build_report, write_build_artifacts, write_report
from .engine import MaterialEngine
from .errors import E_SNAPSHOT_VERSION, Issue, make_issue
from .injection import InjectionRegistry, load_injection_config
from .interactive import InteractiveRunner
from .loader import input_fingerprint, load_input
from .snapshot import SnapshotManager, phase_hashes_from_result
from .trace import build_error_trace, compare_full_rebuild


def _failure_report(loaded: Any, issues: list[Issue]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "report_type": "texture_material_generation_diagnostic",
        "status": "failed",
        "input": {
            "directory": getattr(loaded, "input_dir", ""),
            "textures_loaded": sorted(getattr(loaded, "textures", {})),
        },
        "codes": sorted({issue.code for issue in issues}),
        "issues": [issue.as_dict() for issue in issues],
        "artifacts": {},
    }


def _snapshot_public_manifest(manifest: dict[str, Any] | None) -> dict[str, Any] | None:
    if manifest is None:
        return None
    return {
        "schema_version": manifest.get("schema_version"),
        "global_version": manifest.get("global_version"),
        "input_config_hash": manifest.get("input_config_hash"),
        "config_hash": manifest.get("config_hash"),
        "commands_hash": manifest.get("commands_hash"),
        "quantization": manifest.get("quantization"),
        "command_cursor": manifest.get("command_cursor"),
        "completed_batches_cursor": manifest.get("completed_batches_cursor"),
        "phases": manifest.get("phases"),
        "artifact_versions": {
            kind: {
                "count": len(entries),
                "versions": [entry.get("version") for entry in entries],
            }
            for kind, entries in sorted(manifest.get("artifact_versions", {}).items())
        },
        "state_hash": manifest.get("state_hash"),
        "created_at_utc": manifest.get("created_at_utc"),
    }


def run_sample(args: argparse.Namespace) -> int:
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "diagnostic_report.json"
    loaded = load_input(args.input)
    all_issues: list[Issue] = list(loaded.issues)
    injections, injection_issues = load_injection_config(args.injections)
    all_issues.extend(injection_issues)
    registry = InjectionRegistry(injections)
    for issue in loaded.issues:
        registry.record_observed(issue.code, target=issue.item, stage="load")

    if loaded.config is None or not loaded.materials:
        report = _failure_report(loaded, all_issues)
        write_report(report_path, report)
        print(json.dumps({"status": "failed", "report": str(report_path), "codes": report["codes"]}))
        return 2

    snapshot_dir = Path(args.snapshot_dir).resolve() if args.snapshot_dir else output_dir / "snapshots"
    try:
        snapshot_manager = SnapshotManager(snapshot_dir)
    except Exception as exc:
        all_issues.append(make_issue("E_SNAPSHOT_IO", f"cannot initialize snapshot manager: {exc}", source=str(snapshot_dir)))
        snapshot_manager = None
    if snapshot_manager is not None:
        all_issues.extend(snapshot_manager.recovery_issues)

    cache = CacheStore()
    engine = MaterialEngine(cache=cache, injections=registry)
    input_hash = input_fingerprint(loaded)
    runner = InteractiveRunner(
        engine=engine,
        textures=loaded.textures,
        input_config_hash=input_hash,
        commands=loaded.commands,
        snapshot_manager=snapshot_manager,
        resume=not args.fresh,
        expected_snapshot_version=args.expected_snapshot_version,
    )
    interactive = runner.run(
        loaded.config,
        loaded.materials,
        stop_after=args.stop_after,
        commit_final_snapshot=False,
    )
    for issue in interactive["resume"].get("issues", []):
        if issue.get("code") == E_SNAPSHOT_VERSION:
            registry.record_code_for_action(
                E_SNAPSHOT_VERSION,
                action="snapshot_hash_mismatch",
                stage="snapshot",
            )
            all_issues.append(
                Issue(
                    code=E_SNAPSHOT_VERSION,
                    message=str(issue.get("message", "snapshot version mismatch")),
                    source=str(issue.get("source", snapshot_dir)),
                    item="snapshot",
                    details=dict(issue.get("details", {})),
                )
            )

    final_result = interactive.get("final_result")
    error_trace = None
    error_diffs = None
    full_rebuild = None
    if final_result is not None:
        try:
            error_trace, error_diffs, _ = build_error_trace(
                config=interactive["config"],
                materials=interactive["materials"],
                textures=loaded.textures,
                low_result=final_result,
                cache=cache,
                injections=registry,
            )
        except Exception as exc:
            all_issues.append(make_issue("E_INTERNAL", f"error trace failed: {exc}", source="trace"))
        try:
            full_rebuild = compare_full_rebuild(
                incremental=final_result,
                config=interactive["config"],
                materials=interactive["materials"],
                textures=loaded.textures,
                injections=registry,
            )
        except Exception as exc:
            all_issues.append(make_issue("E_INTERNAL", f"full rebuild verification failed: {exc}", source="verification"))

    artifact_paths = write_build_artifacts(
        output_dir,
        result=final_result,
        error_diffs=error_diffs,
    ) if final_result is not None else {}

    final_snapshot = None
    if snapshot_manager is not None and final_result is not None:
        try:
            final_snapshot = snapshot_manager.commit(
                cache=cache,
                config=interactive["config"],
                input_config_hash=input_hash,
                commands_hash=runner.commands_hash,
                command_cursor=interactive["commands_started_at"] + interactive["commands_processed"],
                completed_batches_cursor=final_result.batches_consumed,
                phase_hashes=phase_hashes_from_result(final_result),
                ledger=final_result.ledger_entries,
            )
        except Exception as exc:
            final_snapshot = {"status": "error", "message": str(exc)}
            all_issues.append(make_issue("E_SNAPSHOT_IO", f"final snapshot failed: {exc}", source=str(snapshot_dir)))

    manifest = snapshot_manager.load_manifest() if snapshot_manager is not None else None
    snapshot_summary = {
        "directory": str(snapshot_dir),
        "resume": interactive["resume"],
        "commit_interval_commands": interactive["config"].snapshot_commit_interval,
        "worker_interval_commit": interactive.get("final_snapshot"),
        "final_commit": final_snapshot,
        "manifest": _snapshot_public_manifest(manifest),
    }
    injection_report = registry.report()
    report = build_report(
        loaded=loaded,
        interactive=interactive,
        error_trace=error_trace,
        full_rebuild=full_rebuild,
        injection_report=injection_report,
        snapshot_summary=snapshot_summary,
        artifact_paths=artifact_paths,
        extra_issues=all_issues,
        input_config_hash=input_hash,
    )
    report["artifacts"]["diagnostic_report"] = str(report_path)
    write_report(report_path, report)
    summary = {
        "status": report["status"],
        "report": str(report_path),
        "final_digest": final_result.final_digest if final_result is not None else None,
        "stale_responses": interactive.get("stale_responses"),
        "within_budget_responses": interactive.get("within_budget_responses"),
        "full_rebuild_exact": full_rebuild.get("exact_match") if full_rebuild else None,
        "error_trace_attribution": error_trace.get("attribution", {}).get("category") if error_trace else None,
        "snapshot_version": manifest.get("global_version") if manifest else None,
        "codes": report["codes"],
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if report["status"] in {"ok", "partial"} else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic texture and material generation component")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-samples", help="regenerate deterministic local samples")
    generate.add_argument("--root", default=str(Path(__file__).resolve().parents[1] / "samples"))

    run = subparsers.add_parser("run", help="run the interactive sample and write a diagnostic report")
    run.add_argument("--input", required=True, help="sample directory")
    run.add_argument("--output", required=True, help="output directory for reports and artifacts")
    run.add_argument("--snapshot-dir", default=None, help="snapshot directory (default: OUTPUT/snapshots)")
    run.add_argument("--injections", default=None, help="optional injection config JSON")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--fresh", action="store_true", help="ignore an existing snapshot and rebuild from blank")
    mode.add_argument("--resume", action="store_true", help="resume a compatible snapshot (default)")
    run.add_argument("--expected-snapshot-version", type=int, default=None)
    run.add_argument("--stop-after", type=int, default=None, help="process only the first N commands")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate-samples":
        import scripts.generate_samples as sample_generator

        sample_generator.main(Path(args.root))
        return 0
    if args.command == "run":
        return run_sample(args)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
