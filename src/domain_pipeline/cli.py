from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .canonical import fingerprint
from .config import load_config_yaml, validate_config
from .edits import EditPlan, load_edit_plan
from .errors import DomainPipelineError, SnapshotStaleError
from .loader import (
    SampleStore,
    parse_samples,
    state_failures,
)
from .pipeline import (
    PipelineRunner,
    _canonical_dependency,
    _diff_paths,
    canonical_comparison_values,
)
from .report import build_run_report, write_run_report, write_stale_report
from .snapshot import SnapshotStore
from .synthetic import generate_demo_workspace


def _empty_plan() -> EditPlan:
    return EditPlan(
        batches=[],
        failures=[],
        fingerprint=fingerprint("edit-plan:v1", {"batches": []}),
    )


def _json_print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _resolve_existing(path: str | None, default: Path) -> Path:
    candidate = Path(path) if path else default
    if not candidate.is_file():
        raise DomainPipelineError(f"required file does not exist: {candidate}")
    return candidate


def command_generate(args: argparse.Namespace) -> int:
    paths = generate_demo_workspace(
        args.workspace, seed=args.seed, reset=args.reset
    )
    _json_print(
        {
            "ok": True,
            "workspace": str(paths["workspace"]),
            "samples": str(paths["samples"]),
            "config": str(paths["config"]),
            "plan": str(paths["plan"]),
        }
    )
    return 0


def command_run(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    store = SnapshotStore(workspace / "snapshots")
    current_exists = store.has_current()
    if not current_exists and args.expect_version is not None:
        raise SnapshotStaleError(
            "no current snapshot exists for the requested version",
            expected=str(args.expect_version),
            actual=None,
        )

    if current_exists:
        expected_config_hash = None
        if args.config:
            supplied_config = validate_config(
                load_config_yaml(args.config)
            ).config_hash
            expected_config_hash = supplied_config
        store.validate_resume(
            expected_version=args.expect_version,
            expected_config_hash=expected_config_hash,
        )
        runner = PipelineRunner.from_snapshot(store)
        if runner is None:
            raise SnapshotStaleError("CURRENT exists but snapshot state cannot be loaded")
    else:
        samples_path = _resolve_existing(
            args.samples, workspace / "samples" / "samples.jsonl"
        )
        config_path = _resolve_existing(
            args.config, workspace / "configs" / "pipeline.yaml"
        )
        runner = PipelineRunner(
            snapshot_store=store,
            sample_store=SampleStore.from_path(samples_path),
            raw_config=load_config_yaml(str(config_path)),
            pipeline_complete=False,
        )

    if args.plan:
        plan_path = Path(args.plan)
    else:
        default_plan = workspace / "plans" / "edits.json"
        plan_path = default_plan if default_plan.is_file() else None
    plan = load_edit_plan(plan_path) if plan_path is not None else _empty_plan()
    result = runner.run_plan(plan, verify_full=args.verify_full)
    report = build_run_report(
        result,
        mode="incremental",
        plan_fingerprint=plan.fingerprint if plan is not None else None,
    )
    report_path, latest_path = write_run_report(workspace, report)
    summary = report["summary"]
    verify_results = [
        batch.get("full_rebuild", {}).get("equal")
        for batch in report["reports"]
        if "full_rebuild" in batch
    ]
    ok = bool(summary["pipeline_complete"]) and not any(
        value is False for value in verify_results
    )
    _json_print(
        {
            "ok": ok,
            "report": str(report_path),
            "latest": str(latest_path),
            "snapshot": summary.get("snapshot"),
            "batch_count": report["batch_count"],
            "unprocessed_batches": report["unprocessed_batches"],
            "failure_codes": summary["failure_codes"],
            "budget_consumed": summary["budget_consumed"],
            "budget_pending": summary["budget_pending"],
            "invalidations_by_stage": summary["invalidations_by_stage"],
            "reuses_by_stage": summary["reuses_by_stage"],
            "full_rebuild_verified": summary["full_rebuild_verified"],
            "pipeline_complete": summary["pipeline_complete"],
        }
    )
    if not summary["pipeline_complete"]:
        return 3
    if any(value is False for value in verify_results):
        return 1
    return 0


def command_demo(args: argparse.Namespace) -> int:
    paths = generate_demo_workspace(
        args.workspace, seed=args.seed, reset=args.reset
    )
    run_args = argparse.Namespace(
        workspace=str(paths["workspace"]),
        samples=str(paths["samples"]),
        config=str(paths["config"]),
        plan=str(paths["plan"]),
        expect_version=None,
        verify_full=not args.no_verify,
    )
    return command_run(run_args)


def command_restore(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    store = SnapshotStore(workspace / "snapshots")
    manifest = store.restore(args.version)
    _json_print(
        {
            "ok": True,
            "code": "SNAPSHOT_RESTORED",
            "global_version": manifest.get("global_version"),
            "version_id": manifest.get("version_id"),
            "config_hash": manifest.get("config_hash"),
            "data_fingerprint": manifest.get("data_fingerprint"),
        }
    )
    return 0


def command_versions(args: argparse.Namespace) -> int:
    store = SnapshotStore(Path(args.workspace).resolve() / "snapshots")
    versions = [
        {
            "global_version": manifest.get("global_version"),
            "version_id": manifest.get("version_id"),
            "config_hash": manifest.get("config_hash"),
            "data_fingerprint": manifest.get("data_fingerprint"),
            "pipeline_complete": manifest.get("pipeline_complete"),
            "stage_versions": manifest.get("stage_versions"),
        }
        for manifest in store.list_versions()
    ]
    _json_print(
        {
            "ok": True,
            "current": store.current_pointer(),
            "versions": versions,
        }
    )
    return 0


def _snapshot_comparison(workspace: Path) -> dict[str, Any]:
    store = SnapshotStore(workspace / "snapshots")
    data = store.load_current()
    if data is None:
        raise DomainPipelineError(f"workspace has no current snapshot: {workspace}")
    config = validate_config(data.raw_config)
    parsed = parse_samples(data.sample_lines, config)
    failures = state_failures(config.failures, parsed.failures)
    canonical = canonical_comparison_values(
        data.stages,
        failures,
        data.ledger,
        pipeline_complete=data.pipeline_complete,
    )
    return {
        "manifest": data.manifest,
        "canonical": canonical,
        "stage_dependency": _canonical_dependency(data.ledger),
    }


def command_compare(args: argparse.Namespace) -> int:
    left = _snapshot_comparison(Path(args.left).resolve())
    right = _snapshot_comparison(Path(args.right).resolve())
    diffs = _diff_paths(left["canonical"], right["canonical"])
    equal = not diffs
    _json_print(
        {
            "ok": equal,
            "equal": equal,
            "diffs": diffs[:100],
            "left": {
                "version": left["manifest"].get("global_version"),
                "config_hash": left["manifest"].get("config_hash"),
                "data_fingerprint": left["manifest"].get("data_fingerprint"),
                "stage_versions": left["manifest"].get("stage_versions"),
            },
            "right": {
                "version": right["manifest"].get("global_version"),
                "config_hash": right["manifest"].get("config_hash"),
                "data_fingerprint": right["manifest"].get("data_fingerprint"),
                "stage_versions": right["manifest"].get("stage_versions"),
            },
        }
    )
    return 0 if equal else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="domain-pipeline",
        description="Deterministic incremental feature/train/inference pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic demo inputs")
    generate.add_argument("--workspace", default="demo_workspace")
    generate.add_argument("--seed", type=int, default=202604)
    generate.add_argument("--reset", action="store_true")
    generate.set_defaults(func=command_generate)

    run = subparsers.add_parser("run", help="run or resume the incremental pipeline")
    run.add_argument("--workspace", default="demo_workspace")
    run.add_argument("--samples")
    run.add_argument("--config")
    run.add_argument("--plan")
    run.add_argument("--expect-version", type=int)
    run.add_argument("--verify-full", action="store_true")
    run.set_defaults(func=command_run)

    demo = subparsers.add_parser("demo", help="generate inputs, replay edits, and verify")
    demo.add_argument("--workspace", default="demo_workspace")
    demo.add_argument("--seed", type=int, default=202604)
    demo.add_argument("--reset", action="store_true")
    demo.add_argument("--no-verify", action="store_true")
    demo.set_defaults(func=command_demo)

    restore = subparsers.add_parser("restore", help="atomically switch CURRENT snapshot")
    restore.add_argument("--workspace", default="demo_workspace")
    restore.add_argument("--version", required=True)
    restore.set_defaults(func=command_restore)

    versions = subparsers.add_parser("versions", help="list immutable snapshot versions")
    versions.add_argument("--workspace", default="demo_workspace")
    versions.set_defaults(func=command_versions)

    compare = subparsers.add_parser("compare", help="compare two current snapshots")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.set_defaults(func=command_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    workspace = Path(getattr(args, "workspace", "demo_workspace")).resolve()
    try:
        version = getattr(args, "version", None)
        if version is not None:
            try:
                args.version = int(version)
            except ValueError:
                pass
        return int(args.func(args))
    except SnapshotStaleError as exc:
        failure = exc.as_failure()
        path = write_stale_report(workspace, failure)
        _json_print({"ok": False, "code": exc.code, "failure": failure, "report": str(path)})
        return 2
    except DomainPipelineError as exc:
        _json_print({"ok": False, "code": exc.code, "message": str(exc)})
        return 1
    except Exception as exc:
        _json_print(
            {
                "ok": False,
                "code": "E_INTERNAL",
                "message": f"{type(exc).__name__}: {exc}",
            }
        )
        return 1
