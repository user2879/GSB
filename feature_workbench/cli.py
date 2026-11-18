from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .canonical import canonical_json, fingerprint
from .engine import Registry, Workbench
from .models import Event, FeatureDef, ScoreConfig, failure, failure_sort_key
from .snapshot import SCHEMA_VERSION, SnapshotMismatch, SnapshotStore, definitions_hash
from .validation import (
    FaultPlan,
    InjectionBundle,
    build_registry,
    load_config,
    load_definitions,
    load_events,
    load_injection_bundle,
    load_plan,
    parse_feature_definition,
    validate_event_payload,
)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _manifest_summary(manifest: dict[str, Any] | None) -> dict[str, Any] | None:
    if manifest is None:
        return None
    return {
        "global_version": manifest.get("global_version"),
        "config_hash": manifest.get("config_hash"),
        "definitions_hash": manifest.get("definitions_hash"),
        "base_definitions_hash": manifest.get("base_definitions_hash"),
        "base_events_hash": manifest.get("base_events_hash"),
        "event_cursor": manifest.get("event_cursor"),
        "applied_batches": manifest.get("applied_batches"),
        "snapshot_dir": manifest.get("snapshot_dir"),
        "artifact_versions": manifest.get("artifact_versions"),
        "status": manifest.get("status"),
    }


def _current_definitions_hash(workbench: Workbench) -> str:
    return definitions_hash(
        [workbench.registry.definitions[name].raw for name in workbench.registry.topo]
    )


def _new_workbench(
    *,
    config: ScoreConfig,
    definition_result: Any,
    event_result: Any,
    faults: FaultPlan,
) -> Workbench:
    return Workbench(
        config=config,
        definitions=definition_result.definitions,
        topo=definition_result.topo,
        downstream=definition_result.downstream,
        events=event_result.events.values(),
        load_failures=definition_result.failures + event_result.failures,
        faults=faults,
    )


def _resume_mismatches(
    manifest: dict[str, Any],
    *,
    config: ScoreConfig,
    definition_result: Any,
    event_result: Any,
    plan_hash: str,
    batch_count: int,
) -> list[str]:
    mismatches: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        mismatches.append("schema_version")
    version = manifest.get("global_version")
    if not isinstance(version, int) or version < 1:
        mismatches.append("global_version")
    if manifest.get("config_hash") != config.config_fingerprint:
        mismatches.append("config_hash")
    if manifest.get("base_definitions_hash") != definition_result.source_hash:
        mismatches.append("base_definitions_hash")
    if manifest.get("base_events_hash") != event_result.source_hash:
        mismatches.append("base_events_hash")
    if manifest.get("plan_hash") != plan_hash:
        mismatches.append("plan_hash")
    applied = manifest.get("applied_batches")
    if not isinstance(applied, int) or applied < 0 or applied > batch_count:
        mismatches.append("event_cursor_or_batch_progress")
    return mismatches


def execute_incremental(
    *,
    definitions_path: Path,
    events_path: Path,
    config_path: Path,
    edits_path: Path | None,
    state_dir: Path,
    report_path: Path,
    injection_config_path: Path | None = None,
    batch_delay_ms: int = 0,
    stop_requested: Callable[[], bool] | None = None,
) -> tuple[dict[str, Any], Workbench | None, ScoreConfig | None]:
    stop_requested = stop_requested or (lambda: False)
    config, config_failure, _ = load_config(config_path)
    if config is None or config_failure is not None:
        report = {
            "status": "failed",
            "mode": "incremental",
            "exit_code": 2,
            "failures": [config_failure] if config_failure else [],
            "features": [],
            "scores": [],
            "pending": [],
        }
        _atomic_write_json(report_path, report)
        return report, None, None

    injection: InjectionBundle = load_injection_bundle(injection_config_path)
    definition_result = load_definitions(definitions_path)
    event_result = load_events(events_path, config.allowed_behaviors)
    if edits_path is None:
        plan, plan_hash, plan_failure = {"batches": []}, fingerprint("edit-plan-v1", {}), None
    else:
        plan, plan_hash, plan_failure = load_plan(edits_path)
    batches = plan.get("batches", []) if isinstance(plan.get("batches", []), list) else []

    store = SnapshotStore(state_dir)
    manifest: dict[str, Any] | None = None
    startup_failures: list[dict[str, Any]] = []
    resumed = False
    applied_batches = 0
    global_version = 0
    try:
        manifest = store.read_manifest()
    except SnapshotMismatch as exc:
        startup_failures.append(
            failure(
                code="E_CONFIG_MISMATCH",
                stage="snapshot_resume",
                message=str(exc),
            )
        )
        manifest = None
    if manifest is not None:
        mismatches = _resume_mismatches(
            manifest,
            config=config,
            definition_result=definition_result,
            event_result=event_result,
            plan_hash=plan_hash,
            batch_count=len(batches),
        )
        if mismatches:
            startup_failures.append(
                failure(
                    code="E_CONFIG_MISMATCH",
                    stage="snapshot_resume",
                    message="snapshot cannot be reused; rebuilding from a blank state",
                    detail={"mismatched_fields": mismatches},
                )
            )
            global_version = int(manifest.get("global_version", 0))
            manifest = None
        else:
            try:
                load_stage, feature_stage, score_stage = store.load_stages(manifest)
                workbench = Workbench.from_stages(
                    config=config,
                    load_stage=load_stage,
                    feature_stage=feature_stage,
                    score_stage=score_stage,
                    faults=injection.faults,
                )
                if workbench.cursor != manifest.get("event_cursor"):
                    raise SnapshotMismatch("loaded event cursor does not match the manifest")
                resumed = True
                applied_batches = int(manifest.get("applied_batches", 0))
                global_version = int(manifest.get("global_version", 0))
            except (SnapshotMismatch, ValueError, KeyError) as exc:
                startup_failures.append(
                    failure(
                        code="E_CONFIG_MISMATCH",
                        stage="snapshot_resume",
                        message=f"snapshot artifacts cannot be reused: {exc}",
                    )
                )
                global_version = int(manifest.get("global_version", 0))
                manifest = None
                resumed = False
                applied_batches = 0
    if not resumed:
        workbench = _new_workbench(
            config=config,
            definition_result=definition_result,
            event_result=event_result,
            faults=injection.faults,
        )

    if plan_failure is not None:
        workbench.operation_failures.append(plan_failure)

    new_batches: list[dict[str, Any]] = []
    stopped = False
    if applied_batches >= len(batches) and workbench.run_count == 0:
        stats = workbench.run_incremental(force_all=True)
        summary = {
            "batch_id": "initial",
            "batch_index": -1,
            "force_all": True,
            **stats.as_dict(),
            "operation_failure_count": 0,
        }
        workbench.batch_summaries.append(summary)
        new_batches.append(summary)

    for batch_index in range(applied_batches, len(batches)):
        if stop_requested():
            stopped = True
            break
        batch = batches[batch_index]
        operation_failures_before = len(workbench.operation_failures)
        force_all, forced_units = workbench.apply_batch(batch, batch_index)
        if batch_delay_ms > 0:
            time.sleep(batch_delay_ms / 1000.0)
        stats = workbench.run_incremental(force_all=force_all, forced_units=forced_units)
        summary = {
            "batch_id": batch.get("batch_id", f"batch-{batch_index}") if isinstance(batch, dict) else f"batch-{batch_index}",
            "batch_index": batch_index,
            "force_all": force_all,
            "forced_unit_count": len(forced_units),
            **stats.as_dict(),
            "operation_failure_count": len(workbench.operation_failures)
            - operation_failures_before,
        }
        workbench.batch_summaries.append(summary)
        new_batches.append(summary)
        applied_batches = batch_index + 1
        if stop_requested():
            stopped = True
            if applied_batches < len(batches):
                break
        if applied_batches % config.snapshot_commit_interval == 0 and applied_batches < len(batches):
            global_version += 1
            manifest = _commit_store(
                store=store,
                workbench=workbench,
                global_version=global_version,
                config=config,
                base_definitions_hash=definition_result.source_hash,
                base_events_hash=event_result.source_hash,
                plan_hash=plan_hash,
                applied_batches=applied_batches,
                status="checkpoint",
            )

    # A graceful signal finishes the in-flight batch; this final commit records that boundary.
    if stopped:
        global_version += 1
        manifest = _commit_store(
            store=store,
            workbench=workbench,
            global_version=global_version,
            config=config,
            base_definitions_hash=definition_result.source_hash,
            base_events_hash=event_result.source_hash,
            plan_hash=plan_hash,
            applied_batches=applied_batches,
            status="stopped",
        )
    elif manifest is None or manifest.get("applied_batches") != applied_batches:
        global_version = max(global_version, int(manifest.get("global_version", 0)) if manifest else 0) + 1
        manifest = _commit_store(
            store=store,
            workbench=workbench,
            global_version=global_version,
            config=config,
            base_definitions_hash=definition_result.source_hash,
            base_events_hash=event_result.source_hash,
            plan_hash=plan_hash,
            applied_batches=applied_batches,
            status="committed",
        )

    combined_failures = workbench.failures() + list(workbench.operation_failures) + startup_failures
    all_failures = sorted(
        {canonical_json(item): item for item in combined_failures}.values(),
        key=failure_sort_key,
    )
    injection_hits = compute_injection_hits(injection.expected, all_failures)
    pending = workbench.pending_public()
    exit_code = 3 if pending else 0
    report = {
        "status": "stopped" if stopped else ("partial" if pending else "ok"),
        "mode": "incremental",
        "exit_code": exit_code,
        "resumed": resumed,
        "startup_failures": startup_failures,
        "batch_summaries": workbench.batch_summaries,
        "current_batch_summaries": new_batches,
        "applied_batches": applied_batches,
        "total_batches": len(batches),
        "event_cursor": workbench.cursor,
        "event_count": len(workbench.events.order),
        "definitions_hash": _current_definitions_hash(workbench),
        "feature_count": len(workbench.feature_values),
        "score_count": len(workbench.score_values),
        "feature_graph": {
            name: {
                "upstream_features": list(workbench.registry.definitions[name].upstream_features),
                "downstream_features": list(workbench.registry.downstream.get(name, ())),
                "window_ms": workbench.registry.definitions[name].window_ms,
            }
            for name in workbench.registry.topo
        },
        "features": workbench.features_public(),
        "scores": workbench.scores_public(),
        "failures": sorted(
            {
                canonical_json(item): item
                for item in workbench.failures() + startup_failures
            }.values(),
            key=failure_sort_key,
        ),
        "operation_failures": workbench.operation_failures,
        "all_failures": all_failures,
        "pending": pending,
        "injection_hits": injection_hits,
        "snapshot": _manifest_summary(manifest),
    }
    _atomic_write_json(report_path, report)
    return report, workbench, config


def _commit_store(
    *,
    store: SnapshotStore,
    workbench: Workbench,
    global_version: int,
    config: ScoreConfig,
    base_definitions_hash: str,
    base_events_hash: str,
    plan_hash: str,
    applied_batches: int,
    status: str,
) -> dict[str, Any]:
    return store.commit(
        global_version=global_version,
        config_hash=config.config_fingerprint,
        base_definitions_hash=base_definitions_hash,
        base_events_hash=base_events_hash,
        definitions_hash=_current_definitions_hash(workbench),
        plan_hash=plan_hash,
        event_cursor=workbench.cursor,
        applied_batches=applied_batches,
        load_stage=workbench.export_load_stage(),
        feature_stage=workbench.export_feature_stage(),
        score_stage=workbench.export_score_stage(),
        status=status,
    )


def compute_injection_hits(
    expected: list[dict[str, Any]], failures: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    for item in failures:
        injection_id = item.get("injection_id")
        if injection_id:
            by_id.setdefault(str(injection_id), []).append(item)
    hits: list[dict[str, Any]] = []
    for item in expected:
        injection_id = str(item.get("id"))
        expected_code = item.get("business_code")
        matched = by_id.get(injection_id, [])
        codes = sorted({str(entry.get("code")) for entry in matched})
        hits.append(
            {
                "id": injection_id,
                "kind": item.get("kind"),
                "expected_code": expected_code,
                "actual_codes": codes,
                "hit": bool(matched) and (expected_code is None or expected_code in codes),
                "failure_count": len(matched),
            }
        )
    return hits


def build_full_from_final(
    incremental: Workbench, config: ScoreConfig, faults: FaultPlan | None = None
) -> Workbench:
    raw_definitions = [
        incremental.registry.definitions[name].raw for name in incremental.registry.topo
    ]
    definitions: dict[str, FeatureDef] = {}
    for index, raw in enumerate(raw_definitions):
        definition, error = parse_feature_definition(raw, index)
        if error or definition is None:
            raise ValueError(f"final definition cannot be rebuilt: {error}")
        definitions[definition.name] = definition
    registry_result = build_registry(definitions)
    if registry_result.failures:
        raise ValueError("final definitions are not a valid dependency graph")
    events: list[Event] = []
    previous_timestamp: int | None = None
    for index, event in enumerate(incremental.events.order):
        parsed, error = validate_event_payload(
            event.public(),
            existing_ids={item.event_id for item in events},
            previous_timestamp=previous_timestamp,
            allowed_behaviors=config.allowed_behaviors,
            source_index=index,
        )
        if error or parsed is None:
            raise ValueError(f"final event cannot be rebuilt: {error}")
        events.append(parsed)
        previous_timestamp = parsed.timestamp_ms
    workbench = Workbench(
        config=config,
        definitions=registry_result.definitions,
        topo=registry_result.topo,
        downstream=registry_result.downstream,
        events=events,
        load_failures=[],
        faults=faults,
    )
    workbench.run_incremental(force_all=True)
    return workbench


def compare_projections(
    incremental: Workbench, full: Workbench
) -> dict[str, Any]:
    left = incremental.projection()
    right = full.projection()
    left_canonical = canonical_json(left)
    right_canonical = canonical_json(right)
    diff_paths = _diff_paths(left, right)
    return {
        "equal": left_canonical == right_canonical,
        "incremental_fingerprint": fingerprint("projection-v1", left),
        "full_fingerprint": fingerprint("projection-v1", right),
        "diff_paths": diff_paths[:50],
        "incremental_feature_count": len(left["features"]),
        "full_feature_count": len(right["features"]),
        "incremental_score_count": len(left["scores"]),
        "full_score_count": len(right["scores"]),
        "incremental_failure_count": len(left["failures"]),
        "full_failure_count": len(right["failures"]),
        "incremental_pending_count": len(left["pending"]),
        "full_pending_count": len(right["pending"]),
    }


def _diff_paths(left: Any, right: Any, prefix: str = "$") -> list[str]:
    if type(left) is not type(right):
        return [prefix]
    if isinstance(left, dict):
        differences: list[str] = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                differences.append(f"{prefix}.{key}")
            else:
                differences.extend(_diff_paths(left[key], right[key], f"{prefix}.{key}"))
            if len(differences) >= 50:
                break
        return differences
    if isinstance(left, list):
        differences = []
        if len(left) != len(right):
            differences.append(f"{prefix}.length")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(_diff_paths(left_item, right_item, f"{prefix}[{index}]"))
            if len(differences) >= 50:
                break
        return differences
    return [] if left == right else [prefix]


def _remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _write_materialized_final(workbench: Workbench, root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    definitions_path = root / "definitions.json"
    events_path = root / "events.jsonl"
    definitions_payload = {
        "features": [
            workbench.registry.definitions[name].raw for name in workbench.registry.topo
        ]
    }
    _atomic_write_json(definitions_path, definitions_payload)
    with events_path.open("w", encoding="utf-8", newline="\n") as handle:
        for event in workbench.events.order:
            handle.write(json.dumps(event.public(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return definitions_path, events_path


def diagnose(
    *,
    sample_dir: Path,
    work_dir: Path,
    report_path: Path,
) -> int:
    _remove_tree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    definitions_path = sample_dir / "definitions.json"
    events_path = sample_dir / "events.jsonl"
    config_path = sample_dir / "scoring.yaml"
    edits_path = sample_dir / "edits.json"
    injection_dir = sample_dir / "injections"

    incremental_report_path = work_dir / "incremental_report.json"
    incremental_report, incremental_workbench, config = execute_incremental(
        definitions_path=definitions_path,
        events_path=events_path,
        config_path=config_path,
        edits_path=edits_path,
        state_dir=work_dir / "incremental_state",
        report_path=incremental_report_path,
    )
    if incremental_workbench is None or config is None:
        report = {
            "status": "failed",
            "exit_code": 1,
            "reason": "incremental run could not start",
            "incremental_report": incremental_report,
        }
        _atomic_write_json(report_path, report)
        return 1

    final_root = work_dir / "final_input"
    final_definitions, final_events = _write_materialized_final(incremental_workbench, final_root)
    empty_plan = final_root / "empty_edits.json"
    _atomic_write_json(empty_plan, {"batches": []})
    full_report, full_workbench, _ = execute_incremental(
        definitions_path=final_definitions,
        events_path=final_events,
        config_path=config_path,
        edits_path=empty_plan,
        state_dir=work_dir / "full_state",
        report_path=work_dir / "full_report.json",
    )
    if full_workbench is None:
        report = {
            "status": "failed",
            "exit_code": 1,
            "reason": "full rebuild could not start",
            "incremental_report": incremental_report,
            "full_report": full_report,
        }
        _atomic_write_json(report_path, report)
        return 1

    comparison = compare_projections(incremental_workbench, full_workbench)
    concurrency_results: list[dict[str, Any]] = []
    baseline_full = full_workbench.projection()
    for concurrency in sorted({1, config.concurrency, max(1, config.concurrency * 2)}):
        variant_config = replace(config, concurrency=concurrency)
        variant = build_full_from_final(incremental_workbench, variant_config)
        variant_projection = variant.projection()
        concurrency_results.append(
            {
                "concurrency": concurrency,
                "equal_to_full": canonical_json(variant_projection) == canonical_json(baseline_full),
                "projection_fingerprint": fingerprint("projection-v1", variant_projection),
            }
        )

    injection_results: dict[str, Any] = {
        "ran": False,
        "all_hit": True,
        "hits": [],
    }
    injection_config = injection_dir / "injections.json"
    injection_defs = injection_dir / "definitions.json"
    injection_events = injection_dir / "events.jsonl"
    injection_plan = injection_dir / "edits.json"
    if all(path.exists() for path in (injection_config, injection_defs, injection_events, injection_plan)):
        injection_report, _, _ = execute_incremental(
            definitions_path=injection_defs,
            events_path=injection_events,
            config_path=config_path,
            edits_path=injection_plan,
            state_dir=work_dir / "injection_state",
            report_path=work_dir / "injection_report.json",
            injection_config_path=injection_config,
        )
        hits = injection_report.get("injection_hits", [])
        injection_results = {
            "ran": True,
            "all_hit": bool(hits) and all(item.get("hit") for item in hits),
            "hits": hits,
            "failure_count": len(injection_report.get("all_failures", [])),
        }

    decimal_places_seen = sorted(
        {
            definition.decimal_places
            for definition in incremental_workbench.registry.definitions.values()
        }
    )
    all_checks = (
        comparison["equal"]
        and all(item["equal_to_full"] for item in concurrency_results)
        and (not injection_results["ran"] or injection_results["all_hit"])
        and not incremental_report.get("pending")
        and not full_report.get("pending")
    )
    report = {
        "status": "ok" if all_checks else "failed",
        "exit_code": 0 if all_checks else 1,
        "environment": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "numpy": _package_version("numpy"),
            "pytest": _package_version("pytest"),
            "pyyaml": _package_version("PyYAML"),
            "note": "All paths are local; no container or remote execution is used.",
        },
        "sample": {
            "directory": str(sample_dir),
            "fixed_seed": 188,
            "decimal_places_seen": decimal_places_seen,
            "concurrency": config.concurrency,
            "point_budget_per_batch": config.point_budget_per_batch,
        },
        "incremental": {
            "status": incremental_report.get("status"),
            "applied_batches": incremental_report.get("applied_batches"),
            "feature_count": incremental_report.get("feature_count"),
            "score_count": incremental_report.get("score_count"),
            "event_cursor": incremental_report.get("event_cursor"),
            "batch_summaries": incremental_report.get("batch_summaries"),
            "pending_count": len(incremental_report.get("pending", [])),
            "failure_count": len(incremental_report.get("failures", [])),
            "report_path": str(incremental_report_path),
        },
        "full_rebuild": {
            "status": full_report.get("status"),
            "feature_count": full_report.get("feature_count"),
            "score_count": full_report.get("score_count"),
            "event_cursor": full_report.get("event_cursor"),
            "pending_count": len(full_report.get("pending", [])),
            "failure_count": len(full_report.get("failures", [])),
            "report_path": str(work_dir / "full_report.json"),
        },
        "equivalence": comparison,
        "concurrency_equivalence": concurrency_results,
        "injections": injection_results,
        "all_checks_passed": all_checks,
    }
    _atomic_write_json(report_path, report)
    return 0 if all_checks else 1


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _install_signal_handlers(state: dict[str, bool]) -> None:
    def handler(signum: int, _frame: Any) -> None:
        state["stop"] = True
        state["signals"].append(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic local feature-service diagnostic workbench")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="apply an incremental edit plan")
    run_parser.add_argument("--definitions", required=True, type=Path)
    run_parser.add_argument("--events", required=True, type=Path)
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--edits", type=Path)
    run_parser.add_argument("--state-dir", required=True, type=Path)
    run_parser.add_argument("--report", required=True, type=Path)
    run_parser.add_argument("--injection-config", type=Path)
    run_parser.add_argument("--batch-delay-ms", type=int, default=0)

    diagnose_parser = subparsers.add_parser("diagnose", help="run equivalence and injection diagnostics")
    diagnose_parser.add_argument("--sample-dir", required=True, type=Path)
    diagnose_parser.add_argument("--work-dir", required=True, type=Path)
    diagnose_parser.add_argument("--report", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "diagnose":
        return diagnose(sample_dir=args.sample_dir, work_dir=args.work_dir, report_path=args.report)
    signal_state = {"stop": False, "signals": []}
    _install_signal_handlers(signal_state)
    report, _, _ = execute_incremental(
        definitions_path=args.definitions,
        events_path=args.events,
        config_path=args.config,
        edits_path=args.edits,
        state_dir=args.state_dir,
        report_path=args.report,
        injection_config_path=args.injection_config,
        batch_delay_ms=args.batch_delay_ms,
        stop_requested=lambda: bool(signal_state["stop"]),
    )
    if signal_state["signals"]:
        report["signals_received"] = signal_state["signals"]
        _atomic_write_json(args.report, report)
    if report.get("status") == "stopped":
        return 0
    return int(report.get("exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
