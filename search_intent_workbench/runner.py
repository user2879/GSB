"""End-to-end deterministic workbench verification orchestration."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, digest
from .faults import run_fault_suite
from .loader import load_commands, load_dataset
from .pipeline import Workbench, semantic_projection
from .report import build_report, write_report
from .snapshot import SimulatedCrash, SnapshotStore
from .state import WorkbenchState, apply_command


def _recompute_counts(run: dict[str, Any]) -> dict[str, int]:
    return {
        stage: len(run["reuse_report"][stage]["recomputed"])
        for stage in ("normalization", "slots", "intent", "rewrite")
    }


def _reused_counts(run: dict[str, Any]) -> dict[str, int]:
    return {
        stage: len(run["reuse_report"][stage]["reused"])
        for stage in ("normalization", "slots", "intent", "rewrite")
    }


def _affected_queries(run: dict[str, Any]) -> list[str]:
    affected: set[str] = set()
    for stage in run["reuse_report"].values():
        affected.update(stage["recomputed"])
    return sorted(affected)


def _simple_recovery(status: str, inspection: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": status,
        "reusable": inspection.get("reusable", False),
        "artifact_count": len(inspection.get("reusable_artifacts", [])),
        "completed_stage": inspection.get("completed_stage"),
        "completed_query_cursor": inspection.get("completed_query_cursor"),
        "errors": inspection.get("errors", []),
    }


def run_workbench_demo(
    *,
    data_dir: str | Path = "data/base",
    injections_file: str | Path = "data/injected/injections.json",
    snapshot_root: str | Path = ".snapshots",
    artifact_dir: str | Path = "artifacts",
    workers: int | None = None,
    with_faults: bool = True,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    snapshot_store = SnapshotStore(snapshot_root)

    dataset = load_dataset(
        data_dir / "queries.jsonl",
        data_dir / "intent_config.json",
        data_dir / "normalization_rules.json",
    )
    if not dataset.config_ok:
        result = {
            "status": "E_CONFIG_INVALID",
            "config_issues": dataset.config_issues,
            "rule_issues": dataset.rule_issues,
            "query_failures": dataset.failures,
        }
        atomic_write_json(artifact_dir / "verification_summary.json", result)
        return result

    commands, command_load_failures = load_commands(data_dir / "commands.json")
    state = WorkbenchState(
        queries=dataset.queries,
        config=dataset.config,
        rules=dataset.rules,
        failures=sorted(
            dataset.failures + command_load_failures,
            key=lambda item: (
                str(item.get("query_id", "")),
                str(item.get("command_id", "")),
                str(item.get("code", "")),
            ),
        ),
    )

    version = 1
    initial_workbench = Workbench(
        state.config,
        state.rules,
        max_workers=workers,
    )
    run = initial_workbench.run(
        state.sorted_queries(),
        input_failures=state.failures,
        global_version=version,
        command_index=0,
    )
    previous_cache = Workbench.stage_cache_from_run(run)
    initial_manifest = snapshot_store.commit(
        version=version,
        state=state,
        run=run,
        report=None,
    )
    evolution: list[dict[str, Any]] = []

    for step, command in enumerate(commands, start=1):
        state, applied, change = apply_command(state, command)
        version += 1
        incremental_workbench = Workbench(
            state.config,
            state.rules,
            max_workers=workers,
            previous_stage_cache=previous_cache,
        )
        incremental = incremental_workbench.run(
            state.sorted_queries(),
            input_failures=state.failures,
            global_version=version,
            command_index=step,
        )
        full_workbench = Workbench(state.config, state.rules, max_workers=workers)
        full = full_workbench.run(
            state.sorted_queries(),
            input_failures=state.failures,
            global_version=version,
            command_index=step,
        )
        equal = semantic_projection(incremental) == semantic_projection(full)
        evolution.append(
            {
                "step": step,
                "command_id": command["command_id"],
                "type": command["type"],
                "applied": applied,
                "change": change,
                "global_version": version,
                "incremental_equals_full": equal,
                "recomputed_counts": _recompute_counts(incremental),
                "reused_counts": _reused_counts(incremental),
                "affected_queries": _affected_queries(incremental),
                "failure_count": len(incremental["failures"]),
            }
        )
        snapshot_store.commit(
            version=version,
            state=state,
            run=incremental,
            report=None,
        )
        previous_cache = Workbench.stage_cache_from_run(incremental)
        run = incremental

    configured_workers = int(state.config.get("max_workers", 1))
    serial_workbench = Workbench(state.config, state.rules, max_workers=1)
    serial_run = serial_workbench.run(
        state.sorted_queries(),
        input_failures=state.failures,
        global_version=version,
        command_index=len(commands),
    )
    concurrent_workbench = Workbench(
        state.config, state.rules, max_workers=configured_workers
    )
    concurrent_run = concurrent_workbench.run(
        state.sorted_queries(),
        input_failures=state.failures,
        global_version=version,
        command_index=len(commands),
    )
    capped_workbench = Workbench(
        state.config, state.rules, max_workers=configured_workers + 16
    )
    capped_run = capped_workbench.run(
        state.sorted_queries(),
        input_failures=state.failures,
        global_version=version,
        command_index=len(commands),
    )
    concurrency_check = {
        "equal": semantic_projection(serial_run) == semantic_projection(concurrent_run),
        "capped_equals_serial": semantic_projection(serial_run)
        == semantic_projection(capped_run),
        "configured_max_workers": configured_workers,
        "serial_effective_workers": serial_workbench.max_workers,
        "serial_peak_workers": serial_run["metadata"]["peak_active_tasks"],
        "concurrent_effective_workers": concurrent_workbench.max_workers,
        "concurrent_peak_workers": concurrent_run["metadata"]["peak_active_tasks"],
        "requested_over_cap": configured_workers + 16,
        "capped_effective_workers": capped_workbench.max_workers,
        "capped_peak_workers": capped_run["metadata"]["peak_active_tasks"],
        "ordering_stable": [
            record["query_id"] for record in serial_run["queries"]
        ]
        == [record["query_id"] for record in concurrent_run["queries"]],
    }

    final_config_hash = run["metadata"]["config_hash"]
    recovery_checks: list[dict[str, Any]] = []
    matching = snapshot_store.inspect(
        expected_version=version, expected_config_hash=final_config_hash
    )
    recovery_checks.append({"name": "matching_version_and_config", **_simple_recovery("RESUME", matching)})

    restored_cache = snapshot_store.load_stage_cache(matching)
    restored_workbench = Workbench(
        state.config,
        state.rules,
        max_workers=workers,
        previous_stage_cache=restored_cache,
    )
    restored_run = restored_workbench.run(
        state.sorted_queries(),
        input_failures=state.failures,
        global_version=version,
        command_index=len(commands),
    )
    restored_reuse = {
        stage: len(restored_run["reuse_report"][stage]["reused"])
        for stage in ("normalization", "slots", "intent", "rewrite")
    }
    recovery_checks.append(
        {
            "name": "restored_stage_cache",
            "status": "RESUME" if matching.get("reusable") else "NO_REUSE",
            "restored_stage_rows": len(restored_cache),
            "all_stages_reused": all(
                restored_reuse[stage] == len(restored_run["queries"])
                for stage in restored_reuse
            ),
            "restored_equals_live": semantic_projection(restored_run)
            == semantic_projection(run),
            "reused_counts": restored_reuse,
            "errors": [],
        }
    )
    stale_config = snapshot_store.inspect(
        expected_version=version,
        expected_config_hash=digest({"mismatch": final_config_hash}),
    )
    recovery_checks.append(
        {
            "name": "config_hash_mismatch",
            **_simple_recovery(stale_config["status"], stale_config),
        }
    )
    stale_version = snapshot_store.inspect(
        expected_version=version + 1,
        expected_config_hash=final_config_hash,
    )
    recovery_checks.append(
        {
            "name": "version_mismatch",
            **_simple_recovery(stale_version["status"], stale_version),
        }
    )

    crash_root = Path(snapshot_root).parent / (Path(snapshot_root).name + "-crash")
    SnapshotStore.remove(crash_root)
    crash_store = SnapshotStore(crash_root)
    crash_version = version + 1
    try:
        crash_store.commit(
            version=crash_version,
            state=state,
            run=run,
            report=None,
            crash_after_stage="slots",
        )
    except SimulatedCrash:
        pass
    crash_recovery = crash_store.inspect(
        expected_version=crash_version,
        expected_config_hash=final_config_hash,
    )
    recovery_checks.append(
        {
            "name": "partial_staging_same_version",
            **_simple_recovery(crash_recovery["status"], crash_recovery),
        }
    )
    crash_stale = crash_store.inspect(
        expected_version=crash_version,
        expected_config_hash=stale_config["expected_config_hash"],
    )
    recovery_checks.append(
        {
            "name": "partial_staging_hash_mismatch",
            **_simple_recovery(crash_stale["status"], crash_stale),
        }
    )

    fault_report = run_fault_suite(injections_file) if with_faults else None
    report = build_report(
        run,
        command_checks=evolution,
        recovery_checks=recovery_checks,
        concurrency_check=concurrency_check,
        fault_report=fault_report,
    )
    write_report(
        report,
        artifact_dir / "diagnostic_report.json",
        artifact_dir / "diagnostic_report.html",
    )
    atomic_write_json(artifact_dir / "command_evolution.json", evolution)
    atomic_write_json(artifact_dir / "serial_concurrency_check.json", concurrency_check)
    atomic_write_json(artifact_dir / "recovery_check.json", recovery_checks)
    if fault_report is not None:
        atomic_write_json(artifact_dir / "fault_report.json", fault_report)
    atomic_write_json(
        artifact_dir / "final_run.json",
        {
            "global_version": run["global_version"],
            "metadata": run.get("metadata", {}),
            "reuse_report": run.get("reuse_report", {}),
            "queries": run.get("queries", []),
            "failures": run.get("failures", []),
        },
    )

    all_commands_equal = all(item["incremental_equals_full"] for item in evolution)
    recovery_ok = (
        recovery_checks[0]["status"] == "RESUME"
        and recovery_checks[1]["restored_equals_live"]
        and recovery_checks[1]["all_stages_reused"]
        and recovery_checks[2]["status"] == "E_SNAPSHOT_STALE"
        and not recovery_checks[2]["reusable"]
        and recovery_checks[2]["artifact_count"] == 0
        and recovery_checks[3]["status"] == "E_SNAPSHOT_STALE"
        and recovery_checks[4]["status"] == "RESUME_STAGING"
        and recovery_checks[5]["status"] == "E_SNAPSHOT_STALE"
        and not recovery_checks[5]["reusable"]
        and recovery_checks[5]["artifact_count"] == 0
    )
    fault_ok = fault_report is None or fault_report["summary"]["all_detected"]
    summary = {
        "status": "PASS"
        if all_commands_equal and concurrency_check["equal"] and recovery_ok and fault_ok
        else "FAIL",
        "global_version": version,
        "config_hash": final_config_hash,
        "rule_fingerprint": run["metadata"]["rule_fingerprint"],
        "dictionary_hash": run["metadata"]["dictionary_hash"],
        "initial_snapshot_version": 1,
        "initial_snapshot_status": initial_manifest["status"],
        "query_count": len(run["queries"]),
        "failure_count": len(run["failures"]),
        "command_count": len(commands),
        "all_commands_equal": all_commands_equal,
        "serial_concurrent_equal": concurrency_check["equal"],
        "recovery_ok": recovery_ok,
        "fault_injections_all_detected": fault_ok,
        "fault_hits": (
            [item["injection_id"] for item in fault_report["injections"] if item["detected"]]
            if fault_report
            else []
        ),
        "snapshot_root": str(Path(snapshot_root)),
        "artifact_dir": str(artifact_dir),
        "report_json": str(artifact_dir / "diagnostic_report.json"),
        "report_html": str(artifact_dir / "diagnostic_report.html"),
    }
    atomic_write_json(artifact_dir / "verification_summary.json", summary)
    return summary


def clean_directory(path: str | Path) -> None:
    target = Path(path)
    if target.exists():
        shutil.rmtree(target)
