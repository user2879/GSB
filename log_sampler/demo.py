"""Reproducible local acceptance verification."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .canonical import atomic_write_bytes, atomic_write_json, fingerprint, read_json
from .evidence import (
    build_evidence_slots,
    convert_evidence,
    parse_evidence,
    serialize_evidence,
)
from .pipeline import Pipeline, RuntimeConfig, compare_logical_results
from .state import SessionStore
from .validation import load_batches, load_config, load_json_file

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fresh(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _list_payload(path: Path, key: str) -> list[dict[str, Any]]:
    payload, issues = load_json_file(path, key)
    if issues:
        raise RuntimeError(issues[0].message)
    values = payload.get(key) if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise RuntimeError(f"{key} must be a list")
    return values


def _run(
    config: dict[str, Any],
    batches: list[dict[str, Any]],
    injections: list[dict[str, Any]],
    state: Path,
    crash_after: str | None = None,
):
    return Pipeline(
        config,
        batches,
        injections,
        RuntimeConfig(state_root=state, crash_after=crash_after),
    ).run()


def _step_cache(command: dict[str, Any], result) -> dict[str, Any]:
    ledger = result.ledger
    misses = [
        item["bucket_id"]
        for item in ledger["buckets"]
        if item["status"] != "cache_hit"
    ]
    hits = [
        item["bucket_id"]
        for item in ledger["buckets"]
        if item["status"] == "cache_hit"
    ]
    return {
        "command_id": command.get("id"),
        "command_type": command.get("type"),
        "global_version": result.manifest["global_version"],
        "bucket_cache_hits": len(hits),
        "bucket_cache_misses": len(misses),
        "affected_buckets": sorted(misses),
        "reused_buckets": sorted(hits),
        "final_sample_cache": ledger["final_sample"]["status"],
        "serialization_cache": {
            version: item["status"]
            for version, item in ledger["serializations"].items()
        },
        "final_sample_count": len(result.logical_result["final_samples"]),
        "failure_count": len(
            result.logical_result["validation"]["failures"]
        ),
    }


def _slots_equal(left, right) -> bool:
    return [item.to_dict() for item in left] == [item.to_dict() for item in right]


def _evidence_roundtrip(
    slots, output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    p3_payload, p3_metadata = serialize_evidence(slots, "p3")
    atomic_write_bytes(output_dir / "source.p3.bin", p3_payload)
    direct = parse_evidence(p3_payload, "p3")
    checks: dict[str, Any] = {}

    p1 = convert_evidence(p3_payload, "p3", "p1")
    p1_back = convert_evidence(p1.payload, "p1", "p3", p1.sidecar)
    atomic_write_bytes(output_dir / "down.p1.txt", p1.payload)
    atomic_write_json(output_dir / "down.p1.metadata.json", p1.sidecar)
    atomic_write_bytes(output_dir / "up.p3.bin", p1_back.payload)
    atomic_write_json(output_dir / "up.p3.metadata.json", p1_back.sidecar)
    checks["p3_to_p1_to_p3"] = {
        "equal": _slots_equal(direct.slots, parse_evidence(p1_back.payload, "p3").slots),
        "down_issues": [item.to_dict() for item in p1.issues],
        "up_issues": [item.to_dict() for item in p1_back.issues],
        "rows": len(p1.slots),
    }

    p2 = convert_evidence(p3_payload, "p3", "p2")
    p2_back = convert_evidence(p2.payload, "p2", "p3", p2.sidecar)
    atomic_write_bytes(output_dir / "down.p2.json", p2.payload)
    atomic_write_json(output_dir / "down.p2.metadata.json", p2.sidecar)
    checks["p3_to_p2_to_p3"] = {
        "equal": _slots_equal(direct.slots, parse_evidence(p2_back.payload, "p3").slots),
        "down_issues": [item.to_dict() for item in p2.issues],
        "up_issues": [item.to_dict() for item in p2_back.issues],
        "rows": len(p2.slots),
    }

    p2_chain = convert_evidence(p2.payload, "p2", "p1", p2.sidecar)
    p3_chain = convert_evidence(p2_chain.payload, "p1", "p3", p2_chain.sidecar)
    atomic_write_bytes(output_dir / "chain.p1.txt", p2_chain.payload)
    atomic_write_json(output_dir / "chain.p1.metadata.json", p2_chain.sidecar)
    atomic_write_bytes(output_dir / "chain.p3.bin", p3_chain.payload)
    checks["p3_to_p2_to_p1_to_p3"] = {
        "equal": _slots_equal(direct.slots, parse_evidence(p3_chain.payload, "p3").slots),
        "middle_issues": [item.to_dict() for item in p2_chain.issues],
        "up_issues": [item.to_dict() for item in p3_chain.issues],
    }
    return {
        "source_slots": len(slots),
        "source_p3_sha256": fingerprint(p3_payload.hex()),
        "checks": checks,
    }


def _probe(
    state: Path,
    config: dict[str, Any],
    batches: list[dict[str, Any]],
    injections: list[dict[str, Any]],
):
    _fresh(state)
    store = SessionStore(state).initialize(config, batches, injections)
    return _run(config, batches, injections, state)


def _bucket_log_ids(bucket: dict[str, Any]) -> list[str]:
    return [slot["log_id"] for slot in bucket["slots"]]


def run_verification(
    *,
    config_path: Path,
    batches_path: Path,
    injections_path: Path | None,
    commands_path: Path | None,
    state_root: Path,
    fresh: bool,
) -> dict[str, Any]:
    if fresh and state_root.exists():
        shutil.rmtree(state_root)
    state_root.mkdir(parents=True, exist_ok=True)
    config, issues = load_config(config_path)
    if issues:
        raise RuntimeError(issues[0].message)
    batches, issues = load_batches(batches_path)
    if issues:
        raise RuntimeError(issues[0].message)
    injections = _list_payload(injections_path, "injections") if injections_path else []
    commands = _list_payload(commands_path, "commands") if commands_path else []

    incremental_state = state_root / "incremental"
    _fresh(incremental_state)
    session = SessionStore(incremental_state).initialize(
        config, batches, injections
    )
    baseline = _run(config, batches, injections, incremental_state)
    step_reports: list[dict[str, Any]] = []
    final_result = baseline
    for command in commands:
        session = SessionStore(incremental_state)
        session.apply_commands([command])
        effective_config, effective_batches, effective_injections = (
            session.effective_input()
        )
        final_result = _run(
            effective_config,
            effective_batches,
            effective_injections,
            incremental_state,
        )
        step_reports.append(_step_cache(command=command, result=final_result))

    effective_config, effective_batches, effective_injections = (
        SessionStore(incremental_state).effective_input()
    )
    replay_state = state_root / "replay"
    replay = _probe(
        replay_state,
        effective_config,
        effective_batches,
        effective_injections,
    )
    replay_diffs = compare_logical_results(
        final_result.logical_result, replay.logical_result
    )

    crash_state = state_root / "crash"
    _fresh(crash_state)
    SessionStore(crash_state).initialize(
        effective_config, effective_batches, effective_injections
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    crash_command = [
        sys.executable,
        "-m",
        "log_sampler",
        "run",
        "--state",
        str(crash_state),
        "--crash-after",
        "bucket:2",
    ]
    crashed = subprocess.run(
        crash_command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    staging_manifest = None
    staging_manifest_path = crash_state / "staging" / "current" / "manifest.json"
    if staging_manifest_path.exists():
        staging_manifest = read_json(staging_manifest_path)
    restart_command = [
        sys.executable,
        "-m",
        "log_sampler",
        "run",
        "--state",
        str(crash_state),
    ]
    restarted = subprocess.run(
        restart_command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    recovered_logical = read_json(crash_state / "latest_result.json")
    recovered_manifest = read_json(crash_state / "manifest.json")
    crash_diffs = compare_logical_results(replay.logical_result, recovered_logical)
    recovered_ledger = read_json(
        crash_state / recovered_manifest["version_dir"] / "ledger.json"
    )
    resumed_buckets = [
        item["bucket_id"]
        for item in recovered_ledger["buckets"]
        if item["status"] == "stage_resume_hit"
    ]

    final_slots = build_evidence_slots(final_result.logical_result["buckets"])
    roundtrip = _evidence_roundtrip(
        final_slots, state_root / "evidence-roundtrip"
    )

    high_budget_config = copy.deepcopy(effective_config)
    high_budget_config["profiles"]["tight"]["global_budget"] = 8
    high_budget = _probe(
        state_root / "probes" / "budget-high",
        high_budget_config,
        effective_batches,
        effective_injections,
    )
    low_budget_ids = {
        item["log_id"] for item in final_result.logical_result["final_samples"]
    }
    high_budget_ids = {
        item["log_id"] for item in high_budget.logical_result["final_samples"]
    }

    high_capacity_config = copy.deepcopy(effective_config)
    high_capacity_config["bucket_overrides"]["edge-a|warning"]["capacity"] = 3
    high_capacity = _probe(
        state_root / "probes" / "capacity-high",
        high_capacity_config,
        effective_batches,
        effective_injections,
    )
    target_bucket = "edge-a|warning"
    low_bucket = final_result.logical_result["buckets"][target_bucket]
    high_bucket = high_capacity.logical_result["buckets"][target_bucket]
    other_bucket_changes = [
        bucket_id
        for bucket_id in sorted(final_result.logical_result["buckets"])
        if bucket_id != target_bucket
        and final_result.logical_result["buckets"][bucket_id]
        != high_capacity.logical_result["buckets"][bucket_id]
    ]

    expected_injection_ids = sorted(
        item["id"] for item in injections if isinstance(item, dict) and "id" in item
    )
    observed_injection_ids = sorted(
        {
            item["injection_id"]
            for item in final_result.logical_result["validation"]["failures"]
            if item.get("injection_id")
        }
    )
    injection_isolation = {
        "expected": expected_injection_ids,
        "observed_in_failure_list": observed_injection_ids,
        "all_recorded": set(expected_injection_ids) <= set(observed_injection_ids),
        "accepted_records_after_failures": len(
            final_result.logical_result["buckets"]
        )
        > 0
        and final_result.logical_result["validation"]["accepted_records"] > 0,
        "rejected_batches": final_result.logical_result["validation"]["rejected_batches"],
        "accepted_batches": final_result.logical_result["validation"]["accepted_batches"],
    }

    single_bucket_steps = [
        report
        for report in step_reports
        if len(report["affected_buckets"]) == 1 and report["bucket_cache_hits"] > 0
    ]
    all_hit_steps = [
        report for report in step_reports if report["bucket_cache_misses"] == 0
    ]
    capacity_steps = [
        report
        for report in step_reports
        if report["command_type"] == "set_bucket_capacity"
    ]
    capacity_precision = bool(
        capacity_steps
        and capacity_steps[0]["affected_buckets"] == [target_bucket]
    )

    checks = {
        "incremental_matches_blank_replay": not replay_diffs,
        "crash_recovery_matches_uninterrupted": not crash_diffs,
        "crash_test_exited_137": crashed.returncode == 137,
        "restart_succeeded": restarted.returncode == 0,
        "resumed_pending_buckets": bool(resumed_buckets),
        "stage_versions_recovered": recovered_manifest["stage_versions"]
        == replay.manifest["stage_versions"],
        "evidence_roundtrips": all(
            item["equal"] and not item["down_issues"] and not item["up_issues"]
            for key, item in roundtrip["checks"].items()
            if key != "p3_to_p2_to_p1_to_p3"
        ),
        "evidence_three_hop_roundtrip": roundtrip["checks"][
            "p3_to_p2_to_p1_to_p3"
        ]["equal"]
        and not roundtrip["checks"]["p3_to_p2_to_p1_to_p3"]["middle_issues"]
        and not roundtrip["checks"]["p3_to_p2_to_p1_to_p3"]["up_issues"],
        "budget_tightening_changes_final_set": low_budget_ids != high_budget_ids,
        "capacity_change_changes_bucket": low_bucket != high_bucket,
        "capacity_change_isolated_to_bucket": not other_bucket_changes,
        "single_bucket_reuse_observed": bool(single_bucket_steps),
        "all_bucket_cache_reuse_observed": bool(all_hit_steps),
        "capacity_invalidates_only_target": capacity_precision,
        "injection_isolation": injection_isolation["all_recorded"]
        and injection_isolation["accepted_records_after_failures"],
        "diagnostics_present": bool(
            final_result.logical_result["diagnostics"]["summary"][
                "reservoir_boundary_count"
            ]
        )
        and bool(
            final_result.logical_result["diagnostics"]["summary"][
                "truncation_boundary_count"
            ]
        )
        and "E_D_FLIP_RANDOM_SOURCE"
        in final_result.logical_result["diagnostics"]["summary"]["business_codes"]
        and "E_D_FLIP_TRUNCATION_TIE"
        in final_result.logical_result["diagnostics"]["summary"]["business_codes"],
    }
    status = "ok" if all(checks.values()) else "failed"
    report = {
        "status": status,
        "checks": checks,
        "metrics": {
            "baseline_version": baseline.manifest["global_version"],
            "final_incremental_version": final_result.manifest["global_version"],
            "replay_version": replay.manifest["global_version"],
            "baseline_final_samples": len(baseline.logical_result["final_samples"]),
            "final_samples": len(final_result.logical_result["final_samples"]),
            "high_budget_final_samples": len(high_budget.logical_result["final_samples"]),
            "candidate_counts": final_result.logical_result["truncation"][
                "candidate_count"
            ],
            "failure_count": len(final_result.logical_result["validation"]["failures"]),
            "bucket_count": len(final_result.logical_result["buckets"]),
            "evidence_slots": len(final_slots),
            "diagnostic_business_codes": final_result.logical_result["diagnostics"][
                "summary"
            ]["business_codes"],
        },
        "incremental": {
            "state": str(incremental_state),
            "manifest": final_result.manifest,
            "steps": step_reports,
            "single_bucket_reuse_steps": [item["command_id"] for item in single_bucket_steps],
            "all_hit_steps": [item["command_id"] for item in all_hit_steps],
        },
        "replay": {
            "state": str(replay_state),
            "difference_count": len(replay_diffs),
            "differences": replay_diffs[:25],
        },
        "crash_recovery": {
            "state": str(crash_state),
            "crash_exit_code": crashed.returncode,
            "restart_exit_code": restarted.returncode,
            "staging_status_after_kill": (
                staging_manifest.get("status") if staging_manifest else None
            ),
            "pending_after_kill": (
                list(staging_manifest.get("pending_buckets", []))
                if staging_manifest
                else []
            ),
            "completed_after_kill": (
                len(staging_manifest.get("completed_buckets", []))
                if staging_manifest
                else 0
            ),
            "resumed_buckets": resumed_buckets,
            "difference_count": len(crash_diffs),
            "differences": crash_diffs[:25],
            "stage_versions": recovered_manifest["stage_versions"],
        },
        "evidence_roundtrip": roundtrip,
        "budget_probe": {
            "low_budget": final_result.logical_result["global_budget"],
            "high_budget": high_budget.logical_result["global_budget"],
            "low_ids": sorted(low_budget_ids),
            "high_ids": sorted(high_budget_ids),
            "changed": low_budget_ids != high_budget_ids,
        },
        "capacity_probe": {
            "bucket_id": target_bucket,
            "low_slots": _bucket_log_ids(low_bucket),
            "high_slots": _bucket_log_ids(high_bucket),
            "other_bucket_changes": other_bucket_changes,
        },
        "injection_isolation": injection_isolation,
        "artifacts": {
            "incremental_result": str(incremental_state / "latest_result.json"),
            "replay_result": str(replay_state / "latest_result.json"),
            "recovered_result": str(crash_state / "latest_result.json"),
            "evidence_roundtrip": str(state_root / "evidence-roundtrip"),
        },
    }
    atomic_write_json(state_root / "verification.json", report)
    return report
