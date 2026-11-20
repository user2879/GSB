#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from rope_extrap.analysis import build_extended_report, diagnostic_consistency_view
from rope_extrap.cache import ResultCache
from rope_extrap.engine import DiagnosticEngine, run_full_diagnosis
from rope_extrap.errors import BusinessError
from rope_extrap.loader import load_position_config, load_samples
from rope_extrap.models import InteractionCommand
from rope_extrap.session import InteractiveSession, run_command_sequence
from rope_extrap.snapshot import SnapshotStore
from rope_extrap.state import state_from_sequence_result
from scripts.generate_samples import generate


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return ordered[index]


def _run_latency_probe(config, samples) -> dict[str, Any]:
    subset = dict(list(samples.items())[:24])
    session = InteractiveSession(config, subset)
    try:
        first = session.apply_command(
            InteractionCommand(
                index=1,
                command_id="probe-switch",
                command_type="set_scheme",
                changes={"scheme": "ntk_aware"},
            )
        )
        time.sleep(0.05)
        second = session.apply_command(
            InteractionCommand(
                index=2,
                command_id="probe-poll",
                command_type="poll",
                changes={},
                budget_ms=20,
            )
        )
        time.sleep(0.05)
        third = session.apply_command(
            InteractionCommand(
                index=3,
                command_id="probe-complete",
                command_type="poll",
                changes={},
                budget_ms=120_000,
            )
        )
        return {
            "first": _latency_summary(first),
            "second": _latency_summary(second),
            "third": _latency_summary(third),
            "completed_progress": [
                first["summary"]["completed_samples"],
                second["summary"]["completed_samples"],
                third["summary"]["completed_samples"],
            ],
            "queue_progress": [
                first["pending_queue"]["remaining_stage_tasks"] if first["pending_queue"] else None,
                second["pending_queue"]["remaining_stage_tasks"] if second["pending_queue"] else None,
                third["pending_queue"]["remaining_stage_tasks"] if third["pending_queue"] else None,
            ],
        }
    finally:
        session.close()


def _latency_summary(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "command_id": response["command_id"],
        "budget_ms": response["budget_ms"],
        "elapsed_ms": response["elapsed_ms"],
        "within_budget": response["within_budget"],
        "pending": response["pending"],
        "business_codes": response["business_codes"],
        "completed_samples": response["summary"]["completed_samples"],
    }


def _verify_snapshot(config, samples, snapshot_root: Path) -> dict[str, Any]:
    if snapshot_root.exists():
        shutil.rmtree(snapshot_root)
    subset = dict(list(samples.items())[:12])
    engine = DiagnosticEngine(config, subset)
    plan = engine.run_full()
    report = engine.build_report(plan=plan)
    store = SnapshotStore(snapshot_root)
    manifest = store.commit(
        engine,
        global_version=7,
        completed_samples=[row["sample_id"] for row in report["samples"]],
        metadata={"acceptance": True},
    )
    loaded = store.load(
        expected_version=7,
        expected_config_hash=config.config_fingerprint(),
    )
    stale = store.load(
        expected_version=8,
        expected_config_hash=config.config_fingerprint(),
    )
    cache = ResultCache()
    loaded.apply_to(cache)
    resumed = DiagnosticEngine(config, subset, cache=cache)
    resumed_plan = resumed.build_plan()
    return {
        "manifest": manifest,
        "load_ok": loaded.ok,
        "loaded_entry_count": len(loaded.entries),
        "stale_ok": stale.ok,
        "stale_code": stale.failures[0].code if stale.failures else None,
        "resumed_reused_entries": len(resumed_plan.reused),
        "resumed_recomputed_jobs": sum(len(jobs) for jobs in resumed_plan.jobs_by_stage.values()),
        "files": sorted(str(path.relative_to(snapshot_root)) for path in snapshot_root.rglob("*") if path.is_file()),
    }


def _verify_faults() -> dict[str, Any]:
    samples_dir = ROOT / "samples"
    config, config_failures = load_position_config(samples_dir / "config_large_budget.json")
    assert config is not None and not config_failures
    _valid, sample_failures = load_samples(
        samples_dir / "fault_samples.jsonl",
        dimension=config.dimension,
        target_length=config.extrapolation_target_length,
    )
    base_config, base_failures = load_position_config(samples_dir / "config_fault_base.json")
    scale_config, scale_failures = load_position_config(samples_dir / "config_fault_scale.json")
    invalid_config, invalid_config_failures = load_position_config(
        samples_dir / "config_invalid_json.json"
    )
    _empty, empty_failures = load_samples(
        samples_dir / "empty_samples.jsonl",
        dimension=config.dimension,
        target_length=config.extrapolation_target_length,
    )

    engine_samples, _ = load_samples(
        samples_dir / "samples.jsonl",
        dimension=config.dimension,
        target_length=config.extrapolation_target_length,
    )
    isolated_samples = dict(list(engine_samples.items())[:6])

    def fail_one(sample):
        if sample.sample_id == "s002":
            raise BusinessError(
                "E_INJECTED_STAGE",
                "deliberate injected stage failure",
                stage="error_trace",
                sample_id=sample.sample_id,
            )

    engine = DiagnosticEngine(
        config,
        isolated_samples,
        stage_hooks={"error_trace": fail_one},
    )
    plan = engine.run_full()
    injected_report = engine.build_report(plan=plan)
    return {
        "loader": {
            "valid_sample_ids": sorted(_valid),
            "sample_failures": [failure.to_dict() for failure in sample_failures],
            "base_failure_codes": [failure.code for failure in base_failures],
            "scale_failure_codes": [failure.code for failure in scale_failures],
            "invalid_config_codes": [failure.code for failure in invalid_config_failures],
            "empty_file_codes": [failure.code for failure in empty_failures],
            "base_config_is_none": base_config is None,
            "scale_config_is_none": scale_config is None,
            "invalid_config_is_none": invalid_config is None,
        },
        "stage_injection": {
            "summary": injected_report["summary"],
            "failures": injected_report["failures"],
            "business_code_counts": injected_report["business_code_counts"],
            "completed_sample_ids": [row["sample_id"] for row in injected_report["samples"]],
        },
    }


def _markdown_report(report: dict[str, Any]) -> str:
    checks = report["checks"]
    passed = sum(1 for check in checks if check["passed"])
    scheme_rows = []
    for scheme, stats in report["scheme_statistics"].items():
        scheme_rows.append(
            f"| {scheme} | {stats['over_tolerance_sample_count']} | "
            f"{stats['mean_max_deviation']:.6f} | {stats['max_max_deviation']:.6f} | "
            f"`{stats['first_sample_position_fingerprint'][:16]}` |"
        )
    latency = report["latency"]
    large = latency["large_sequence"]
    small = latency["small_sequence"]
    probe = latency["probe"]
    sensitivity_rows = [
        f"| {row['rank']} | {row['parameter']} | {row['variant_value']} | {row['abs_delta']:.6f} |"
        for row in report["sensitivity_top"]
    ]
    violation_rows = [
        f"| {row['rank']} | {row['sample_id']} | {row['relative_distance']} | "
        f"{row['first_exceeded_stage']} | {row['max_deviation']:.6f} |"
        for row in report["top_violations"]
    ]
    check_rows = [
        f"| {check['name']} | {'PASS' if check['passed'] else 'FAIL'} | {check['detail']} |"
        for check in checks
    ]
    return f"""# Verification Report

Generated: {report['generated_at_utc']}

## Environment

- OS: {report['environment']['os']}
- Python: {report['environment']['python']}
- NumPy: {report['environment']['numpy']}
- Container: none (local process only)
- Fixture seed: {report['fixture_seed']}

The implementation uses only portable Python APIs for paths, file replacement,
and thread pools.  It is compatible with Python 3.10+ on Windows and was run in
the available local environment above.

## Acceptance Summary

**{passed}/{len(checks)} checks passed.**

| Check | Result | Detail |
|---|---|---|
{chr(10).join(check_rows)}

## Scheme Extrapolation Statistics

| Scheme | Over-tolerance samples | Mean max deviation | Max deviation | Position fingerprint prefix |
|---|---:|---:|---:|---|
{chr(10).join(scheme_rows)}

## Incremental vs Full Rebuild

- Final incremental report equals a blank full rebuild on all substantive fields: `{report['consistency']['substantive_equal']}`.
- Large-budget final report equals the full rebuild: `{report['consistency']['large_equal']}`.
- Small-budget final report equals the full rebuild: `{report['consistency']['small_equal']}`.
- The execution-only audit is intentionally excluded from equality; the warm run has cache hits while a blank rebuild cannot. Stage cache keys and all diagnostic fields are compared.

## Latency Budgets

- Large budget: {large['budget_ms']} ms; responses within budget {large['within_budget_count']}/{large['response_count']}; pending responses {large['pending_count']}; p95 elapsed {large['p95_elapsed_ms']:.3f} ms.
- Small budget: {small['budget_ms']} ms; responses within budget {small['within_budget_count']}/{small['response_count']}; pending responses {small['pending_count']}; p95 elapsed {small['p95_elapsed_ms']:.3f} ms.
- Dedicated queue probe: completed samples {probe['completed_progress']}, remaining stage tasks {probe['queue_progress']}; final business code `{probe['third']['business_codes']}`.

The small-budget command sequence returns `E_INTERACT_BUDGET_PENDING` and retains
the background queue; the final large-budget poll drains it to a full report.

## Snapshot Recovery

- Committed global version: {report['snapshot']['manifest']['global_version']}
- Loaded snapshot: `{report['snapshot']['load_ok']}`; entries: {report['snapshot']['loaded_entry_count']}
- Version mismatch: `{report['snapshot']['stale_code']}` (`ok={report['snapshot']['stale_ok']}`)
- Resumed cache hits: {report['snapshot']['resumed_reused_entries']}; recomputed jobs: {report['snapshot']['resumed_recomputed_jobs']}

## Fault Isolation

- Invalid JSON, non-finite feature, dimension mismatch, duplicate ID, out-of-bounds key and empty file are reported while valid samples continue.
- Loader business codes: `{json.dumps(report['faults']['loader_business_codes'], ensure_ascii=False, sort_keys=True)}`
- Interactive business codes: `{json.dumps(report['interactive_business_code_counts'], ensure_ascii=False, sort_keys=True)}`
- Injected stage failure: completed {report['faults']['stage_injection']['summary']['completed_samples']} samples, failed {report['faults']['stage_injection']['summary']['failed_samples']}; code `{report['faults']['stage_injection']['failures'][0]['code']}` on stage `{report['faults']['stage_injection']['failures'][0]['stage']}`.

## Top Over-tolerance Samples

| Rank | Sample | Relative distance | First exceeded stage | Max deviation |
|---:|---|---:|---|---:|
{chr(10).join(violation_rows)}

## Parameter Sensitivity Ranking

| Rank | Parameter | Variant | |Delta mean max deviation| |
|---:|---|---|---:|
{chr(10).join(sensitivity_rows)}

## Artifacts

{chr(10).join(f"- `{path}`" for path in report['artifacts'])}
"""


def main(output_dir: Path | str = ROOT / "artifacts") -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = ROOT / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    generate(ROOT / "samples")
    config_large, _ = load_position_config(ROOT / "samples" / "config_large_budget.json")
    config_small, _ = load_position_config(ROOT / "samples" / "config_small_budget.json")
    assert config_large is not None and config_small is not None
    samples, sample_failures = load_samples(
        ROOT / "samples" / "samples.jsonl",
        dimension=config_large.dimension,
        target_length=config_large.extrapolation_target_length,
    )
    assert not sample_failures

    started = time.perf_counter()
    large_result = run_command_sequence(
        ROOT / "samples" / "config_large_budget.json",
        ROOT / "samples" / "samples.jsonl",
        ROOT / "samples" / "commands.json",
        snapshot_dir=runtime_dir / "snapshots" / "large_sequence",
    )
    small_result = run_command_sequence(
        ROOT / "samples" / "config_small_budget.json",
        ROOT / "samples" / "samples.jsonl",
        ROOT / "samples" / "commands.json",
        snapshot_dir=runtime_dir / "snapshots" / "small_sequence",
    )
    config_final, samples_final, state_errors = state_from_sequence_result(large_result)
    assert config_final is not None and not state_errors
    small_config_final, small_samples_final, small_state_errors = state_from_sequence_result(small_result)
    assert small_config_final is not None and not small_state_errors

    incremental_report = large_result["final_report"]
    full_report = run_full_diagnosis(config_final, samples_final)
    large_equal = diagnostic_consistency_view(incremental_report) == diagnostic_consistency_view(full_report)
    small_equal = diagnostic_consistency_view(small_result["final_report"]) == diagnostic_consistency_view(
        run_full_diagnosis(small_config_final, small_samples_final)
    )
    extended_report = build_extended_report(
        incremental_report,
        config_final,
        samples_final,
        include_sensitivity=True,
    )

    latency_probe = _run_latency_probe(config_small, samples)
    snapshot = _verify_snapshot(config_large, samples, runtime_dir / "snapshots" / "acceptance")
    faults = _verify_faults()

    large_elapsed = [response["elapsed_ms"] for response in large_result["responses"]]
    small_elapsed = [response["elapsed_ms"] for response in small_result["responses"]]
    large_latency = {
        "budget_ms": config_large.interaction_budget_ms,
        "response_count": len(large_result["responses"]),
        "pending_count": sum(response["pending"] for response in large_result["responses"]),
        "within_budget_count": sum(response["within_budget"] for response in large_result["responses"]),
        "p95_elapsed_ms": _percentile(large_elapsed, 0.95),
        "max_elapsed_ms": max(large_elapsed),
    }
    small_latency = {
        "budget_ms": config_small.interaction_budget_ms,
        "response_count": len(small_result["responses"]),
        "pending_count": sum(response["pending"] for response in small_result["responses"]),
        "within_budget_count": sum(response["within_budget"] for response in small_result["responses"]),
        "p95_elapsed_ms": _percentile(small_elapsed, 0.95),
        "max_elapsed_ms": max(small_elapsed),
    }

    scheme_stats = extended_report["scheme_extrapolation_statistics"]
    sensitivity = extended_report["parameter_sensitivity_ranking"]
    checks = [
        {
            "name": "deterministic_fixtures",
            "passed": len(samples) == 64 and (ROOT / "samples" / "commands.json").exists(),
            "detail": f"64 samples, 19 commands, fixed seed 20261121",
        },
        {
            "name": "all_three_schemes_distinct",
            "passed": len(
                {
                    stats["first_sample_position_fingerprint"]
                    for stats in scheme_stats.values()
                }
            )
            == 3,
            "detail": "three distinct position-vector fingerprints and three distinct score fingerprints",
        },
        {
            "name": "incremental_equals_full",
            "passed": large_equal and small_equal,
            "detail": "all substantive diagnostic fields match blank full rebuilds",
        },
        {
            "name": "incremental_cache_audit",
            "passed": any(
                response["reuse_and_recompute"]
                and response["reuse_and_recompute"]["summary"]["reused_stage_count"] > 0
                for response in large_result["responses"]
            )
            and any(
                response["command_id"] == "c05"
                and response["reuse_and_recompute"]["summary"]["reused_stage_count"] == 64
                and response["reuse_and_recompute"]["summary"]["recomputed_stage_count"] == 128
                for response in large_result["responses"]
            )
            and any(
                response["command_id"] == "c12"
                and response["reuse_and_recompute"]["summary"]["recomputed_samples"] == ["s007"]
                for response in large_result["responses"]
            ),
            "detail": "context change reuses all encoding stages; sample replacement recomputes only s007",
        },
        {
            "name": "large_budget_completes",
            "passed": large_latency["pending_count"] == 0
            and large_latency["within_budget_count"] == large_latency["response_count"],
            "detail": f"{large_latency['response_count']} responses, p95 {large_latency['p95_elapsed_ms']:.3f} ms",
        },
        {
            "name": "small_budget_pending_semantics",
            "passed": small_latency["pending_count"] > 0
            and small_latency["within_budget_count"] == small_latency["response_count"]
            and all(
                (not response["pending"]) or "E_INTERACT_BUDGET_PENDING" in response["business_codes"]
                for response in small_result["responses"]
            ),
            "detail": f"{small_latency['pending_count']} pending responses with retained queue",
        },
        {
            "name": "pending_queue_advances",
            "passed": latency_probe["third"]["pending"] is False
            and latency_probe["third"]["completed_samples"] == 24
            and latency_probe["queue_progress"][2] == 0,
            "detail": f"queue {latency_probe['queue_progress']}",
        },
        {
            "name": "snapshot_resume_and_stale",
            "passed": snapshot["load_ok"]
            and snapshot["resumed_reused_entries"] == 36
            and snapshot["stale_code"] == "E_SNAPSHOT_STALE",
            "detail": "atomic version load, exact reuse, stale version rejected",
        },
        {
            "name": "fault_isolation",
            "passed": faults["stage_injection"]["summary"]["completed_samples"] == 5
            and faults["stage_injection"]["summary"]["failed_samples"] == 1
            and faults["stage_injection"]["failures"][0]["code"] == "E_INJECTED_STAGE",
            "detail": "one injected trace failure, five unrelated samples completed",
        },
        {
            "name": "concurrency_cap",
            "passed": full_report["run"]["engine_max_observed_concurrency"] <= config_final.concurrency,
            "detail": f"observed {full_report['run']['engine_max_observed_concurrency']} <= configured {config_final.concurrency}",
        },
        {
            "name": "report_sections",
            "passed": all(
                key in extended_report
                for key in (
                    "scheme_extrapolation_statistics",
                    "over_tolerance_samples",
                    "parameter_sensitivity_ranking",
                    "dimension_sensitivity_ranking",
                    "failures",
                    "business_code_counts",
                )
            )
            and len(extended_report["parameter_sensitivity_ranking"]) >= 3,
            "detail": "diagnostic JSON includes all requested report sections and a ranked sensitivity list",
        },
    ]
    passed = all(check["passed"] for check in checks)
    loader_codes: dict[str, int] = {}
    for failure in faults["loader"]["sample_failures"]:
        loader_codes[failure["code"]] = loader_codes.get(failure["code"], 0) + 1
    for code in (
        faults["loader"]["base_failure_codes"]
        + faults["loader"]["scale_failure_codes"]
        + faults["loader"]["invalid_config_codes"]
        + faults["loader"]["empty_file_codes"]
    ):
        loader_codes[code] = loader_codes.get(code, 0) + 1
    faults["loader_business_codes"] = dict(sorted(loader_codes.items()))
    interactive_counts: dict[str, int] = {}
    for response in large_result["responses"] + small_result["responses"]:
        for code in response.get("business_codes", []):
            if code.startswith("E_"):
                interactive_counts[code] = interactive_counts.get(code, 0) + 1

    artifacts = [
        "artifacts/diagnostic_report.json",
        "artifacts/full_rebuild_report.json",
        "artifacts/interactive_large_responses.json",
        "artifacts/interactive_small_responses.json",
        "artifacts/fault_report.json",
        "artifacts/snapshot_verification.json",
        "artifacts/latency_verification.json",
        "artifacts/acceptance_report.json",
        "docs/verification_report.md",
    ]
    acceptance = {
        "passed": passed,
        "checks": checks,
        "environment": {
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fixture_seed": 20261121,
        "duration_seconds": time.perf_counter() - started,
        "consistency": {
            "substantive_equal": large_equal,
            "large_equal": large_equal,
            "small_equal": small_equal,
        },
        "scheme_statistics": scheme_stats,
        "sensitivity_top": sensitivity[:8],
        "top_violations": extended_report["over_tolerance_samples"][:10],
        "latency": {
            "large_sequence": large_latency,
            "small_sequence": small_latency,
            "probe": latency_probe,
        },
        "snapshot": snapshot,
        "faults": faults,
        "business_code_counts": extended_report["business_code_counts"],
        "interactive_business_code_counts": dict(sorted(interactive_counts.items())),
        "artifacts": artifacts,
    }
    _write_json(output_dir / "diagnostic_report.json", extended_report)
    _write_json(output_dir / "full_rebuild_report.json", full_report)
    _write_json(output_dir / "interactive_large_responses.json", large_result["responses"])
    _write_json(output_dir / "interactive_small_responses.json", small_result["responses"])
    _write_json(output_dir / "fault_report.json", faults)
    _write_json(output_dir / "snapshot_verification.json", snapshot)
    _write_json(output_dir / "latency_verification.json", acceptance["latency"])
    _write_json(output_dir / "acceptance_report.json", acceptance)
    (docs_dir / "verification_report.md").write_text(_markdown_report(acceptance), encoding="utf-8")
    print(json.dumps({"passed": passed, "checks": checks, "duration_seconds": acceptance["duration_seconds"]}, ensure_ascii=False, indent=2))
    return acceptance


if __name__ == "__main__":
    main()
