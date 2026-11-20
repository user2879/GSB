#!/usr/bin/env python3
"""Repeatable end-to-end demonstration for the replication-lag workbench."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

from workbench.errors import LogValidationError, ParameterError, SnapshotError, WorkbenchError
from workbench.jsonlog import StructuredLogger
from workbench.service import WorkbenchService
from workbench.snapshot import snapshot_sequence
from workbench.util import (
    atomic_write_json,
    canonical_json,
    read_json,
    sha256_file,
    sha256_json,
)
from workbench.validation import (
    load_master_records,
    load_replay_config,
    validate_aggregation_spec,
    validate_master_records,
    validate_replay_params,
)


ROOT = Path(__file__).resolve().parent


def _json_write(path: Path, value: Any, *, read_only: bool = False) -> None:
    atomic_write_json(path, value, pretty=True, read_only=read_only)


def _sample_digest(service: WorkbenchService, param_version: str) -> str:
    samples = service.repo.get_samples(param_version)
    return sha256_json(snapshot_sequence([sample.to_dict() for sample in samples]))


def _estimation_bytes(estimation: dict[str, Any]) -> bytes:
    return canonical_json(estimation).encode("utf-8")


def _fixture_check(fixture_dir: Path) -> dict[str, Any]:
    manifest_path = fixture_dir / "fixtures_manifest.json"
    manifest = read_json(manifest_path)
    results: dict[str, Any] = {}
    for filename, expected in sorted(manifest["files"].items()):
        actual = sha256_file(fixture_dir / filename)
        results[filename] = {
            "expected": expected,
            "actual": actual,
            "match": actual == expected,
        }
    return {
        "manifest": str(manifest_path),
        "files": results,
        "all_match": all(item["match"] for item in results.values()),
    }


def _write_error_injection(
    logger: StructuredLogger,
    report_path: Path,
    *,
    expected_code: str,
    exc: WorkbenchError,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "code": exc.code,
        "injection": True,
        "expected_code": expected_code,
        "observed_code": exc.code,
        "matched": exc.code == expected_code,
        "error": exc.as_dict(),
    }
    if extra:
        report.update(extra)
    logger.event("injection_reported", **report)
    _json_write(report_path, report)
    return report


def run_injection_log_truncation(
    service: WorkbenchService,
    fixture_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    document = read_json(fixture_dir / "writes.json")
    records = list(document["records"])
    records.pop(4)  # create a gap after seq 4; whole batch must be rejected
    before = service.repo.replay_state_counts()
    try:
        validate_master_records(records)
    except LogValidationError as exc:
        after = service.repo.replay_state_counts()
        assert before == after
        return _write_error_injection(
            service.logger,
            report_path,
            expected_code="E_LOG",
            exc=exc,
            extra={
                "injected_change": "removed middle commit_seq 5",
                "state_unchanged": before == after,
                "state_counts": before,
            },
        )
    raise AssertionError("E_LOG injection unexpectedly accepted")


def run_injection_invalid_param(
    service: WorkbenchService,
    fixture_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    document = read_json(fixture_dir / "replay_params.json")
    document["replay"]["rate_rps"] = 0
    before = service.repo.replay_state_counts()
    try:
        spec = validate_aggregation_spec(document["aggregation"])
        validate_replay_params(document["replay"], spec)
    except ParameterError as exc:
        after = service.repo.replay_state_counts()
        assert before == after
        return _write_error_injection(
            service.logger,
            report_path,
            expected_code="E_PARAM",
            exc=exc,
            extra={
                "injected_change": "replay.rate_rps = 0",
                "state_unchanged": before == after,
                "state_counts": before,
            },
        )
    raise AssertionError("E_PARAM injection unexpectedly accepted")


def run_injection_snapshot_scope(
    service: WorkbenchService,
    source_snapshot: Path,
    report_path: Path,
) -> dict[str, Any]:
    document = read_json(source_snapshot)
    document["estimator"]["version"] = "fixed-window-v0-incompatible"
    bad_snapshot = report_path.with_name("injected_bad_estimator_snapshot.json")
    _json_write(bad_snapshot, document, read_only=False)
    os.chmod(bad_snapshot, 0o644)
    before = service.repo.replay_state_counts()
    try:
        service.restore_snapshot(bad_snapshot)
    except SnapshotError as exc:
        after = service.repo.replay_state_counts()
        assert before == after
        return _write_error_injection(
            service.logger,
            report_path,
            expected_code="E_SNAPSHOT",
            exc=exc,
            extra={
                "injected_change": "estimator.version changed in a copied snapshot",
                "bad_snapshot": str(bad_snapshot),
                "state_unchanged": before == after,
                "state_counts": before,
            },
        )
    raise AssertionError("E_SNAPSHOT injection unexpectedly accepted")


def run_injection_replay_tamper(
    service: WorkbenchService,
    source_snapshot: Path,
    report_path: Path,
) -> dict[str, Any]:
    document = read_json(source_snapshot)
    document["samples"][0]["delay_ns"] += 1
    bad_snapshot = report_path.with_name("injected_tampered_sample_snapshot.json")
    _json_write(bad_snapshot, document, read_only=False)
    os.chmod(bad_snapshot, 0o644)
    report = service.verify_snapshot(bad_snapshot)
    report.update(
        {
            "code": report.get("code", "E_REPLAY"),
            "injection": True,
            "expected_code": "E_REPLAY",
            "observed_code": report.get("code"),
            "matched": report.get("code") == "E_REPLAY",
            "injected_change": "samples[0].delay_ns += 1",
            "bad_snapshot": str(bad_snapshot),
        }
    )
    assert report["matched"]
    _json_write(report_path, report)
    service.logger.event("injection_reported", **report)
    return report


def _run_crash_case(
    fixture_dir: Path,
    main_artifacts: Path,
    expected_sample_digest: str,
    expected_result_fingerprint: str,
) -> dict[str, Any]:
    crash_workspace = main_artifacts / "crash-workspace"
    if crash_workspace.exists():
        shutil.rmtree(crash_workspace)
    crash_workspace.mkdir(parents=True)
    command = [
        sys.executable,
        "-m",
        "workbench.cli",
        "run",
        "--workspace",
        str(crash_workspace),
        "--writes",
        str(fixture_dir / "writes.json"),
        "--config",
        str(fixture_dir / "replay_params.json"),
        "--crash-after",
        "2",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    child = subprocess.run(
        command,
        cwd=str(ROOT),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert child.returncode == 137, (
        f"expected forced exit 137, got {child.returncode}: {child.stderr}"
    )

    params, spec, _ = load_replay_config(fixture_dir / "replay_params.json")
    service = WorkbenchService(crash_workspace)
    try:
        recovery = service.recovery_report
        progress = service.repo.progress_for(params.version)
        assert progress is not None
        unknown = [batch for batch in service.repo.get_batches(params.version) if batch.status == "UNKNOWN"]
        samples_before_resume = service.repo.get_samples(params.version)
        assert recovery["requires_rerun"]
        assert progress.status == "REQUIRES_RERUN"
        assert len(unknown) == 1
        assert len(samples_before_resume) == 4
        assert not recovery["exit_report_available"]

        resumed = service.run_replay(params, spec)
        estimation = service.estimate(params.version)
        resumed_digest = _sample_digest(service, params.version)
        assert resumed_digest == expected_sample_digest
        assert estimation["result_fingerprint"] == expected_result_fingerprint
        return {
            "child_returncode": child.returncode,
            "recovery": recovery,
            "progress_before_resume": progress.to_dict(),
            "unknown_batches": [batch.to_dict() for batch in unknown],
            "samples_before_resume": len(samples_before_resume),
            "resumed": resumed,
            "sample_sequence_digest_before": expected_sample_digest,
            "sample_sequence_digest_after": resumed_digest,
            "result_fingerprint_before": expected_result_fingerprint,
            "result_fingerprint_after": estimation["result_fingerprint"],
            "exact_after_resume": (
                resumed_digest == expected_sample_digest
                and estimation["result_fingerprint"] == expected_result_fingerprint
            ),
            "post_restart_only": True,
            "crash_exit_report_claimed": False,
        }
    finally:
        service.close()


def run_full_demo(
    workspace: Path,
    *,
    include_crash: bool,
    include_injections: bool,
) -> dict[str, Any]:
    fixture_dir = workspace / "samples"
    if not fixture_dir.exists():
        fixture_dir = ROOT / "samples"
    artifacts = workspace / "artifacts" / "workbench"
    if artifacts.exists():
        shutil.rmtree(artifacts)
    report_dir = artifacts / "reports"
    service = WorkbenchService(workspace)
    try:
        fixtures = _fixture_check(fixture_dir)
        assert fixtures["all_match"]

        records = load_master_records(fixture_dir / "writes.json")
        log_stats = service.import_master_log(records, source=str(fixture_dir / "writes.json"))
        params_a, spec_a, _ = load_replay_config(fixture_dir / "replay_params.json")
        params_b, spec_b, _ = load_replay_config(fixture_dir / "replay_params_changed.json")

        # Scenario 1: first deterministic replay and estimate.
        run_a = service.run_replay(params_a, spec_a)
        estimate_a = service.estimate(params_a.version)
        digest_a = _sample_digest(service, params_a.version)
        report_a = {
            "scenario": "01_initial_replay",
            "fixtures": fixtures,
            "log": log_stats,
            "run": run_a,
            "sample_sequence_digest": digest_a,
            "estimate": estimate_a,
        }
        _json_write(report_dir / "01_initial_estimate.json", report_a)

        # Scenario 2: rate/concurrency change must produce an isolated sample set.
        run_b = service.run_replay(params_b, spec_b)
        estimate_b = service.estimate(params_b.version)
        samples_a = service.repo.get_samples(params_a.version)
        samples_b = service.repo.get_samples(params_b.version)
        isolation_ok = (
            all(sample.param_version == params_a.version for sample in samples_a)
            and all(sample.param_version == params_b.version for sample in samples_b)
            and params_a.version != params_b.version
            and estimate_a["result_fingerprint"] != estimate_b["result_fingerprint"]
        )
        assert isolation_ok
        digest_b = _sample_digest(service, params_b.version)
        report_b = {
            "scenario": "02_parameter_change_recompute",
            "run": run_b,
            "sample_sequence_digest": digest_b,
            "estimate": estimate_b,
            "parameter_isolation": {
                "param_version_a": params_a.version,
                "param_version_b": params_b.version,
                "samples_a": len(samples_a),
                "samples_b": len(samples_b),
                "result_a": estimate_a["result_fingerprint"],
                "result_b": estimate_b["result_fingerprint"],
                "isolated": isolation_ok,
            },
        }
        _json_write(report_dir / "02_recomputed_estimate.json", report_b)

        # Scenario 3: immutable snapshot, whole replacement, historical replay.
        snapshot = service.save_snapshot(param_version=params_b.version)
        snapshot_path = Path(snapshot["snapshot_path"])
        source_digest_before = sha256_file(snapshot_path)
        service.reset_replay_state()
        service.run_replay(params_a, spec_a)
        before_restore = {
            "progress": [item.to_dict() for item in service.repo.list_progress()],
            "sample_count": len(service.repo.list_samples()),
            "param_versions": sorted({sample.param_version for sample in service.repo.list_samples()}),
        }
        restore_report = service.restore_snapshot(snapshot_path)
        verify_report = service.verify_snapshot(snapshot_path)
        restored_estimate = service.estimate(params_b.version)
        source_digest_after = sha256_file(snapshot_path)
        restored_digest_b = _sample_digest(service, params_b.version)
        restore_ok = (
            restore_report["mode"] == "WHOLE_REPLACEMENT"
            and not restore_report["merge"]
            and restore_report["snapshot_retained_readonly"]
            and verify_report["reproducible"]
            and restored_estimate["result_fingerprint"] == estimate_b["result_fingerprint"]
            and restored_digest_b == digest_b
            and source_digest_before == source_digest_after
        )
        assert restore_ok
        report_c = {
            "scenario": "03_snapshot_restore_and_historical_replay",
            "snapshot": snapshot,
            "snapshot_digest_before_restore": source_digest_before,
            "snapshot_digest_after_restore": source_digest_after,
            "before_restore": before_restore,
            "restore_report": restore_report,
            "historical_replay": verify_report,
            "restored_estimate": restored_estimate,
            "expected_result_fingerprint": estimate_b["result_fingerprint"],
            "expected_sample_sequence_digest": digest_b,
            "restored_result_fingerprint": restored_estimate["result_fingerprint"],
            "restored_sample_sequence_digest": restored_digest_b,
            "exact_match": restore_ok,
        }
        _json_write(report_dir / "03_snapshot_restore_replay.json", report_c)

        crash_report = None
        if include_crash:
            crash_report = _run_crash_case(
                fixture_dir,
                artifacts,
                digest_a,
                estimate_a["result_fingerprint"],
            )
            _json_write(report_dir / "04_crash_recovery.json", crash_report)

        injection_reports: dict[str, Any] = {}
        if include_injections:
            injection_reports["E_LOG"] = run_injection_log_truncation(
                service, fixture_dir, report_dir / "injection_E_LOG.json"
            )
            injection_reports["E_PARAM"] = run_injection_invalid_param(
                service, fixture_dir, report_dir / "injection_E_PARAM.json"
            )
            injection_reports["E_SNAPSHOT"] = run_injection_snapshot_scope(
                service, snapshot_path, report_dir / "injection_E_SNAPSHOT.json"
            )
            injection_reports["E_REPLAY"] = run_injection_replay_tamper(
                service, snapshot_path, report_dir / "injection_E_REPLAY.json"
            )

        # Re-aggregating the same sample set after order reversal must be byte-identical.
        from workbench.estimator import SampleInput, estimate_windows

        samples = service.repo.get_samples(params_b.version)
        records_for_b = service.repo.get_master_log()
        inputs = [
            SampleInput(sample.commit_seq, sample.replay_completed_ns, sample.delay_ns, sample.fixed_delay)
            for sample in samples
        ]
        first = estimate_windows(records_for_b, inputs, params_b, spec_b)
        second = estimate_windows(records_for_b, list(reversed(inputs)), params_b, spec_b)
        order_invariant = _estimation_bytes(first) == _estimation_bytes(second)
        assert order_invariant

        checks = {
            "fixture_digests_match": fixtures["all_match"],
            "first_run_complete": run_a["progress"]["status"] == "COMPLETE",
            "parameter_isolation": isolation_ok,
            "snapshot_whole_replacement": restore_report["mode"] == "WHOLE_REPLACEMENT",
            "snapshot_source_unchanged": source_digest_before == source_digest_after,
            "historical_replay_exact": verify_report["reproducible"],
            "restored_estimate_byte_exact": (
                restored_estimate["result_fingerprint"] == estimate_b["result_fingerprint"]
            ),
            "sample_order_invariant": order_invariant,
            "crash_recovery_exact": crash_report is None or crash_report["exact_after_resume"],
            "all_injected_codes_matched": (
                not injection_reports or all(item["matched"] for item in injection_reports.values())
            ),
        }
        assert all(checks.values())
        environment = {
            "python": sys.version,
            "sqlite3": sqlite3.sqlite_version,
            "numpy_available": importlib.util.find_spec("numpy") is not None,
            "tkinter_available": importlib.util.find_spec("tkinter") is not None,
            "workspace": str(workspace),
        }
        summary = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "environment": environment,
            "paths": {
                "sqlite": str(service.db_path),
                "structured_log": str(service.log_path),
                "reports": str(report_dir),
                "snapshots": str(service.snapshot_dir),
            },
            "scenario_fingerprints": {
                "initial": estimate_a["result_fingerprint"],
                "changed": estimate_b["result_fingerprint"],
                "restored": restored_estimate["result_fingerprint"],
            },
            "sample_sequence_digests": {
                "initial": digest_a,
                "changed": digest_b,
                "restored": restored_digest_b,
            },
            "injection_codes": {
                key: {"expected": value["expected_code"], "observed": value["observed_code"]}
                for key, value in injection_reports.items()
            },
            "crash_recovery": crash_report,
            "report_files": sorted(path.name for path in report_dir.glob("*.json")),
        }
        _json_write(report_dir / "demo_summary.json", summary)
        service.logger.event("demo_completed", status=summary["status"], checks=checks)
        return summary
    finally:
        service.close()


def run_single_injection(workspace: Path, kind: str) -> dict[str, Any]:
    fixture_dir = workspace / "samples"
    if not fixture_dir.exists():
        fixture_dir = ROOT / "samples"
    injection_workspace = workspace / "artifacts" / "workbench" / f"injection-{kind.lower()}"
    if injection_workspace.exists():
        shutil.rmtree(injection_workspace)
    service = WorkbenchService(injection_workspace)
    try:
        records = load_master_records(fixture_dir / "writes.json")
        service.import_master_log(records, source=str(fixture_dir / "writes.json"))
        params, spec, _ = load_replay_config(fixture_dir / "replay_params.json")
        service.run_replay(params, spec)
        snapshot = service.save_snapshot(param_version=params.version)
        report_dir = service.report_dir
        if kind == "E_LOG":
            result = run_injection_log_truncation(
                service, fixture_dir, report_dir / "injection_E_LOG.json"
            )
        elif kind == "E_PARAM":
            result = run_injection_invalid_param(
                service, fixture_dir, report_dir / "injection_E_PARAM.json"
            )
        elif kind == "E_SNAPSHOT":
            result = run_injection_snapshot_scope(
                service, Path(snapshot["snapshot_path"]), report_dir / "injection_E_SNAPSHOT.json"
            )
        elif kind == "E_REPLAY":
            result = run_injection_replay_tamper(
                service, Path(snapshot["snapshot_path"]), report_dir / "injection_E_REPLAY.json"
            )
        else:
            raise ValueError(f"unknown injection kind: {kind}")
        result["report_dir"] = str(report_dir)
        result["structured_log"] = str(service.log_path)
        return result
    finally:
        service.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=str(ROOT))
    parser.add_argument(
        "--inject",
        choices=("E_LOG", "E_PARAM", "E_SNAPSHOT", "E_REPLAY"),
        help="run only one isolated exception-injection scenario",
    )
    parser.add_argument("--skip-crash", action="store_true")
    parser.add_argument("--skip-injections", action="store_true")
    args = parser.parse_args(argv)
    workspace = Path(args.workspace).resolve()
    if args.inject:
        result = run_single_injection(workspace, args.inject)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0 if result.get("matched") else 1
    summary = run_full_demo(
        workspace,
        include_crash=not args.skip_crash,
        include_injections=not args.skip_injections,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
