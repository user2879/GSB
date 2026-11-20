from __future__ import annotations

import copy
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from workbench.errors import LogValidationError, ParameterError, SnapshotError
from workbench.estimator import SampleInput, estimate_windows
from workbench.models import AggregationSpec, MasterRecord, ReplayParams
from workbench.service import WorkbenchService
from workbench.snapshot import snapshot_sequence
from workbench.util import atomic_write_json, canonical_json, fixed_from_ns, read_json, sha256_file, sha256_json
from workbench.validation import (
    load_master_records,
    load_replay_config,
    validate_aggregation_spec,
    validate_master_records,
    validate_progress,
    validate_replay_params,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def _records(count: int, *, timestamp_step: int = 100) -> list[MasterRecord]:
    return [
        MasterRecord(
            commit_seq=index + 1,
            txn_id=f"txn-{index + 1:04d}",
            commit_ts_ms=index * timestamp_step,
            txn_bytes=512 + index,
        )
        for index in range(count)
    ]


def _spec(**overrides: int) -> AggregationSpec:
    values = {
        "window_size": 4,
        "max_samples_per_window": 4,
        "min_samples": 1,
        "latency_ceiling_ms": 1000,
        "scale_power": 3,
        "dominant_threshold_pct": 60,
    }
    values.update(overrides)
    return AggregationSpec(**values)


def _params(spec: AggregationSpec, *, rate: int = 10, concurrency: int = 2) -> ReplayParams:
    return ReplayParams(
        rate_rps=rate,
        concurrency=concurrency,
        clock_mode="virtual_monotonic_ns",
        clock_base_ms=1000,
        estimator_digest=spec.digest,
    )


def _samples_for_delays(delays_ms: list[int], spec: AggregationSpec) -> list[SampleInput]:
    return [
        SampleInput(
            commit_seq=index + 1,
            replay_completed_ns=(index + 1) * 1_000_000 + delay_ms * 1_000_000,
            delay_ns=delay_ms * 1_000_000,
            fixed_delay=fixed_from_ns(delay_ms * 1_000_000, spec.scale_power),
        )
        for index, delay_ms in enumerate(delays_ms)
    ]


def _digest_for_service(service: WorkbenchService, param_version: str) -> str:
    samples = service.repo.get_samples(param_version)
    return sha256_json(snapshot_sequence([sample.to_dict() for sample in samples]))


class ValidationTests(unittest.TestCase):
    def test_log_gap_is_rejected_with_error_seq_and_no_mutation(self) -> None:
        raw = read_json(SAMPLES / "writes.json")["records"]
        with tempfile.TemporaryDirectory() as temporary:
            service = WorkbenchService(temporary)
            try:
                stats_before = service.import_master_records(raw)
                with self.assertRaises(LogValidationError) as caught:
                    service.import_master_records(raw[:4] + raw[5:])
                self.assertEqual(caught.exception.code, "E_LOG")
                self.assertEqual(caught.exception.seq, 6)
                stats_after = service.repo.log_stats()
                self.assertEqual(stats_before["digest"], stats_after["digest"])
                self.assertEqual(len(service.repo.get_master_log()), 12)
            finally:
                service.close()

    def test_negative_timestamp_and_nonpositive_bytes_are_e_log(self) -> None:
        raw = read_json(SAMPLES / "writes.json")["records"]
        bad_time = copy.deepcopy(raw)
        bad_time[2]["commit_ts_ms"] = -1
        with self.assertRaises(LogValidationError) as caught:
            validate_master_records(bad_time)
        self.assertEqual((caught.exception.code, caught.exception.seq), ("E_LOG", 3))
        bad_bytes = copy.deepcopy(raw)
        bad_bytes[3]["txn_bytes"] = 0
        with self.assertRaises(LogValidationError) as caught:
            validate_master_records(bad_bytes)
        self.assertEqual((caught.exception.code, caught.exception.seq), ("E_LOG", 4))

    def test_progress_ahead_is_rejected_and_database_trigger_enforces_it(self) -> None:
        with self.assertRaises(LogValidationError):
            validate_progress(13, 12)
        with tempfile.TemporaryDirectory() as temporary:
            service = WorkbenchService(temporary)
            try:
                records = load_master_records(SAMPLES / "writes.json")
                service.import_master_log(records)
                params, spec, _ = load_replay_config(SAMPLES / "replay_params.json")
                service.run_replay(params, spec)
                import sqlite3

                with self.assertRaises(sqlite3.IntegrityError):
                    with service.repo.transaction() as connection:
                        connection.execute(
                            "UPDATE replay_progress SET last_replayed_seq = 99 "
                            "WHERE param_version = ?",
                            (params.version,),
                        )
                self.assertEqual(service.repo.progress_for(params.version).last_replayed_seq, 12)
            finally:
                service.close()

    def test_invalid_replay_parameter_is_e_param(self) -> None:
        document = read_json(SAMPLES / "replay_params.json")
        document["replay"]["rate_rps"] = 0
        spec = validate_aggregation_spec(document["aggregation"])
        with self.assertRaises(ParameterError) as caught:
            validate_replay_params(document["replay"], spec)
        self.assertEqual(caught.exception.code, "E_PARAM")
        self.assertEqual(caught.exception.field, "replay.rate_rps")

    def test_fixture_manifest_matches(self) -> None:
        manifest = read_json(SAMPLES / "fixtures_manifest.json")
        for filename, digest in manifest["files"].items():
            self.assertEqual(sha256_file(SAMPLES / filename), digest)


class EstimatorTests(unittest.TestCase):
    def test_fixed_point_stats_and_error_segments(self) -> None:
        spec = _spec()
        params = _params(spec)
        records = _records(4)
        samples = _samples_for_delays([100, 200, 300, 400], spec)
        report = estimate_windows(records, samples, params, spec)
        window = report["windows"][0]
        self.assertTrue(window["determinable"])
        self.assertEqual(window["statistics"]["mean"], {"fixed": 250000, "scale": 1000, "ms": "250.000"})
        self.assertEqual(window["statistics"]["p95"]["fixed"], 400000)
        self.assertEqual(window["statistics"]["p95"]["rank_one_based"], 4)
        self.assertEqual(window["statistics"]["max"]["fixed"], 400000)
        budget = window["error_budget"]["mean"]
        self.assertEqual(budget["segments"]["sampling_truncation"]["upper_bound_fixed"], 0)
        self.assertEqual(budget["segments"]["fixed_point_truncation"]["upper_bound_fixed"], 1)
        self.assertEqual(
            budget["total_upper_bound_fixed"],
            sum(item["upper_bound_fixed"] for item in budget["segments"].values()),
        )
        self.assertIn(
            budget["dominant_segment"],
            ("sampling_truncation", "fixed_point_truncation", "replay_rate_discretization", "MIXED"),
        )

    def test_order_and_concurrency_metadata_do_not_change_bytes(self) -> None:
        spec = _spec(max_samples_per_window=3)
        params1 = _params(spec, rate=4, concurrency=2)
        params2 = _params(spec, rate=4, concurrency=9)
        records = _records(12, timestamp_step=30)
        delays = [100, 130, 190, 230, 310, 370, 410, 470, 530, 610, 680, 710]
        samples = _samples_for_delays(delays, spec)
        first = estimate_windows(records, samples, params1, spec)
        reversed_order = estimate_windows(records, list(reversed(samples)), params1, spec)
        different_metadata = estimate_windows(records, samples, params2, spec)
        self.assertEqual(canonical_json(first), canonical_json(reversed_order))
        self.assertEqual(canonical_json(first), canonical_json(different_metadata))
        self.assertEqual(first["result_fingerprint"], different_metadata["result_fingerprint"])

    def test_missing_latency_seq_is_undecidable_not_zero_filled(self) -> None:
        spec = _spec()
        params = _params(spec)
        records = _records(4)
        samples = _samples_for_delays([100, 200, 300, 400], spec)
        del samples[1]
        report = estimate_windows(records, samples, params, spec)
        window = report["windows"][0]
        self.assertFalse(window["determinable"])
        self.assertIn("MISSING_LATENCY_SAMPLE", window["decision_reasons"])
        self.assertEqual(window["basis"]["missing_commit_seqs"], [2])
        self.assertIsNone(window["statistics"]["mean"])
        self.assertIsNone(window["statistics"]["p95"])
        self.assertIsNone(window["statistics"]["max"])
        self.assertIsNone(window["error_budget"])

    def test_below_min_samples_is_undecidable(self) -> None:
        spec = _spec(max_samples_per_window=2, min_samples=3)
        params = _params(spec)
        records = _records(4)
        samples = _samples_for_delays([100, 200, 300, 400], spec)
        window = estimate_windows(records, samples, params, spec)["windows"][0]
        self.assertFalse(window["determinable"])
        self.assertIn("SAMPLE_COUNT_BELOW_MIN", window["decision_reasons"])
        self.assertIsNone(window["statistics"]["mean"])


class ReplayAndSnapshotTests(unittest.TestCase):
    def _setup_run(self, workspace: str) -> tuple[WorkbenchService, ReplayParams, AggregationSpec, dict]:
        service = WorkbenchService(workspace)
        records = load_master_records(SAMPLES / "writes.json")
        service.import_master_log(records)
        params, spec, _ = load_replay_config(SAMPLES / "replay_params.json")
        run = service.run_replay(params, spec)
        return service, params, spec, run

    def test_replay_is_deterministic_and_parameter_sets_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            service1, params1, spec1, _ = self._setup_run(one)
            service2, params2, spec2, _ = self._setup_run(two)
            try:
                self.assertEqual(params1.version, params2.version)
                self.assertEqual(
                    _digest_for_service(service1, params1.version),
                    _digest_for_service(service2, params2.version),
                )
                changed, changed_spec, _ = load_replay_config(SAMPLES / "replay_params_changed.json")
                service1.run_replay(changed, changed_spec)
                samples_a = service1.repo.get_samples(params1.version)
                samples_b = service1.repo.get_samples(changed.version)
                self.assertEqual(len(samples_a), 12)
                self.assertEqual(len(samples_b), 12)
                self.assertTrue(all(item.param_version == params1.version for item in samples_a))
                self.assertTrue(all(item.param_version == changed.version for item in samples_b))
                self.assertNotEqual(params1.version, changed.version)
            finally:
                service1.close()
                service2.close()

    def test_snapshot_whole_replacement_and_historical_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service, params, spec, _ = self._setup_run(temporary)
            try:
                expected = service.estimate(params.version)
                expected_digest = _digest_for_service(service, params.version)
                snapshot = service.save_snapshot(param_version=params.version)
                snapshot_path = Path(snapshot["snapshot_path"])
                source_digest = sha256_file(snapshot_path)
                self.assertFalse(os.access(snapshot_path, os.W_OK) and bool(os.stat(snapshot_path).st_mode & 0o222))
                service.reset_replay_state()
                changed, changed_spec, _ = load_replay_config(SAMPLES / "replay_params_changed.json")
                service.run_replay(changed, changed_spec)
                restore = service.restore_snapshot(snapshot_path)
                self.assertEqual(restore["mode"], "WHOLE_REPLACEMENT")
                self.assertFalse(restore["merge"])
                self.assertTrue(restore["log_digest_match"])
                self.assertTrue(restore["snapshot_retained_readonly"])
                self.assertEqual(sha256_file(snapshot_path), source_digest)
                self.assertEqual(_digest_for_service(service, params.version), expected_digest)
                self.assertEqual(
                    service.estimate(params.version)["result_fingerprint"],
                    expected["result_fingerprint"],
                )
                verification = service.verify_snapshot(snapshot_path)
                self.assertTrue(verification["reproducible"])
                self.assertIsNone(verification["first_mismatch"])
                fields = [item["field"] for item in restore["replaced_fields"]]
                self.assertIn("replay_progress", fields)
                self.assertIn("replay_samples", fields)
            finally:
                service.close()

    def test_snapshot_scope_mismatch_is_e_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service, params, spec, _ = self._setup_run(temporary)
            try:
                snapshot = service.save_snapshot(param_version=params.version)
                document = read_json(snapshot["snapshot_path"])
                document["estimator"]["version"] = "fixed-window-v0"
                bad_path = Path(temporary) / "bad-scope.json"
                atomic_write_json(bad_path, document)
                state_before = service.repo.replay_state_counts()
                with self.assertRaises(SnapshotError) as caught:
                    service.restore_snapshot(bad_path)
                self.assertEqual(caught.exception.code, "E_SNAPSHOT")
                self.assertEqual(service.repo.replay_state_counts(), state_before)
                replay_report = service.verify_snapshot(bad_path)
                self.assertEqual(replay_report["code"], "E_REPLAY")
            finally:
                service.close()

    def test_tampered_sample_reports_first_seq_and_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service, params, spec, _ = self._setup_run(temporary)
            try:
                snapshot = service.save_snapshot(param_version=params.version)
                document = read_json(snapshot["snapshot_path"])
                document["samples"][0]["delay_ns"] += 1
                document["samples"][0]["fixed_delay"] = fixed_from_ns(
                    document["samples"][0]["delay_ns"], spec.scale_power
                )
                bad_path = Path(temporary) / "tampered-sample.json"
                atomic_write_json(bad_path, document)
                verification = service.verify_snapshot(bad_path)
                self.assertEqual(verification["code"], "E_REPLAY")
                self.assertFalse(verification["reproducible"])
                self.assertEqual(verification["first_mismatch"]["commit_seq"], 1)
                self.assertEqual(verification["first_mismatch"]["field"], "sample.delay_ns")
            finally:
                service.close()

    def test_tampered_statistics_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service, params, spec, _ = self._setup_run(temporary)
            try:
                snapshot = service.save_snapshot(param_version=params.version)
                document = read_json(snapshot["snapshot_path"])
                document["estimation"]["windows"][0]["statistics"]["mean"]["fixed"] += 1
                bad_path = Path(temporary) / "tampered-stat.json"
                atomic_write_json(bad_path, document)
                verification = service.verify_snapshot(bad_path)
                self.assertEqual(verification["code"], "E_REPLAY")
                self.assertIsNone(verification["first_mismatch"]["commit_seq"])
                self.assertIn("statistics.mean.fixed", verification["first_mismatch"]["field"])
            finally:
                service.close()

    def test_log_digest_mismatch_restores_wholesale_but_is_not_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service, params, spec, _ = self._setup_run(temporary)
            try:
                snapshot = service.save_snapshot(param_version=params.version)
                changed_records = load_master_records(SAMPLES / "writes.json")
                changed_records[0] = MasterRecord(
                    commit_seq=1,
                    txn_id=changed_records[0].txn_id,
                    commit_ts_ms=changed_records[0].commit_ts_ms + 1,
                    txn_bytes=changed_records[0].txn_bytes,
                )
                service.import_master_log(changed_records)
                self.assertEqual(service.repo.list_samples(), [])
                restore = service.restore_snapshot(snapshot["snapshot_path"])
                self.assertFalse(restore["log_digest_match"])
                self.assertEqual(restore["status_override"], "RESTORED_STALE_LOG_MISMATCH")
                self.assertEqual(restore["mode"], "WHOLE_REPLACEMENT")
                self.assertEqual(len(service.repo.get_samples(params.version)), 12)
                verification = service.verify_snapshot(snapshot["snapshot_path"])
                self.assertEqual(verification["code"], "E_REPLAY")
                self.assertEqual(verification["reason"], "快照日志摘要与当前磁盘日志不一致")
            finally:
                service.close()

    def test_forced_termination_recovery_and_resume_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            normal_workspace = root_path / "normal"
            crash_workspace = root_path / "crash"
            normal, params, spec, _ = self._setup_run(str(normal_workspace))
            try:
                normal_digest = _digest_for_service(normal, params.version)
                normal_result = normal.estimate(params.version)["result_fingerprint"]
            finally:
                normal.close()

            command = [
                sys.executable,
                "-m",
                "workbench.cli",
                "run",
                "--workspace",
                str(crash_workspace),
                "--writes",
                str(SAMPLES / "writes.json"),
                "--config",
                str(SAMPLES / "replay_params.json"),
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
            self.assertEqual(child.returncode, 137, child.stderr)
            service = WorkbenchService(str(crash_workspace))
            try:
                self.assertTrue(service.recovery_report["requires_rerun"])
                self.assertFalse(service.recovery_report["exit_report_available"])
                progress = service.repo.progress_for(params.version)
                self.assertEqual(progress.status, "REQUIRES_RERUN")
                self.assertEqual(len(service.repo.get_samples(params.version)), 4)
                unknown = [batch for batch in service.repo.get_batches(params.version) if batch.status == "UNKNOWN"]
                self.assertEqual(len(unknown), 1)
                service.run_replay(params, spec)
                self.assertEqual(_digest_for_service(service, params.version), normal_digest)
                self.assertEqual(
                    service.estimate(params.version)["result_fingerprint"], normal_result
                )
            finally:
                service.close()


if __name__ == "__main__":
    unittest.main()
