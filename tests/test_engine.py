from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from feature_backfill.engine import (
    BackfillEngine,
    RunOptions,
    compare_reports,
    execution_comparison,
)
from feature_backfill.errors import AppError, E_BATCH_QUOTA_EXHAUSTED, E_SNAPSHOT_STALE
from feature_backfill.fingerprints import sha256_json

from tests.helpers import (
    ROOT,
    empty_plan,
    load_fixture,
    load_injection_set,
)


def run_engine(
    snapshot_root: Path,
    *,
    fixture: dict | None = None,
    config=None,
    plan=None,
    injections=None,
    resume: bool = False,
    crash_batch: int | None = None,
):
    fixture = fixture or load_fixture()
    engine = BackfillEngine(
        base_rows=fixture["rows"],
        base_features=fixture["features"],
        raw_fields=fixture["raw_fields"],
        config=config or fixture["config"],
        plan=fixture["plan"] if plan is None else plan,
        base_failures=fixture["row_failures"],
        injections=injections or [],
        snapshot_root=snapshot_root,
    )
    return engine.run(
        RunOptions(
            resume=resume,
            crash_before_commit_batch=crash_batch,
            write_report=False,
        )
    )


class EngineTests(unittest.TestCase):
    def test_incremental_plan_equals_final_full_backfill(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            incremental = run_engine(temp_path / "incremental")
            final_fixture = load_fixture(
                ROOT / "data" / "final",
                plan_path=ROOT / "data" / "scenarios" / "empty_plan.json",
            )
            full = run_engine(temp_path / "full", fixture=final_fixture)
            comparison = compare_reports(incremental, full)
            self.assertTrue(comparison["equal"], comparison)
            self.assertEqual(incremental["status"], "completed")
            self.assertEqual(full["status"], "completed")
            self.assertEqual(len(incremental["results"]), 259)

    def test_parallel_and_serial_have_identical_execution_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            loaded = load_fixture()
            parallel = run_engine(temp_path / "parallel", fixture=loaded)
            serial_config = replace(loaded["config"], max_workers=1)
            serial = run_engine(
                temp_path / "serial",
                fixture=loaded,
                config=serial_config,
            )
            self.assertEqual(
                execution_comparison(parallel), execution_comparison(serial)
            )
            self.assertEqual(parallel["run"]["max_observed_concurrency"], 4)
            self.assertEqual(serial["run"]["max_observed_concurrency"], 1)
            self.assertLessEqual(
                parallel["run"]["max_observed_concurrency"],
                parallel["run"]["max_workers"],
            )

    def test_repeated_runs_are_field_for_field_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            first = run_engine(temp_path / "first")
            second = run_engine(temp_path / "second")
            self.assertTrue(compare_reports(first, second)["equal"])
            self.assertTrue(execution_comparison(first) == execution_comparison(second))

    def test_invalidation_manifest_matches_append_correction_window_rules(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            report = run_engine(Path(temp) / "precise")
            events = report["invalidation_manifest"]
            e001_appends = [
                event
                for event in events
                if event.get("batch_id") == 1
                and event.get("operation") == "append"
                and event.get("entity_id") == "E001"
            ]
            self.assertEqual(len(e001_appends), 2)
            self.assertTrue(all(not event["invalidated_blocks"] for event in e001_appends))
            self.assertTrue(
                all(len(event["new_targets"]) == 7 for event in e001_appends)
            )
            correction = next(
                event
                for event in events
                if event.get("batch_id") == 2 and event.get("operation") == "correct"
            )
            self.assertEqual(correction["changed_fields"], ["amount"])
            self.assertEqual(
                set(correction["affected_features"]),
                {
                    "amount_chain_5000",
                    "amount_count_5000",
                    "amount_latest_5000",
                    "amount_mean_5000",
                    "amount_sum_5000",
                },
            )
            self.assertEqual(len(correction["invalidated_blocks"]), 15)
            adjustment = next(
                event
                for event in events
                if event.get("batch_id") == 3 and event.get("operation") == "window_adjust"
            )
            self.assertEqual(adjustment["affected_features"], ["amount_sum_5000"])
            self.assertEqual(adjustment["entities"], [f"E{index:03d}" for index in range(1, 9)])
            score_adjustment = next(
                event
                for event in events
                if event.get("batch_id") == 7 and event.get("operation") == "window_adjust"
            )
            self.assertEqual(score_adjustment["affected_features"], ["score_mean_5000"])
            self.assertEqual(len(score_adjustment["entities"]), 9)
            persistent_reuse = [
                event
                for event in report["cache"]["block_executions"]
                if event.get("kind") == "reuse"
                and event.get("reuse_source") == "persistent_snapshot"
            ]
            self.assertTrue(persistent_reuse)

    def test_failure_order_is_identical_serial_and_parallel(self) -> None:
        loaded = load_fixture()
        injections = load_injection_set("reference")
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            parallel = run_engine(
                temp_path / "parallel",
                fixture=loaded,
                plan=empty_plan(),
                injections=injections,
            )
            serial = run_engine(
                temp_path / "serial",
                fixture=loaded,
                config=replace(loaded["config"], max_workers=1),
                plan=empty_plan(),
                injections=injections,
            )
            self.assertEqual(parallel["failures"], serial["failures"])
            self.assertEqual(parallel["pending"], serial["pending"])
            self.assertEqual(
                [hit["injection_id"] for hit in parallel["injection_hits"]],
                [hit["injection_id"] for hit in serial["injection_hits"]],
            )

    def test_quota_exhaustion_keeps_stable_pending_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            loaded = load_fixture()
            quota_config = replace(loaded["config"], batch_point_quota=60)
            report = run_engine(
                Path(temp) / "quota",
                fixture=loaded,
                config=quota_config,
            )
            self.assertEqual(report["status"], "partial")
            self.assertEqual(
                report["pending"],
                ["E002", "E003", "E004", "E005", "E006", "E007", "E008", "E009"],
            )
            self.assertTrue(
                all(point <= 60 for point in report["consumed_points_by_batch"].values())
            )
            quota_failures = [
                failure
                for failure in report["failures"]
                if failure["code"] == E_BATCH_QUOTA_EXHAUSTED
            ]
            self.assertEqual(len(quota_failures), 8)
            self.assertTrue(
                all(failure["entity_id"] in report["pending"] for failure in quota_failures)
            )
            allocations = report["quota_allocations"]
            for batch_id in {item["batch_id"] for item in allocations}:
                batch_items = [item for item in allocations if item["batch_id"] == batch_id]
                self.assertLessEqual(max(item["cumulative_points"] for item in batch_items), 60)

    def test_snapshot_rejects_config_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            loaded = load_fixture()
            run_engine(temp_path / "snap", fixture=loaded)
            with self.assertRaises(AppError) as context:
                run_engine(
                    temp_path / "snap",
                    fixture=loaded,
                    config=replace(loaded["config"], max_workers=1),
                    resume=True,
                )
            self.assertEqual(context.exception.code, E_SNAPSHOT_STALE)

    def test_snapshot_rejects_lease_version_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            loaded = load_fixture()
            run_engine(temp_path / "snap", fixture=loaded)
            current = json.loads((temp_path / "snap" / "CURRENT").read_text())
            generation = current["generation"]
            gen_dir = temp_path / "snap" / "generations" / f"gen-{generation:08d}"
            manifest = json.loads((gen_dir / "manifest.json").read_text())
            manifest["lease_version"] = "0" * 32
            manifest.pop("manifest_sha256", None)
            manifest_sha = sha256_json(manifest)
            manifest["manifest_sha256"] = manifest_sha
            (gen_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            (gen_dir / "COMMITTED").write_text(manifest_sha + "\n")
            current["manifest_sha256"] = manifest_sha
            (temp_path / "snap" / "CURRENT").write_text(
                json.dumps(current, sort_keys=True, separators=(",", ":"))
            )
            with self.assertRaises(AppError) as context:
                run_engine(temp_path / "snap", fixture=loaded, resume=True)
            self.assertEqual(context.exception.code, E_SNAPSHOT_STALE)

    def test_snapshot_rejects_state_hash_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            loaded = load_fixture()
            run_engine(temp_path / "snap", fixture=loaded)
            current = json.loads((temp_path / "snap" / "CURRENT").read_text())
            generation = current["generation"]
            state_path = (
                temp_path
                / "snap"
                / "generations"
                / f"gen-{generation:08d}"
                / "state.json"
            )
            state_path.write_text(state_path.read_text() + " ")
            with self.assertRaises(AppError) as context:
                run_engine(temp_path / "snap", fixture=loaded, resume=True)
            self.assertEqual(context.exception.code, E_SNAPSHOT_STALE)

    def test_forced_process_exit_resumes_from_last_committed_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            loaded = load_fixture()
            full = run_engine(temp_path / "full", fixture=loaded)
            snapshot = temp_path / "crash-snap"
            report_path = temp_path / "resume-report.json"
            command = [
                sys.executable,
                "-m",
                "feature_backfill",
                "run",
                "--samples",
                str(ROOT / "data" / "default" / "samples.jsonl"),
                "--features",
                str(ROOT / "data" / "default" / "features.json"),
                "--config",
                str(ROOT / "data" / "default" / "config.yaml"),
                "--plan",
                str(ROOT / "data" / "default" / "plan.json"),
                "--snapshot-dir",
                str(snapshot),
                "--crash-before-commit-batch",
                "3",
            ]
            crashed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(crashed.returncode, 137, crashed.stderr)
            pointer = json.loads((snapshot / "CURRENT").read_text())
            self.assertEqual(pointer["generation"], 2)
            self.assertTrue(
                list(snapshot.glob(".tmp-crash-*")),
                "crash hook should leave an uncommitted temp generation",
            )
            resumed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "feature_backfill",
                    "run",
                    "--samples",
                    str(ROOT / "data" / "default" / "samples.jsonl"),
                    "--features",
                    str(ROOT / "data" / "default" / "features.json"),
                    "--config",
                    str(ROOT / "data" / "default" / "config.yaml"),
                    "--plan",
                    str(ROOT / "data" / "default" / "plan.json"),
                    "--snapshot-dir",
                    str(snapshot),
                    "--resume",
                    "--report",
                    str(report_path),
                ],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr + resumed.stdout)
            report = json.loads(report_path.read_text())
            self.assertTrue(report["run"]["resumed"])
            self.assertEqual(report["run"]["resumed_from_generation"], 2)
            self.assertTrue(report["run"]["uncommitted_temporary_generations"])
            self.assertTrue(compare_reports(full, report)["equal"])

    def test_each_injection_is_isolated_and_reports_its_id(self) -> None:
        expected = {
            "nonfinite": ("INJ-001-NONFINITE", "E_ROW_INVALID"),
            "timestamp_reversal": ("INJ-002-TIMESTAMP-REVERSAL", "E_ROW_INVALID"),
            "lease_ttl_zero": ("INJ-003-LEASE-TTL-ZERO", "E_CONFIG_INVALID"),
            "cache_budget_zero": ("INJ-004-CACHE-BUDGET-ZERO", "E_CONFIG_INVALID"),
        }
        loaded = load_fixture()
        with tempfile.TemporaryDirectory() as temp:
            for name, (injection_id, code) in expected.items():
                with self.subTest(injection=name):
                    report = run_engine(
                        Path(temp) / name,
                        fixture=loaded,
                        plan=empty_plan(),
                        injections=load_injection_set(name),
                    )
                    self.assertEqual(len(report["injection_hits"]), 1)
                    hit = report["injection_hits"][0]
                    self.assertEqual(hit["injection_id"], injection_id)
                    self.assertEqual(hit["observed_code"], code)
                    self.assertEqual(report["pending"], [])
                    self.assertEqual(report["status"], "completed_with_failures")
                    matching = [
                        failure
                        for failure in report["failures"]
                        if failure.get("injection_id") == injection_id
                    ]
                    self.assertEqual(len(matching), 1)
                    self.assertEqual(matching[0]["code"], code)
                    self.assertEqual(len(report["completed_entities"]), 7)


if __name__ == "__main__":
    unittest.main()
