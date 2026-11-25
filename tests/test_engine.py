from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from fixedseason.engine import FitEngine
from fixedseason.experiment import compare_reports
from fixedseason.model import FitConfig, SeriesRecord


def make_series(series_id: str, length: int, base: float = 100.0) -> SeriesRecord:
    timestamps = tuple(1000.0 + index for index in range(length))
    values = tuple(
        base + 0.4 * index + (2.0 if index % 4 == 0 else -1.0)
        for index in range(length)
    )
    return SeriesRecord(series_id, timestamps, values)


def coefficient_actions(report: dict) -> dict[str, str]:
    return {
        item["series_id"]: item["action"]
        for item in report["cache_ledger"]
        if item["stage"] == "coefficients"
    }


class BudgetAndConcurrencyTests(unittest.TestCase):
    def test_global_pending_and_batch_limits_are_stable(self) -> None:
        series = [make_series("a", 10), make_series("b", 12), make_series("c", 8)]
        config = FitConfig(
            harmonic_order=1,
            global_fit_points=20,
            per_series_quota=20,
            batch_observation_points=20,
            thread_pool_workers=2,
        )
        with tempfile.TemporaryDirectory() as temp:
            report = FitEngine(temp, config, series).run(version=1, force_rerun=True)
        self.assertEqual([item["series_id"] for item in report["series"]], ["a", "c"])
        self.assertEqual([item["series_id"] for item in report["pending"]], ["b"])
        self.assertEqual(report["pending"][0]["code"], "E_POINT_BUDGET_PENDING")
        self.assertEqual(report["budget"]["consumed_fit_points"], 18)
        self.assertEqual(report["budget"]["remaining_fit_points"], 2)
        self.assertEqual(report["budget"]["batches"], [["a", "c"]])

    def test_per_series_quota_failure_does_not_abort_other_series(self) -> None:
        series = [make_series("a", 10), make_series("b", 12), make_series("c", 8)]
        config = FitConfig(
            harmonic_order=1,
            global_fit_points=100,
            per_series_quota=10,
            batch_observation_points=20,
            thread_pool_workers=2,
        )
        with tempfile.TemporaryDirectory() as temp:
            report = FitEngine(temp, config, series).run(version=1, force_rerun=True)
        self.assertEqual([item["series_id"] for item in report["series"]], ["a", "c"])
        codes = [item["code"] for item in report["fit_failures"]]
        self.assertEqual(codes, ["E_FIT_BUDGET_EXCEEDED"])
        self.assertEqual(report["fit_failures"][0]["series_id"], "b")

    def test_serial_and_concurrent_runs_are_field_identical(self) -> None:
        series = [make_series(f"s{index}", 14 + index) for index in range(5)]
        config = FitConfig(
            harmonic_order=2,
            global_fit_points=500,
            per_series_quota=50,
            batch_observation_points=36,
            thread_pool_workers=3,
        )
        with tempfile.TemporaryDirectory() as serial_dir, tempfile.TemporaryDirectory() as concurrent_dir:
            serial = FitEngine(serial_dir, config, series, mode="serial").run(
                version=1, force_rerun=True
            )
            concurrent = FitEngine(
                concurrent_dir, config, series, mode="concurrent"
            ).run(version=1, force_rerun=True)
        self.assertEqual(serial["series"], concurrent["series"])
        self.assertEqual(serial["cache_ledger"], concurrent["cache_ledger"])
        self.assertEqual(serial["budget"]["ledger"], concurrent["budget"]["ledger"])
        self.assertEqual(serial["pending"], concurrent["pending"])
        self.assertEqual(serial["failures"], concurrent["failures"])
        self.assertTrue(compare_reports(serial, concurrent)[0])


class CacheInvalidationTests(unittest.TestCase):
    def test_single_series_edit_reuses_all_other_series(self) -> None:
        original = [make_series("a", 20), make_series("b", 20), make_series("c", 20)]
        config = FitConfig(harmonic_order=2, thread_pool_workers=2, version=1)
        with tempfile.TemporaryDirectory() as temp:
            first = FitEngine(temp, config, original).run(version=1, force_rerun=True)
            self.assertTrue(all(value == "miss" for value in coefficient_actions(first).values()))
            changed = [
                original[0].with_value(4, 777.25),
                original[1],
                original[2],
            ]
            second_config = config.with_changes(version=2)
            second = FitEngine(temp, second_config, changed).run(
                version=2, force_rerun=True
            )
        actions = coefficient_actions(second)
        self.assertEqual(actions, {"a": "miss", "b": "hit", "c": "hit"})
        self.assertEqual(second["summary"]["cache_hit_series_count"], 2)

    def test_harmonic_change_invalidates_all_but_budget_edit_does_not(self) -> None:
        series = [make_series("a", 20), make_series("b", 20)]
        config = FitConfig(harmonic_order=2, thread_pool_workers=2, version=1)
        with tempfile.TemporaryDirectory() as temp:
            FitEngine(temp, config, series).run(version=1, force_rerun=True)
            harmonic_config = config.with_changes(harmonic_order=1, version=2)
            harmonic = FitEngine(temp, harmonic_config, series).run(
                version=2, force_rerun=True
            )
            self.assertEqual(coefficient_actions(harmonic), {"a": "miss", "b": "miss"})
            budget_config = harmonic_config.with_changes(
                global_fit_points=9000, version=3
            )
            budget = FitEngine(temp, budget_config, series).run(
                version=3, force_rerun=True
            )
            self.assertEqual(coefficient_actions(budget), {"a": "hit", "b": "hit"})
            self.assertTrue(
                budget["run"]["snapshot_status"].startswith("invalidated_")
            )
            config_edit = budget_config.with_changes(regularization_strength=1.5)
            invalidated = FitEngine(temp, config_edit, series).run(
                version=3, force_rerun=True
            )
            self.assertEqual(
                invalidated["run"]["snapshot_status"], "invalidated_config_hash"
            )


class ResumeAndSnapshotTests(unittest.TestCase):
    def test_interrupted_concurrent_run_resumes_to_exact_one_shot_result(self) -> None:
        series = [make_series(f"s{index}", 20, base=80 + index) for index in range(5)]
        config = FitConfig(
            harmonic_order=2,
            global_fit_points=200,
            per_series_quota=50,
            batch_observation_points=100,
            thread_pool_workers=2,
            version=1,
        )
        with tempfile.TemporaryDirectory() as interrupted_dir, tempfile.TemporaryDirectory() as one_shot_dir:
            partial = FitEngine(interrupted_dir, config, series, mode="concurrent").run(
                version=1, stop_after=2, force_rerun=True
            )
            self.assertFalse(partial["run"]["complete"])
            self.assertEqual(partial["summary"]["unfinished_count"], 3)
            resumed = FitEngine(interrupted_dir, config, series, mode="concurrent").run(
                version=1
            )
            one_shot = FitEngine(one_shot_dir, config, series, mode="concurrent").run(
                version=1, force_rerun=True
            )
        self.assertTrue(resumed["run"]["resumed"])
        self.assertEqual(resumed["series"], one_shot["series"])
        self.assertEqual(resumed["budget"]["ledger"], one_shot["budget"]["ledger"])
        self.assertEqual(resumed["cache_ledger"], one_shot["cache_ledger"])
        self.assertTrue(compare_reports(resumed, one_shot)[0])

    def test_snapshot_manifest_and_stage_files(self) -> None:
        series = [make_series("a", 16), make_series("b", 18)]
        config = FitConfig(harmonic_order=1, thread_pool_workers=2, version=7)
        with tempfile.TemporaryDirectory() as temp:
            report = FitEngine(temp, config, series).run(
                version=7, force_rerun=True
            )
            current = Path(temp) / "snapshot" / "current"
            manifest = json.loads((current / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["global_version"], 7)
            self.assertEqual(manifest["config_hash"], report["run"]["config_hash"])
            self.assertEqual(manifest["bit_width_tier"], config.bit_tier)
            self.assertEqual(manifest["completed_series_cursor"], 2)
            self.assertTrue(manifest["complete"])
            for stage in (
                "loading",
                "interpolation",
                "basis",
                "normal_equations",
                "elimination",
                "back_substitution",
            ):
                self.assertTrue((current / "stages" / f"{stage}.json").exists())
            self.assertTrue((current / "report.json").exists())

    def test_interrupted_directory_swap_recovers_backup(self) -> None:
        series = [make_series("a", 16)]
        config = FitConfig(harmonic_order=1, thread_pool_workers=1, version=1)
        with tempfile.TemporaryDirectory() as temp:
            initial = FitEngine(temp, config, series).run(
                version=1, force_rerun=True
            )
            snapshot_root = Path(temp) / "snapshot"
            current = snapshot_root / "current"
            backup = snapshot_root / ".backup-test.tmp"
            shutil.move(str(current), str(backup))
            recovered = FitEngine(temp, config, series).run(version=1)
        self.assertEqual(recovered["series"], initial["series"])
        self.assertFalse(backup.exists())


if __name__ == "__main__":
    unittest.main()
