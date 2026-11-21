from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from driftbench.engine import DiagnosticEngine, comparable_report
from driftbench.fingerprint import canonical_json, digest
from driftbench.injections import load_injections, run_injection_suite
from driftbench.loading import load_workspace
from driftbench.simulation import run_edit_sequence
from tests.common import SAMPLES


class EngineSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="driftbench-tests-")
        cls.initial = load_workspace(SAMPLES)
        cls.result = run_edit_sequence(cls.initial, snapshot_dir=cls.temp.name)
        cls.final_report = cls.result["final_report"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_every_edit_matches_full_rebuild(self) -> None:
        self.assertTrue(self.result["all_equal"])
        self.assertEqual(len(self.result["batches"]), 8)
        for row in self.result["batches"]:
            self.assertTrue(row["equal"], row)
            self.assertEqual(row["incremental_digest"], row["full_digest"])
            self.assertTrue(row["reuse_consistent"])
            self.assertEqual(
                row["reuse"]["actual_compute_calls"],
                row["reuse"]["recomputed"],
            )

    def test_invalidation_is_scoped_to_affected_stages(self) -> None:
        rows = {row["batch"]: row for row in self.result["batches"]}
        # A single-row correction touches one group's metrics and downstream artifacts.
        fixed = rows[2]["reuse_by_stage"]
        self.assertEqual(fixed["bucketing"], {"reused": 1, "recomputed": 0})
        self.assertEqual(fixed["metrics"], {"reused": 3, "recomputed": 3})
        self.assertEqual(fixed["drift"], {"reused": 1, "recomputed": 1})
        self.assertEqual(fixed["attribution"], {"reused": 1, "recomputed": 1})
        # Switching active version reuses all per-version metrics but changes the drift pair.
        switched = rows[3]["reuse_by_stage"]
        self.assertEqual(switched["metrics"], {"reused": 6, "recomputed": 0})
        self.assertEqual(switched["drift"], {"reused": 0, "recomputed": 2})
        self.assertEqual(switched["attribution"], {"reused": 0, "recomputed": 2})
        # Seed change invalidates bucketing and therefore all assignment-dependent stages.
        seeded = rows[4]["reuse_by_stage"]
        self.assertEqual(seeded["bucketing"], {"reused": 0, "recomputed": 1})
        self.assertEqual(seeded["metrics"]["recomputed"], 6)
        self.assertEqual(seeded["drift"]["recomputed"], 2)
        self.assertEqual(seeded["attribution"]["recomputed"], 2)
        # Returning to an already computed active pair is an exact cache hit.
        final = rows[7]["reuse"]
        self.assertEqual(final["reused"], 11)
        self.assertEqual(final["recomputed"], 0)

    def test_report_contains_metrics_drift_attribution_reuse_and_failures(self) -> None:
        report = self.final_report
        self.assertEqual(report["active_spec_version"], "v3")
        self.assertEqual(report["active_pair"], {"from_version": "v2", "to_version": "v3"})
        self.assertEqual(len(report["metrics"]["active"]["groups"]), 2)
        self.assertEqual(len(report["drift"]["groups"]), 2)
        self.assertEqual(len(report["attribution"]["groups"]), 2)
        self.assertTrue(report["reuse"]["consistent"])
        self.assertIn("counts", report["reuse"])
        self.assertIn("failures", report)
        for group in report["drift"]["groups"]:
            self.assertEqual(
                set(group["sources"]),
                {"truncation_rank", "normalization", "dedup_count", "sample_filter"},
            )
            self.assertTrue(group["significant"])
            self.assertTrue(group["sources"]["sample_filter"]["impacted_count"] > 0)
            self.assertTrue(group["sources"]["dedup_count"]["impacted_count"] > 0)
        for group in report["attribution"]["groups"]:
            self.assertEqual(group["residual_fixed"], 0)
            self.assertTrue(group["within_tolerance"])
            self.assertTrue(group["target_check_ok"])
            self.assertEqual(group["contribution_sum_fixed"], group["total_delta_fixed"])
        self.assertTrue(any(f["code"] == "E_ATTRIB_AMBIGUOUS" for f in report["failures"]))

    def test_serial_and_parallel_reports_are_identical(self) -> None:
        final_state = self.result["final_state"]
        serial_state = final_state.clone()
        serial_state.config = replace(serial_state.config, concurrency=1)
        parallel_state = final_state.clone()
        serial = DiagnosticEngine(full_rebuild=True, check_snapshot=False).run(serial_state)
        parallel = DiagnosticEngine(full_rebuild=True, check_snapshot=False).run(parallel_state)
        self.assertEqual(canonical_json(comparable_report(serial)), canonical_json(comparable_report(parallel)))
        self.assertEqual(digest(comparable_report(serial)), digest(comparable_report(parallel)))
        self.assertLessEqual(serial["concurrency"]["max_observed_active"], 1)
        self.assertLessEqual(
            parallel["concurrency"]["max_observed_active"],
            parallel_state.config.concurrency,
        )


class SnapshotTests(unittest.TestCase):
    def test_snapshot_resume_and_stale_rebuild(self) -> None:
        state = load_workspace(SAMPLES)
        with tempfile.TemporaryDirectory(prefix="driftbench-snapshot-test-") as temporary:
            first = DiagnosticEngine(snapshot_dir=temporary).run(state)
            self.assertEqual(first["snapshot"]["status"], "committed")
            manifest_path = Path(temporary) / "MANIFEST.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(manifest["stages"]),
                {"bucketing", "metrics", "drift", "attribution"},
            )
            self.assertTrue(all("product_version" in value for value in manifest["stages"].values()))

            resumed = DiagnosticEngine(snapshot_dir=temporary).run(state)
            self.assertEqual(resumed["reuse"]["counts"]["recomputed"], 0)
            self.assertEqual(resumed["reuse"]["counts"]["reused"], 11)
            self.assertTrue(all(row["origin"] == "disk" for row in resumed["reuse"]["reused"]))

            changed = state.clone()
            changed.config = replace(changed.config, bucket_hash_seed="stale-seed")
            rebuilt = DiagnosticEngine(snapshot_dir=temporary).run(changed)
            stale = [failure for failure in rebuilt["failures"] if failure["code"] == "E_SNAPSHOT_STALE"]
            self.assertEqual(len(stale), 1)
            self.assertEqual(rebuilt["snapshot"]["status"], "rebuilt")
            self.assertGreater(rebuilt["reuse"]["counts"]["recomputed"], 0)

    def test_injection_suite_hits_all_five_cases(self) -> None:
        state = load_workspace(SAMPLES)
        items = load_injections(SAMPLES / "injections.json")
        result = run_injection_suite(state, items)
        self.assertTrue(result["all_hit"])
        self.assertEqual(result["hit_count"], 5)
        self.assertEqual(
            [item["injection_id"] for item in result["injections"]],
            ["I01", "I02", "I03", "I04", "I05"],
        )
        expected = {
            "I01": "E_LOG_INVALID",
            "I02": "E_LOG_INVALID",
            "I03": "E_CONFIG_INVALID",
            "I04": "E_CONFIG_INVALID",
            "I05": "E_CONFIG_INVALID",
        }
        for item in result["injections"]:
            self.assertIn(expected[item["injection_id"]], item["observed_codes"])
            self.assertIsNotNone(item["matched_failure"])


if __name__ == "__main__":
    unittest.main()
