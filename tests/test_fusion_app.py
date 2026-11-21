from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from candidate_fusion.edits import apply_edit_sequence
from candidate_fusion.engine import run_workspace
from candidate_fusion.io_utils import read_json, write_json
from candidate_fusion.samples import generate_samples
from candidate_fusion.snapshot import SimulatedCrash, SnapshotStore


class FusionApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_root = tempfile.TemporaryDirectory(prefix="fusion-tests-")
        cls.samples = generate_samples(Path(cls.temp_root.name) / "samples", force=True)
        cls.base_workspace = cls.samples / "base_workspace"
        cls.sequence_path = cls.samples / "edits" / "sequence.json"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_root.cleanup()

    def setUp(self) -> None:
        self.workspace = Path(self.temp_root.name) / self._testMethodName
        shutil.rmtree(self.workspace, ignore_errors=True)
        shutil.copytree(self.base_workspace, self.workspace)

    def _config(self) -> dict:
        return read_json(self.workspace / "config.json")

    def _write_config(self, config: dict) -> None:
        write_json(self.workspace / "config.json", config)

    def test_injection_mapping_and_isolation(self) -> None:
        expected = {
            "INJ-001": "E_NONFINITE_SCORE",
            "INJ-002": "E_CONFIG_INVALID",
            "INJ-003": "E_CONFIG_INVALID",
            "INJ-004": "E_CONFIG_INVALID",
        }
        for injection_id, code in expected.items():
            shutil.rmtree(self.workspace / "state", ignore_errors=True)
            report = run_workspace(
                self.workspace,
                injection_path=self.samples / "injections" / f"{injection_id}.json",
            ).report
            hit = next(item for item in report["injections_hit"] if item["id"] == injection_id)
            self.assertEqual(hit["mapped_code"], code)
            self.assertTrue(any(item["injection_id"] == injection_id and item["code"] == code for item in report["failures"]))

    def test_load_injection_after_commit_does_not_reuse_removed_candidate(self) -> None:
        baseline = run_workspace(self.workspace).report
        injected = run_workspace(
            self.workspace,
            injection_path=self.samples / "injections" / "INJ-001.json",
        ).report
        self.assertEqual(injected["mode"], "incremental")
        self.assertEqual(injected["invalidation"]["counts"]["reused_candidates"], 79)
        self.assertEqual(
            injected["invalidation"]["counts"]["invalidated_candidates"],
            54,
        )
        self.assertEqual(
            {
                item.split("/", 1)[0]
                for item in injected["invalidation"]["invalidated_candidates"]
            },
            {"ch_alpha"},
        )
        normalized = SnapshotStore(self.workspace).load().stages["normalize"]
        self.assertEqual(len(normalized["channels"]["ch_alpha"]["candidates"]), 54)
        self.assertNotEqual(
            baseline["budget"]["consumed"],
            injected["budget"]["consumed"],
        )
        self.assertIn("E_NONFINITE_SCORE", {item["code"] for item in injected["failures"]})

    def test_zero_denominator_records_event_and_quantizes_zero(self) -> None:
        fragment = self.workspace / "fragments" / "ch_delta" / "frag_000.jsonl"
        records = [json.loads(line) for line in fragment.read_text(encoding="utf-8").splitlines() if line]
        for record in records:
            record["score"] = 1.0
        fragment.write_text(
            "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
            encoding="utf-8",
        )
        report = run_workspace(self.workspace).report
        self.assertTrue(
            any(
                event["channel"] == "ch_delta" and event["code"] == "E_NORMALIZE_ZERO_DENOM"
                for event in report["normalization_events"]
            )
        )
        normalize_stage = SnapshotStore(self.workspace).load().stages["normalize"]
        for candidate in normalize_stage["channels"]["ch_delta"]["candidates"].values():
            self.assertEqual(candidate["fixed"], 0)
            self.assertEqual(candidate["residual"], 0.0)

    def test_single_fragment_append_has_precise_invalidation_and_reuse(self) -> None:
        run_workspace(self.workspace)
        apply_edit_sequence(self.workspace, self.sequence_path, edit_ids={"E01"})
        report = run_workspace(self.workspace).report
        self.assertEqual(report["invalidation"]["counts"]["invalidated_candidates"], 61)
        self.assertEqual(report["invalidation"]["counts"]["reused_fusions"], 12)
        invalidated_channels = {
            item.split("/", 1)[0] for item in report["invalidation"]["invalidated_candidates"]
        }
        self.assertEqual(invalidated_channels, {"ch_alpha"})
        unchanged = run_workspace(self.workspace).report
        self.assertEqual(unchanged["invalidation"]["counts"]["invalidated_fusions"], 0)
        self.assertEqual(unchanged["invalidation"]["counts"]["reused_fusions"], 73)
        self.assertEqual(unchanged["budget"]["consumed"], 0)

    def test_budget_allocation_is_stable_and_bounded(self) -> None:
        config = self._config()
        config["channel_quota"] = 20
        config["global_budget"] = 100
        self._write_config(config)
        first = run_workspace(self.workspace).report
        second = run_workspace(self.workspace, reset=True).report
        for field in ("audit", "consumed", "pending_queue", "remaining"):
            self.assertEqual(first["budget"][field], second["budget"][field])
        self.assertLessEqual(first["budget"]["consumed"], 100)
        self.assertTrue(all(item["granted"] <= 20 for item in first["budget"]["audit"]))
        self.assertLessEqual(first["budget"]["concurrency"]["max_observed"], config["max_workers"])
        self.assertGreaterEqual(first["budget"]["concurrency"]["max_observed"], 2)
        self.assertIn(
            "E_CHANNEL_QUOTA_EXHAUSTED",
            {item["code"] for item in first["failures"]},
        )
        self.assertTrue(first["budget"]["pending_queue"])

        config["channel_quota"] = 10000
        config["global_budget"] = 50
        self._write_config(config)
        global_report = run_workspace(self.workspace, reset=True).report
        self.assertEqual(global_report["budget"]["remaining"], 0)
        self.assertIn("E_GLOBAL_BUDGET_EXHAUSTED", {item["code"] for item in global_report["failures"]})

    def test_snapshot_crash_recovery_and_stale_fallback(self) -> None:
        run_workspace(self.workspace)
        store = SnapshotStore(self.workspace)
        version = store.load().manifest_version
        with self.assertRaises(SimulatedCrash):
            run_workspace(self.workspace, simulate_crash_after="normalize")
        after_crash = store.load()
        self.assertEqual(after_crash.manifest_version, version)
        self.assertEqual(after_crash.ignored_temp_snapshots, 1)
        recovered = run_workspace(self.workspace).report
        self.assertTrue(recovered["snapshot"]["resumed"])
        self.assertFalse(recovered["snapshot"]["stale"])
        self.assertEqual(recovered["invalidation"]["counts"]["invalidated_candidates"], 0)

        current_name = (self.workspace / "state" / "CURRENT").read_text(encoding="utf-8").strip()
        load_path = self.workspace / "state" / "snapshots" / current_name / "load.json"
        tampered = read_json(load_path)
        tampered["tampered"] = True
        write_json(load_path, tampered)
        stale = run_workspace(self.workspace).report
        self.assertTrue(stale["snapshot"]["stale"])
        self.assertIn("E_SNAPSHOT_STALE", {item["code"] for item in stale["failures"]})
        self.assertGreater(stale["invalidation"]["counts"]["invalidated_candidates"], 0)

    def test_final_incremental_result_matches_clean_full_rebuild(self) -> None:
        run_workspace(self.workspace)
        edit_ids = [edit["id"] for edit in read_json(self.sequence_path)["edits"]]
        incremental = None
        for edit_id in edit_ids:
            apply_edit_sequence(self.workspace, self.sequence_path, edit_ids={edit_id})
            incremental = run_workspace(self.workspace).report
        assert incremental is not None

        clean_workspace = Path(self.temp_root.name) / f"{self._testMethodName}_clean"
        shutil.rmtree(clean_workspace, ignore_errors=True)
        shutil.copytree(self.workspace, clean_workspace)
        shutil.rmtree(clean_workspace / "state", ignore_errors=True)
        full = run_workspace(clean_workspace).report
        for key in (
            "config_echo",
            "failures",
            "fusion_summary",
            "normalization_events",
            "observability",
            "quantization_evidence",
            "invalidation",
        ):
            self.assertEqual(incremental[key], full[key], key)
        for key in ("audit", "consumed", "pending_queue", "remaining"):
            self.assertEqual(incremental["budget"][key], full["budget"][key], key)

    def test_trunc_zero_returns_empty_and_failure_event(self) -> None:
        config = self._config()
        config["trunc_k"] = 0
        self._write_config(config)
        report = run_workspace(self.workspace).report
        self.assertEqual(report["fusion_summary"]["kept"], [])
        self.assertIn("E_TRUNC_ZERO", {item["code"] for item in report["failures"]})


if __name__ == "__main__":
    unittest.main()
