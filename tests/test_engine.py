from __future__ import annotations

import unittest
from unittest import mock

from near_dup.config import config_from_overrides
from near_dup.demo import _digest
from near_dup.engine import NearDuplicateEngine, SimulatedCrash
from near_dup.errors import E_CLUSTER_ID_CONFLICT, E_SIGNATURE_BUDGET_PENDING, E_SNAPSHOT_VERSION_MISMATCH, BusinessError
from near_dup.plan import apply_edit_batch, load_injections

from tests.helpers import SampleData


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = SampleData()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.data.close()

    def _apply_all_batches(self):
        records = dict(self.data.record_map)
        config = self.data.config
        failures = []
        for _, operations in sorted(self.data.plans.items()):
            records, config, batch_failures = apply_edit_batch(records, config, operations)
            failures.extend(batch_failures)
        return records, config, failures

    def test_initial_clusters_boundary_and_singletons(self) -> None:
        engine = NearDuplicateEngine(self.data.fresh_snapshot("initial"))
        result = engine.sync(self.data.records, self.data.config)
        cluster_map = {cluster["cluster_id"]: cluster for cluster in result["clusters"]}
        self.assertEqual(cluster_map["T001"]["members"], ["T001", "T002", "T003", "T004", "T005"])
        self.assertEqual(cluster_map["T006"]["members"], ["T006", "T007"])
        self.assertTrue(cluster_map["T008"]["singleton"])
        self.assertTrue(any(item["text_ids"] == ["T001", "T005"] for item in result["boundary_texts"]))
        boundary = next(item for item in result["boundary_texts"] if item["text_ids"] == ["T001", "T005"])
        self.assertEqual(boundary["similarity_fixed"], "0.7500")
        self.assertEqual(boundary["decision"], "merge")
        self.assertTrue(all(cluster["members"] == sorted(cluster["members"]) for cluster in result["clusters"]))

    def test_incremental_edit_sequence_equals_full_rebuild(self) -> None:
        incremental = NearDuplicateEngine(self.data.fresh_snapshot("incremental"))
        incremental.sync(self.data.records, self.data.config)
        current_records = dict(self.data.record_map)
        current_config = self.data.config
        plan_failures = []
        result = None
        for _, operations in sorted(self.data.plans.items()):
            current_records, current_config, batch_failures = apply_edit_batch(
                current_records, current_config, operations
            )
            plan_failures.extend(batch_failures)
            result = incremental.sync(
                current_records.values(),
                current_config,
                external_failures=plan_failures,
            )
        self.assertIsNotNone(result)
        result = incremental.sync(
            current_records.values(), current_config, external_failures=plan_failures
        )
        incremental_projection = incremental.state_projection(result)

        full = NearDuplicateEngine(self.data.fresh_snapshot("full"))
        full_result = full.sync(
            current_records.values(), current_config, external_failures=plan_failures
        )
        full_projection = full.state_projection(full_result)
        self.assertEqual(incremental_projection, full_projection)
        self.assertEqual(_digest(incremental_projection), _digest(full_projection))

    def test_threshold_only_edit_reuses_all_signatures(self) -> None:
        engine = NearDuplicateEngine(self.data.fresh_snapshot("threshold"))
        engine.sync(self.data.records, self.data.config)
        changed_config, failures = config_from_overrides(
            self.data.config, [("similarity_threshold", 0.70)], source="test"
        )
        self.assertEqual(failures, [])
        original_compute = __import__("near_dup.engine", fromlist=["compute_minhash"]).compute_minhash
        with mock.patch("near_dup.engine.compute_minhash", wraps=original_compute) as wrapped:
            result = engine.sync(self.data.records, changed_config)
        self.assertEqual(wrapped.call_count, 0)
        self.assertEqual(result["audit"]["signature_invalidated"], [])
        self.assertEqual(len(result["audit"]["signature_reused"]), len(self.data.records))
        self.assertTrue(result["audit"]["cluster_invalidated_buckets"])

    def test_replacement_recomputes_only_affected_signature(self) -> None:
        engine = NearDuplicateEngine(self.data.fresh_snapshot("replacement"))
        engine.sync(self.data.records, self.data.config)
        changed = dict(self.data.record_map)
        from near_dup.text_store import TextRecord

        changed["T007"] = TextRecord("T007", tuple(["庚"] * 12), "wiki", 4)
        result = engine.sync(changed.values(), self.data.config)
        self.assertEqual(result["audit"]["signature_invalidated"], ["T007"])
        self.assertEqual(result["audit"]["signature_computed"], ["T007"])
        self.assertEqual(len(result["audit"]["signature_reused"]), len(self.data.records) - 1)

    def test_bucket_bits_edit_reuses_signatures(self) -> None:
        engine = NearDuplicateEngine(self.data.fresh_snapshot("bucket-bits"))
        engine.sync(self.data.records, self.data.config)
        changed_config, failures = config_from_overrides(
            self.data.config, [("bucket_bits", 9)], source="test"
        )
        self.assertEqual(failures, [])
        original_compute = __import__("near_dup.engine", fromlist=["compute_minhash"]).compute_minhash
        with mock.patch("near_dup.engine.compute_minhash", wraps=original_compute) as wrapped:
            result = engine.sync(self.data.records, changed_config)
        self.assertEqual(wrapped.call_count, 0)
        self.assertEqual(result["audit"]["signature_invalidated"], [])
        self.assertEqual(len(result["audit"]["bucket_invalidated"]), len(self.data.records))
        self.assertTrue(result["audit"]["cluster_invalidated_buckets"])

    def test_budget_pending_priority_and_failure_code(self) -> None:
        budget_config, failures = config_from_overrides(
            self.data.config,
            [("signature_batch_budget", self.data.config.signature_dimension)],
            source="test",
        )
        self.assertEqual(failures, [])
        engine = NearDuplicateEngine(self.data.fresh_snapshot("budget"))
        first = engine.sync(self.data.records, budget_config)
        second = engine.sync(self.data.records, budget_config)
        self.assertEqual(first["audit"]["signature_computed"], ["T001"])
        self.assertEqual(first["pending_signatures"][0], "T002")
        self.assertEqual(second["audit"]["signature_computed"], ["T002"])
        self.assertEqual(second["pending_signatures"], first["pending_signatures"][1:])
        self.assertTrue(any(item["code"] == E_SIGNATURE_BUDGET_PENDING for item in first["failures"]))
        self.assertEqual(first["manifest"]["signature_budget"]["batch_consumed"], 32)
        self.assertEqual(second["manifest"]["signature_budget"]["consumed_total"], 64)

    def test_delete_from_pair_leaves_singleton_and_no_residue(self) -> None:
        engine = NearDuplicateEngine(self.data.fresh_snapshot("delete"))
        engine.sync(self.data.records, self.data.config)
        changed = dict(self.data.record_map)
        changed.pop("T007")
        result = engine.sync(changed.values(), self.data.config)
        cluster_map = {cluster["cluster_id"]: cluster for cluster in result["clusters"]}
        self.assertEqual(cluster_map["T006"]["members"], ["T006"])
        self.assertTrue(cluster_map["T006"]["singleton"])
        self.assertNotIn("T007", result["signatures"])
        self.assertNotIn("T007", result["buckets"])
        self.assertFalse(any("T007" in cluster["members"] for cluster in result["clusters"]))
        self.assertNotIn("T007", result["pending_signatures"])

    def test_concurrency_equals_serial(self) -> None:
        serial_config, serial_failures = config_from_overrides(
            self.data.config, [("max_workers", 1)], source="test"
        )
        concurrent_config, concurrent_failures = config_from_overrides(
            self.data.config, [("max_workers", 5)], source="test"
        )
        self.assertEqual(serial_failures + concurrent_failures, [])
        serial = NearDuplicateEngine(self.data.fresh_snapshot("serial"))
        concurrent = NearDuplicateEngine(self.data.fresh_snapshot("concurrent"))
        serial_result = serial.sync(self.data.records, serial_config)
        concurrent_result = concurrent.sync(self.data.records, concurrent_config)
        self.assertEqual(serial.state_projection(serial_result), concurrent.state_projection(concurrent_result))

    def test_snapshot_crash_resume_and_mismatch(self) -> None:
        records, final_config, _ = self._apply_all_batches()
        recovery_dir = self.data.fresh_snapshot("recovery")
        NearDuplicateEngine(recovery_dir).sync(self.data.records, self.data.config)
        crash_engine = NearDuplicateEngine(recovery_dir, crash_after_stage="signatures")
        with self.assertRaises(SimulatedCrash):
            crash_engine.sync(records.values(), final_config)
        partial = NearDuplicateEngine(recovery_dir).latest_snapshot()
        self.assertEqual(len(partial["manifest"]["completed_texts"]), 3)
        self.assertTrue(partial["manifest"]["pending_signatures"])
        resumed_engine = NearDuplicateEngine(recovery_dir)
        resumed = resumed_engine.sync(records.values(), final_config)
        full_engine = NearDuplicateEngine(self.data.fresh_snapshot("recovery-full"))
        full = full_engine.sync(records.values(), final_config)
        self.assertEqual(resumed_engine.state_projection(resumed), full_engine.state_projection(full))
        with self.assertRaises(BusinessError) as version_error:
            resumed_engine.recover_manifest(expected_version=999999)
        self.assertEqual(version_error.exception.code, E_SNAPSHOT_VERSION_MISMATCH)
        with self.assertRaises(BusinessError) as hash_error:
            resumed_engine.recover_manifest(expected_config_hash="bad-hash")
        self.assertEqual(hash_error.exception.code, E_SNAPSHOT_VERSION_MISMATCH)

    def test_cluster_id_conflict_injection_is_isolated(self) -> None:
        injections, _ = load_injections(str(self.data.root / "samples" / "injections.json"))
        conflict = next(item for item in injections if item.kind == "cluster_id_conflict")
        engine = NearDuplicateEngine(self.data.fresh_snapshot("injection"))
        with self.assertRaises(BusinessError) as error:
            engine.sync(
                self.data.records,
                self.data.config,
                runtime_injection=conflict.payload,
            )
        self.assertEqual(error.exception.code, E_CLUSTER_ID_CONFLICT)
        clean_engine = NearDuplicateEngine(self.data.fresh_snapshot("injection-clean"))
        clean = clean_engine.sync(self.data.records, self.data.config)
        self.assertFalse(any(cluster.get("injected_conflict") for cluster in clean["clusters"]))


if __name__ == "__main__":
    unittest.main()
