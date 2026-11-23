from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mixedmem.ledger import Ledger
from mixedmem.runtime import InferenceRuntime
from mixedmem.schema import parse_plan

from .helpers import plan_copy


class IncrementalTests(unittest.TestCase):
    def test_only_invalidated_cone_is_recomputed(self) -> None:
        plan = parse_plan(plan_copy())
        with tempfile.TemporaryDirectory() as directory:
            ledger = Ledger(Path(directory) / "ledger.db")
            runtime = InferenceRuntime(plan, 8, batch_size=4, ledger=ledger)
            self.assertTrue(runtime.run(worker_count=2)["ok"])
            runtime.hold_version_leases("b0000")
            before = runtime.layer_summaries()
            before_versions = dict(runtime.layer_versions)
            change = runtime.apply_change("hidden", {"scale": 256})
            after = runtime.layer_summaries()
            self.assertEqual(change["invalidated"], ["hidden", "compress", "head"])
            self.assertEqual(change["unaffected"], ["embed"])
            for layer_id in change["unaffected"]:
                self.assertEqual(before[layer_id]["float_digest"], after[layer_id]["float_digest"])
                self.assertEqual(before[layer_id]["quant_digest"], after[layer_id]["quant_digest"])
                self.assertEqual(before[layer_id]["combined_digest"], after[layer_id]["combined_digest"])
                self.assertEqual(before[layer_id]["lease_range"], after[layer_id]["lease_range"])
                self.assertEqual(before_versions[layer_id], runtime.layer_versions[layer_id])
            self.assertEqual(len(change["lease_replacement"]["revoked"]), 4)
            self.assertTrue(change["lease_replacement"]["atomic"])
            self.assertTrue(runtime.verify_incremental()["ok"])
            all_leases = ledger.load_leases()
            self.assertEqual(
                len([row for row in all_leases if row["status"] == "revoked"]),
                4,
            )
            ledger.close()

    def test_multi_layer_dimension_edit_is_atomic_and_versioned_once(self) -> None:
        plan = parse_plan(plan_copy())
        with tempfile.TemporaryDirectory() as directory:
            ledger = Ledger(Path(directory) / "ledger.db")
            runtime = InferenceRuntime(plan, 8, batch_size=4, ledger=ledger)
            runtime.run(worker_count=2)
            change = runtime.apply_changes(
                {
                    "hidden": {"output_dim": 6},
                    "compress": {"input_dim": 6, "output_dim": 4},
                }
            )
            self.assertEqual(change["new_plan_version"], 2)
            self.assertEqual(runtime.plan.version, 2)
            self.assertEqual(runtime.plan.layer_by_id["hidden"].output_dim, 6)
            self.assertEqual(runtime.plan.layer_by_id["compress"].input_dim, 6)
            self.assertEqual(change["invalidated"], ["hidden", "compress", "head"])
            self.assertTrue(runtime.verify_incremental()["ok"])
            ledger.close()

    def test_memory_derivation_change_invalidates_descendants_only(self) -> None:
        plan = parse_plan(plan_copy())
        with tempfile.TemporaryDirectory() as directory:
            ledger = Ledger(Path(directory) / "ledger.db")
            runtime = InferenceRuntime(plan, 8, batch_size=4, ledger=ledger)
            runtime.run(worker_count=2)
            before = runtime.layer_summaries()
            change = runtime.apply_change(
                "compress",
                {"memory_derivation": {"base_bytes": 64, "terms": []}},
            )
            self.assertEqual(change["invalidated"], ["compress", "head"])
            self.assertEqual(change["unaffected"], ["embed", "hidden"])
            self.assertTrue(runtime.verify_incremental()["ok"])
            self.assertGreaterEqual(
                runtime.layer_summaries()["compress"]["occupancy_bytes"],
                before["compress"]["occupancy_bytes"],
            )
            ledger.close()


if __name__ == "__main__":
    unittest.main()
