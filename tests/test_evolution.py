from __future__ import annotations

import unittest

from mlpipe_workbench.engine import Engine
from mlpipe_workbench.ledger import Ledger
from mlpipe_workbench.loader import prepare_from_objects

from tests.helpers import TempDirectoryMixin, base_objects, prepared_base


class EvolutionTests(TempDirectoryMixin, unittest.TestCase):
    def test_sample_and_stage_version_evolution_replaces_view_and_reuses_shards(self) -> None:
        base = prepared_base()
        data, pipeline = base_objects()
        current = float(data["sources"][0]["columns"][1]["values"][50])
        data["sources"][0]["columns"][1]["values"][50] = f"{current + 0.125:.6f}".rstrip("0").rstrip(".")
        pipeline["stages"][1]["version"] = 2
        evolved = prepare_from_objects(data, pipeline)
        with Ledger(self.tempdir / "evolution.db") as ledger:
            first = Engine(ledger).run(base, run_id="old", batch_size=32)
            second = Engine(ledger).run(evolved, run_id="new", batch_size=32)
            conclusion = ledger.current_conclusion()
            counts = ledger.table_counts()

        report = second.evolution or {}
        self.assertEqual(first.spec_version, 1)
        self.assertEqual(second.spec_version, 2)
        self.assertNotEqual(first.fingerprint, second.fingerprint)
        self.assertTrue(report["dataset_changed"])
        self.assertEqual(
            [stage["stage_id"] for stage in report["config_changed_stages"]],
            ["fill_missing"],
        )
        self.assertEqual(report["old_shard_count"], 5)
        self.assertEqual(report["new_shard_count"], 5)
        self.assertEqual(report["invalidated_shard_count"], 1)
        self.assertEqual(report["preserved_shard_count"], 4)
        self.assertEqual(report["preserved_shard_indices"], [0, 2, 3, 4])
        self.assertEqual(conclusion["fingerprint"], evolved.fingerprint)
        self.assertEqual(conclusion["view"]["run_id"], "new")
        self.assertEqual(counts["conclusion_history"], 2)
        self.assertEqual(sum(1 for shard in second.shards if shard["reused_from_spec_version"] == 1), 4)

    def test_sharding_only_change_preserves_upstream_outputs(self) -> None:
        base = prepared_base()
        data, pipeline = base_objects()
        pipeline["sharding"]["target_rows"] = 40
        changed = prepare_from_objects(data, pipeline)
        with Ledger(self.tempdir / "sharding.db") as ledger:
            first = Engine(ledger).run(base, run_id="old", batch_size=17)
            second = Engine(ledger).run(changed, run_id="new", batch_size=17)
        report = second.evolution or {}
        self.assertEqual([stage["stage_id"] for stage in report["affected_stages"]], ["pack"])
        self.assertEqual(len(report["unaffected_stages"]), 4)
        self.assertTrue(report["unaffected_outputs_stable"])
        self.assertEqual(report["unaffected_output_mismatches"], [])
        old_outputs = {stage["stage_id"]: stage["output_digest"] for stage in first.stage_statuses}
        new_outputs = {stage["stage_id"]: stage["output_digest"] for stage in second.stage_statuses}
        for stage_id in ("canonicalize", "fill_missing", "normalize_amount", "clip_amount"):
            self.assertEqual(old_outputs[stage_id], new_outputs[stage_id])
        self.assertEqual(report["old_shard_count"], 5)
        self.assertEqual(report["new_shard_count"], 4)
        self.assertEqual(report["invalidated_shard_count"], 5)
        self.assertEqual(report["preserved_shard_count"], 0)

    def test_complete_run_twice_with_same_spec_replaces_the_single_view(self) -> None:
        prepared = prepared_base()
        with Ledger(self.tempdir / "same.db") as ledger:
            Engine(ledger).run(prepared, run_id="run-1")
            Engine(ledger).run(prepared, run_id="run-2")
            conclusion = ledger.current_conclusion()
            counts = ledger.table_counts()
        self.assertEqual(conclusion["view"]["run_id"], "run-2")
        self.assertEqual(counts["conclusion_history"], 2)
        rows = conclusion["view"]
        self.assertEqual(len(rows["stages"]), 5)
        self.assertEqual(len(rows["shards"]), 5)


if __name__ == "__main__":
    unittest.main()
