from __future__ import annotations

import json
import unittest

from bucketpack.canonical import dumps
from bucketpack.changes import parse_changes
from bucketpack.engine import run_full
from bucketpack.errors import E_REF, BusinessError
from bucketpack.incremental import run_incremental

from .common import ROOT, sample_spec


class IncrementalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = sample_spec()
        cls.baseline = run_full(cls.spec)

    def _load_change(self, name: str):
        raw = json.loads((ROOT / "samples" / name).read_text(encoding="utf-8"))
        return parse_changes(raw, self.spec)

    def test_value_and_null_change_recomputes_only_affected_subset(self) -> None:
        outcome = run_incremental(
            self.spec,
            self._load_change("changes_value.json"),
            baseline=self.baseline,
        )
        report = outcome.report
        self.assertTrue(report["equivalence"]["equal"])
        self.assertEqual(
            report["invalidation"]["bucket_columns"],
            [
                "events.latency_band",
                "events.latency_fine",
                "events.weight_band",
                "orders.region_band",
                "orders.spend_band",
                "orders.spend_half",
            ],
        )
        self.assertEqual(
            report["recomputed"]["packs"],
            ["events:p00000", "events:p00001", "orders:p00000", "orders:p00001"],
        )
        self.assertTrue(report["preserved"]["packs"])
        for item in report["preserved"]["packs"]:
            self.assertTrue(item["same"])
            self.assertEqual(item["baseline_digest"], item["incremental_digest"])
        unaffected = outcome.state.expansions["orders"].buckets["orders.items_band"]
        baseline_unaffected = self.baseline.expansions["orders"].buckets["orders.items_band"]
        self.assertEqual(unaffected.version, baseline_unaffected.version)
        self.assertEqual(dumps(unaffected.to_dict()), dumps(baseline_unaffected.to_dict()))

    def test_boundary_change_invalidates_whole_source_chain(self) -> None:
        outcome = run_incremental(
            self.spec,
            self._load_change("changes_buckets.json"),
            baseline=self.baseline,
        )
        report = outcome.report
        self.assertTrue(report["equivalence"]["equal"])
        self.assertEqual(
            report["invalidation"]["bucket_columns"],
            [
                "events.latency_band",
                "events.latency_fine",
                "orders.spend_band",
                "orders.spend_half",
            ],
        )
        self.assertEqual(report["recomputed"]["reused_packs"], [])
        preserved_buckets = {item["bucket_column"] for item in report["preserved"]["bucket_columns"]}
        self.assertIn("orders.items_band", preserved_buckets)
        self.assertNotIn("orders.spend_band", preserved_buckets)

    def test_leaf_rule_deletion_changes_width_and_repacks_source(self) -> None:
        outcome = run_incremental(
            self.spec,
            self._load_change("changes_delete.json"),
            baseline=self.baseline,
        )
        self.assertTrue(outcome.report["equivalence"]["equal"])
        self.assertNotIn("orders.items_aux", outcome.state.expansions["orders"].buckets)
        self.assertEqual(outcome.state.expansions["orders"].row_width, 13)
        self.assertTrue(all(pack.source_id != "orders" or pack.pack_id in outcome.report["recomputed"]["packs"] for pack in outcome.state.packs))
        self.assertEqual(
            outcome.report["preserved"]["packs"][0]["pack_id"].split(":", 1)[0],
            "events",
        )

    def test_deleting_referenced_rule_is_ref_error(self) -> None:
        with self.assertRaises(BusinessError) as caught:
            run_incremental(
                self.spec,
                self._load_change("changes_ref_error.json"),
                baseline=self.baseline,
            )
        self.assertEqual(caught.exception.code, E_REF)
        self.assertTrue(caught.exception.details["referrers"])


if __name__ == "__main__":
    unittest.main()
