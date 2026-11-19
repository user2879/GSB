from __future__ import annotations

import copy
import unittest

from cdn_edge.content import ContentRepository, topology_from_versions
from cdn_edge.spec import parse_spec
from cdn_edge.stats import BatchRequestRecord, compute_batch_metrics

from tests.helpers import sample_raw


def record(seq: int, node_id: str, consistent: bool, ttl_ms: int = 2000) -> BatchRequestRecord:
    return BatchRequestRecord(
        seq=seq,
        request_id=f"r{seq}",
        node_id=node_id,
        path="/catalog/base.json",
        query="q",
        at_ms=seq * 10,
        hit_source="cache_hit",
        returned_etag="v1" if consistent else "v0",
        authoritative_etag="v1",
        version_consistent=consistent,
        remaining_ttl_ms=ttl_ms,
        entry_version=1,
        stale=False,
        error_code=None,
    )


class StatsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = parse_spec(sample_raw())

    def test_derived_weights_and_multilevel_bound(self) -> None:
        metrics = compute_batch_metrics(
            self.spec,
            self.spec.grid,
            1,
            [
                record(0, "edge-a", True),
                record(1, "edge-b", True),
                record(2, "edge-c", True),
            ],
        )
        derived = metrics["derived_weights"]["paths"]
        bundle = derived["/bundle.js"]["leaves"]
        self.assertEqual(bundle["/catalog/base.json"]["fixed_weight_units"], 400)
        self.assertEqual(bundle["/catalog/theme.json"]["fixed_weight_units"], 600)
        self.assertEqual(bundle["/catalog/base.json"]["error_bound_units"], 2)
        all_leaf = derived["/all.js"]["leaves"]
        self.assertEqual(all_leaf["/catalog/base.json"]["fixed_weight_units"], 266)
        self.assertEqual(all_leaf["/catalog/theme.json"]["fixed_weight_units"], 399)
        self.assertEqual(all_leaf["/promo.txt"]["fixed_weight_units"], 333)
        self.assertEqual(derived["/all.js"]["normalization_shortfall_units"], 2)
        self.assertEqual(derived["/all.js"]["max_error_bound_units"], 3)
        self.assertIn("derive:/all.js->/bundle.js", all_leaf["/catalog/base.json"]["dominant_error_segment"])

    def test_cross_node_aggregation_error_propagation(self) -> None:
        records = [
            record(0, "edge-a", True),
            record(1, "edge-a", False),
            record(2, "edge-a", False),
            record(3, "edge-b", True),
            record(4, "edge-c", True),
            record(5, "edge-c", False),
        ]
        metrics = compute_batch_metrics(self.spec, self.spec.grid, 2, records)
        self.assertEqual(metrics["nodes"]["edge-a"]["metrics"]["version_consistency_rate"]["value_units"], 333)
        self.assertEqual(metrics["nodes"]["edge-b"]["metrics"]["version_consistency_rate"]["value_units"], 1000)
        self.assertEqual(metrics["nodes"]["edge-c"]["metrics"]["version_consistency_rate"]["value_units"], 500)
        aggregate = metrics["aggregate"]["metrics"]["version_consistency_rate"]
        self.assertEqual(aggregate["value_units"], 499)
        self.assertEqual(aggregate["error_bound_units"], 2)
        self.assertEqual(aggregate["dominant_error_source"], "aggregate:final_truncation")
        self.assertTrue(aggregate["segments"])

    def test_batch_topology_from_source_versions_controls_derived_weights(self) -> None:
        raw = sample_raw()
        raw["source_updates"] = [
            {
                "at_ms": 1800,
                "path": "/bundle.js",
                "derivation": {
                    "inputs": [
                        {"path": "/catalog/base.json", "coefficient": 2},
                        {"path": "/promo.txt", "coefficient": 1},
                    ]
                },
            }
        ]
        spec = parse_spec(raw)
        repository = ContentRepository(spec)
        for update in spec.source_updates:
            repository.apply_update(update)
        versions = repository.versions_at(2600)
        topology = topology_from_versions(versions)
        metrics = compute_batch_metrics(
            spec,
            spec.grid,
            2,
            [record(0, "edge-a", True), record(1, "edge-b", True), record(2, "edge-c", True)],
            topology,
        )
        bundle = metrics["derived_weights"]["paths"]["/bundle.js"]["leaves"]
        self.assertEqual(set(bundle), {"/catalog/base.json", "/promo.txt"})
        self.assertEqual(bundle["/catalog/base.json"]["fixed_weight_units"], 666)
        self.assertEqual(bundle["/promo.txt"]["fixed_weight_units"], 333)

    def test_node_order_does_not_change_attribution(self) -> None:
        raw = sample_raw()
        raw_reversed = copy.deepcopy(raw)
        raw_reversed["nodes"] = list(reversed(raw_reversed["nodes"]))
        spec_reversed = parse_spec(raw_reversed)
        records = [
            record(0, "edge-a", True),
            record(1, "edge-a", False),
            record(2, "edge-b", True),
            record(3, "edge-c", True),
            record(4, "edge-c", False),
        ]
        left = compute_batch_metrics(self.spec, self.spec.grid, 1, records)
        right = compute_batch_metrics(spec_reversed, spec_reversed.grid, 1, records)
        self.assertEqual(left["nodes"], right["nodes"])
        self.assertEqual(left["aggregate"], right["aggregate"])
        self.assertEqual(left["derived_weights"], right["derived_weights"])
        self.assertEqual(left["cross_node_comparisons"], right["cross_node_comparisons"])

    def test_pairwise_difference_within_bound_is_annotated(self) -> None:
        records = [
            record(0, "edge-a", True),
            record(1, "edge-a", False),
            record(2, "edge-a", False),
            record(3, "edge-b", True),
            record(4, "edge-b", True),
            record(5, "edge-b", False),
            record(6, "edge-b", False),
            record(7, "edge-b", False),
            record(8, "edge-b", False),
            record(9, "edge-c", True),
        ]
        metrics = compute_batch_metrics(self.spec, self.spec.grid, 1, records)
        comparisons = metrics["cross_node_comparisons"]["version_consistency_rate"]
        self.assertTrue(comparisons["pairwise_within_bound"])
        for pair in comparisons["pairwise_within_bound"]:
            self.assertLessEqual(pair["abs_delta_units"], 2)
            self.assertEqual(pair["stable_tie_order"], sorted(pair["stable_tie_order"]))


if __name__ == "__main__":
    unittest.main()
