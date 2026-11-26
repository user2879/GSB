from __future__ import annotations

import copy
import unittest

from sharddiag.deterministic import compute_shard_summary_local
from sharddiag.deterministic import (
    aggregate_partial,
    bind_summary,
    generate_samples,
    merge_partials,
    sample_digest,
)
from sharddiag.errors import SpecError
from sharddiag.fixedpoint import (
    attribute_errors,
    budget_margin_decision,
    quantize_shard_demand,
    saturating_mul,
    water_gap_decision,
)
from sharddiag.spec import load_spec, validate_spec


SAMPLE = "config/spec.sample.json"


class SpecValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = load_spec(SAMPLE)
        self.raw = copy.deepcopy(self.spec.raw)

    def expect_spec(self, mutate, field_fragment: str | None = None) -> SpecError:
        data = copy.deepcopy(self.raw)
        mutate(data)
        with self.assertRaises(SpecError) as ctx:
            validate_spec(data)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertTrue(ctx.exception.location)
        if field_fragment:
            self.assertIn(field_fragment, ctx.exception.field or "")
        return ctx.exception

    def test_sample_is_valid(self) -> None:
        self.assertEqual(8, len(self.spec.shards))
        self.assertIn("s07_sink", self.spec.topo_order[-1])

    def test_duplicate_node_id(self) -> None:
        def mutate(data):
            node = copy.deepcopy(data["nodes"][0])
            node["port"] = 39999
            data["nodes"].append(node)

        self.expect_spec(mutate, "node_id")

    def test_host_must_be_literal_loopback(self) -> None:
        self.expect_spec(lambda data: data["nodes"][0].__setitem__("host", "localhost"), "host")

    def test_ports_unique_and_range(self) -> None:
        def duplicate(data):
            data["nodes"][1]["port"] = data["nodes"][0]["port"]

        self.expect_spec(duplicate, "port")
        self.expect_spec(lambda data: data["nodes"][0].__setitem__("port", 70000), "port")

    def test_non_negative_and_positive_integers(self) -> None:
        self.expect_spec(
            lambda data: data["nodes"][0].__setitem__("concurrency_quota", -1),
            "concurrency_quota",
        )
        self.expect_spec(
            lambda data: data["shards"][0].__setitem__("sample_rows", 0),
            "sample_rows",
        )
        self.expect_spec(
            lambda data: data["shards"][0].__setitem__("resource_units", 0),
            "resource_units",
        )
        self.expect_spec(lambda data: data.__setitem__("session_scale", 4), "session_scale")

    def test_dependency_reference_self_and_cycle(self) -> None:
        self.expect_spec(
            lambda data: data["shards"][0]["depends_on"].append("s00_a_side"),
            "depends_on",
        )

        def missing(data):
            data["shards"][0]["depends_on"] = ["missing_shard"]

        self.expect_spec(missing, "depends_on")

        def cycle(data):
            by_id = {row["shard_id"]: row for row in data["shards"]}
            by_id["s01_a_ingest"]["depends_on"] = ["s00_a_side"]
            by_id["s00_a_side"]["depends_on"] = ["s01_a_ingest"]

        error = self.expect_spec(cycle, "depends_on")
        self.assertIn("cycle", error.message)

    def test_capacity_chain_parent_and_order(self) -> None:
        def bad_parent(data):
            data["capacity_versions"][1]["parent"] = None

        self.expect_spec(bad_parent, "parent")

        def bad_order(data):
            data["capacity_versions"][1]["version"] = 1

        self.expect_spec(bad_order, "version")


class DeterministicAggregateTests(unittest.TestCase):
    def test_batch_splits_and_delivery_order_are_bitwise_identical(self) -> None:
        common = dict(
            node_id="node-a",
            node_seed=110101,
            shard_id="s01_a_ingest",
            sample_rows=97,
            capacity_version=1,
            fixed_point_decimals=2,
        )
        digests = []
        sample_digests = set()
        for batch_size in (1, 3, 7, 16, 32, 128):
            _, summary, basis = compute_shard_summary_local(batch_size=batch_size, **common)
            digests.append(summary["summary_digest"])
            sample_digests.add(basis["sample_digest"])
            self.assertEqual(97, summary["sample_rows"])
            self.assertEqual(97, summary["participating_rows"])
            self.assertEqual("node-a", summary["node_id"])
            self.assertEqual("s01_a_ingest", summary["shard_id"])
            self.assertEqual(1, summary["capacity_version"])
        self.assertEqual(1, len(set(digests)))
        self.assertEqual(1, len(sample_digests))

    def test_seed_and_shard_change_samples(self) -> None:
        args = dict(sample_rows=32, capacity_version=1, fixed_point_decimals=0, batch_size=8)
        first, first_summary, _ = compute_shard_summary_local(
            node_id="a", node_seed=1, shard_id="s1", **args
        )
        second, second_summary, _ = compute_shard_summary_local(
            node_id="a", node_seed=1, shard_id="s2", **args
        )
        third, third_summary, _ = compute_shard_summary_local(
            node_id="b", node_seed=2, shard_id="s1", **args
        )
        self.assertNotEqual(first_summary["sample_digest"], second_summary["sample_digest"])
        self.assertNotEqual(first_summary["sample_digest"], third_summary["sample_digest"])

    def test_reordered_batch_partials_merge_to_same_summary(self) -> None:
        samples = generate_samples(99, "reorder-me", 41)
        partials = []
        batch_size = 7
        for index, start in enumerate(range(0, 41, batch_size)):
            end = min(41, start + batch_size)
            batch = {name: column[start:end] for name, column in samples.items()}
            partials.append(aggregate_partial(batch, fixed_point_decimals=2, batch_index=index))
        merged_order = bind_summary(
            merge_partials(partials),
            node_id="node-a",
            shard_id="reorder-me",
            sample_rows=41,
            capacity_version=1,
            digest_of_samples=sample_digest(samples),
            batch_size=batch_size,
        )
        merged_reverse = bind_summary(
            merge_partials(reversed(partials)),
            node_id="node-a",
            shard_id="reorder-me",
            sample_rows=41,
            capacity_version=1,
            digest_of_samples=sample_digest(samples),
            batch_size=batch_size,
        )
        self.assertEqual(merged_order["summary_digest"], merged_reverse["summary_digest"])


class FixedPointTests(unittest.TestCase):
    def test_truncation_and_error_attribution(self) -> None:
        item = quantize_shard_demand(
            shard_id="s", resource_units=7, budget_units=24, scale_power=1
        )
        self.assertEqual(2, item["budget_pressure_q"])
        self.assertEqual("22/24", item["truncation_error_q_exact"])
        attribution = attribute_errors(
            candidate_count=3,
            budget_units=24,
            scale_power=1,
            lease_attribution_units=2,
            dependency_wait_units=10,
        )
        self.assertEqual(3, attribution["segments_q"]["quantization_truncation"])
        self.assertEqual(1, attribution["segments_q"]["lease_attribution"])
        self.assertEqual(5, attribution["segments_q"]["dependency_wait"])
        self.assertEqual("dependency_wait", attribution["dominant_segment"])
        self.assertEqual(9, attribution["synthetic_error_bound_q"])

    def test_margin_and_water_closeness_use_exact_integer_boundary(self) -> None:
        attribution = attribute_errors(candidate_count=1, budget_units=100, scale_power=2)
        close_budget = budget_margin_decision(
            used_units=91, candidate_units=8, budget_units=100, attribution=attribution
        )
        self.assertEqual(1, close_budget["budget_margin_units"])
        self.assertTrue(close_budget["within_synthetic_error_bound"])
        gap = water_gap_decision(water_a_q=12, water_b_q=13, attribution=attribution)
        self.assertTrue(gap["within_synthetic_error_bound"])

    def test_overflow_saturation(self) -> None:
        value, saturated = saturating_mul(10**18, 10**18)
        self.assertTrue(saturated)
        self.assertLessEqual(value, 2**63 - 1)


if __name__ == "__main__":
    unittest.main()
