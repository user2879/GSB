from __future__ import annotations

import copy
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from shard_diag.determinism import (
    compute_shard_summary,
    generate_samples,
    partition_ranges,
)
from shard_diag.errors import SpecError
from shard_diag.spec import load_spec, parse_spec
from shard_diag.topology import stable_topological_order

from tests.factories import ROOT, sample_spec_dict


class SpecValidationTests(unittest.TestCase):
    def test_sample_spec_is_valid_and_ports_are_literal_loopback(self) -> None:
        spec = load_spec(ROOT / "config" / "spec.json")
        self.assertEqual([node.host for node in spec.nodes], ["127.0.0.1", "127.0.0.1"])
        self.assertEqual(len({node.port for node in spec.nodes}), len(spec.nodes))
        self.assertIn("s3-join", stable_topological_order(spec.shards)[2:4])

    def test_first_invalid_object_and_field_are_reported(self) -> None:
        cases = [
            ("duplicate_node", lambda raw: raw["nodes"][1].__setitem__("node_id", "node-a"), "node_id"),
            ("bad_host", lambda raw: raw["nodes"][0].__setitem__("host", "localhost"), "host"),
            ("duplicate_port", lambda raw: raw["nodes"][1].__setitem__("port", raw["nodes"][0]["port"]), "port"),
            ("zero_quota", lambda raw: raw["nodes"][0].__setitem__("concurrency_quota", 0), "concurrency_quota"),
            ("zero_lease", lambda raw: raw["nodes"][0].__setitem__("lease_limit", 0), "lease_limit"),
            ("duplicate_shard", lambda raw: raw["shards"][1].__setitem__("shard_id", raw["shards"][0]["shard_id"]), "shard_id"),
            ("missing_dependency", lambda raw: raw["shards"][2]["dependencies"].append("missing"), "dependencies"),
            ("revision_order", lambda raw: raw["revision_chain"].__setitem__(1, 201), "revision_chain"),
            ("replay_ahead", lambda raw: raw["delay_batches"][0].__setitem__("replayed_revision", 204), "replayed_revision"),
            ("scale_range", lambda raw: raw.__setitem__("scale_power", 4), "scale_power"),
        ]
        for name, mutate, field in cases:
            with self.subTest(name=name):
                raw = sample_spec_dict()
                mutate(raw)
                with self.assertRaises(SpecError) as raised:
                    parse_spec(raw)
                self.assertEqual(raised.exception.code, "E_SPEC")
                self.assertEqual(raised.exception.field, field)
                self.assertIsNotNone(raised.exception.object)

    def test_dependency_cycle_is_espec(self) -> None:
        raw = sample_spec_dict()
        by_id = {shard["shard_id"]: shard for shard in raw["shards"]}
        by_id["s1-extract-a"]["dependencies"] = ["s4-aggregate"]
        with self.assertRaises(SpecError) as raised:
            parse_spec(raw)
        self.assertEqual(raised.exception.field, "dependencies")
        self.assertIn("cycle", raised.exception.message)


class DeterminismTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = parse_spec(sample_spec_dict())
        self.node = self.spec.nodes[0]
        self.shard = self.spec.shard("s1-extract-a")

    def test_sample_bytes_are_bitwise_deterministic(self) -> None:
        first = generate_samples(self.node.seed, self.shard.shard_id, self.shard.sample_rows)
        second = generate_samples(self.node.seed, self.shard.shard_id, self.shard.sample_rows)
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(first.dtype, second.dtype)

    def test_arbitrary_batch_splits_match_single_batch(self) -> None:
        whole = compute_shard_summary(self.node, self.shard, split_points=[])
        split_points = [1, 7, 129, self.shard.sample_rows - 1]
        split = compute_shard_summary(self.node, self.shard, split_points=split_points)
        self.assertEqual(split.to_dict(), whole.to_dict())
        self.assertEqual(split.participating_rows, self.shard.sample_rows)
        self.assertEqual(
            partition_ranges(self.shard.sample_rows, split_points)[0],
            (0, 1),
        )

    def test_group_keys_are_explicitly_sorted(self) -> None:
        summary = compute_shard_summary(self.node, self.shard)
        keys = [(group["group_a"], group["group_b"]) for group in summary.groups]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(summary.node_id, self.node.node_id)
        self.assertEqual(summary.shard_id, self.shard.shard_id)
        self.assertEqual(summary.capacity_version, self.node.capacity_version)
        self.assertEqual(summary.participating_rows, summary.sample_rows)

    def test_concurrent_and_dispatch_order_produce_same_summary(self) -> None:
        shards = list(self.spec.shards)

        def calculate(index: int):
            shard = shards[index % len(shards)]
            node = self.spec.node(shard.node_id)
            return compute_shard_summary(node, shard).to_dict()

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(calculate, range(80)))
        by_shard = {}
        for result in results:
            expected = compute_shard_summary(
                self.spec.node(result["node_id"]), self.spec.shard(result["shard_id"])
            ).to_dict()
            by_shard.setdefault(result["shard_id"], result)
            self.assertEqual(result, expected)
        self.assertEqual(len(by_shard), len(self.spec.shards))


if __name__ == "__main__":
    unittest.main()
