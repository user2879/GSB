from __future__ import annotations

import unittest

from lease_arbitrator.arbitration import ClusterCapacity, arbitrate_candidates
from lease_arbitrator.content import build_shard


class ContentTests(unittest.TestCase):
    def test_digest_is_golden_and_order_independent(self) -> None:
        first = build_shard(node_id="node-a", node_seed=1101, artifact_key="core", shard_index=0, byte_count=96)
        second = build_shard(node_id="node-a", node_seed=1101, artifact_key="core", shard_index=0, byte_count=96)
        self.assertEqual(first["digest"], second["digest"])
        self.assertEqual(
            first["digest"],
            "1c8ffa12a00d2b7777da358ed51bd665700a893c071fbfcd503f32cf54bada5a",
        )
        self.assertEqual(first["shard_bytes"], 96)
        self.assertEqual(first["source_version"], 1)

    def test_digest_binds_all_fields(self) -> None:
        base = build_shard(node_id="a", node_seed=1, artifact_key="k", shard_index=0, byte_count=32)
        changed_node = build_shard(node_id="b", node_seed=1, artifact_key="k", shard_index=0, byte_count=32)
        changed_key = build_shard(node_id="a", node_seed=1, artifact_key="k2", shard_index=0, byte_count=32)
        changed_shard = build_shard(node_id="a", node_seed=1, artifact_key="k", shard_index=1, byte_count=32)
        self.assertNotEqual(base["digest"], changed_node["digest"])
        self.assertNotEqual(base["digest"], changed_key["digest"])
        self.assertNotEqual(base["digest"], changed_shard["digest"])


class ArbitrationCapacityTests(unittest.TestCase):
    def test_stable_ranking_and_tie_marker(self) -> None:
        decision = arbitrate_candidates(
            [
                {"node_id": "node-c", "digest_match": True, "remaining_slots": 2},
                {"node_id": "node-a", "digest_match": True, "remaining_slots": 2},
                {"node_id": "node-b", "digest_match": False, "remaining_slots": 9},
            ]
        )
        self.assertEqual(decision["winner_node_id"], "node-a")
        self.assertEqual(decision["indistinguishable_node_ids"], ["node-a", "node-c"])
        by_node = {item["node_id"]: item for item in decision["candidates"]}
        self.assertIn("NODE_ID_TIE_BREAK", by_node["node-c"]["reasons"])
        self.assertIn("DIGEST_MISMATCH", by_node["node-b"]["reasons"])

    def test_reservation_rollback_is_exact(self) -> None:
        capacity = ClusterCapacity.from_specs(
            [
                type("N", (), {"node_id": "a", "capacity_slots": 2, "lease_limit": 2})(),
                type("N", (), {"node_id": "b", "capacity_slots": 1, "lease_limit": 1})(),
            ]
        )
        capacity.reserve("r1", "a")
        capacity.reserve("r2", "a")
        self.assertEqual(capacity.nodes["a"].used_slots, 2)
        with self.assertRaises(OverflowError):
            capacity.reserve("r3", "a")
        capacity.rollback_reservation("r2")
        self.assertEqual(capacity.nodes["a"].used_slots, 1)
        self.assertIn("r1", capacity.reservations)
        capacity.rollback_reservation("r1")
        self.assertEqual(capacity.nodes["a"].used_slots, 0)

    def test_watermark_truncation_error_bound(self) -> None:
        capacity = ClusterCapacity.from_specs(
            [type("N", (), {"node_id": "a", "capacity_slots": 3, "lease_limit": 2})()]
        )
        capacity.reserve("r1", "a")
        watermark = capacity.nodes["a"].watermark(1000)
        self.assertEqual(watermark["watermark_fixed"], 333)
        self.assertEqual(watermark["quantization_error_bound_ratio_denominator"], 1000)


if __name__ == "__main__":
    unittest.main()
