import unittest

from shardbench.algorithms import HashRing, RingNode, slot_to_target
from shardbench.config import load_config
from shardbench.determinism import all_physical_targets, stable_hash
from shardbench.predicate import parse_predicate
from shardbench.router import Router

from .test_validation import sample_paths


class PredicateNormalizationTests(unittest.TestCase):
    def test_whitespace_case_in_order_and_order_insensitive(self):
        variants = [
            "user_id in (3, 1, 2) AND tenant_id = 7",
            "  USER_ID   IN (1, 3, 2)  and   TENANT_ID=7 ",
            "tenant_id = 7 AND user_id IN (2, 3, 1)",
        ]
        parsed = [parse_predicate(text) for text in variants]
        normalized = {item.normalized_text() for item in parsed}
        route_keys = {item.route_key() for item in parsed}
        self.assertEqual(len(normalized), 1)
        self.assertEqual(len(route_keys), 1)

    def test_covers_full_composite_key(self):
        full = parse_predicate("user_id = 1 AND tenant_id = 2")
        partial = parse_predicate("user_id = 1")
        self.assertTrue(full.covers_key(("user_id", "tenant_id")))
        self.assertFalse(partial.covers_key(("user_id", "tenant_id")))
        self.assertIn("tenant_id", partial.broadcast_reason(("user_id", "tenant_id")))


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(*sample_paths())
        self.router = Router(self.config)

    def test_modulo_target_matches_deterministic_hash(self):
        plan = self.router.compile("u_user_eq")
        table = self.config.tables["t_user"]
        expected_slot = stable_hash("user_id=42") % table.params["base"]
        expected = slot_to_target(expected_slot, table.db_count, table.tables_per_db)
        self.assertEqual(plan.targets, [expected])
        self.assertFalse(plan.is_broadcast)
        self.assertEqual(plan.decisions[0]["algorithm"], "modulo")

    def test_missing_shard_key_is_explicit_broadcast(self):
        plan = self.router.compile("u_user_broadcast")
        table = self.config.tables["t_user"]
        self.assertTrue(plan.is_broadcast)
        self.assertEqual(
            plan.targets,
            all_physical_targets(table.db_count, table.tables_per_db),
        )
        self.assertIn("user_id", plan.broadcast_reason)

    def test_partial_composite_key_broadcasts(self):
        plan = self.router.compile("u_order_partial")
        table = self.config.tables["t_order"]
        self.assertTrue(plan.is_broadcast)
        self.assertEqual(
            len(plan.targets), table.db_count * table.tables_per_db
        )

    def test_in_list_union_is_sorted_and_deterministic(self):
        first = self.router.compile("u_user_in")
        second = self.router.compile("u_user_in")
        self.assertEqual(first.targets, second.targets)
        self.assertEqual(first.route_key, second.route_key)
        self.assertEqual(first.targets, sorted(first.targets))

    def test_range_boundary_ownership_declared(self):
        plan = self.router.compile("u_log_eq")
        decision = plan.decisions[0]
        self.assertEqual(decision["algorithm"], "range")
        self.assertEqual(decision["boundary_ownership"], "lower")
        # event_time=200 在 lower 归属下属于 [200,300) 槽位 2
        self.assertEqual(decision["slots"], [2])

    def test_range_interval_enumerates_slots(self):
        plan = self.router.compile("u_log_range")
        slots = sorted(
            slot
            for decision in plan.decisions
            for slot in decision["slots"]
        )
        # [100,400) 覆盖槽位 1 [100,200), 2 [200,300), 3 [300,+inf)
        self.assertEqual(slots, [1, 2, 3])

    def test_consistent_hash_ring_wraparound_and_version(self):
        ring = self.config.rings["t_cart_ring"]
        node_id, _ = ring.locate("cart_id='cart-777'")
        self.assertIn(node_id, ring.node_ids)
        plan = self.router.compile("u_ring_eq")
        self.assertEqual(plan.decisions[0]["ring_version"], ring.ring_version)
        self.assertEqual(len(plan.targets), 1)

    def test_ring_manual_wraparound(self):
        ring = HashRing((RingNode("a", 1), RingNode("b", 1)), "ring-test")
        max_position = max(point.position for point in ring.points)
        # 键位置超过所有虚拟点时应回绕到首个（位置最小的）节点
        node_id, point_position = ring.locate_position(max_position + 1)
        self.assertEqual(point_position, ring.points[0].position)
        self.assertIn(node_id, ring.node_ids)

    def test_derived_unit_union_targets(self):
        plans = {}
        for unit_id in self.config.topo_order:
            plans[unit_id] = self.router.compile(unit_id)
        # 通过引擎合并逻辑的等价手工核验在 test_engine 中完成，
        # 这里确保派生单元自身谓词可独立编译。
        self.assertIn("u_derived_user", plans)

    def test_route_reason_contains_parameters(self):
        plan = self.router.compile("u_user_eq")
        reason = plan.decisions[0]["reason"]
        self.assertIn("base", reason) if False else self.assertIn("% 8", reason)


if __name__ == "__main__":
    unittest.main()
