import unittest

from graphidx import GraphError, StaleError
from graphidx.canon import canonical

from tests.helpers import make_session


class TestTraversalQueries(unittest.TestCase):
    def setUp(self):
        self.session = make_session()

    def test_adjacency_traversal_deterministic_order(self):
        result = self.session.query.traverse_adjacency(1)
        # 节点 1 的出边按权重升序：e2(w2)->3, e1(w5)->2
        self.assertEqual(result["sequence"], [1, 3, 2, 5])
        self.assertEqual(result["bindings"]["indexes"]["by_weight"], {"1": 1, "2": 1, "3": 1})

    def test_label_filtered_traversal(self):
        result = self.session.query.traverse_by_label(4, "Person")
        self.assertEqual(result["sequence"], [4, 1, 2])
        self.assertIn("Person", result["bindings"]["indexes"]["by_label"])

    def test_label_traversal_start_label_mismatch(self):
        result = self.session.query.traverse_by_label(1, "City")
        self.assertEqual(result["sequence"], [])

    def test_view_reachability_query(self):
        result = self.session.query.view_reachability("reach_team")
        self.assertEqual(result["reachable"], [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(
            result["bindings"]["views"], {"reach_from_1": 1, "reach_team": 1}
        )

    def test_repeated_query_byte_identical_after_sync(self):
        self.session.apply_changeset({"id": "cs1", "ops": [
            {"op": "set_edge_weight", "edge": "e1", "weight": 8},
        ]})
        first = canonical(self.session.query.traverse_adjacency(1))
        second = canonical(self.session.query.traverse_adjacency(1))
        self.assertEqual(first, second)

    def test_unknown_start_node_rejected(self):
        with self.assertRaises(GraphError):
            self.session.query.traverse_adjacency(999)


class TestStalenessProtection(unittest.TestCase):
    def test_stale_index_entry_raises_e_stale(self):
        session = make_session(budget=1)  # 预算只够重算 1 个任务
        # cs1 影响 3 个索引条目 + 2 个视图，预算 1 -> 其余排队保持失效
        session.apply_changeset({"id": "cs1", "ops": [
            {"op": "set_edge_weight", "edge": "e1", "weight": 8},
            {"op": "set_node_label", "node": 4, "label": "Org"},
        ]})
        # 预算 1 只重算了 by_label:Org；by_label:Person 与 by_weight:1 仍失效
        with self.assertRaises(StaleError) as ctx:
            session.query.traverse_adjacency(1)
        err = ctx.exception
        self.assertEqual(err.code, "E_STALE")
        self.assertTrue(err.details)
        self.assertTrue(all(d["field"] == "status" for d in err.details))

    def test_stale_view_raises_e_stale(self):
        session = make_session(budget=3)  # 3 个索引条目可重算，视图排队
        report = session.apply_changeset({"id": "cs1", "ops": [
            {"op": "set_edge_weight", "edge": "e1", "weight": 8},
            {"op": "set_node_label", "node": 4, "label": "Org"},
        ]})
        self.assertEqual(report["skipped"], 2)  # 两个视图在排队
        with self.assertRaises(StaleError) as ctx:
            session.query.view_reachability("reach_team")
        refs = [d["element"] for d in ctx.exception.details]
        self.assertIn("view:reach_from_1", refs)
        self.assertIn("view:reach_team", refs)


if __name__ == "__main__":
    unittest.main()
