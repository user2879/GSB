import unittest

from graphidx import DependencyError, ViewDef, ViewStore

from tests.helpers import load_sample_graph, make_session, sample_view_defs


class TestViewComputation(unittest.TestCase):
    def test_reachability_content(self):
        session = make_session()
        self.assertEqual(session.views.states["reach_from_1"].content["reachable"], [1, 2, 3, 5])
        # reach_team = 从 4 可达 {4,1,2,3,5,6,7} ∪ 上游 {1,2,3,5}
        self.assertEqual(session.views.states["reach_team"].content["reachable"], [1, 2, 3, 4, 5, 6, 7])

    def test_stable_topo_order(self):
        store = ViewStore(sample_view_defs())
        self.assertEqual(store.order, ["reach_from_1", "reach_team"])


class TestDependencyRules(unittest.TestCase):
    def test_cycle_rejected_with_cycle_sequence(self):
        with self.assertRaises(DependencyError) as ctx:
            ViewStore([
                ViewDef("a", [1], ["c"]),
                ViewDef("b", [1], ["a"]),
                ViewDef("c", [1], ["b"]),
            ])
        self.assertEqual(ctx.exception.code, "E_DEPEND")
        element = ctx.exception.details[0]["element"]
        for name in ("a", "b", "c"):
            self.assertIn(name, element)
        self.assertIn("->", element)

    def test_unknown_dependency_rejected(self):
        with self.assertRaises(DependencyError):
            ViewStore([ViewDef("a", [1], ["ghost"])])

    def test_remove_referenced_view_rejected_with_referrers(self):
        store = ViewStore(sample_view_defs())
        with self.assertRaises(DependencyError) as ctx:
            store.remove_def("reach_from_1")
        self.assertIn("reach_team", ctx.exception.details[0]["message"])
        # 未被引用的视图可删除
        store.remove_def("reach_team")
        self.assertNotIn("reach_team", store.defs)


class TestInvalidationPropagation(unittest.TestCase):
    def test_direct_and_indirect_invalidation_in_topo_order(self):
        session = make_session()
        report = session.apply_changeset({"id": "cs1", "ops": [
            {"op": "set_edge_weight", "edge": "e1", "weight": 8},
        ]})
        # e1 端点 {1,2} 在 reach_from_1 可达区域内 -> 直接失效；reach_team 间接失效
        self.assertEqual(report["invalid_views"], ["reach_from_1", "reach_team"])
        for name in ("reach_from_1", "reach_team"):
            self.assertEqual(session.views.states[name].status, "valid")  # 已重算
            self.assertEqual(session.views.states[name].version, 2)

    def test_unrelated_change_does_not_invalidate(self):
        session = make_session()
        report = session.apply_changeset({"id": "cs1", "ops": [
            {"op": "set_edge_weight", "edge": "e7", "weight": 60},
        ]})
        # e7 端点 {6,7}：不在 reach_from_1 区域内，但在 reach_team 区域内
        self.assertEqual(report["invalid_views"], ["reach_team"])
        self.assertEqual(session.views.states["reach_from_1"].version, 1)

    def test_label_only_change_does_not_invalidate_views(self):
        session = make_session()
        report = session.apply_changeset({"id": "cs1", "ops": [
            {"op": "set_node_label", "node": 1, "label": "Org"},
        ]})
        self.assertEqual(report["invalid_views"], [])

    def test_view_digest_updates_after_recompute(self):
        session = make_session()
        before = session.views.states["reach_from_1"].digest
        session.apply_changeset({"id": "cs1", "ops": [
            {"op": "add_node", "node": 8, "label": "Person"},
            {"op": "add_edge", "edge": "e9", "src": 2, "dst": 8, "weight": 1, "label": "knows"},
        ]})
        state = session.views.states["reach_from_1"]
        self.assertNotEqual(state.digest, before)
        self.assertIn(8, state.content["reachable"])


if __name__ == "__main__":
    unittest.main()
