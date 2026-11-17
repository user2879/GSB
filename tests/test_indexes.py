import unittest

from graphidx import IndexDef, IndexStore, PropertyGraph

from tests.helpers import load_sample_graph, sample_index_defs


class TestFullBuild(unittest.TestCase):
    def setUp(self):
        self.graph = load_sample_graph()
        self.store = IndexStore(sample_index_defs())
        self.report = self.store.build_full(self.graph)

    def test_label_inverted_entries(self):
        entries = self.store.entries["by_label"]
        self.assertEqual(sorted(entries), ["City", "Org", "Person"])
        self.assertEqual(entries["Person"].content, {"label": "Person", "nodes": [1, 2, 4]})
        self.assertEqual(entries["City"].content, {"label": "City", "nodes": [5, 7]})

    def test_weight_adjacency_ordering_rule(self):
        entry = self.store.entries["by_weight"]["1"]
        # e2 (w2) 排在 e1 (w5) 前：按 (weight, dst, edge_id) 升序
        self.assertEqual(
            entry.content,
            {"src": 1, "edges": [[3, 2, "works_at", "e2"], [2, 5, "knows", "e1"]]},
        )

    def test_entry_versions_and_def_version(self):
        for entries in self.store.entries.values():
            for entry in entries.values():
                self.assertEqual(entry.entry_version, 1)
                self.assertEqual(entry.def_version, 1)
                self.assertEqual(entry.status, "valid")

    def test_deterministic_across_build_orders(self):
        # 同一图内容、不同加载顺序 -> 逐字节一致的索引摘要
        reversed_graph = PropertyGraph.from_dict({
            "nodes": [{"id": n, "label": l} for n, l in reversed(self.graph.to_content()["nodes"])],
            "edges": [
                {"id": e, "src": s, "dst": d, "weight": w, "label": l}
                for e, s, d, w, l in reversed(self.graph.to_content()["edges"])
            ],
        })
        other = IndexStore(sample_index_defs())
        other_report = other.build_full(reversed_graph)
        self.assertEqual(self.report["indexes"], other_report["indexes"])
        self.assertEqual(self.report["graph_digest"], other_report["graph_digest"])

    def test_build_report_contains_graph_and_index_digests(self):
        self.assertIn("graph_digest", self.report)
        self.assertIn("index_def_version", self.report)
        self.assertEqual(set(self.report["indexes"]), {"by_label", "by_weight"})
        self.assertTrue(all(d.startswith("sha256:") for d in self.report["indexes"].values()))


if __name__ == "__main__":
    unittest.main()
