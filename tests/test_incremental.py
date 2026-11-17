import unittest

from tests.helpers import make_session


class TestIncrementalMaintenance(unittest.TestCase):
    def test_affected_set_exact_and_stable(self):
        session = make_session()
        report = session.apply_changeset({"id": "cs1", "ops": [
            {"op": "set_edge_weight", "edge": "e1", "weight": 8},
            {"op": "set_node_label", "node": 4, "label": "Org"},
        ]})
        self.assertEqual(
            report["affected_entries"],
            [["by_label", "Org"], ["by_label", "Person"], ["by_weight", "1"]],
        )

    def test_unaffected_entries_keep_version_and_digest(self):
        session = make_session()
        before = {
            (name, key): (e.entry_version, e.digest)
            for name, entries in session.indexes.entries.items()
            for key, e in entries.items()
        }
        session.apply_changeset({"id": "cs1", "ops": [
            {"op": "set_edge_weight", "edge": "e1", "weight": 8},
            {"op": "set_node_label", "node": 4, "label": "Org"},
        ]})
        affected = {("by_label", "Org"), ("by_label", "Person"), ("by_weight", "1")}
        for (name, key), (version, digest) in before.items():
            entry = session.indexes.entries[name][key]
            if (name, key) in affected:
                self.assertEqual(entry.entry_version, version + 1, (name, key))
            else:
                self.assertEqual(entry.entry_version, version, (name, key))
                self.assertEqual(entry.digest, digest, (name, key))

    def test_no_unrelated_entries_recomputed(self):
        session = make_session()
        report = session.apply_changeset({"id": "cs1", "ops": [
            {"op": "set_edge_weight", "edge": "e1", "weight": 8},
        ]})
        index_tasks = [e["target"] for e in report["ledger"] if e["kind"] == "index"]
        self.assertEqual(index_tasks, ["by_weight:1"])

    def test_entry_removal_and_creation(self):
        session = make_session()
        # 删除节点 4 的全部关联边后删除节点 4 -> Person 条目变化；新增标签 VIP -> 新条目
        session.apply_changeset({"id": "cs1", "ops": [
            {"op": "remove_edge", "edge": "e5"},
            {"op": "remove_edge", "edge": "e6"},
            {"op": "remove_node", "node": 4},
            {"op": "add_node", "node": 9, "label": "VIP"},
        ]})
        labels = session.indexes.entries["by_label"]
        self.assertIn("VIP", labels)
        self.assertEqual(labels["VIP"].entry_version, 1)
        self.assertEqual(labels["Person"].content["nodes"], [1, 2])
        # 节点 4 的邻接条目被移除
        self.assertNotIn("4", session.indexes.entries["by_weight"])

    def test_label_only_change_does_not_touch_adjacency(self):
        session = make_session()
        report = session.apply_changeset({"id": "cs1", "ops": [
            {"op": "set_node_label", "node": 2, "label": "Org"},
        ]})
        self.assertEqual(
            report["affected_entries"],
            [["by_label", "Org"], ["by_label", "Person"]],
        )
        self.assertEqual(report["invalid_views"], [])


if __name__ == "__main__":
    unittest.main()
