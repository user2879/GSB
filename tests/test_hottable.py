import unittest

from hotblock_diag.hottable import HotTable


class HotTableTests(unittest.TestCase):
    def test_capacity_eviction_uses_count_then_kb(self):
        table = HotTable(capacity=2, keep=2)
        selections0 = {0: [5, 7], 1: [5]}
        scores = {(0, 5): 1.0, (0, 7): 1.0, (1, 5): 1.0}
        table.apply_round(0, selections0, scores)
        self.assertEqual({entry.kb for entry in table.entries.values()}, {5, 7})

        selections1 = {0: [9], 1: [9]}
        scores1 = {(0, 9): 1.0, (1, 9): 1.0}
        snapshot = table.apply_round(1, selections1, scores1)
        self.assertTrue(snapshot["evictions"])
        victim = snapshot["evictions"][0]["kb"]
        self.assertEqual(victim, 7)

    def test_evicted_block_reaccumulates_from_zero(self):
        table = HotTable(capacity=1, keep=1)
        table.apply_round(0, {0: [3]}, {(0, 3): 2.0})
        table.apply_round(1, {0: [4]}, {(0, 4): 1.0})
        self.assertNotIn(3, table.entries)
        table.apply_round(2, {0: [3]}, {(0, 3): 5.0})
        self.assertEqual(table.entries[3].selected_count, 1)
        self.assertEqual(table.entries[3].accumulated_score, 5.0)

    def test_keep_suppression_and_rebuild(self):
        table = HotTable(capacity=4, keep=2)
        selections = {0: [1, 2, 3]}
        scores = {(0, kb): 1.0 for kb in (1, 2, 3)}
        snapshot = table.apply_round(0, selections, scores)
        self.assertEqual([entry["kb"] for entry in snapshot["entries"]], [1, 2])
        self.assertTrue(snapshot["suppressions"])

        rounds = [(0, selections, scores)]
        rebuilt = HotTable.rebuild(4, 2, rounds)
        self.assertEqual(rebuilt.state_dict()["entries"],
                         table.state_dict()["entries"])

    def test_keep_larger_than_capacity_clamped(self):
        table = HotTable(capacity=2, keep=5)
        snapshot = table.apply_round(0, {0: [1, 2, 3]},
                                     {(0, 1): 1.0, (0, 2): 1.0, (0, 3): 1.0})
        self.assertLessEqual(len(snapshot["entries"]), 2)
        self.assertTrue(snapshot["evictions"])


if __name__ == "__main__":
    unittest.main()
