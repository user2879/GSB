from __future__ import annotations

import unittest

from story_workbench.merge import check_all_merge_nodes, check_merge_node

try:
    from tests.helpers import merge
except ModuleNotFoundError:  # supports `unittest discover -s tests`
    from helpers import merge


class MergeConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = merge()
        cls.report = check_all_merge_nodes(cls.spec)

    def test_consistent_merge_reports_tiny_reproducible_difference(self):
        check = next(item for item in self.report["checks"] if item["target_node"] == "m_ok")
        self.assertEqual(check["status"], "consistent")
        self.assertEqual(check["feasible_path_count"], 2)
        self.assertIsNone(check["first_divergence"])
        pair = check["pairwise_differences"][0]
        self.assertTrue(pair["within_tolerance"])
        trust_diff = next(item for item in pair["component_differences"] if item["component"] == "trust")
        self.assertEqual(trust_diff["path_a_value"], 0.30000000000000004)
        self.assertEqual(trust_diff["path_b_value"], 0.3)
        self.assertEqual(trust_diff["absolute_difference"], 5.551115123125783e-17)

    def test_divergent_merge_reports_first_component_and_both_edge_sequences(self):
        check = next(item for item in self.report["checks"] if item["target_node"] == "m_bad")
        self.assertEqual(check["status"], "divergent")
        divergence = check["first_divergence"]
        self.assertEqual(divergence["component"], "trust")
        self.assertEqual(divergence["path_a_edges"], ["e_a_start", "e_a_end"])
        self.assertEqual(divergence["path_b_edges"], ["e_b_start", "e_b_end"])
        self.assertGreater(divergence["absolute_difference"], divergence["tolerance"])
        self.assertEqual(divergence["absolute_difference"], 0.09999999999999998)
        self.assertEqual(divergence["recompute"]["formula"], "abs(path_a_state[component] - path_b_state[component])")

    def test_each_path_exposes_stepwise_recomputation_data(self):
        check = check_merge_node(self.spec, "m_bad")
        self.assertEqual(len(check["paths"]), 2)
        for path in check["paths"]:
            self.assertEqual(len(path["steps"]), 2)
            self.assertTrue(path["condition_feasible"])
            self.assertEqual(path["steps"][0]["state_before"]["trust"], 0.0)
        self.assertEqual(check["report_hash"], check_merge_node(self.spec, "m_bad")["report_hash"])


if __name__ == "__main__":
    unittest.main()
