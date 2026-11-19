from __future__ import annotations

import unittest

from story_workbench.dependency import IncrementalDependencyGraph
from story_workbench.errors import E_DEPENDENCY_INVALID, WorkbenchError

try:
    from tests.helpers import normal
except ModuleNotFoundError:  # supports `unittest discover -s tests`
    from helpers import normal


class DependencyClosureTests(unittest.TestCase):
    def test_coefficient_edit_closure_is_exact_and_outside_hashes_are_reused(self):
        graph = IncrementalDependencyGraph(normal(), "normal_path")
        key = "e_intro:f_intro_rep:coef:0"
        before_versions = {node.id: graph.cache[node.id].version for node in graph.topology}
        before_hashes = {node.id: graph.cache[node.id].content_hash for node in graph.topology}
        closure = graph.closure(key)
        report = graph.set_coefficient(key, 0.25)
        self.assertEqual(sorted(report["recomputed_nodes"]), sorted(closure))
        self.assertTrue(report["actual_recomputed_equals_closure"])
        self.assertEqual(set(report["closure_nodes"]), set(report["recomputed_nodes"]))
        for node_id in set(before_hashes) - set(closure):
            self.assertEqual(graph.cache[node_id].content_hash, before_hashes[node_id])
            self.assertEqual(graph.cache[node_id].version, before_versions[node_id])
        for node_id in closure:
            self.assertGreater(graph.cache[node_id].version, before_versions[node_id])
        self.assertTrue(report["reused_sources"])
        self.assertEqual(graph.cache["ending:normal_path"].value["ending"], "alliance")
        self.assertEqual(graph.cache["ending:normal_path"].value["final_state"]["reputation"], 0.65)

    def test_threshold_edit_invalidates_judgment_path_and_ending(self):
        graph = IncrementalDependencyGraph(normal(), "normal_path")
        key = "e_ally:c_trust_ally:threshold"
        closure = graph.closure(key)
        self.assertIn("condition:normal_path:1:e_ally:c_trust_ally", closure)
        self.assertIn("ending:normal_path", closure)
        report = graph.set_threshold(key, 0.5)
        self.assertTrue(report["actual_recomputed_equals_closure"])
        ending = graph.cache["ending:normal_path"].value
        self.assertEqual(ending["status"], "blocked")
        self.assertEqual(ending["ending"], "blocked:s_choice")
        changed = {item["node_id"] for item in report["changed_nodes"] if item["changed"]}
        self.assertIn("condition:normal_path:1:e_ally:c_trust_ally", changed)
        self.assertIn("ending:normal_path", changed)

    def test_unknown_trigger_key_is_structured_error(self):
        graph = IncrementalDependencyGraph(normal(), "normal_path")
        with self.assertRaises(WorkbenchError) as caught:
            graph.closure("missing:key")
        self.assertEqual(caught.exception.code, E_DEPENDENCY_INVALID)

    def test_graph_manifest_is_deterministic(self):
        first = IncrementalDependencyGraph(normal(), "normal_path").node_manifest()
        second = IncrementalDependencyGraph(normal(), "normal_path").node_manifest()
        self.assertEqual(first, second)
        self.assertTrue(all(node["content_hash"] for node in first["nodes"]))


if __name__ == "__main__":
    unittest.main()
