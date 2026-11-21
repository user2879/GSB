import copy
import unittest

from graphdiag.core.fixedpoint import Quantizer
from graphdiag.core.loader import load_declaration
from graphdiag.core.scheduler import TraversalBudget
from graphdiag.core.traversal import (
    CONFIRMED_OVER,
    ERROR_CROSS,
    UNREACHABLE,
    WITHIN,
    run_traversal,
)

TINY = "data/tiny_graph.json"


class DeterminismTest(unittest.TestCase):
    def test_byte_identical_under_insertion_shuffle(self):
        decl_a = load_declaration(TINY)
        import json
        with open(TINY, encoding="utf-8") as src:
            raw = json.load(src)
        shuffled = copy.deepcopy(raw)
        # Reverse node and edge insertion order; logical content is unchanged.
        shuffled["nodes"] = list(reversed(shuffled["nodes"]))
        shuffled["edges"] = list(reversed(shuffled["edges"]))
        import tempfile, os
        path = tempfile.mktemp(suffix=".json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(shuffled, handle)
        decl_b = load_declaration(path)
        os.unlink(path)

        qz = Quantizer(decl_a.scale_exponent)
        for query_a, query_b in zip(decl_a.queries, decl_b.queries):
            ra = run_traversal(decl_a.graph, query_a, qz, TraversalBudget(10 ** 9))
            rb = run_traversal(decl_b.graph, query_b, qz, TraversalBudget(10 ** 9))
            self.assertEqual(ra.visited_sequence, rb.visited_sequence)
            self.assertEqual(
                [(t.targets_key if False else t.target, t.dist_fp, t.path_nodes,
                  t.exact_bound_micro, t.classification)
                 for t in ra.targets.values()],
                [(t.target, t.dist_fp, t.path_nodes,
                  t.exact_bound_micro, t.classification)
                 for t in rb.targets.values()],
            )

    def test_pruning_records_cap_breach(self):
        decl = load_declaration(TINY)
        qz = Quantizer(0)
        result = run_traversal(
            decl.graph, decl.queries[0], qz, TraversalBudget(10 ** 9)
        )
        reasons = {p.reason for p in result.pruned}
        self.assertIn("cap_exceeded", reasons)
        for prune in result.pruned:
            self.assertGreater(prune.candidate_fp, prune.cap_fp)

    def test_budget_halts_traversal(self):
        decl = load_declaration(TINY)
        result = run_traversal(
            decl.graph, decl.queries[0], Quantizer(0), TraversalBudget(2)
        )
        self.assertTrue(result.budget_exhausted)
        self.assertLessEqual(len(result.visited_sequence), 2)

    def test_indistinct_and_within_classification(self):
        decl = load_declaration(TINY)
        qz = Quantizer(0)
        by_id = {q.query_id: q for q in decl.queries}
        small = run_traversal(
            decl.graph, by_id["q-small-cap"], qz, TraversalBudget(10 ** 9)
        )
        # target 4: dist 2.0 + exact error crosses cap 4.5 in conservative view
        t4 = small.targets[4]
        self.assertIn(
            (t4.classification_exact, t4.classification_conservative),
            [(WITHIN, ERROR_CROSS), (ERROR_CROSS, ERROR_CROSS)],
        )
        indistinct = run_traversal(
            decl.graph, by_id["q-indistinct"], qz, TraversalBudget(10 ** 9)
        )
        target = indistinct.targets[4]
        self.assertTrue(target.error_affected)

    def test_dominant_error_segment_reported(self):
        decl = load_declaration(TINY)
        result = run_traversal(
            decl.graph, decl.queries[0], Quantizer(0), TraversalBudget(10 ** 9)
        )
        target = result.targets[3]
        self.assertEqual(target.dominant_edge, (2, 3))
        self.assertEqual(target.dominant_remainder_micro, 900_000)

    def test_unreachable_target_explicit_row(self):
        decl = load_declaration(TINY)
        query = decl.queries[0]
        tiny_budget = run_traversal(
            decl.graph, query, Quantizer(0), TraversalBudget(1)
        )
        statuses = {t.target: t.classification for t in tiny_budget.targets.values()}
        self.assertIn(UNREACHABLE, statuses.values())


if __name__ == "__main__":
    unittest.main()
