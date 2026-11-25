import unittest

from coljoin.canonical import canonical_bytes
from coljoin.cost import CostEstimator, choose_plan
from coljoin.errors import SpecError
from coljoin.model import JoinPredicate, QuerySpec, SessionSpec
from coljoin.planner import (
    JoinGraph,
    enumerate_connected_orders,
    enumerate_plans,
)
from coljoin.storage import load_generated_specs
from coljoin.loader import load_query, load_weights, load_session

DATA_DIR = "workspace/data"
CFG = "workspace/configs"


def load_context():
    groups = load_generated_specs(DATA_DIR)
    query = load_query(f"{CFG}/query.json", groups)
    weights = load_weights(f"{CFG}/weights.json")
    session = load_session(f"{CFG}/session.json")
    return groups, query, weights, session


class PlannerDeterminismTest(unittest.TestCase):
    def test_plan_set_byte_identical_across_shuffled_declaration(self):
        groups, query, weights, session = load_context()
        plans_a = enumerate_plans(query)
        shuffled_groups = list(reversed(groups))
        query_b = load_query(f"{CFG}/query.json", shuffled_groups)
        plans_b = enumerate_plans(query_b)
        self.assertEqual(
            canonical_bytes(plans_a), canonical_bytes(plans_b)
        )

    def test_plan_ids_sorted_and_unique(self):
        _, query, _, _ = load_context()
        plans = enumerate_plans(query)
        ids = [plan["plan_id"] for plan in plans]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(ids), len(set(ids)))

    def test_connected_orders_prune_cartesian(self):
        groups, query, _, _ = load_context()
        graph = JoinGraph(query)
        orders = enumerate_connected_orders(graph)
        for order in orders:
            used = {order[0]}
            for node in order[1:]:
                self.assertTrue(
                    any(neighbor in used for neighbor in graph.adjacency[node])
                )
                used.add(node)

    def test_disconnected_graph_rejected(self):
        query = QuerySpec(
            "disconnected",
            [
                JoinPredicate("a", "x", "b", "y"),
                JoinPredicate("c", "z", "d", "w"),
            ],
            [],
            [],
            [],
            [],
        )
        with self.assertRaises(SpecError):
            JoinGraph(query)

    def test_three_algorithms_present(self):
        _, query, _, _ = load_context()
        plans = enumerate_plans(query)
        algorithms = set()
        for plan in plans:
            for step in plan["body"]["steps"]:
                if step["op"] == "join":
                    algorithms.add(step["algorithm"])
        self.assertEqual(algorithms, {"hash", "sort_merge", "nested_loop"})


class FixedPointCostTest(unittest.TestCase):
    def test_all_integer_metrics_and_determinism(self):
        groups, query, weights, session = load_context()
        plans = enumerate_plans(query)
        estimator = CostEstimator(groups, weights, session)
        evals = [estimator.evaluate(plan) for plan in plans]
        for evaluation in evals:
            for metric in evaluation["vector"].values():
                self.assertIsInstance(metric, int)
            self.assertEqual(evaluation["truncation"], "floor")
            self.assertGreaterEqual(evaluation["error_interval"]["epsilon"], 0)
        reordered = CostEstimator(list(reversed(groups)), weights, session)
        evals_b = [reordered.evaluate(plan) for plan in plans]
        self.assertEqual(
            canonical_bytes(evals), canonical_bytes(evals_b)
        )

    def test_winner_stable_under_scale_width(self):
        groups, query, weights, _ = load_context()
        winners = set()
        for scale in (1, 10, 1000, 100000):
            session = SessionSpec(scale, 128, "stats-v1")
            plans = enumerate_plans(query)
            estimator = CostEstimator(groups, weights, session)
            decision = choose_plan([estimator.evaluate(plan) for plan in plans])
            winners.add(decision["chosen_plan_id"])
        self.assertEqual(len(winners), 1)

    def test_tie_break_flags_error_influenced(self):
        groups, query, weights, session = load_context()
        plans = enumerate_plans(query)
        estimator = CostEstimator(groups, weights, session)
        decision = choose_plan([estimator.evaluate(plan) for plan in plans])
        self.assertIn("error_influenced", decision)
        self.assertIn("ties", decision)
        if decision["ties"]:
            tie = decision["ties"][0]
            self.assertLessEqual(tie["delta"], tie["tolerance"])
            self.assertEqual(
                tie["resolution"], ["plan_id_order", "op_digest_order"]
            )

    def test_saturation_rule_declared(self):
        groups, query, weights, session = load_context()
        plans = enumerate_plans(query)
        evaluation = CostEstimator(groups, weights, session).evaluate(plans[0])
        from coljoin.fixedpoint import SAT_MAX

        self.assertEqual(evaluation["saturation_limit"], SAT_MAX)


if __name__ == "__main__":
    unittest.main()
