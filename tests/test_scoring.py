import unittest

from k8ssim.constants import PLUGIN_ORDER
from k8ssim.loader import load_nodes, load_pods
from k8ssim.scheduler.engine import SchedulingEngine
from k8ssim.scheduler.plugins import quantize

NODES = load_nodes("samples/nodes.json")
PODS = load_pods("samples/pods.json", NODES)


class FixedPointTest(unittest.TestCase):
    def test_quantize_bounds_and_exact_error(self) -> None:
        for divisor in range(1, 25):
            for numerator in range(divisor + 1):
                for scale in (1, 7, 100, 10000):
                    raw, error_mp = quantize(numerator, divisor, scale)
                    self.assertGreaterEqual(raw, 0)
                    self.assertLessEqual(raw, 100)
                    real = 100 * numerator / divisor
                    exact_error_mp = abs(real - raw) * 1_000_000
                    self.assertLessEqual(exact_error_mp, error_mp + 1)
                    if exact_error_mp == 0:
                        self.assertEqual(error_mp, 0)
                    else:
                        self.assertGreater(error_mp, 0)

    def test_zero_remainder_has_zero_error(self) -> None:
        raw, error_mp = quantize(1, 2, 123)
        self.assertEqual((raw, error_mp), (50, 0))


class ScoreCompositionTest(unittest.TestCase):
    def test_plugin_scores_are_integers_in_range(self) -> None:
        engine = SchedulingEngine(nodes=NODES, pods=PODS)
        report = engine.run()
        for decision in report.decisions:
            for node_score in decision.node_scores:
                self.assertIsInstance(node_score.total_mp, int)
                for plugin in node_score.plugins:
                    self.assertIsInstance(plugin.raw_score, int)
                    self.assertGreaterEqual(plugin.raw_score, 0)
                    self.assertLessEqual(plugin.raw_score, 100)
                    self.assertEqual(plugin.truncation, "floor_div")
                    self.assertGreaterEqual(plugin.quant_error_mp, 0)

    def test_repeated_runs_are_identical(self) -> None:
        first = SchedulingEngine(nodes=NODES, pods=PODS).run()
        second = SchedulingEngine(nodes=NODES, pods=PODS).run()
        self.assertEqual(
            [d.to_dict() for d in first.decisions],
            [d.to_dict() for d in second.decisions],
        )

    def test_plugin_computation_order_independence(self) -> None:
        from k8ssim.scheduler import scoring
        from k8ssim.models import NodeScore

        engine = SchedulingEngine(nodes=NODES, pods=PODS)
        engine.run()
        original = scoring.compose_node_score

        def reversed_compose(node_name, outputs):
            reversed_outputs = {key: outputs[key] for key in reversed(list(outputs))}
            result = original(node_name, reversed_outputs)
            # Re-order plugins back to canonical order for comparison.
            canonical = sorted(result.plugins, key=lambda p: PLUGIN_ORDER.index(p.plugin))
            return NodeScore(
                node=result.node,
                plugins=tuple(canonical),
                total_mp=result.total_mp,
                total_points=result.total_points,
                comp_trunc_error_mp=result.comp_trunc_error_mp,
                error_bound_mp=result.error_bound_mp,
                error_bound_points=result.error_bound_points,
            )

        scoring.compose_node_score = reversed_compose
        try:
            shuffled_report = engine.run()
        finally:
            scoring.compose_node_score = original
        canonical_report = engine.run()
        for left, right in zip(shuffled_report.decisions, canonical_report.decisions):
            self.assertEqual(left.bound_node, right.bound_node)
            self.assertEqual(left.error_influenced, right.error_influenced)
            self.assertEqual(left.tie_reason, right.tie_reason)
            self.assertEqual(left.score_gap_mp, right.score_gap_mp)

    def test_error_influenced_flag_is_stable(self) -> None:
        engine = SchedulingEngine(nodes=NODES, pods=PODS)
        report = engine.run()
        flagged = {d.pod_id for d in report.decisions if d.error_influenced}
        self.assertIn("web-1", flagged)
        for decision in report.decisions:
            if decision.error_influenced:
                self.assertEqual(decision.tie_reason, "error_bound_tie")
                self.assertIsNotNone(decision.runner_up)
                self.assertLessEqual(
                    decision.score_gap_mp, decision.combined_error_mp
                )

    def test_exact_tie_breaks_by_plugin_vector_then_name(self) -> None:
        from k8ssim.scheduler.scoring import compose_node_score, select_binding
        from k8ssim.scheduler.plugins import PluginOutput

        def output(plugin, score):
            return PluginOutput(
                plugin=plugin, numerator=score, divisor=100, raw_score=score,
                weight=2500, contribution_mp=score * 2500, quant_error_mp=0,
            )

        def make_score(name, vector):
            outputs = {
                plugin: output(plugin, vector[index])
                for index, plugin in enumerate(PLUGIN_ORDER)
            }
            return compose_node_score(name, outputs)

        score_z = make_score("node-z", (50, 50, 50, 50))
        score_a = make_score("node-a", (50, 50, 50, 50))
        decision = select_binding(
            ["node-a", "node-z"], [score_z, score_a], "p", 1, []
        )
        self.assertEqual(decision.bound_node, "node-a")
        self.assertEqual(decision.tie_reason, "exact_tie")
        self.assertFalse(decision.error_influenced)

    def test_vector_order_outranks_name_on_non_exact_tie(self) -> None:
        from k8ssim.scheduler.scoring import compose_node_score, select_binding
        from k8ssim.scheduler.plugins import PluginOutput

        def output(plugin, score):
            return PluginOutput(
                plugin=plugin, numerator=score, divisor=100, raw_score=score,
                weight=2500, contribution_mp=score * 2500, quant_error_mp=0,
            )

        # Equal weighted totals; the lexicographic plugin vector decides.
        high_first = compose_node_score(
            "node-z", {p: output(p, (60, 40, 40, 40)[i])
                       for i, p in enumerate(PLUGIN_ORDER)}
        )
        low_first = compose_node_score(
            "node-a", {p: output(p, (40, 60, 40, 40)[i])
                       for i, p in enumerate(PLUGIN_ORDER)}
        )
        decision = select_binding(
            ["node-a", "node-z"], [high_first, low_first], "p", 1, []
        )
        self.assertEqual(decision.bound_node, "node-z")


if __name__ == "__main__":
    unittest.main()
