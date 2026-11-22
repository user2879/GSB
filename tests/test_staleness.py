import unittest

from k8ssim.loader import load_nodes, load_pods
from k8ssim.scheduler.engine import SchedulingEngine

NODES = load_nodes("samples/nodes.json")
PODS = load_pods("samples/pods.json", NODES)


class StalenessTest(unittest.TestCase):
    def test_version_increases_and_old_report_marked_stale(self) -> None:
        engine = SchedulingEngine(nodes=NODES, pods=PODS)
        first = engine.run()
        self.assertEqual(engine.params.version, 1)
        engine.update_params(scale_factor=100)
        self.assertEqual(engine.params.version, 2)
        self.assertEqual(first.staleness, "stale")
        stale = engine.stale_reports()
        self.assertEqual([r.parameter_version for r in stale], [1])

    def test_replaced_entries_list_is_stable(self) -> None:
        engine = SchedulingEngine(nodes=NODES, pods=PODS)
        engine.run()
        engine.update_params(scale_factor=100)
        second = engine.run()
        self.assertEqual(
            list(second.replaced_entries), [pod.pod_id for pod in PODS]
        )
        self.assertEqual(second.staleness, "current")

    def test_unchanged_params_keep_version(self) -> None:
        engine = SchedulingEngine(nodes=NODES, pods=PODS)
        before = engine.params.version
        engine.update_params(weights=dict(engine.params.weights),
                             scale_factor=engine.params.scale_factor)
        self.assertEqual(engine.params.version, before)

    def test_weight_change_replaces_entire_view(self) -> None:
        engine = SchedulingEngine(nodes=NODES, pods=PODS)
        engine.run()
        weights = dict(engine.params.weights)
        weights["label_locality"] += 500
        weights["resource_balance"] -= 500
        engine.update_params(weights=weights)
        report = engine.run()
        self.assertEqual(report.parameter_version, 2)
        self.assertEqual(len(report.decisions), len(PODS))
        # Every decision belongs to the new version.
        self.assertTrue(
            all(decision.parameter_version == 2 for decision in report.decisions)
        )


if __name__ == "__main__":
    unittest.main()
