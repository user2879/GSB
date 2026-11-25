import unittest

from arbplatform.propagation import (
    CHANGE_OBS_REPLACE,
    CHANGE_VIEW_POLICY,
    Change,
    propagate,
)
from tests.test_engine import manifest


class PropagationTest(unittest.TestCase):
    def setUp(self):
        self.m = manifest("latest_revision")

    def test_observation_change_propagates_to_referencing_views(self):
        change = Change(CHANGE_OBS_REPLACE, env_id="e1", key="k")
        plan = propagate(self.m, {}, [change])
        self.assertEqual(plan.affected_views, ("v",))
        step = plan.steps[0]
        self.assertEqual(step.affected_keys, ("k",))
        self.assertEqual((step.from_version, step.to_version), (0, 1))
        self.assertIn("e1", step.basis)

    def test_versions_monotonic_over_repeated_changes(self):
        versions: dict[str, int] = {}
        for _ in range(3):
            plan = propagate(
                self.m, versions,
                [Change(CHANGE_OBS_REPLACE, env_id="e2", key="k")],
            )
            versions = plan.view_versions
        self.assertEqual(versions["v"], 3)

    def test_policy_change_only_invalidates_its_key(self):
        # Add a second key via a fresh manifest inline.
        data = {
            "scale_power": 2, "global_concurrency": 2,
            "environments": [
                {"env_id": "e1", "host": "127.0.0.1", "port": 49021,
                 "lease_capacity": 1, "seed": 1, "weight": 1}],
            "instances": [
                {"instance_id": "i1", "env_ref": "e1", "seq_low": 1,
                 "seq_high": 2, "time_base_ms": 1}],
            "views": [
                {"view_id": "v", "env_refs": ["e1"],
                 "policies": {"a": "latest_revision",
                              "b": "weighted_majority"}, "tolerance": 0}],
        }
        from arbplatform.spec import parse_manifest
        m = parse_manifest(data)
        plan = propagate(m, {}, [Change(CHANGE_VIEW_POLICY,
                                        view_id="v", key="b")])
        self.assertEqual(plan.steps[0].affected_keys, ("b",))

    def test_output_order_stable(self):
        first = propagate(
            self.m, {},
            [Change(CHANGE_OBS_REPLACE, env_id="e1", key="k")]).as_dict()
        second = propagate(
            self.m, {},
            [Change(CHANGE_OBS_REPLACE, env_id="e1", key="k")]).as_dict()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()


class ViewLifecyclePropagationTest(unittest.TestCase):
    def test_view_added_and_removed(self):
        from arbplatform.propagation import (
            CHANGE_VIEW_ADDED,
            CHANGE_VIEW_REMOVED,
        )
        m = manifest("latest_revision")
        added = propagate(m, {}, [
            Change(CHANGE_VIEW_ADDED, view_id="v")])
        self.assertEqual(added.affected_views, ("v",))
        self.assertEqual(added.steps[0].affected_keys, ("k",))

        removed = propagate(m, {"v": 4}, [
            Change(CHANGE_VIEW_REMOVED, view_id="v")])
        self.assertEqual(removed.removed_views, ("v",))
        self.assertEqual(removed.affected_views, ())
        self.assertEqual(removed.steps, [])
