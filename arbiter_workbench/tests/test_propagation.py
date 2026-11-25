import unittest
from pathlib import Path

from arbiter.propagation import ChangeSet, PropagationEngine
from arbiter.sample import DEFAULT_KEYS
from arbiter.spec import load_declarations

DECLS = Path(__file__).resolve().parents[1] / "decls"


class PropagationTest(unittest.TestCase):
    def setUp(self):
        self.decls = load_declarations(DECLS, DEFAULT_KEYS)
        self.engine = PropagationEngine(self.decls)

    def test_versions_monotonic_and_sorted_output(self):
        first = self.engine.propagate(ChangeSet(replaced_obs_envs=("env-prod-a",)))
        view_ids = [item.view_id for item in first]
        self.assertEqual(view_ids, sorted(view_ids))
        self.assertTrue(all(item.new_version == 1 for item in first))
        second = self.engine.propagate(ChangeSet(replaced_obs_envs=("env-prod-a",)))
        self.assertTrue(all(item.new_version == 2 for item in second))

    def test_only_participating_views_invalidated(self):
        result = self.engine.propagate(
            ChangeSet(replaced_obs_envs=("env-dr-d",))
        )
        affected = {item.view_id for item in result}
        # env-dr-d participates only in view-all-envs and view-dr-quorum
        self.assertEqual(affected, {"view-all-envs", "view-dr-quorum"})
        edge = result[0].steps[0]
        self.assertEqual(edge.edge, "env->view")
        self.assertIn("observations replaced", edge.basis)

    def test_policy_change_dirties_only_named_keys(self):
        result = self.engine.propagate(
            ChangeSet(changed_policies={"view-dr-quorum": ("pool.max_size",)})
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].dirty_keys, ("pool.max_size",))
        self.assertTrue(any(s.edge == "policy->key" for s in result[0].steps))

    def test_removed_env_propagates_to_referencing_views(self):
        # env-canary-c currently participates in edge and all-envs views;
        # dr-quorum never references it and must stay untouched.
        result = self.engine.propagate(
            ChangeSet(removed_envs=("env-canary-c",))
        )
        affected = {i.view_id for i in result}
        self.assertEqual(affected, {"view-edge-policy", "view-all-envs"})
        self.assertTrue(
            all(any(s.edge == "env->view" for s in item.steps) for item in result)
        )


if __name__ == "__main__":
    unittest.main()
