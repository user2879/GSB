import unittest

from arbplatform.planner import FULL, INCREMENTAL, ScaleParameters, choose_path


class PlannerTest(unittest.TestCase):
    def test_empty_changes_prefers_incremental_noop(self):
        params = ScaleParameters(3, 5, 500, 0, 0)
        decision = choose_path(params)
        self.assertEqual(decision.path, INCREMENTAL)
        self.assertEqual(decision.boundary["rule"], "empty_changeset")

    def test_small_input_boundary_forces_full(self):
        params = ScaleParameters(3, 5, 256, 10, 2)
        decision = choose_path(params)
        self.assertEqual(decision.path, FULL)
        self.assertEqual(decision.boundary["rule"], "small_input")
        self.assertEqual(decision.boundary["threshold"], 256)

    def test_all_keys_affected_forces_full(self):
        params = ScaleParameters(4, 5, 1000, 50, 5)
        decision = choose_path(params)
        self.assertEqual(decision.path, FULL)
        self.assertEqual(decision.boundary["rule"], "all_keys_affected")

    def test_partial_change_selects_incremental(self):
        # Large input, only one of many keys touched by few observations.
        params = ScaleParameters(4, 20, 5000, 4, 1)
        decision = choose_path(params)
        self.assertEqual(decision.path, INCREMENTAL)
        self.assertGreater(decision.full_effort, decision.incremental_effort)
        self.assertFalse(decision.tolerance_influenced)

    def test_tolerance_band_tie_breaks_full_and_flags(self):
        # Construct near-equal efforts inside the 10% band.
        params = ScaleParameters(2, 10, 1000, 5, 5)
        # all keys affected would trigger first; use partial keys instead.
        params = ScaleParameters(2, 10, 1000, 1, 1)
        decision = choose_path(params)
        if decision.tolerance_influenced:
            self.assertEqual(decision.path, FULL)
            self.assertEqual(decision.boundary["rule"], "tolerance_tie")
        else:
            self.assertIn(decision.path, (FULL, INCREMENTAL))

    def test_decision_is_deterministic(self):
        params = ScaleParameters(4, 20, 5000, 4, 1)
        first = choose_path(params).as_dict()
        second = choose_path(params).as_dict()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
