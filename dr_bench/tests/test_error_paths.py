import unittest

from dr_bench.canonical import first_diff
from dr_bench.errors import DivergenceError, ReplayError
from dr_bench.recovery import compare_path_outputs


class ErrorReportingTests(unittest.TestCase):
    def test_first_diff_locates_instance_and_field(self):
        left = {"round_index": 1, "instance_states": {"i-1": {"state": "healthy"}}}
        right = {"round_index": 1, "instance_states": {"i-1": {"state": "unhealthy"}}}
        path, expected, actual = first_diff(left, right)
        self.assertEqual(path, "$.instance_states.i-1.state")
        self.assertEqual(expected, "healthy")
        self.assertEqual(actual, "unhealthy")

    def test_compare_outputs_raises_diverge_with_location(self):
        good = [{"round_index": 0, "instance_states": {"i": {"state": "healthy"}}}]
        bad = [{"round_index": 0, "instance_states": {"i": {"state": "unknown"}}}]
        with self.assertRaises(DivergenceError) as ctx:
            compare_path_outputs({"a": good, "b": bad})
        self.assertEqual(ctx.exception.code, "E_DIVERGE")
        self.assertEqual(
            ctx.exception.detail["first_diff"][0],
            "$.instance_states.i.state",
        )

    def test_replay_error_carries_code(self):
        error = ReplayError("boom", "rounds[2]", {"expected": "x", "actual": "y"})
        payload = error.to_dict()
        self.assertEqual(payload["code"], "E_REPLAY")
        self.assertEqual(payload["path"], "rounds[2]")
        self.assertEqual(payload["detail"]["expected"], "x")


if __name__ == "__main__":
    unittest.main()
