import unittest

from arbiter.fixedpoint import FixedPoint
from arbiter.merge import (
    EnvVote,
    merge_interval_midpoint,
    merge_key,
    merge_latest,
    merge_weighted_majority,
)
from arbiter.sample import Observation


def vote(env_id, rev, ts, value, weight=1, digest="d"):
    fp = FixedPoint(2)
    return EnvVote(env_id, rev, ts, fp.quantize(value).q, str(value), weight, digest)


class MergePolicyTest(unittest.TestCase):
    def test_latest_revision_then_ts_tiebreak(self):
        votes = [
            vote("env-a", 3, 100, "10"),
            vote("env-b", 5, 100, "20"),
            vote("env-c", 5, 90, "30"),
        ]
        result = merge_latest(votes, "k")
        self.assertEqual(result.authority_q, 2000)
        self.assertEqual(result.adopted_envs, ("env-b",))
        self.assertEqual(result.rejected_envs, ("env-a", "env-c"))

    def test_weighted_majority_groups_and_tiebreak(self):
        votes = [
            vote("env-a", 1, 1, "10", weight=5),
            vote("env-b", 1, 1, "10", weight=3),
            vote("env-c", 1, 1, "20", weight=4),
        ]
        result = merge_weighted_majority(votes, "k", FixedPoint(2))
        self.assertEqual(result.authority_q, 1000)
        self.assertEqual(result.adopted_envs, ("env-a", "env-b"))
        self.assertEqual(result.evidence["group_weights"], {"1000": 8, "2000": 4})

    def test_interval_midpoint_integer_truncation(self):
        votes = [
            vote("env-a", 1, 1, "10"),
            vote("env-b", 1, 1, "13"),
            vote("env-c", 1, 1, "20"),
        ]
        result = merge_interval_midpoint(votes, "k", FixedPoint(2))
        self.assertEqual(result.authority_q, 1500)
        self.assertEqual(result.evidence["lower_q"], 1000)
        self.assertEqual(result.evidence["upper_q"], 2000)
        self.assertEqual(result.evidence["span_even"], True)

    def test_order_independence_across_input_permutation(self):
        fp = FixedPoint(2)
        base = [
            vote("env-a", 2, 10, "10.5", weight=2),
            vote("env-b", 4, 20, "12.25", weight=3),
            vote("env-c", 4, 15, "12.25", weight=1),
            vote("env-d", 1, 5, "9.0", weight=4),
        ]
        for policy in ("latest", "weighted_majority", "interval_midpoint"):
            outputs = []
            for ordering in (base, list(reversed(base)), [base[2], base[0], base[3], base[1]]):
                if policy == "latest":
                    r = merge_latest(list(ordering), "k")
                elif policy == "weighted_majority":
                    r = merge_weighted_majority(list(ordering), "k", fp)
                else:
                    r = merge_interval_midpoint(list(ordering), "k", fp)
                outputs.append((r.authority_q, r.adopted_envs, r.rejected_envs))
            self.assertEqual(len(set(outputs)), 1, policy)

    def test_merge_key_with_observation_chains(self):
        fp = FixedPoint(1)
        obs = {
            "env-a": [
                Observation("env-a", "i1", "k", 1, 10, "10.0", 8),
                Observation("env-a", "i1", "k", 2, 20, "11.0", 8),
            ],
            "env-b": [
                Observation("env-b", "i2", "k", 1, 11, "12.0", 8),
            ],
        }
        result = merge_key(
            "latest", "k", ("env-a", "env-b"), obs,
            {"env-a": 1, "env-b": 1}, {}, fp,
        )
        self.assertEqual(result.authority_q, 110)


if __name__ == "__main__":
    unittest.main()
