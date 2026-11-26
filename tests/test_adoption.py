"""Requirement 5: adoption winner rule, rejection reasons and order independence."""
import unittest

from shardmeta.adoption import decide_all, decisions_canonical_blob
from shardmeta.models import MergeCandidate


def candidate(key, index, *, merge_id, meta_version, source_format, merge_seq,
              digest=None, content_check=None, node_id="n1"):
    return MergeCandidate(
        object_key=key, shard_index=index,
        shard_digest=digest or f"d-{merge_id}",
        content_check=content_check, merge_id=merge_id,
        parent_merge_id=None, shard_bytes=8, meta_version=meta_version,
        source_format=source_format, merge_seq=merge_seq,
        origin="test", node_id=node_id,
    )


class AdoptionTest(unittest.TestCase):
    def test_winner_prefers_higher_meta_version_then_format_then_seq(self):
        candidates = [
            candidate("k", 0, merge_id="old", meta_version=1,
                      source_format=3, merge_seq=1),
            candidate("k", 0, merge_id="mid", meta_version=2,
                      source_format=2, merge_seq=2),
            candidate("k", 0, merge_id="win", meta_version=2,
                      source_format=3, merge_seq=3),
        ]
        decisions = decide_all(candidates)
        decision = decisions["k"][0]
        self.assertEqual(decision.winner.merge_id, "win")
        reasons = {item.candidate.merge_id: item.reason for item in decision.rejected}
        self.assertEqual(reasons["old"], "lower-meta-version")
        self.assertEqual(reasons["mid"], "lower-source-format")

    def test_dispatch_and_dict_order_independent_bytes(self):
        candidates = [
            candidate("b", 0, merge_id="b0", meta_version=1, source_format=1, merge_seq=4),
            candidate("a", 0, merge_id="a0", meta_version=1, source_format=2, merge_seq=2),
            candidate("a", 0, merge_id="a1", meta_version=1, source_format=3, merge_seq=1),
            candidate("a", 1, merge_id="a1x", meta_version=3, source_format=1, merge_seq=3),
        ]
        baseline = decisions_canonical_blob(decide_all(candidates))
        shuffled = list(reversed(candidates))
        # Insert in a different order; conclusion bytes must stay identical.
        reordered = [shuffled[2], shuffled[0], shuffled[3], shuffled[1]]
        again = decisions_canonical_blob(decide_all(reordered))
        self.assertEqual(baseline, again)
        decisions = decide_all(candidates)
        self.assertEqual(decisions["a"][0].winner.merge_id, "a1")

    def test_mixed_formats_compared_after_normalization(self):
        # Same shard, c3 record wins over c2 even with a later merge sequence.
        c2 = candidate("k", 0, merge_id="c2-id", meta_version=2,
                       source_format=2, merge_seq=1, digest="same")
        c3 = candidate("k", 0, merge_id="c3-id", meta_version=2,
                       source_format=3, merge_seq=9, digest="same")
        decision = decide_all([c2, c3])["k"][0]
        self.assertEqual(decision.winner.merge_id, "c3-id")
        self.assertEqual(decision.rejected[0].reason, "lower-source-format")


if __name__ == "__main__":
    unittest.main()
