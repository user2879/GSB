import unittest

from gsb.adoption import adjudicate, attribute_segments
from gsb.receipts import normalize, encode_x1, encode_x2, encode_x3

BATCH = "b" * 32
RECORD = {
    "dispatch_id": "d1",
    "subscriber_id": "sub-alpha",
    "event_seqs": [1, 2],
    "digest": "x" * 64,
    "completion_ms": 1000,
    "caliber_version": "v1",
    "covered_refs": [],
}


def _candidates(summary=500, merge_seq=1, duration=10):
    make = [
        ("x1", encode_x1(BATCH, summary)),
        ("x2", encode_x2(BATCH, "sub-alpha", 1000, summary)),
        ("x3", encode_x3(BATCH, "sub-alpha", 1000, summary, [1, 2], [], "v1")),
    ]
    out = []
    for idx, (fmt, text) in enumerate(make):
        normalized = normalize(text, lambda _id: RECORD)
        out.append(
            {
                "normalized": normalized,
                "merge_seq": merge_seq,
                "attempt": 1,
                "fixed_point_duration_ms": duration + idx,
                "summary_digest": "s" * 64,
            }
        )
    return out


class AdoptionTest(unittest.TestCase):
    def test_x3_wins_by_format_descending(self):
        decision = adjudicate(_candidates(), error_margin_ms=2)
        self.assertEqual(decision["winning_format"], "x3")
        self.assertEqual(
            {r["format_version"] for r in decision["rejected_formats"]}, {"x1", "x2"}
        )

    def test_error_margin_flagged(self):
        decision = adjudicate(_candidates(duration=10), error_margin_ms=5)
        self.assertTrue(decision["error_margin_applied"])

    def test_attribution_segments(self):
        attribution = attribute_segments(queue_ms=70, backfill_ms=5, processing_ms=25)
        self.assertEqual(attribution["dominant_segment"], "queue")
        self.assertAlmostEqual(
            sum(attribution["contribution_ratio"].values()), 1.0, places=5
        )


if __name__ == "__main__":
    unittest.main()
