import unittest

from intent_router.attribution import (
    FLIP_DEDUP,
    FLIP_QUANT,
    FLIP_THRESHOLD,
    FLIP_WEIGHT,
    FLIP_NONE,
    attribute_flip,
)
from intent_router.dedup import DedupMember
from intent_router.scoring import IntentPath, ScoreResult


def score(decision, paths, eligible=None):
    result = ScoreResult("T", decision)
    result.paths = dict(paths)
    result.eligible = list(eligible if eligible is not None else paths.keys())
    return result


class AttributionTest(unittest.TestCase):
    def test_no_flip(self):
        s = score("IA", {"IA": IntentPath("IA", 10, 10, 1, [], 0)})
        flip = attribute_flip("T", s, s, None, None)
        self.assertFalse(flip.flipped)
        self.assertEqual(flip.source, FLIP_NONE)

    def test_quantization_source(self):
        # Identical reference maps; only quantized values move and break a tie.
        b = score(
            "IA",
            {
                "IA": IntentPath("IA", 1000, 1000, 1, [], 0),
                "IB": IntentPath("IB", 1000, 1000, 1, [], 0),
            },
        )
        a = score(
            "IB",
            {
                "IA": IntentPath("IA", 1000, 1000, 1, [], 0),
                "IB": IntentPath("IB", 1000, 1004, 1, [], 0),
            },
        )
        pure = attribute_flip("T", b, a, None, None)
        flip = pure
        self.assertEqual(pure.source, FLIP_QUANT)
        self.assertIn("quantized_delta", flip.evidence)

    def test_threshold_source(self):
        before = score("FB", {"IA": IntentPath("IA", 1000, 1000, 1, [], 1500)}, eligible=[])
        after = score("IA", {"IA": IntentPath("IA", 1000, 1000, 1, [], 500)}, eligible=["IA"])
        flip = attribute_flip("T", before, after, None, None)
        self.assertEqual(flip.source, FLIP_THRESHOLD)

    def test_dedup_source_takes_precedence(self):
        before = score("IA", {"IA": IntentPath("IA", 10, 10, 1, [], 0)})
        after = score("IB", {"IB": IntentPath("IB", 20, 20, 1, [], 0)})
        member = DedupMember("T", "REP", 800, False)
        flip = attribute_flip("T", before, after, None, member)
        self.assertEqual(flip.source, FLIP_DEDUP)

    def test_weight_change_source(self):
        before = score("FB", {"IA": IntentPath("IA", 10, 10, 1, [], 100)})
        after = score("IA", {"IA": IntentPath("IA", 500, 500, 1, [], 100)})
        flip = attribute_flip("T", before, after, None, None)
        self.assertEqual(flip.source, FLIP_WEIGHT)
        self.assertNotEqual(
            flip.evidence["reference_delta"], 0
        )


if __name__ == "__main__":
    unittest.main()
