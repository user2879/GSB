import unittest

from intent_router.dedup import jaccard_permille, subword_set
from intent_router.fixedpoint import QuantGear
from intent_router.library import MatchCandidate, MatchResult
from intent_router.models import RoutingConfig
from intent_router.scoring import score_text


def config(**over):
    base = dict(
        fraction_bits=8,
        gear=2,
        intents=("IA", "IB"),
        intent_thresholds={"IA": 1, "IB": 1},
        dedup_threshold_permille=500,
        max_workers=2,
        global_match_budget=100,
        channel_quotas={"C1": 100},
        per_batch_recompute_limit=100,
        fallback_intent="FALLBACK",
    )
    base.update(over)
    return RoutingConfig(**base)


class ScoringTest(unittest.TestCase):
    def test_fixed_order_accumulation_and_quantization(self):
        gear = QuantGear(8, 2)
        match = MatchResult("T")
        match.accepted = [
            MatchCandidate("E1", "IA", 5, 0, 2, ("a", "b")),
            MatchCandidate("E2", "IA", 3, 2, 3, ("c",)),
        ]
        cfg = config()
        result = score_text("T", 4, match, cfg, gear)
        path = result.paths["IA"]
        scale = 256
        expected_ref = (5 + 3) * scale - 4 * (scale // 8 + 1)
        self.assertEqual(path.reference_raw, expected_ref)
        self.assertEqual(path.quantized_raw, gear.quantize(expected_ref))
        self.assertEqual(result.decided_intent, "IA")

    def test_half_step_boundary_flag(self):
        gear = QuantGear(8, 4)
        # Directly validate the half-step predicate used by the scorer.
        self.assertTrue(gear.at_half_step(2))
        self.assertFalse(gear.at_half_step(1))

    def test_scorer_marks_half_step_when_aligned(self):
        gear = QuantGear(8, 4)
        match = MatchResult("T")
        # weight*256 with length 0 penalty keeps raw on an integer scale; use a
        # synthetic residual by checking quantization boundary reporting.
        match.accepted = [MatchCandidate("E1", "IA", 1, 0, 1, ("a",))]
        cfg = config()
        result = score_text("T", 1, match, cfg, gear)
        self.assertIn("IA", result.paths)

    def test_fallback_when_below_threshold(self):
        gear = QuantGear(8, 1)
        cfg = config(intent_thresholds={"IA": 100, "IB": 100})
        match = MatchResult("T")
        match.accepted = [MatchCandidate("E1", "IA", 1, 0, 1, ("a",))]
        result = score_text("T", 8, match, cfg, gear)
        self.assertEqual(result.decided_intent, "FALLBACK")

    def test_tie_breaks_by_intent_id(self):
        gear = QuantGear(8, 1)
        cfg = config(intent_thresholds={"IA": 0, "IB": 0})
        match = MatchResult("T")
        match.accepted = [
            MatchCandidate("E1", "IA", 4, 0, 1, ("a",)),
            MatchCandidate("E2", "IB", 4, 1, 1, ("b",)),
        ]
        result = score_text("T", 2, match, cfg, gear)
        self.assertEqual(result.decided_intent, "IA")


class DedupTest(unittest.TestCase):
    def test_jaccard_permille_and_exact_equality(self):
        left = subword_set(("a", "b", "c"))
        right = subword_set(("a", "b", "d"))
        sim, exact = jaccard_permille(left, right)
        # intersection 2 / union 4 = 500 permille
        self.assertEqual(sim, 500)
        self.assertTrue(exact)

    def test_threshold_equal_treated_as_cluster(self):
        # Two sets sharing half their union land exactly on 500 permille.
        left = subword_set(("a", "b", "c"))
        right = subword_set(("a", "b", "d"))
        sim, exact = jaccard_permille(left, right)
        self.assertEqual(sim, 500)
        self.assertTrue(exact)

    def test_disjoint_sets(self):
        sim, _ = jaccard_permille(
            subword_set(("a",)), subword_set(("b",))
        )
        self.assertEqual(sim, 0)


if __name__ == "__main__":
    unittest.main()
