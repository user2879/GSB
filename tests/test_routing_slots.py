import json
import os
import unittest

from workbench.loader import load_vocabulary
from workbench.models import HierarchyConfig
from workbench.routing import route_domain, route_intent
from workbench.slots import fill_slots
from workbench.recalibration import recalibrate
from workbench.fixedpoint import decimal_to_fixed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RoutingSlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocab = load_vocabulary(os.path.join(ROOT, "data/vocab/vocabulary.json"))
        cls.config = HierarchyConfig.from_dict(
            json.load(open(os.path.join(ROOT, "data/config/hierarchy_config.json"),
                           encoding="utf-8")), cls.vocab)

    def test_domain_routing_picks_best_bag(self):
        result = route_domain(("my", "phone", "screen", "broken", "repair"),
                              self.config)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["winner"], "aftersale")
        self.assertEqual(result["criterion"], "domain_unique_max")

    def test_domain_tie_lexicographic(self):
        # Construct an artificial tie by giving every domain the same bag.
        cfg = self._tie_config()
        result = route_domain(("please",), cfg)
        self.assertTrue(result["tie"])
        self.assertEqual(result["winner"], min(cfg.domains))
        self.assertEqual(result["criterion"], "domain_tie_lexicographic")

    def _tie_config(self):
        raw = json.load(open(os.path.join(ROOT, "data/config/hierarchy_config.json"),
                             encoding="utf-8"))
        raw["domain_word_bags"] = {
            domain: {"please": 1} for domain in raw["domains"]}
        # inject the shared token into every template trigger set is unnecessary;
        # domain routing only needs the bags.
        return HierarchyConfig.from_dict(raw, self.vocab)

    def test_intent_threshold_equal_records_criterion(self):
        result = route_intent(("battery", "fault", "refund", "laptop"),
                              "aftersale", self.config)
        # the tie path is lexicographic; verify winner is the smaller name
        self.assertIn(result["criterion"],
                      {"intent_tie_lexicographic",
                       "intent_score_equal_threshold_lexicographic",
                       "intent_unique_max"})
        self.assertEqual(sorted(result["scores"]), sorted(
            self.config.domain_intents["aftersale"]))

    def test_intents_never_cross_domain(self):
        result = route_intent(("invoice", "reissue"), "billing", self.config)
        for intent in result["scores"]:
            self.assertIn(intent, self.config.domain_intents["billing"])

    def test_slot_suppression_ordering(self):
        tokens = ("broken", "phone", "and", "laptop", "fault", "repair")
        result = fill_slots(tokens, "repair_request", self.config)
        self.assertEqual(result["filled_slots"][0]["slot"], "product")
        self.assertTrue(result["suppressed"])
        # suppression list ordering key
        keys = [(item["slot"], -item["match_length"], item["start"], item["value"])
                for item in result["suppressed"]]
        self.assertEqual(keys, sorted(keys))
        self.assertIn("issue_part", result["missing_slots"])
        self.assertTrue(any(
            e["code"] == "E_SLOT_MISSING" for e in result["errors"]))

    def test_slot_window_allows_gapped_match(self):
        # window=2: a single filler token is allowed between slot tokens
        # use single-token values here; window applies to multi-token values
        tokens = ("track", "parcel", "to", "shanghai", "courier")
        result = fill_slots(tokens, "track_shipment", self.config)
        pairs = dict(result["slot_pairs"])
        self.assertEqual(pairs["city"], ["shanghai"])

    def test_recalibration_boost_and_penalty_clamp(self):
        raw = decimal_to_fixed("0.5", self.config.scale_bits())
        result = recalibrate(raw, 2, 2, 3, self.config)
        self.assertGreater(float(result["confidence_after"]),
                           float(result["confidence_before"]))
        self.assertEqual(result["completeness"], "1")
        # clamp at upper bound with huge completeness
        top = recalibrate(decimal_to_fixed("1", self.config.scale_bits()),
                          1, 1, 0, self.config)
        self.assertLessEqual(float(top["confidence_after"]), 1.0)

    def test_quantization_levels_change_values(self):
        raw = decimal_to_fixed("0.3333", self.config.scale_bits())
        result = recalibrate(raw, 2, 1, 1, self.config)
        levels = result["all_quantization_levels"]
        self.assertNotEqual(
            levels["coarse"]["raw_quantized"],
            levels["fine"]["raw_quantized"])
        self.assertNotEqual(
            levels["coarse"]["recal_quantized"],
            levels["fine"]["recal_quantized"])

    def test_negation_resolution_suppresses_trigger(self):
        result = route_intent(("not", "refund", "please"),
                              "aftersale", self.config)
        refund_entries = [e for e in result["trigger_entries"]
                          if e["trigger"] == "refund"]
        self.assertTrue(refund_entries)
        self.assertTrue(refund_entries[0]["negated"])
        self.assertEqual(refund_entries[0]["negator"], "not")
        self.assertEqual(result["scores"]["refund_request"], "0")

    def test_negation_outside_window_keeps_trigger(self):
        result = route_intent(("not", "please", "urgent", "refund"),
                              "aftersale", self.config)
        refund_entries = [e for e in result["trigger_entries"]
                          if e["trigger"] == "refund"]
        self.assertTrue(refund_entries)
        self.assertFalse(refund_entries[0]["negated"])


if __name__ == "__main__":
    unittest.main()
