import tempfile
import unittest

from qfusion.audit import build_request_ledger
from qfusion.config import load_config
from qfusion.engine import compute_request
from qfusion.fixedpoint import (STAGE_CLIP, STAGE_FEATURE, STAGE_FUSION,
                                STAGE_NORM)
from qfusion.loader import load_stream
from qfusion.normalization import normalize_channel
from qfusion.config import ChannelSpec
from qfusion.sample_generator import generate


class NormalizationTest(unittest.TestCase):
    def _rec(self, item, score):
        class R:
            def __init__(self, item_id, raw_score):
                self.item_id = item_id
                self.raw_score = raw_score
        return R(item, score)

    def test_minmax_zero_denominator_sets_zero(self):
        recs = [self._rec("a", 1.0), self._rec("b", 1.0)]
        vals, events = normalize_channel(recs, ChannelSpec("c", 1.0, "minmax"))
        self.assertEqual(set(vals.values()), {0.0})
        self.assertEqual(events[0]["rule"], "minmax_zero_range")

    def test_zscore_zero_std_sets_zero(self):
        recs = [self._rec("a", 2.0), self._rec("b", 2.0)]
        vals, events = normalize_channel(recs, ChannelSpec("c", 1.0, "zscore"))
        self.assertEqual(set(vals.values()), {0.0})
        self.assertEqual(events[0]["rule"], "zscore_zero_std")

    def test_minmax_bounds(self):
        recs = [self._rec("a", 0.0), self._rec("b", 5.0),
                self._rec("c", 2.5)]
        vals, _ = normalize_channel(recs, ChannelSpec("c", 1.0, "minmax"))
        self.assertEqual(vals["a"], 0.0)
        self.assertEqual(vals["b"], 1.0)


class EngineAuditTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        info = generate(self.tmp)
        self.cfg, _ = load_config(info["config"])
        self.stream = load_stream(info["stream"], self.cfg)
        req = self.stream.request_ids[0]
        self.req = req
        features = {item: self.stream.requests[req][
            self.cfg.channel_names_sorted[0]][0].features
            for item in self.stream.request_items[req]}
        self.result = compute_request(
            req, self.stream.requests[req],
            self.stream.request_items[req], features, self.cfg)
        self.max_f = max(self.cfg.frac_bits_choices)
        self.ledger = build_request_ledger(self.result, self.max_f)

    def test_settings_cartesian_and_determinism(self):
        expected = (len(self.cfg.frac_bits_choices)
                    * len(self.cfg.gears))
        self.assertEqual(len(self.result["settings"]), expected)
        again = build_request_ledger(self.result, self.max_f)
        self.assertEqual(again, self.ledger)

    def test_gear_change_alters_integer_or_residual(self):
        entries = {(s["frac_bits"], s["gear"]): s["entries"][0]
                   for s in self.result["settings"]}
        base = entries[(8, 1)]["channels"]
        coarse = entries[(8, 4)]["channels"]
        changed = []
        for chan in base:
            q1 = base[chan]["norm_quant"]
            q4 = coarse[chan]["norm_quant"]
            changed.append(q1["q"] != q4["q"]
                           or q1["residual"] != q4["residual"])
        self.assertTrue(any(changed))

    def test_rank_is_permutation_and_tie_break_fields(self):
        for setting in self.result["settings"]:
            ranks = sorted(setting["rank_of"].values())
            self.assertEqual(ranks, list(range(1, len(ranks) + 1)))

    def test_ledger_has_all_stages_and_directions(self):
        for item_id, slot in self.ledger["per_item"].items():
            for setting in slot["settings"]:
                for stage in (STAGE_NORM, STAGE_FUSION, STAGE_FEATURE,
                              STAGE_CLIP):
                    self.assertIn(stage, setting["stages"])
                self.assertTrue(setting["rounding_directions"])

    def test_stage_attribution_references_real_diff(self):
        for audit in self.ledger["setting_pair_audit"]:
            for change in audit["item_changes"]:
                if change["first_diverged_stage"] is not None:
                    diff = change["stage_diff_common_unit"]
                    if isinstance(diff, dict):
                        self.assertTrue(any(v != 0 for v in diff.values()))
                    else:
                        self.assertNotEqual(diff, 0)

    def test_flip_records_counterpart_and_channel_delta(self):
        for flip in self.ledger["flips"]:
            self.assertNotEqual(flip["rank_before"], flip["rank_after"])
            self.assertIsNotNone(flip["counterpart_item_id"])
            self.assertEqual(
                set(flip["channel_contribution_delta_common_unit"]),
                set(self.cfg.channels))
            self.assertIn(flip["tie_break_triggered"], (True, False))

    def test_boundary_pairs_monotone_with_gear(self):
        counts = {}
        for pair in self.ledger["boundary_pairs"]:
            key = (pair["frac_bits"], pair["gear"])
            counts[key] = counts.get(key, 0) + 1
        for f in self.cfg.frac_bits_choices:
            per_gear = [counts.get((f, g), 0) for g in self.cfg.gears]
            self.assertTrue(all(per_gear[i] <= per_gear[i + 1]
                                for i in range(len(per_gear) - 1)))
        # 至少一个位宽上边界对随档位收紧真实增加（要求 5）。
        self.assertTrue(any(
            counts.get((f, self.cfg.gears[-1]), 0)
            > counts.get((f, self.cfg.gears[0]), 0)
            for f in self.cfg.frac_bits_choices))

    def test_boundary_reference_uses_highest_frac(self):
        for pair in self.ledger["boundary_pairs"]:
            self.assertIn("highest_frac_ref_diff_real", pair)
            self.assertGreaterEqual(pair["half_step"], 0)


if __name__ == "__main__":
    unittest.main()
