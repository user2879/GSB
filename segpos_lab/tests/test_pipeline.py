import math
import unittest

from segpos.models import Config, Lexicon, LexiconEntry, ProbTable, Sentence
from segpos.oov import process_oov
from segpos.segmenter import FmmSegmenter
from segpos.tagger import _fixed_quantize, viterbi


def make_config(**overrides):
    base = dict(
        concurrency=4,
        first_response_budget_ms=50.0,
        snapshot_commit_interval_commands=3,
        low_confidence_threshold=0.5,
        fixed_point_bits=16,
        quantization_levels=1024,
        smoothing=1e-4,
        oov_rule="equal",
        oov_max_length=3,
    )
    base.update(overrides)
    return Config(**base)


def make_table(tags=("a", "b")):
    start = {tag: math.log(0.5) for tag in tags}
    transition = {t: {u: math.log(0.5) for u in tags} for t in tags}
    emission = {"天": {"a": math.log(0.7), "b": math.log(0.3)}}
    return ProbTable(list(tags), start, transition, emission, smoothing=1e-4)


class FmmTests(unittest.TestCase):
    def test_longest_match_and_positions(self):
        lexicon = Lexicon(
            {
                "天地": LexiconEntry("天地", 10, "n"),
                "天地人": LexiconEntry("天地人", 20, "n"),
            }
        )
        segmenter = FmmSegmenter(lexicon)
        tokens, meta = segmenter.segment(Sentence("S1", "天地人和", 1))
        self.assertEqual([t.text for t in tokens], ["天地人", "和"])
        self.assertEqual([(t.start, t.end) for t in tokens], [(0, 3), (3, 4)])
        self.assertEqual(tokens[0].source, "lexicon")
        self.assertEqual(tokens[1].source, "single")
        self.assertEqual(meta["hit_entries"], ["天地人"])

    def test_same_length_tie_freq_then_lexicographic(self):
        lexicon = Lexicon(
            {
                "天平": LexiconEntry("天平", 5, "n"),
                "天气": LexiconEntry("天气", 9, "n"),
            }
        )
        tokens, _ = FmmSegmenter(lexicon).segment(Sentence("S1", "天气", 1))
        self.assertEqual(tokens[0].text, "天气")
        lexicon = Lexicon(
            {
                "天平": LexiconEntry("天平", 9, "n"),
                "天气": LexiconEntry("天气", 9, "n"),
            }
        )
        tokens, _ = FmmSegmenter(lexicon).segment(Sentence("S1", "天气", 1))
        self.assertEqual(tokens[0].text, "天气")

    def test_deterministic_fingerprint_across_runs(self):
        lexicon = Lexicon({"天地": LexiconEntry("天地", 10, "n")})
        segmenter = FmmSegmenter(lexicon)
        first = segmenter.segment(Sentence("S1", "天地间", 1))
        second = segmenter.segment(Sentence("S1", "天地间", 1))
        self.assertEqual(first[1]["content_fingerprint"], second[1]["content_fingerprint"])
        self.assertEqual([t.to_dict() for t in first[0]], [t.to_dict() for t in second[0]])


class OovTests(unittest.TestCase):
    def _seg(self, text):
        lexicon = Lexicon({"已知": LexiconEntry("已知", 5, "n")})
        return FmmSegmenter(lexicon).segment(Sentence("S1", text, 1))

    def test_rules_change_boundaries(self):
        tokens, meta = self._seg("已知春夏秋冬风雨雷电")
        equal_tokens, equal_meta = process_oov(tokens, meta, make_config(oov_rule="equal"))
        fixed_tokens, fixed_meta = process_oov(tokens, meta, make_config(oov_rule="fixed_max"))
        equal_bounds = [(t.start, t.end) for t in equal_tokens if t.source == "oov"]
        fixed_bounds = [(t.start, t.end) for t in fixed_tokens if t.source == "oov"]
        self.assertNotEqual(equal_bounds, fixed_bounds)
        self.assertNotEqual(equal_meta["content_fingerprint"], fixed_meta["content_fingerprint"])
        self.assertTrue(all(t.segment_basis for t in equal_tokens if t.source == "oov"))

    def test_hard_cap_segmentation(self):
        tokens, meta = self._seg("金木水火土")
        out, out_meta = process_oov(tokens, meta, make_config(oov_rule="fixed_max", oov_max_length=3))
        oov = [t for t in out if t.source == "oov"]
        self.assertTrue(all(len(t.text) <= 3 for t in oov))
        self.assertEqual("".join(t.text for t in oov), "金木水火土")
        self.assertTrue(any("hard_cap" in t.segment_basis for t in oov))
        self.assertEqual(out_meta["rule"], "fixed_max")


class ViterbiTests(unittest.TestCase):
    def test_deterministic_and_smoothed_missing_emission(self):
        config = make_config()
        table = make_table()
        tokens, seg_meta = FmmSegmenter(Lexicon({})).segment(Sentence("S1", "天地", 1))
        tokens, oov_meta = process_oov(tokens, seg_meta, config)
        tagged_a, meta_a = viterbi([t for t in tokens], table, config, oov_meta)
        tagged_b, meta_b = viterbi([t for t in tokens], table, config, oov_meta)
        self.assertEqual(meta_a["tags"], meta_b["tags"])
        self.assertEqual(meta_a["normalized_log_prob"], meta_b["normalized_log_prob"])
        self.assertTrue(meta_a["smoothed_positions"])
        self.assertTrue(all(t.tag in table.tags for t in tagged_a))

    def test_tie_breaks_by_tag_lexicographic(self):
        config = make_config()
        table = make_table()
        tokens, seg_meta = FmmSegmenter(Lexicon({})).segment(Sentence("S1", "天", 1))
        tokens, oov_meta = process_oov(tokens, seg_meta, config)
        tagged, meta = viterbi(tokens, table, config, oov_meta)
        self.assertEqual(tagged[0].tag, "a")

    def test_fixed_quantization_monotone(self):
        values = [0.0, 0.25, 0.5, 0.75, 1.0]
        quantized = [_fixed_quantize(v, 16, 1024) for v in values]
        self.assertEqual(quantized[0], 0)
        self.assertEqual(quantized[-1], (1 << 16) - 1)
        self.assertEqual(quantized, sorted(quantized))


if __name__ == "__main__":
    unittest.main()
