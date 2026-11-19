import unittest
from pathlib import Path
import tempfile

from segpos import ComponentConfig, Dictionary, ProbabilityTable
from segpos.config import ComponentConfig as Cfg
from segpos.errors import ConfigError
from segpos.segmenter import maximum_match
from segpos.tagger import analyze

from tests._fixtures import TAGS, build_table_data, write_workspace


class SegmenterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.paths = write_workspace(Path(self.tmp.name))
        self.dictionary = Dictionary.load(
            self.paths["dictionary"], allowed_pos=set(TAGS)
        )
        self.table = ProbabilityTable(build_table_data(), version=1)

    def tearDown(self):
        self.tmp.cleanup()

    def test_max_match_prefers_longest_then_freq_then_lexical(self):
        segments = maximum_match("研究生学习", self.dictionary)
        self.assertEqual([s.text for s in segments], ["研究生", "学习"])
        self.assertTrue(all(s.hit for s in segments))

    def test_oov_single_chars_are_marked(self):
        segments = maximum_match("喵星人", self.dictionary)
        self.assertEqual([s.text for s in segments], ["喵", "星", "人"])
        self.assertTrue(all(not s.hit for s in segments))
        self.assertTrue(all(s.match_basis == "oov_single_char" for s in segments))

    def test_tagger_pos_and_logprob_deterministic(self):
        first = analyze("研究生物", self.dictionary, self.table, -20.0)
        second = analyze("研究生物", self.dictionary, self.table, -20.0)
        self.assertEqual(
            [t.pos for t in first.tokens], [t.pos for t in second.tokens]
        )
        self.assertEqual(
            [t.local_path_score for t in first.tokens],
            [t.local_path_score for t in second.tokens],
        )
        self.assertEqual(first.sentence_logprob, second.sentence_logprob)
        self.assertTrue(all(t.pos in TAGS for t in first.tokens))

    def test_smoothing_is_counted_when_emission_missing(self):
        result = analyze("喵星人", self.dictionary, self.table, -20.0)
        self.assertGreater(result.smoothing_hits, 0)


class ConfigTests(unittest.TestCase):
    def test_workers_must_be_positive_int(self):
        with self.assertRaises(ConfigError):
            Cfg.from_dict({"workers": 0})
        with self.assertRaises(ConfigError):
            Cfg.from_dict({"workers": 1.5})
        with self.assertRaises(ConfigError):
            Cfg.from_dict({"workers": True})

    def test_smoothing_must_be_negative_log(self):
        with self.assertRaises(ConfigError):
            Cfg.from_dict({"smoothing": 0.0})

    def test_dictionary_rejects_bad_pos(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.txt"
            path.write_text("词 10 x\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                Dictionary.load(path, allowed_pos=set(TAGS))


if __name__ == "__main__":
    unittest.main()
