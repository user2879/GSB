import unittest
from pathlib import Path

from tests.context import ROOT
from segpos.errors import (
    E_CONFIG_INVALID,
    E_LEXICON_INVALID,
    E_PROB_TABLE_INVALID,
    E_SENTENCE_INVALID,
)
from segpos.loader import (
    load_charset,
    load_commands,
    load_config,
    load_lexicon,
    load_prob_table,
    load_sentences,
)

BASE = ROOT / "samples" / "base"
INVALID = ROOT / "samples" / "invalid"


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.charset = load_charset(BASE / "charset.txt")
        self.table, self.table_failures = load_prob_table(BASE / "prob_table.json")

    def test_sentences_unique_id_and_charset(self):
        sentences, failures = load_sentences(BASE / "sentences.txt", self.charset)
        self.assertTrue(sentences)
        self.assertEqual(len({s.sid for s in sentences}), len(sentences))
        self.assertEqual(failures, [])

    def test_invalid_sentences(self):
        _, failures = load_sentences(INVALID / "sentences.txt", self.charset)
        codes = {f.code for f in failures}
        self.assertEqual(codes, {E_SENTENCE_INVALID})
        reasons = " ".join(f.reason for f in failures)
        self.assertIn("duplicate", reasons)
        self.assertIn("outside controlled", reasons)
        self.assertIn("empty", reasons)
        self.assertEqual([f.source_line for f in failures], sorted(f.source_line for f in failures))

    def test_lexicon_validation_and_duplicate_accumulation(self):
        lexicon, failures = load_lexicon(BASE / "lexicon.txt", set(self.table.tags))
        self.assertTrue(any(len(e.sources) >= 2 for e in lexicon.entries.values()))
        bad_lexicon, bad_failures = load_lexicon(INVALID / "lexicon.txt", set(self.table.tags))
        self.assertEqual({f.code for f in bad_failures}, {E_LEXICON_INVALID})
        self.assertTrue(bad_lexicon.entries)

    def test_prob_table_invalid_entry(self):
        table, failures = load_prob_table(INVALID / "prob_table.json")
        self.assertIsNotNone(table)
        self.assertTrue(any(f.code == E_PROB_TABLE_INVALID for f in failures))
        self.assertIn("start.n", [f.key for f in failures])

    def test_config_validation(self):
        config, failures = load_config(BASE / "config.json")
        self.assertEqual(failures, [])
        self.assertGreater(config.concurrency, 0)
        self.assertGreater(config.smoothing, 0.0)

    def test_config_rejects_bad_values(self):
        import json
        import tempfile

        good = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
        cases = [
            {"concurrency": 0},
            {"concurrency": 99},
            {"low_confidence_threshold": 1.5},
            {"fixed_point_bits": 7},
            {"quantization_levels": 12},
            {"smoothing": -1.0},
            {"first_response_budget_ms": -2},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for override in cases:
                data = {**good, **override}
                path = Path(tmp) / "config.json"
                path.write_text(json.dumps(data), encoding="utf-8")
                config, failures = load_config(path)
                self.assertIsNone(config, override)
                self.assertTrue(all(f.code == E_CONFIG_INVALID for f in failures), override)

    def test_command_stream_keeps_valid_and_flags_bad(self):
        commands, failures = load_commands(BASE / "commands.json")
        self.assertTrue(commands)
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
