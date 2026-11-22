import shutil
import tempfile
import unittest
from pathlib import Path

from tests.context import ROOT
from segpos.errors import (
    E_LEXICON_INVALID,
    E_PROB_TABLE_INVALID,
    E_SENTENCE_INVALID,
)
from segpos.loader import load_charset, load_lexicon, load_prob_table, load_sentences
from segpos.runner import build_engine, prepare_bundle

BASE = ROOT / "samples" / "base"
INVALID = ROOT / "samples" / "invalid"


class InvalidScenarioTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_invalid_entries_collected_but_valid_ones_processed(self):
        charset = load_charset(INVALID / "charset.txt")
        sentences, sentence_failures = load_sentences(INVALID / "sentences.txt", charset)
        table, table_failures = load_prob_table(INVALID / "prob_table.json")
        lexicon, lexicon_failures = load_lexicon(INVALID / "lexicon.txt", set(table.tags))
        self.assertTrue(any(f.code == E_SENTENCE_INVALID for f in sentence_failures))
        self.assertTrue(any(f.code == E_PROB_TABLE_INVALID for f in table_failures))
        self.assertTrue(any(f.code == E_LEXICON_INVALID for f in lexicon_failures))
        self.assertTrue(sentences)
        self.assertTrue(lexicon.entries)

    def test_engine_runs_with_partial_failures(self):
        bundle, _ = prepare_bundle(BASE)
        engine = build_engine(bundle, Path(self.tmp) / "snap")
        response = engine.run_due(set(engine.states))
        self.assertEqual(response["code"], "OK")
        report = engine.build_report(0)
        self.assertEqual(report["success_count"], report["total_sentences"])
        sample_sid = sorted(report["sentences"])[0]
        tag_meta = report["sentences"][sample_sid]["tag_meta"]
        self.assertIn("smoothed_positions", tag_meta)
        self.assertIn("normalized_log_prob", tag_meta)
        oov_meta = report["sentences"][sample_sid]["oov_meta"]
        self.assertIn(oov_meta["rule"], ("equal", "fixed_max"))
        self.assertIn("content_fingerprint", oov_meta)
        for token in report["sentences"][sample_sid]["tag"]:
            self.assertIn(token["source"], ("lexicon", "oov"))
            self.assertIsNotNone(token["tag"])
            self.assertIsNotNone(token["confidence_fixed"])

    def test_failure_order_is_stable(self):
        charset = load_charset(INVALID / "charset.txt")
        _, failures = load_sentences(INVALID / "sentences.txt", charset)
        self.assertEqual(
            [(f.source_line, f.code, f.key) for f in failures],
            sorted((f.source_line, f.code, f.key) for f in failures),
        )


if __name__ == "__main__":
    unittest.main()
