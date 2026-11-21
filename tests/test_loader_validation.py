import json
import os
import tempfile
import unittest

from workbench.loader import load_samples, load_vocabulary
from workbench.models import ConfigError, HierarchyConfig
from workbench.errors import (
    E_DOMAIN_UNKNOWN,
    E_INTENT_OUT_OF_DOMAIN,
    E_SAMPLE_INVALID,
    E_SLOT_SCHEMA_INVALID,
    E_THRESHOLD_NOT_FINITE,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB_PATH = os.path.join(ROOT, "data/vocab/vocabulary.json")
CONFIG_PATH = os.path.join(ROOT, "data/config/hierarchy_config.json")


class LoaderValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocab = load_vocabulary(VOCAB_PATH)
        cls.config = HierarchyConfig.from_dict(
            json.load(open(CONFIG_PATH, encoding="utf-8")), cls.vocab)

    def _write(self, rows):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                          encoding="utf-8")
        for row in rows:
            tmp.write(json.dumps(row) + "\n")
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)
        return tmp.name

    def test_illegal_token_sample_invalid(self):
        path = self._write([{
            "utterance_id": "X1", "tokens": ["price", "phone", "HACK"],
            "domain": "presale", "intent": "price_inquiry", "gold_slots": []}])
        _, failures = load_samples(path, self.vocab, self.config)
        self.assertEqual([f["code"] for f in failures], [E_SAMPLE_INVALID])

    def test_unknown_domain(self):
        path = self._write([{
            "utterance_id": "X2", "tokens": ["repair"],
            "domain": "mars", "intent": "repair_request", "gold_slots": []}])
        _, failures = load_samples(path, self.vocab, self.config)
        self.assertEqual([f["code"] for f in failures], [E_DOMAIN_UNKNOWN])

    def test_intent_out_of_domain(self):
        path = self._write([{
            "utterance_id": "X3", "tokens": ["repair"],
            "domain": "aftersale", "intent": "price_inquiry", "gold_slots": []}])
        _, failures = load_samples(path, self.vocab, self.config)
        self.assertEqual([f["code"] for f in failures], [E_INTENT_OUT_OF_DOMAIN])

    def test_duplicate_id_and_bad_gold_slot(self):
        path = self._write([
            {"utterance_id": "DUP", "tokens": ["price", "phone"],
             "domain": "presale", "intent": "price_inquiry", "gold_slots": []},
            {"utterance_id": "DUP", "tokens": ["price", "phone"],
             "domain": "presale", "intent": "price_inquiry", "gold_slots": []},
            {"utterance_id": "X4", "tokens": ["price", "phone"],
             "domain": "presale", "intent": "price_inquiry",
             "gold_slots": [["ghost", ["phone"], 0]]},
        ])
        _, failures = load_samples(path, self.vocab, self.config)
        codes = [f["code"] for f in failures]
        self.assertIn(E_SAMPLE_INVALID, codes)
        self.assertIn(E_SLOT_SCHEMA_INVALID, codes)

    def test_non_finite_threshold_rejected(self):
        raw = json.load(open(CONFIG_PATH, encoding="utf-8"))
        raw["thresholds"]["intent"] = "NaN"
        with self.assertRaises(ConfigError) as ctx:
            HierarchyConfig.from_dict(raw, self.vocab)
        codes = {e["code"] for e in ctx.exception.errors}
        self.assertIn(E_THRESHOLD_NOT_FINITE, codes)

    def test_undeclared_slot_in_pattern_rejected(self):
        raw = json.load(open(CONFIG_PATH, encoding="utf-8"))
        raw["intent_required_slots"]["price_inquiry"] = ["ghost_slot"]
        with self.assertRaises(ConfigError) as ctx:
            HierarchyConfig.from_dict(raw, self.vocab)
        self.assertIn(E_SLOT_SCHEMA_INVALID,
                      {e["code"] for e in ctx.exception.errors})


if __name__ == "__main__":
    unittest.main()
