import json
import tempfile
import unittest
from pathlib import Path

from rexapp.config import ConfigError, load_config, parse_config
from rexapp.documents import load_documents
from rexapp.errors import ErrorCode

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


class LoadingTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(SAMPLES / "config.json")

    def test_sample_failures_have_expected_codes(self):
        result = load_documents(SAMPLES / "documents.jsonl", set(self.config.entity_vocab))
        codes = {failure.doc_id: failure.code for failure in result.failures}
        self.assertEqual(codes["doc-empty"], ErrorCode.E_DOC_INVALID)
        self.assertEqual(codes["line-15"], ErrorCode.E_JSON_PARSE)
        self.assertEqual(codes["doc-bad-pos"], ErrorCode.E_POSITION_OUT_OF_RANGE)
        self.assertEqual(codes["doc-unknown-type"], ErrorCode.E_ENTITY_TYPE_UNKNOWN)
        self.assertEqual(len(result.documents), 13)

    def test_duplicate_doc_id_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.jsonl"
            path.write_text(
                json.dumps({"doc_id": "x", "words": ["Acme"], "positions": [[0, 4]], "text": "Acme"})
                + "\n"
                + json.dumps({"doc_id": "x", "words": ["Bob"], "positions": [[0, 3]], "text": "Bob"})
                + "\n",
                encoding="utf-8",
            )
            result = load_documents(path, set(self.config.entity_vocab))
            self.assertEqual(len(result.documents), 1)
            self.assertEqual(result.failures[0].code, ErrorCode.E_DOC_INVALID)

    def test_non_monotonic_event_thresholds_rejected(self):
        raw = json.loads((SAMPLES / "config.json").read_text(encoding="utf-8"))
        raw["event_band_thresholds"] = [0.7, 0.5]
        with self.assertRaises(ConfigError) as ctx:
            parse_config(raw)
        self.assertEqual(ctx.exception.code, ErrorCode.E_DOC_INVALID)

    def test_unknown_entity_type_in_trigger_rejected(self):
        raw = json.loads((SAMPLES / "config.json").read_text(encoding="utf-8"))
        raw["relation_triggers"][0]["head_type"] = "WEIRD"
        with self.assertRaises(ConfigError) as ctx:
            parse_config(raw)
        self.assertEqual(ctx.exception.code, ErrorCode.E_ENTITY_TYPE_UNKNOWN)


if __name__ == "__main__":
    unittest.main()
