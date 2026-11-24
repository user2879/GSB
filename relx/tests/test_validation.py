import json
import os
import tempfile
import unittest

from relx.config import load_config
from relx.corpus import load_corpus
from relx.errors import (
    E_ENTITY_REF_INVALID,
    E_ENTITY_STRUCTURE_INVALID,
    E_JSONL_INVALID,
    E_SENTENCE_DUPLICATE,
    E_TOKEN_OOV,
)
from relx.model import load_model
from relx.pipeline import build_runtime


class ValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state, cls.failures = build_runtime("data")
        cls.code_set = {item.code for item in cls.failures}

    def test_invalid_lines_enter_failure_list_without_stopping(self):
        for code in (E_JSONL_INVALID, E_SENTENCE_DUPLICATE, E_TOKEN_OOV,
                     E_ENTITY_STRUCTURE_INVALID, E_ENTITY_REF_INVALID):
            self.assertIn(code, self.code_set)
        self.assertEqual(len(self.state.sentences), 24)

    def test_relation_types_match_templates_one_to_one(self):
        self.assertEqual(
            set(self.state.config.relation_types),
            {tpl.relation_type for tpl in self.state.model.templates},
        )

    def test_invalid_config_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("max_term_length: 0\n")
            with self.assertRaises(Exception):
                load_config(path)

    def test_model_bucket_bounds_validated(self):
        config = self.state.config
        with tempfile.TemporaryDirectory() as tmp:
            pkg_path = os.path.join(tmp, "model.json")
            pkg = {
                "model_version": "x",
                "feature_dim": 0,
                "hash_buckets": 4,
                "term_hash": {"张伟": 0},
                "templates": [],
            }
            with open(pkg_path, "w", encoding="utf-8") as handle:
                json.dump(pkg, handle, ensure_ascii=False)
            with self.assertRaises(Exception):
                load_model(pkg_path, config)

    def test_unique_sentence_ids_among_loaded(self):
        ids = list(self.state.sentences.keys())
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
