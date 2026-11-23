from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from ner_editor.constants import E_COMMAND_INVALID, E_CORPUS_INVALID, E_MODEL_PARAM_INVALID
from ner_editor.loader import load_commands, load_corpus, load_model


class LoaderValidationTests(unittest.TestCase):
    def test_corpus_rejects_each_structural_error_and_keeps_valid_rows(self):
        rows = [
            {"id": "ok", "chars": "张 三", "labels": "B E", "types": "PER PER"},
            {"id": "ok", "chars": "李 四", "labels": "B E", "types": "PER PER"},
            {"id": "len", "chars": "张 三", "labels": "B", "types": "PER"},
            {"id": "label", "chars": "张", "labels": "X", "types": "PER"},
            {"id": "span", "chars": "张 三", "labels": "B O", "types": "PER PER"},
            {"id": "type", "chars": "张 三", "labels": "B E", "types": "PER LOC"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corpus.jsonl"
            path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
            samples, failures = load_corpus(path)
        self.assertEqual(list(samples), ["ok"])
        self.assertGreaterEqual(len(failures), 5)
        self.assertTrue(all(failure.code == E_CORPUS_INVALID for failure in failures))
        self.assertTrue(any("duplicate" in failure.message for failure in failures))
        self.assertTrue(any("equal length" in failure.message for failure in failures))
        self.assertTrue(any("BIESO" in failure.message for failure in failures))

    def test_model_rejects_dimensions_and_transition_shape(self):
        base = json.loads((Path(__file__).resolve().parents[1] / "samples" / "model.json").read_text(encoding="utf-8"))
        cases = []
        bad_dim = copy.deepcopy(base)
        bad_dim["feature_dim"] = 0
        cases.append(bad_dim)
        bad_embedding = copy.deepcopy(base)
        bad_embedding["char_embeddings"]["张"] = [0.0]
        cases.append(bad_embedding)
        bad_transition = copy.deepcopy(base)
        bad_transition["transition_weights"] = bad_transition["transition_weights"][:-1]
        cases.append(bad_transition)
        for index, value in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "model.json"
                path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
                model, failures = load_model(path)
                self.assertIsNone(model)
                self.assertTrue(failures)
                self.assertTrue(all(f.code == E_MODEL_PARAM_INVALID for f in failures))

    def test_command_validation_requires_enum_targets_and_weight_fields(self):
        commands = [
            {"type": "unknown"},
            {"type": "emission_adjust", "target": {"dimension": 0, "label": "B"}},
            {"type": "transition_adjust", "target": {"from": "O", "to": "B"}},
            {"type": "corpus_delete"},
            {"type": "fixed_point_set", "fixed_decimals": 99},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "commands.json"
            path.write_text(json.dumps(commands), encoding="utf-8")
            parsed, failures = load_commands(path)
        self.assertEqual(parsed, [])
        self.assertEqual(len(failures), len(commands))
        self.assertTrue(all(f.code == E_COMMAND_INVALID for f in failures))


if __name__ == "__main__":
    unittest.main()
