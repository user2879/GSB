import unittest
from pathlib import Path

from rexapp.config import load_config
from rexapp.documents import Document, load_documents
from rexapp.extraction import generate_relation_candidates, recognize_entities
from rexapp.pipeline import run_full_pipeline

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


class ExtractionTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(SAMPLES / "config.json")
        self.loaded = load_documents(SAMPLES / "documents.jsonl", set(self.config.entity_vocab))

    def test_ner_longest_match_and_order(self):
        doc = next(doc for doc in self.loaded.documents if doc.doc_id == "doc-004")
        entities = recognize_entities(doc.words, self.config)
        self.assertTrue(all(entities[i].start <= entities[i + 1].start for i in range(len(entities) - 1)))
        self.assertTrue(all(entity.entity_type in self.config.entity_vocab for entity in entities))

    def test_relation_window_excludes_pairs(self):
        long_window = self.config.with_updates(relation_window=10)
        doc = self.loaded.documents[0]
        entities = recognize_entities(doc.words, long_window)
        wide_relations, wide_excluded = generate_relation_candidates(doc.doc_id, doc.words, entities, long_window)
        narrow = self.config.with_updates(relation_window=1)
        narrow_entities = recognize_entities(doc.words, narrow)
        narrow_relations, narrow_excluded = generate_relation_candidates(doc.doc_id, doc.words, narrow_entities, narrow)
        self.assertGreaterEqual(len(wide_relations), len(narrow_relations))
        self.assertTrue(all(item.reason == "outside_window" for item in narrow_excluded))

    def test_repeated_runs_are_field_identical(self):
        doc = self.loaded.documents[0]
        first = run_full_pipeline(doc, self.config).as_dict()
        second = run_full_pipeline(doc, self.config).as_dict()
        self.assertEqual(first, second)

    def test_event_missing_required_role_records_code(self):
        config = self.config
        words = ("Acme", "invested", "in", "Shanghai")
        positions = tuple((i * 8, i * 8 + len(w)) for i, w in enumerate(words))
        doc = Document(doc_id="d-missing", words=words, positions=positions, text=" ".join(words))
        result = run_full_pipeline(doc, config)
        event = result.events[0]
        self.assertIn("target", event.missing_roles)

    def test_boundary_samples_present_in_generated_data(self):
        boundaries = []
        for doc in self.loaded.documents:
            boundaries.extend(run_full_pipeline(doc, self.config).boundaries)
        self.assertTrue(boundaries)
        self.assertTrue(all(item.drift is not None for item in boundaries))


if __name__ == "__main__":
    unittest.main()
