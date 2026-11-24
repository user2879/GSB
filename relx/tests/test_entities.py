import unittest

from relx.entities import (
    SuppressedEntry,
    build_lexical_type_index,
    extract_entities,
)
from relx.pipeline import build_runtime


class EntityResolutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state, cls.failures = build_runtime("data")
        cls.lex_index = build_lexical_type_index(cls.state.model)

    def _record(self, sid):
        return self.state.sentences[sid]

    def test_gold_and_lexical_merge_same_span(self):
        entities, suppressed, _ = extract_entities(
            self._record("S01"), self.state.config, self.lex_index
        )
        ids = {item.candidate_id: item for item in entities}
        self.assertIn("E1", ids)
        self.assertEqual(ids["E1"].source, "gold")
        self.assertEqual(ids["E1"].entity_type, "PER")

    def test_longest_span_wins_over_nested(self):
        entities, suppressed, _ = extract_entities(
            self._record("S17"), self.state.config, self.lex_index
        )
        spans = {(item.start, item.end): item for item in entities}
        self.assertIn((0, 4), spans)
        self.assertEqual(spans[(0, 4)].candidate_id, "E1")
        self.assertTrue(any(entry.swallowed_by for entry in suppressed))

    def test_suppressed_entries_are_ordered_and_auditable(self):
        entities, suppressed, fingerprint_a = extract_entities(
            self._record("S17"), self.state.config, self.lex_index
        )
        _entities2, suppressed2, fingerprint_b = extract_entities(
            self._record("S17"), self.state.config, self.lex_index
        )
        order_key = lambda item: (item.start, item.end, item.entity_type,
                                  item.source, item.swallowed_by)
        self.assertEqual([entry.to_dict() for entry in suppressed],
                         sorted((entry.to_dict() for entry in suppressed),
                                key=lambda row: (row["start"], row["end"],
                                                 row["type"], row["source"],
                                                 row["swallowed_by"])))
        self.assertEqual(fingerprint_a, fingerprint_b)
        for entry in suppressed:
            self.assertTrue(entry.swallowed_by)

    def test_deterministic_candidate_ids(self):
        first, _supp, fp1 = extract_entities(
            self._record("S20"), self.state.config, self.lex_index
        )
        second, _s2, fp2 = extract_entities(
            self._record("S20"), self.state.config, self.lex_index
        )
        self.assertEqual([item.to_dict() for item in first],
                         [item.to_dict() for item in second])
        self.assertEqual(fp1, fp2)


if __name__ == "__main__":
    unittest.main()
