import unittest

from relx.entities import build_lexical_type_index, extract_entities
from relx.relation import (
    MISS_BELOW_THRESHOLD,
    MISS_NO_COMPATIBLE_TEMPLATE,
    MISS_NO_ENTITY,
    MISS_NO_TRIGGER,
    extract_relations,
)
from relx.pipeline import build_runtime


class RelationScoringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state, cls.failures = build_runtime("data")
        cls.lex_index = build_lexical_type_index(cls.state.model)

    def _extract(self, sid):
        record = self.state.sentences[sid]
        entities, _supp, _fp = extract_entities(
            record, self.state.config, self.lex_index
        )
        return record, entities, extract_relations(
            entities, record.tokens, self.state.model, self.state.config
        )

    def test_work_for_relation_kept_with_fixed_point_score(self):
        record, entities, (hits, relations, misses, refs, trigger_fp) = self._extract("S01")
        self.assertTrue(relations)
        rel = relations[0]
        self.assertEqual(rel.relation_type, "work_for")
        self.assertEqual(rel.template_id, "T01")
        self.assertIsInstance(rel.score, int)
        self.assertEqual(rel.threshold_q, 15720)
        self.assertTrue(all(hit.template_id == "T01" for hit in rel.trigger_hits))

    def test_boundary_equal_threshold_kept_and_flagged(self):
        record, entities, (_h, relations, _m, _r, _fp) = self._extract("S13")
        boundary = [rel for rel in relations if rel.boundary]
        self.assertTrue(boundary)
        for rel in boundary:
            self.assertEqual(rel.score, rel.threshold_q)

    def test_no_entity_sentence_kept_with_miss_entry(self):
        record, entities, (hits, relations, misses, refs, _fp) = self._extract("S23")
        self.assertEqual(entities, [])
        self.assertEqual(relations, ())
        self.assertEqual(len(misses), 1)
        self.assertEqual(misses[0].reason_code, MISS_NO_ENTITY)

    def test_miss_reasons_cover_three_categories(self):
        codes = set()
        for sid in self.state.sentence_ids_sorted():
            _r, _e, (_h, _rel, misses, _refs, _fp) = self._extract(sid)
            codes.update(miss.reason_code for miss in misses)
        self.assertIn(MISS_NO_TRIGGER, codes)
        self.assertIn(MISS_NO_COMPATIBLE_TEMPLATE, codes)

    def test_miss_order_is_stable(self):
        _r, _e, (_h, _rel, misses, _refs, _fp) = self._extract("S20")
        keys = [(item.head_start, item.tail_start, item.head_id,
                 item.tail_id, item.template_id) for item in misses]
        self.assertEqual(keys, sorted(keys))

    def test_quant_switch_changes_integer_and_keeps_domain_int(self):
        from dataclasses import replace
        record, entities, _ = self._extract("S01")
        scores = set()
        for width, quant in ((8, 2), (16, 4), (32, 8)):
            cfg = replace(self.state.config,
                          fixedpoint_width=width, quant_level=quant)
            _h, relations, _m, _r, _fp = extract_relations(
                entities, record.tokens, self.state.model, cfg
            )
            self.assertIsInstance(relations[0].score, int)
            scores.add(relations[0].score)
        self.assertEqual(len(scores), 3)


if __name__ == "__main__":
    unittest.main()
