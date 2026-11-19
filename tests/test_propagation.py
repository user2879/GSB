import random
import unittest

from cache_invalidation.engine import Engine
from cache_invalidation.errors import EClock
from cache_invalidation.model import EntryID
from cache_invalidation.store import Store
from tests.support import sample_graph


class PropagationTest(unittest.TestCase):
    def setUp(self):
        self.graph = sample_graph()

    def test_explicit_invalidation_propagates_transitive_closure(self):
        engine = Engine(self.graph, store=Store())
        result = engine.explicit_invalidate(EntryID("catalog", 100), 100)
        self.assertEqual(
            list(result.affected),
            ["catalog/100", "inventory/300", "pricing/200",
             "storefront/400", "search/500"])
        for entry_id in result.affected:
            self.assertEqual(engine.versions[EntryID.parse(entry_id)], 1)

    def test_refresh_expands_inheritance_chain(self):
        engine = Engine(self.graph, store=Store())
        result = engine.refresh_namespace("catalog", 100)
        self.assertEqual(
            list(result.affected),
            ["catalog/100", "catalog/101", "inventory/300", "pricing/200",
             "storefront/400", "search/500"])
        self.assertEqual(
            result.round_record.refresh_triggers,
            tuple(sorted(result.round_record.refresh_triggers)))

    def test_expired_reap_uses_injected_clock(self):
        engine = Engine(self.graph, store=Store())
        result = engine.reap_expired(15000)
        self.assertEqual(
            list(result.affected),
            ["catalog/100", "inventory/300", "pricing/200",
             "storefront/400", "search/500"])
        self.assertEqual(
            list(result.round_record.expired_triggers),
            ["catalog/100", "inventory/300", "pricing/200"])

    def test_clock_rollback_is_rejected_without_state_change(self):
        engine = Engine(self.graph, store=Store())
        engine.explicit_invalidate(EntryID("catalog", 100), 100)
        before = dict(engine.versions)
        with self.assertRaises(EClock) as caught:
            engine.reap_expired(99)
        self.assertEqual(caught.exception.code, "E_CLOCK")
        self.assertEqual(engine.versions, before)

    def test_repeated_runs_produce_byte_identical_results(self):
        outputs = []
        for _ in range(5):
            engine = Engine(sample_graph(), store=Store())
            engine.explicit_invalidate(EntryID("catalog", 100), 100)
            engine.refresh_namespace("catalog", 200)
            outputs.append([
                record.to_dict() for record in engine.records
            ])
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], outputs[-1])

    def test_traversal_shuffle_does_not_change_result(self):
        baseline = Engine(self.graph, store=Store())
        baseline.explicit_invalidate(EntryID("catalog", 100), 100)
        expected = list(baseline.records[0].affected)

        shuffled_graph = sample_graph()
        for targets in shuffled_graph.annotation_edges.values():
            randomized = list(targets)
            random.Random(7).shuffle(randomized)
            targets.clear()
            targets.update(randomized)
        shuffled_engine = Engine(shuffled_graph, store=Store())
        result = shuffled_engine.explicit_invalidate(
            EntryID("catalog", 100), 100)
        self.assertEqual(list(result.affected), expected)

    def test_dead_entry_is_not_invalidated_twice(self):
        engine = Engine(self.graph, store=Store())
        first = engine.explicit_invalidate(EntryID("catalog", 100), 100)
        second = engine.explicit_invalidate(EntryID("catalog", 100), 200)
        self.assertIsNotNone(first.round_record)
        self.assertIsNone(second.round_record)
        self.assertEqual(engine.versions[EntryID("catalog", 100)], 1)


if __name__ == "__main__":
    unittest.main()
