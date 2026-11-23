import tempfile
import unittest
from pathlib import Path

from diagconsole.engine import Engine, NodeKey
from diagconsole.ledger import Ledger
from diagconsole.model import load_spec

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


class InvalidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.spec = load_spec(SAMPLES / "clusters.json",
                              SAMPLES / "tasks.json",
                              SAMPLES / "revisions.json")
        self.ledger = Ledger(Path(self.tmp.name) / "ledger.db")
        self.engine = Engine(self.spec, self.ledger)

    def tearDown(self):
        self.ledger.close()
        self.tmp.cleanup()

    def test_value_change_invalidates_descendants_lane_wise(self):
        changed = {"values": [3], "strategy": [], "derives": []}
        report = self.engine.analyze_invalidation(changed)
        versions = sorted({n.version for n in report.invalid})
        self.assertEqual(versions, [3, 4, 5, 6])
        self.assertTrue(report.steps)
        self.assertTrue(all(s["edge"] in ("parent", "derives")
                            for s in report.steps))

    def test_derives_edge_crosses_lanes(self):
        changed = {"values": [], "strategy": [], "derives": [2]}
        report = self.engine.analyze_invalidation(changed)
        edges = {s["edge"] for s in report.steps}
        self.assertIn("derives", edges)
        self.assertTrue(report.invalid)

    def test_amended_chain_only_recomputes_invalid_subset(self):
        self.engine.apply_chain(None, "batch-1")
        amended = load_spec(SAMPLES / "clusters.json",
                            SAMPLES / "tasks.json",
                            SAMPLES / "revisions_amended.json")
        previous = [{"version": r.version, "values": r.values,
                     "strategy": r.strategy, "derives": list(r.derives)}
                    for r in self.spec.revisions]
        self.engine.spec = amended
        result = self.engine.apply_chain(previous, "batch-2")
        self.assertTrue(result["unchanged_verified"]["all_byte_identical"])
        invalid_versions = {n["revision"]
                            for n in result["invalidation"]["invalid_set"]}
        self.assertIn(3, invalid_versions)
        self.assertNotIn(1, invalid_versions)

    def test_propagation_ordering_is_stable(self):
        changed = {"values": [1], "strategy": [1], "derives": []}
        report = self.engine.analyze_invalidation(changed)
        tuples = [n.as_tuple() for n in report.invalid]
        self.assertEqual(tuples, sorted(tuples))


if __name__ == "__main__":
    unittest.main()
