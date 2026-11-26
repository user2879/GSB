import unittest

from intent_router.errors import FailureCollector
from intent_router.library import match_text, synthesize_library
from intent_router.models import (
    Revision,
    RevisionChain,
    TextItem,
    TriggerOp,
)


def chain_with(rev_id, parent, ops, head=None):
    return RevisionChain(
        root_id="ROOT",
        head_id=head or rev_id,
        revisions=(
            Revision(
                revision_id=rev_id,
                parent_id=parent,
                sequence=1,
                ops=tuple(ops),
            ),
        ),
    )


class LibrarySynthesisTest(unittest.TestCase):
    def test_override_and_delete_provenance(self):
        ops1 = [
            TriggerOp("upsert", "E1", ("a", "b"), 3, "I1"),
            TriggerOp("upsert", "E2", ("c",), 2, "I2"),
        ]
        r1 = Revision("R1", "ROOT", 1, tuple(ops1))
        ops2 = [
            TriggerOp("upsert", "E1", ("a", "b"), 9, "I1"),
            TriggerOp("delete", "E2"),
        ]
        r2 = Revision("R2", "R1", 2, tuple(ops2))
        chain = RevisionChain("ROOT", "R2", (r1, r2))
        library = synthesize_library(chain, FailureCollector())
        self.assertNotIn("E2", library)
        entry = library["E1"]
        self.assertEqual(entry.weight, 9)
        self.assertEqual(entry.born_revision, "R1")
        self.assertEqual(entry.last_revision, "R2")
        self.assertEqual(entry.overrides, ("R2",))

    def test_invalid_weight_skipped_not_fatal(self):
        failures = FailureCollector()
        ops = [
            TriggerOp("upsert", "GOOD", ("a",), 1, "I1"),
            TriggerOp("upsert", "BAD", ("b",), -2, "I1"),
        ]
        chain = chain_with("R1", "ROOT", ops)
        library = synthesize_library(chain, failures)
        self.assertIn("GOOD", library)
        self.assertNotIn("BAD", library)
        self.assertTrue(any(f.code == "E_BAD_WEIGHT" for f in failures.items))


class LongestMatchTest(unittest.TestCase):
    def _library(self):
        ops = [
            TriggerOp(op="upsert", entry_id="SHORT", term=("预", "约"), weight=1, intent="I1"),
            TriggerOp(op="upsert", entry_id="LONG", term=("预", "约", "订", "票"), weight=2, intent="I1"),
            TriggerOp(op="upsert", entry_id="OTHER", term=("订", "票"), weight=3, intent="I2"),
        ]
        chain = chain_with("R1", "ROOT", ops)
        return synthesize_library(chain, FailureCollector())

    def test_longest_span_swallows_overlaps(self):
        library = self._library()
        text = TextItem("T1", ("预", "约", "订", "票", "x"), "C1", "I1")
        result = match_text(text, library)
        accepted_ids = [c.entry_id for c in result.accepted]
        self.assertEqual(accepted_ids, ["LONG"])
        suppressed = {(s.entry_id, s.swallowed_by) for s in result.suppressed}
        self.assertIn(("SHORT", "LONG"), suppressed)
        self.assertIn(("OTHER", "LONG"), suppressed)

    def test_tie_break_lex_then_start(self):
        ops = [
            TriggerOp(op="upsert", entry_id="A", term=("x", "y"), weight=1, intent="I1"),
            TriggerOp(op="upsert", entry_id="B", term=("x", "y"), weight=1, intent="I1"),
        ]
        chain = chain_with("R1", "ROOT", ops)
        library = synthesize_library(chain, FailureCollector())
        text = TextItem("T2", ("x", "y"), "C1", "I1")
        result = match_text(text, library)
        # Same span and term: entry_id order decides which single match survives.
        self.assertEqual(len(result.accepted), 1)
        self.assertEqual(result.accepted[0].entry_id, "A")
        self.assertEqual(result.suppressed[0].swallowed_by, "A")


if __name__ == "__main__":
    unittest.main()
