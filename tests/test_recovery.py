import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from intent_router.fixedpoint import QuantGear
from intent_router.loader import load_inputs
from intent_router.pipeline import Pipeline, full_rebuild
from intent_router.snapshot import SnapshotStore


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.corpus, self.chain, self.config, _ = load_inputs(
            "data/corpus.yaml", "data/revisions.yaml", "data/config.yaml"
        )
        self.tight = replace(
            self.config, global_match_budget=12, per_batch_recompute_limit=4
        )

    def _serialize(self, result, config):
        q = QuantGear(config.fraction_bits, config.gear)
        return {
            "scores": {
                tid: result.scores[tid].to_dict(q)
                for tid in sorted(result.scores)
            },
            "matches": {
                tid: result.matches[tid].to_dict()
                for tid in sorted(result.matches)
            },
            "clusters": [c.to_dict() for c in result.clusters],
            "removed": {
                tid: m.to_dict()
                for tid, m in sorted(result.removed.items())
            },
        }

    def test_resume_matches_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SnapshotStore(Path(tmp) / "snap")
            runner = Pipeline(
                self.tight, self.chain, self.corpus.by_id(),
                self.corpus.vocab, store=store,
            )
            runner.run_batch()  # forced stop after one committed batch

            resumed = Pipeline.resume_from_snapshot(
                self.tight, self.chain, self.corpus.by_id(),
                self.corpus.vocab, store,
            )
            resumed_result = resumed.run_to_completion()

        uninterrupted = full_rebuild(
            self.tight, self.chain, self.corpus.by_id(), self.corpus.vocab
        )
        self.assertEqual(
            self._serialize(resumed_result, self.tight),
            self._serialize(uninterrupted, self.tight),
        )
        self.assertEqual(resumed_result.pending, [])


if __name__ == "__main__":
    unittest.main()
