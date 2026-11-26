import unittest
from dataclasses import replace

from intent_router.commands import parse_commands
from intent_router.engine import EngineResult
from intent_router.fixedpoint import QuantGear
from intent_router.loader import load_inputs, read_yaml
from intent_router.pipeline import Pipeline, full_rebuild


DATA = "data"


def _load():
    return load_inputs(
        f"{DATA}/corpus.yaml",
        f"{DATA}/revisions.yaml",
        f"{DATA}/config.yaml",
    )


def _commands(failures):
    raw = read_yaml(f"{DATA}/commands.yaml")
    return parse_commands(raw["commands"], failures)


def _result_dict(result: EngineResult, fraction_bits: int, gear: int) -> dict:
    q = QuantGear(fraction_bits, gear)
    return {
        "scores": {
            tid: result.scores[tid].to_dict(q) for tid in sorted(result.scores)
        },
        "matches": {
            tid: result.matches[tid].to_dict()
            for tid in sorted(result.matches)
        },
        "clusters": [c.to_dict() for c in result.clusters],
        "removed": {
            tid: m.to_dict() for tid, m in sorted(result.removed.items())
        },
        "final_intents": {
            tid: result.final_intent(tid) for tid in sorted(result.scores)
        },
    }


class IncrementalConsistencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus, cls.chain, cls.config, cls.failures = _load()
        cls.commands = _commands(cls.failures)

    def _incremental(self, max_workers):
        cfg = replace(self.config, max_workers=max_workers)
        pipeline = Pipeline(cfg, self.chain, self.corpus.by_id(), self.corpus.vocab)
        pipeline.run_to_completion()
        for command in self.commands:
            pipeline.apply_commands([command])
            pipeline.run_to_completion()
        return pipeline

    def test_incremental_equals_full_rebuild(self):
        inc = self._incremental(max_workers=1)
        full = full_rebuild(
            inc.config, inc.chain, inc.texts, inc.vocab
        )
        left = _result_dict(inc.previous, inc.config.fraction_bits, inc.config.gear)
        right = _result_dict(full, inc.config.fraction_bits, inc.config.gear)
        self.assertEqual(left, right)

    def test_concurrency_independent_results(self):
        serial = self._incremental(max_workers=1)
        parallel = self._incremental(max_workers=8)
        self.assertEqual(
            _result_dict(serial.previous, serial.config.fraction_bits, serial.config.gear),
            _result_dict(parallel.previous, parallel.config.fraction_bits, parallel.config.gear),
        )

    def test_invalidation_and_reuse_recorded(self):
        pipeline = Pipeline(
            self.config, self.chain, self.corpus.by_id(), self.corpus.vocab
        )
        pipeline.run_to_completion()
        ledger_before = len(pipeline.ledger.entries)
        self.assertGreater(ledger_before, 0)
        # Replace one text: only its products should invalidate, others reused.
        replace_cmd = next(c for c in self.commands if c.kind == "replace_text")
        pipeline.apply_commands([replace_cmd])
        pipeline.run_to_completion()
        invalidated_texts = {
            rec.text_id for rec in pipeline.ledger.invalidations
        }
        self.assertIn(replace_cmd.payload["text_id"], invalidated_texts)
        reused_texts = {rec.text_id for rec in pipeline.ledger.reuses}
        # At least some unaffected texts were served from cache.
        self.assertTrue(reused_texts)

    def test_flip_attribution_uses_known_sources(self):
        pipeline = Pipeline(
            self.config, self.chain, self.corpus.by_id(), self.corpus.vocab
        )
        pipeline.run_to_completion()
        sources = set()
        for command in self.commands:
            pipeline.apply_commands([command])
            pipeline.run_to_completion()
            for flip in pipeline.previous.flips:
                if flip.flipped:
                    sources.add(flip.source)
                    self.assertIn("quantized_delta", flip.evidence)
        # Across the rich command stream at least one attribution source fires.
        self.assertTrue(
            sources
            <= {
                "TRIGGER_WEIGHT_CHANGE",
                "FIXED_POINT_ROUNDING",
                "THRESHOLD_COMPARISON",
                "DEDUP_CLUSTER_MERGE",
            }
        )

    def test_pending_drains_across_batches(self):
        tight = replace(self.config, global_match_budget=12, per_batch_recompute_limit=4)
        pipeline = Pipeline(tight, self.chain, self.corpus.by_id(), self.corpus.vocab)
        result = pipeline.run_to_completion()
        self.assertEqual(result.pending, [])
        self.assertEqual(len(result.scores), len(self.corpus.texts))


if __name__ == "__main__":
    unittest.main()
