import copy
import os
import tempfile
import unittest

from sft_diag.generator import CandidateGenerator, build_reference_index
from sft_diag.loader import (
    load_config,
    load_score_params,
    load_seeds,
    load_slot_sources,
    load_vocab,
)
from sft_diag.models import WorkspaceState
from sft_diag.scorer import QualityScorer
from sft_diag.tokenizer import Tokenizer
from tests._helpers import minimal_sample


def _load_state(sample_dir: str) -> WorkspaceState:
    failures = []
    seeds = load_seeds(os.path.join(sample_dir, "seeds.jsonl"), failures)
    slots = load_slot_sources(os.path.join(sample_dir, "slot_sources.jsonl"), failures)
    vocab = load_vocab(os.path.join(sample_dir, "vocab.json"), failures)
    config = load_config(os.path.join(sample_dir, "build_config.json"), failures)
    params = load_score_params(os.path.join(sample_dir, "score_params.json"), failures)
    assert not failures, failures
    return WorkspaceState(seeds, slots, config, params, vocab)


class DeterminismTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.sample_dir = minimal_sample(self.temp.name)
        self.state = _load_state(self.sample_dir)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_candidate_sequence_identical_across_runs(self) -> None:
        first = CandidateGenerator(self.state).generate_all()
        second = CandidateGenerator(copy.deepcopy(self.state)).generate_all()
        self.assertEqual(first, second)
        self.assertEqual([c["sample_order"] for c in first], list(range(10)))
        self.assertTrue(all("template_occurrence" in c for c in first))
        occurrences = {}
        for candidate in first:
            if candidate["template_id"] == "tpl-a0":
                occurrences.setdefault(candidate["template_id"], []).append(
                    candidate["template_occurrence"]
                )
        seq = occurrences["tpl-a0"]
        self.assertEqual(seq, list(range(len(seq))))
        self.assertTrue(all(c["slot_sources"] for c in first))

    def test_reference_index_covers_every_source(self) -> None:
        candidates = CandidateGenerator(self.state).generate_all()
        index = build_reference_index(candidates)
        referenced = {cid for ids in index["slot_source"].values() for cid in ids}
        self.assertEqual(referenced, {c["candidate_id"] for c in candidates})

    def test_scoring_bit_stable_and_float64(self) -> None:
        tokenizer = Tokenizer(self.state.vocab)
        scorer = QualityScorer(self.state, tokenizer)
        candidates = CandidateGenerator(self.state).generate_all()
        run_one = [scorer.score_one(candidate) for candidate in candidates]
        run_two = [scorer.score_one(candidate) for candidate in candidates]
        self.assertEqual(run_one, run_two)
        for record in run_one:
            self.assertIsInstance(record["total_score"], float)
            for dim, value in record["dimensions"].items():
                self.assertIsInstance(value, float)
        self.assertEqual(
            [r["candidate_id"] for r in run_one],
            sorted(r["candidate_id"] for r in run_one),
        )

    def test_concurrent_and_serial_scoring_identical(self) -> None:
        tokenizer = Tokenizer(self.state.vocab)
        candidates = CandidateGenerator(self.state).generate_all()
        serial, serial_status = QualityScorer(self.state, tokenizer).score_all(
            candidates, max_workers=1
        )
        parallel, parallel_status = QualityScorer(
            copy.deepcopy(self.state), tokenizer
        ).score_all(candidates, max_workers=8)
        self.assertEqual(serial, parallel)
        self.assertEqual(set(serial_status), set(parallel_status))

    def test_tie_break_is_candidate_id_lexicographic(self) -> None:
        tokenizer = Tokenizer(self.state.vocab)
        scorer = QualityScorer(self.state, tokenizer)
        candidates = CandidateGenerator(self.state).generate_all()
        scores = [scorer.score_one(candidate) for candidate in candidates]
        totals = [r["total_score"] for r in scores]
        if len(set(totals)) < len(totals):
            tied = {t for t in totals if totals.count(t) > 1}
            for total in tied:
                ids = [r["candidate_id"] for r in scores if r["total_score"] == total]
                self.assertEqual(ids, sorted(ids))


if __name__ == "__main__":
    unittest.main()
