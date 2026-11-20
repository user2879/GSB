import tempfile
import unittest

from sft_diag.dedup import Deduplicator, jaccard
from sft_diag.loader import load_vocab
from sft_diag.models import BuildConfig, ScoreParams, DimensionSpec, WorkspaceState
from sft_diag.tokenizer import Tokenizer
from tests._helpers import write_json


def _state(threshold: float, rule: str) -> WorkspaceState:
    config = BuildConfig(
        task_types=("alpha",),
        target_counts={"alpha": 0},
        template_weights={"alpha": {}},
        slot_weights={"alpha": {}},
        dedup_threshold=threshold,
        boundary_rule=rule,
        max_workers=4,
        snapshot_interval_batches=1,
    )
    params = ScoreParams(
        dimensions=("clarity", "correctness", "format", "length"),
        specs={
            name: DimensionSpec(1.0, 0.5)
            for name in ("clarity", "correctness", "format", "length")
        },
        length_lower=1,
        length_upper=100,
    )
    return WorkspaceState(seeds={}, slot_sources={}, config=config,
                          score_params=params, vocab=None)  # type: ignore[arg-type]


class DedupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        vocab_path = f"{self.temp.name}/vocab.json"
        tokens = ["a", "b", "c", "d", "e", "f", "g", "h"]
        write_json(vocab_path, {"tokens": tokens, "rules": {}})
        self.vocab = load_vocab(vocab_path, [])
        self.tokenizer = Tokenizer(self.vocab)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _candidates(self) -> list[dict]:
        # 集合：c0={a,b,c,d}, c1={a,b,c,e} -> J=3/5=0.6 恰为阈值
        # c2={f,g,h} 与前两者 J=0
        texts = ["a b c d", "a b c e", "f g h"]
        return [
            {
                "candidate_id": f"alpha#00000{i}",
                "task_type": "alpha",
                "instruction": text,
                "total_score": 0.9 - i * 0.1,
            }
            for i, text in enumerate(texts)
        ]

    def test_boundary_merge_rule(self) -> None:
        state = _state(0.6, "merge")
        state.vocab = self.vocab
        dedup = Deduplicator(state, self.tokenizer)
        result = dedup.deduplicate_task("alpha", self._candidates(), max_workers=4)
        self.assertEqual(len(result["groups"]), 2)
        big_group = next(group for group in result["groups"] if len(group["members"]) == 2)
        self.assertEqual(big_group["representative_id"], "alpha#000000")
        self.assertIn("alpha#000000", big_group["boundary_member_ids"])
        self.assertIn("alpha#000001", big_group["boundary_member_ids"])
        removal = next(item for item in result["removals"]
                       if item["candidate_id"] == "alpha#000001")
        self.assertAlmostEqual(removal["similarity"], 0.6, places=12)
        self.assertEqual(removal["representative_id"], "alpha#000000")
        self.assertEqual(result["retained_ids"], ["alpha#000000", "alpha#000002"])

    def test_boundary_split_rule(self) -> None:
        state = _state(0.6, "split")
        state.vocab = self.vocab
        dedup = Deduplicator(state, self.tokenizer)
        result = dedup.deduplicate_task("alpha", self._candidates(), max_workers=1)
        self.assertEqual(len(result["groups"]), 3)
        edge = next(e for e in result["edges"]
                    if {e["left"], e["right"]} == {"alpha#000000", "alpha#000001"})
        self.assertTrue(edge["boundary"])
        self.assertFalse(edge["duplicate"])
        self.assertEqual(
            result["boundary_candidate_ids"], ["alpha#000000", "alpha#000001"]
        )

    def test_representative_uses_score_then_id(self) -> None:
        candidates = self._candidates()
        candidates[0]["total_score"] = 0.5
        candidates[1]["total_score"] = 0.5
        state = _state(0.6, "merge")
        state.vocab = self.vocab
        result = Deduplicator(state, self.tokenizer).deduplicate_task(
            "alpha", candidates, max_workers=2
        )
        group = next(g for g in result["groups"] if len(g["members"]) == 2)
        self.assertEqual(group["representative_id"], "alpha#000000")

    def test_jaccard_basic(self) -> None:
        self.assertEqual(jaccard(frozenset(), frozenset()), 1.0)
        self.assertEqual(jaccard(frozenset("a"), frozenset("b")), 0.0)
        self.assertAlmostEqual(
            jaccard(frozenset("abc"), frozenset("abd")), 2 / 4, places=12
        )


if __name__ == "__main__":
    unittest.main()
