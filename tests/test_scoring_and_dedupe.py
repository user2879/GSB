from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from edge_ledger.config import parse_config
from edge_ledger.dedupe import DedupeInput, build_dedupe, similarity_permille
from edge_ledger.engine import build_workspace
from edge_ledger.loader import load_raw_workspace, prepare_workspace
from edge_ledger.scoring import compute_quality

from .helpers import make_sample_workspace


class ScoringAndDedupeTests(unittest.TestCase):
    def test_fixed_point_quantization_residual_and_repeatability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = make_sample_workspace(Path(temporary) / "workspace")
            loaded = prepare_workspace(load_raw_workspace(root))
            sample = loaded.valid_samples[0]
            answer = "A deliberately long answer used to trigger deterministic length decay. " * 2
            first = compute_quality(sample, answer, loaded.config)
            second = compute_quality(sample, answer, loaded.config)
            self.assertEqual(first, second)
            self.assertEqual(first.residual, first.raw_score - first.quantized_score)
            self.assertEqual(first.quantum_step, 1_000)
            self.assertEqual(first.quantized_score % first.quantum_step, 0)
            self.assertNotEqual(first.residual, 0)
            result = build_workspace(root)
            scores = [row["quality_score"] for row in result.result["samples"] if row["status"] == "ok"]
            residuals = [row["quality_residual"] for row in result.result["samples"] if row["status"] == "ok"]
            self.assertGreater(len(set(scores)), 1)
            self.assertTrue(any(value != 0 for value in residuals))

    def test_dedupe_quality_keeper_and_boundary_equality(self) -> None:
        equal_a = frozenset({"one", "two", "three", "four"})
        equal_b = frozenset({"one", "two", "three", "five"})
        self.assertEqual(similarity_permille(equal_a, equal_b), 600)
        result = build_dedupe(
            [
                DedupeInput("A", 500, equal_a),
                DedupeInput("B", 900, equal_b),
            ],
            600,
        )
        self.assertEqual(result.clusters[0].keeper_id, "B")
        self.assertEqual(result.removed_count, 1)
        self.assertEqual(result.boundary_pairs[0]["similarity_permille"], 600)
        self.assertTrue(result.clusters[0].removed[0].boundary)
        self.assertEqual(result.clusters[0].removed[0].keeper_id, "B")

    def test_dedupe_tie_breaks_by_sample_id(self) -> None:
        tokens = frozenset({"a", "b", "c"})
        result = build_dedupe(
            [
                DedupeInput("Z", 100, tokens),
                DedupeInput("A", 100, tokens),
            ],
            1000,
        )
        self.assertEqual(result.clusters[0].keeper_id, "A")
        self.assertEqual(result.clusters[0].removed[0].sample_id, "Z")

    def test_quantization_changes_scores_not_renders(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = make_sample_workspace(Path(temporary) / "workspace")
            first = build_workspace(root)
            render_digests = {row["sample_id"]: row.get("render_digest") for row in first.result["samples"]}
            config_path = root / "config" / "pipeline.yaml"
            import yaml

            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config["fixed_point"] = {"decimal_places": 6, "quantization": "Q10000"}
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            second = build_workspace(root)
            for row in second.result["samples"]:
                self.assertEqual(row.get("render_digest"), render_digests[row["sample_id"]])
            score_actions = [
                item
                for item in second.result["execution"]["cache_ledger"]
                if item["stage"] == "score" and item["entity_id"] != "S-FAIL-NONFINITE"
            ]
            self.assertTrue(all(item["action"] == "recompute" for item in score_actions))


if __name__ == "__main__":
    unittest.main()

