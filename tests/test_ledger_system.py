from __future__ import annotations

import copy
import unittest
from pathlib import Path

from preference_ledger.engine import boundary_decision, summarize
from preference_ledger.incremental import IncrementalStore
from preference_ledger.ledger import parse_g1_line
from preference_ledger.loading import load_input
from preference_ledger.pipeline import semantic_comparison_report
from preference_ledger.util import canonical_json, fnv1a_64
from preference_ledger.verify import run_verification


ROOT = Path(__file__).resolve().parents[1]


class LedgerSystemTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.verification = run_verification(ROOT)

    def test_full_verification_report_passes(self) -> None:
        self.assertTrue(self.verification["ok"], self.verification)

    def test_fnv_hash_is_known_and_stable(self) -> None:
        self.assertEqual(fnv1a_64("preference-ledger"), fnv1a_64("preference-ledger"))
        self.assertIsInstance(fnv1a_64("preference-ledger"), int)

    def test_boundary_threshold_marks_borderline(self) -> None:
        loaded = load_input(
            ROOT / "samples/preferences.jsonl",
            ROOT / "samples/annotators.yaml",
            ROOT / "samples/config.yaml",
        )
        assert loaded.config is not None
        decision = boundary_decision(0, 2, loaded.config)
        self.assertTrue(decision["exact_threshold_match"])
        self.assertTrue(decision["boundary"])
        self.assertEqual(decision["verdict_code"], 0)

    def test_unknown_g1_version_code(self) -> None:
        entry, issues = parse_g1_line("G9|unknown")
        self.assertIsNone(entry)
        self.assertEqual(issues[0].code, "E_VERSION_UNKNOWN")

    def test_all_conflict_annotator_is_retained(self) -> None:
        summary = summarize(
            [],
            [
                {
                    "session_id": "S1",
                    "turn": 1,
                    "annotator_id": "ann_a",
                    "quantized_scaled": -8,
                    "decimal_places": 4,
                    "category": "conflict",
                },
                {
                    "session_id": "S1",
                    "turn": 2,
                    "annotator_id": "ann_a",
                    "quantized_scaled": -3,
                    "decimal_places": 4,
                    "category": "conflict",
                },
            ],
            [],
            [],
            {"ann_a": 5000, "ann_b": 1},
        )
        annotators = summary["annotator_summary"]
        self.assertEqual([item["annotator_id"] for item in annotators], ["ann_a", "ann_b"])
        self.assertTrue(annotators[0]["degenerate_all_conflict"])
        self.assertEqual(annotators[0]["conflict_ratio"], 1.0)
        self.assertEqual(annotators[1]["participated"], 0)

    def test_second_deterministic_edit_sequence(self) -> None:
        samples = ROOT / "samples"
        base = IncrementalStore.from_loaded_files(
            samples / "preferences.jsonl", samples / "annotators.yaml", samples / "config.yaml"
        )
        base.run(save=False)
        commands = [
            {"op": "set_weight", "annotator_id": "ann_chen", "weight_bp": 1},
            {"op": "set_quant", "quant_scheme": "Q_BALANCED", "fixed_decimal_places": 6},
            {"op": "set_window_lengths", "value": [3, 4]},
            {"op": "set_partition_boundaries", "value": [3, 6]},
            {"op": "set_exchange_version", "value": "g2"},
            {
                "op": "replace_record",
                "record": {
                    "session_id": "S003",
                    "turn": 2,
                    "prompt": "analysis method logic",
                    "chosen": "clear coherent logic",
                    "rejected": "weak vague analysis",
                    "annotator_id": "ann_chen",
                    "event_ms": 1764102069999,
                },
            },
            {"op": "set_weight", "annotator_id": "ann_chen", "weight_bp": 9999},
        ]
        for command in commands:
            base.apply_command(copy.deepcopy(command))
            incremental = base.run(workers=2, save=False)
            oracle = IncrementalStore(base.raw_records, base.raw_weights, base.raw_config)
            full = oracle.run(workers=1, save=False)
            self.assertEqual(
                canonical_json(semantic_comparison_report(incremental["report"])),
                canonical_json(semantic_comparison_report(full["report"])),
                command,
            )


if __name__ == "__main__":
    unittest.main()
