from __future__ import annotations

import unittest
from decimal import Decimal

from ner_editor.constants import E_EMISSION_INVALID, E_LATENCY_BUDGET_EXCEEDED
from ner_editor.engine import LocalNerEngine
from tests.helpers import command, ledger_statuses, make_engine


class IncrementalInvalidationTests(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine(sample_limit=5, budget_ms=60000, workers=4)
        self.engine.settle()

    def tearDown(self):
        self.engine.close()

    def _all_stages(self, status: str) -> dict[str, dict[str, str]]:
        return {
            sample_id: {stage: status for stage in ("feature", "emission", "decode", "merge")}
            for sample_id in self.engine.samples
        }

    def test_stage_level_reuse_and_recompute(self):
        self.assertEqual(ledger_statuses(self.engine), self._all_stages("recompute"))

        self.engine.apply_command(command(1, "noop"))
        self.engine.settle()
        self.assertEqual(ledger_statuses(self.engine), self._all_stages("reuse"))

        self.engine.apply_command(
            command(
                2,
                "vocab_add",
                entry={"id": "v999", "text": "完全不存在", "type": "LOC"},
            )
        )
        self.engine.settle()
        self.assertEqual(ledger_statuses(self.engine), self._all_stages("reuse"))

        matching = next(
            entry
            for entry in self.engine.model.vocabulary.values()
            if any(entry.text in sample.text for sample in self.engine.samples.values())
        )
        replacement_type = "PER" if matching.type_code != "PER" else "LOC"
        self.engine.apply_command(
            command(
                3,
                "vocab_replace",
                entry_id=matching.entry_id,
                entry={
                    "id": matching.entry_id,
                    "text": matching.text,
                    "type": replacement_type,
                },
            )
        )
        self.engine.settle()
        statuses = ledger_statuses(self.engine)
        affected = [
            sample_id
            for sample_id, sample in self.engine.samples.items()
            if matching.text in sample.text
        ]
        self.assertTrue(affected)
        for sample_id in affected:
            self.assertEqual(
                statuses[sample_id],
                {
                    "feature": "reuse",
                    "emission": "reuse",
                    "decode": "reuse",
                    "merge": "recompute",
                },
            )

        self.engine.apply_command(
            command(
                4,
                "emission_adjust",
                target={"dimension": 0, "label": "B"},
                delta=0.01,
            )
        )
        self.engine.settle()
        for sample_id in self.engine.samples:
            self.assertEqual(
                ledger_statuses(self.engine)[sample_id],
                {
                    "feature": "reuse",
                    "emission": "recompute",
                    "decode": "recompute",
                    "merge": "recompute",
                },
            )

        self.engine.apply_command(
            command(
                5,
                "transition_adjust",
                target={"from": "O", "to": "B"},
                delta=0.01,
            )
        )
        self.engine.settle()
        for sample_id in self.engine.samples:
            self.assertEqual(
                ledger_statuses(self.engine)[sample_id],
                {
                    "feature": "reuse",
                    "emission": "reuse",
                    "decode": "recompute",
                    "merge": "recompute",
                },
            )

        self.engine.apply_command(command(6, "fixed_point_set", fixed_decimals=5))
        self.engine.settle()
        for sample_id in self.engine.samples:
            self.assertEqual(
                ledger_statuses(self.engine)[sample_id],
                {
                    "feature": "reuse",
                    "emission": "recompute",
                    "decode": "recompute",
                    "merge": "recompute",
                },
            )

    def test_label_only_corpus_replacement_reuses_feature_emission_decode(self):
        sample_id = sorted(self.engine.samples)[0]
        original = self.engine.samples[sample_id]
        replacement = {
            "id": sample_id,
            "chars": list(original.chars),
            "labels": ["O"] * len(original.chars),
            "types": ["O"] * len(original.chars),
        }
        self.engine.apply_command(
            command(10, "corpus_replace", sample_id=sample_id, sample=replacement)
        )
        self.engine.settle()
        statuses = ledger_statuses(self.engine)
        self.assertEqual(
            statuses[sample_id],
            {
                "feature": "reuse",
                "emission": "reuse",
                "decode": "reuse",
                "merge": "recompute",
            },
        )
        for other_id in self.engine.samples:
            if other_id != sample_id:
                self.assertEqual(
                    statuses[other_id],
                    {
                        "feature": "reuse",
                        "emission": "reuse",
                        "decode": "reuse",
                        "merge": "reuse",
                    },
                )

    def test_each_stable_command_point_matches_a_blank_cache_full_run(self):
        matching = next(
            entry
            for entry in self.engine.model.vocabulary.values()
            if any(entry.text in sample.text for sample in self.engine.samples.values())
        )
        commands = [
            command(
                1,
                "emission_adjust",
                target={"dimension": 2, "label": "E"},
                delta=0.02,
            ),
            command(
                2,
                "transition_adjust",
                target={"from": "I", "to": "E"},
                delta=0.04,
            ),
            command(3, "fixed_point_set", fixed_decimals=6),
            command(4, "quantization_set", quantization_level=8),
            command(
                5,
                "vocab_replace",
                entry_id=matching.entry_id,
                entry={
                    "id": matching.entry_id,
                    "text": matching.text,
                    "type": "ORG" if matching.type_code != "ORG" else "LOC",
                },
            ),
        ]
        for item in commands:
            self.engine.apply_command(item)
            self.engine.settle()
            incremental = self.engine.business_projection()
            full = LocalNerEngine(
                dict(self.engine.samples),
                self.engine.model.clone(),
                self.engine.config.clone(),
            )
            try:
                full.settle()
                self.assertEqual(
                    incremental,
                    full.business_projection(),
                    f"drift after command {item.position}",
                )
            finally:
                full.close()

    def test_final_incremental_projection_matches_blank_cache_full_run(self):
        matching = next(
            entry
            for entry in self.engine.model.vocabulary.values()
            if any(entry.text in sample.text for sample in self.engine.samples.values())
        )
        commands = [
            command(1, "noop"),
            command(
                2,
                "emission_adjust",
                target={"dimension": 1, "label": "I"},
                delta=0.07,
            ),
            command(
                3,
                "transition_adjust",
                target={"from": "B", "to": "E"},
                delta=-0.03,
            ),
            command(4, "fixed_point_set", fixed_decimals=5),
            command(5, "quantization_set", quantization_level=2),
            command(
                6,
                "vocab_replace",
                entry_id=matching.entry_id,
                entry={
                    "id": matching.entry_id,
                    "text": matching.text,
                    "type": "TIME" if matching.type_code != "TIME" else "ORG",
                },
            ),
        ]
        for item in commands:
            self.engine.apply_command(item)
        self.engine.settle()
        projection = self.engine.business_projection()

        full = LocalNerEngine(
            dict(self.engine.samples),
            self.engine.model.clone(),
            self.engine.config.clone(),
        )
        try:
            full.settle()
            self.assertEqual(projection, full.business_projection())
        finally:
            full.close()


class ConcurrencyAndFailureTests(unittest.TestCase):
    def test_one_and_four_workers_produce_identical_business_projection(self):
        engine_one = make_engine(sample_limit=5, budget_ms=60000, workers=1)
        engine_four = make_engine(sample_limit=5, budget_ms=60000, workers=4)
        try:
            engine_one.settle()
            engine_four.settle()
            self.assertEqual(
                engine_one.business_projection(),
                engine_four.business_projection(),
            )
        finally:
            engine_one.close()
            engine_four.close()

    def test_zero_budget_returns_stale_latency_code_and_later_settles(self):
        engine = make_engine(budget_ms=0, workers=4)
        try:
            response = engine.apply_command(command(1, "noop"))
            self.assertEqual(response["status"], "stale")
            self.assertIn(E_LATENCY_BUDGET_EXCEEDED, response["codes"])
            self.assertLess(response["elapsed_ms"], 500)
            self.assertEqual(response["summary"]["summary_version"], 0)
            self.assertTrue(response["summary"]["progress"]["total"] > 0)
            engine.settle()
            self.assertEqual(engine.version, 1)
            self.assertEqual(
                engine.business_projection()["sample_order"],
                sorted(engine.samples),
            )
        finally:
            engine.close()

    def test_non_finite_parameter_fails_only_at_emission_and_is_ordered(self):
        engine = make_engine(sample_limit=3, budget_ms=60000, workers=3)
        try:
            engine.model.emission_weights[0][0] = Decimal("NaN")
            engine.settle()
            failures = engine.current_failures()
            self.assertEqual([failure.sample_id for failure in failures], ["s001", "s002", "s003"])
            self.assertTrue(all(failure.code == E_EMISSION_INVALID for failure in failures))
            for sample_id in engine.samples:
                report = engine.sample_report(sample_id)
                self.assertEqual(report["code"], E_EMISSION_INVALID)
                self.assertEqual(report["entities"], [])
        finally:
            engine.close()


if __name__ == "__main__":
    unittest.main()
