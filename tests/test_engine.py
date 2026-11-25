from __future__ import annotations

import unittest

from evidence_diag.diagnostics import build_diagnostic_report
from evidence_diag.engine import DiagnosticEngine, run_sequence
from evidence_diag.models import LimitsConfig

from .helpers import sample_objects


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config, cls.session, cls.commands = sample_objects()
        cls.incremental = run_sequence(
            cls.config, cls.session, cls.commands, mode="incremental"
        )
        cls.full = run_sequence(
            cls.config, cls.session, cls.commands, mode="full"
        )

    def test_every_cursor_matches_full_replay(self) -> None:
        self.assertEqual(len(self.incremental["trace"]), len(self.full["trace"]))
        for incremental, full in zip(
            self.incremental["trace"], self.full["trace"]
        ):
            self.assertEqual(
                incremental["result"],
                full["result"],
                msg=f"cursor {incremental['cursor']} differs",
            )
        self.assertEqual(
            self.incremental["final_result"], self.full["final_result"]
        )

    def test_query_replacement_is_precisely_invalidated(self) -> None:
        invalidation = self.incremental["trace"][2]["result"]["invalidation"]
        self.assertEqual(invalidation["invalidated_turns"], [2])
        self.assertIn(1, invalidation["reused_turns"])
        self.assertIn(3, invalidation["reused_turns"])
        self.assertEqual(len(invalidation["invalidated_topic_ids"]), 1)

    def test_evidence_weight_change_reuses_recall(self) -> None:
        execution = self.incremental["trace"][4]["execution"]
        by_stage = {item["stage"]: item for item in execution["stages"]}
        self.assertEqual(by_stage["recall"]["cache_misses"], 0)
        self.assertEqual(
            by_stage["recall"]["cache_hits"], by_stage["recall"]["task_count"]
        )
        self.assertGreater(by_stage["confidence"]["cache_misses"], 0)

    def test_single_evidence_edit_only_invalidates_citing_topic(self) -> None:
        config, session, _ = sample_objects()
        engine = DiagnosticEngine(config, session)
        engine.compute()
        outcome = engine.apply_command(
            {
                "seq": 1,
                "op": "set_evidence_weight",
                "evidence_id": "e07",
                "weight": "0.99",
            }
        )
        self.assertTrue(outcome["valid"])
        computed = engine.compute()
        invalidation = computed.result["invalidation"]
        self.assertEqual(invalidation["invalidated_turns"], [4, 5, 6])
        self.assertEqual(invalidation["reused_turns"], [1, 2, 3, 7, 8])
        self.assertEqual(len(invalidation["invalidated_topic_ids"]), 1)
        by_stage = {
            item["stage"]: item for item in computed.execution["stages"]
        }
        self.assertEqual(by_stage["recall"]["cache_misses"], 0)
        self.assertEqual(by_stage["dedupe"]["cache_misses"], 0)
        self.assertEqual(by_stage["confidence"]["cache_misses"], 3)

    def test_budget_pending_then_drained(self) -> None:
        pending = self.incremental["trace"][9]["result"]["todo"]
        self.assertEqual(
            [item["turn_id"] for item in pending], [2, 3, 4, 5, 6, 7, 8]
        )
        self.assertEqual({item["code"] for item in pending}, {"E_BUDGET_PENDING"})
        self.assertEqual(self.incremental["final_result"]["todo"], [])

    def test_invalid_command_is_quarantined_and_replay_matches(self) -> None:
        result = self.incremental["final_result"]
        self.assertEqual(
            [(item["seq"], item["code"]) for item in result["failures"]],
            [(13, "E_INPUT_INVALID")],
        )
        trace13 = self.incremental["trace"][13]["result"]
        self.assertEqual(trace13["invalidation"]["invalidated_turns"], [])
        self.assertTrue(trace13["invalidation"]["reused_turns"])

    def test_fixed_point_switch_changes_integer_and_confidence(self) -> None:
        before = self.incremental["trace"][8]["result"]
        after = self.incremental["trace"][9]["result"]
        before_turn = before["turns"][0]
        after_turn = after["turns"][0]
        self.assertNotEqual(
            before_turn["recall"]["evidence"][0]["authority_score_units"],
            after_turn["recall"]["evidence"][0]["authority_score_units"],
        )
        self.assertNotEqual(
            before_turn["confidence"]["confidence"],
            after_turn["confidence"]["confidence"],
        )

    def test_concurrency_degree_does_not_change_result(self) -> None:
        serial_config = self.config.with_limits(
            LimitsConfig(
                max_workers=1,
                evidence_budget=self.config.limits.evidence_budget,
            )
        )
        parallel_config = self.config.with_limits(
            LimitsConfig(
                max_workers=8,
                evidence_budget=self.config.limits.evidence_budget,
            )
        )
        serial = run_sequence(
            serial_config, self.session, self.commands, mode="incremental"
        )
        parallel = run_sequence(
            parallel_config, self.session, self.commands, mode="incremental"
        )
        for key in (
            "turns",
            "topics",
            "evidence_attribution",
            "stage_versions",
            "todo",
            "failures",
            "invalidation",
        ):
            self.assertEqual(
                serial["final_result"][key],
                parallel["final_result"][key],
                msg=key,
            )
        execution = parallel["trace"][5]["execution"]
        for stage in execution["stages"]:
            self.assertLessEqual(stage["max_workers"], 8)
            self.assertEqual(
                stage["sorted_task_keys"], sorted(stage["sorted_task_keys"])
            )

    def test_diagnostic_has_stage_diffs_first_exceed_and_flip_causes(self) -> None:
        final_engine = DiagnosticEngine.from_state(self.incremental["state"])
        report = build_diagnostic_report(
            final_engine.config, final_engine.session
        )
        self.assertEqual(
            set(report["stages"]),
            {"recall", "dedupe", "topic", "confidence"},
        )
        self.assertEqual(len(report["first_exceed_by_turn"]), 8)
        self.assertGreater(report["stage_summaries"]["confidence"]["exceed_count"], 0)
        self.assertEqual(report["flip_count"], 1)
        flip = report["flips"][0]
        causes = {item["cause"] for item in flip["causes"]}
        self.assertEqual(
            causes,
            {"dedupe_merge_order", "cross_topic_penalty", "rounding_direction"},
        )
        self.assertIn(flip["primary_cause"], causes)
        self.assertNotEqual(
            flip["actual_conclusion"], flip["reference_conclusion"]
        )


if __name__ == "__main__":
    unittest.main()
