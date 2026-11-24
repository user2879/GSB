import os
import unittest

from relx.commands import Command, load_commands
from relx.pipeline import (
    build_runtime,
    compare_artifacts,
    incremental_run,
    rebuild_from_state,
)
from relx.engine import ExtractionEngine


class IncrementalConsistencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = "data"
        cls.engine, cls.state, cls.baseline, cls.reports, cls.failures = \
            incremental_run(cls.data)

    def test_final_artifacts_equal_blank_rebuild(self):
        rebuilt = rebuild_from_state(self.state)
        identical, _left, _right = compare_artifacts(
            rebuilt.to_dict(), self.reports[-1]["report"]
        )
        self.assertTrue(identical)

    def test_token_edit_only_invalidates_target_sentence(self):
        state, _ = build_runtime(self.data)
        engine = ExtractionEngine(state)
        engine.run()
        cmd = Command(seq=1, kind="edit_tokens",
                      payload={"sentence_id": "S21",
                               "tokens": ["张伟", "就职", "于", "研究院"]},
                      comment="")
        invalidated = engine.apply(cmd)
        sentences = {row[0] for row in invalidated}
        self.assertEqual(sentences, {"S21"})

    def test_template_weight_invalidates_only_referencing_sentences(self):
        state, _ = build_runtime(self.data)
        engine = ExtractionEngine(state)
        engine.run()
        referencing = {
            sid for sid, result in engine.results.items()
            if "T04" in result.referenced_templates
        }
        cmd = Command(seq=2, kind="set_template_weight",
                      payload={"template_id": "T04", "weight": 0.31}, comment="")
        invalidated = engine.apply(cmd)
        affected = {row[0] for row in invalidated}
        self.assertTrue(affected)
        self.assertTrue(affected.issubset(referencing))

    def test_threshold_change_keeps_entity_cache(self):
        state, _ = build_runtime(self.data)
        engine = ExtractionEngine(state)
        engine.run()
        cmd = Command(seq=3, kind="set_threshold",
                      payload={"threshold": 1.2}, comment="")
        invalidated = engine.apply(cmd)
        stages = {row[1] for row in invalidated}
        self.assertNotIn("entity", stages)
        self.assertIn("score", stages)

    def test_quant_switch_invalidates_all_stages(self):
        state, _ = build_runtime(self.data)
        engine = ExtractionEngine(state)
        engine.run()
        cmd = Command(seq=4, kind="set_quant",
                      payload={"fixedpoint_width": 8, "quant_level": 2}, comment="")
        invalidated = engine.apply(cmd)
        stages = {row[1] for row in invalidated}
        self.assertEqual(stages, {"entity", "trigger", "score", "miss"})

    def test_reuse_and_recompute_audit_lists_present(self):
        final = self.reports[-1]["report"]
        for sentence in final["sentences"]:
            self.assertEqual(len(sentence["reuse_recompute"]), 4)
            actions = {row["stage"]: row["action"] for row in sentence["reuse_recompute"]}
            self.assertEqual(set(actions), {"entity", "trigger", "score", "miss"})

    def test_command_stream_loads_and_replays_deterministically(self):
        commands = load_commands(os.path.join(self.data, "commands.json"))
        self.assertEqual([cmd.seq for cmd in commands],
                         sorted(cmd.seq for cmd in commands))
        self.assertEqual(len(commands), 9)


if __name__ == "__main__":
    unittest.main()
