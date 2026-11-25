import unittest

from fxpdiag.edits import run_command_sequence
from fxpdiag.models import EditCommand
from fxpdiag.pipeline import PipelineRunner
from tests.test_pipeline import make_state


class ReuseSequenceTests(unittest.TestCase):
    def test_noop_second_command_reuses_everything(self):
        commands = [
            EditCommand(1, "set_tier", {"tier": "fine"}),
            EditCommand(2, "set_tier", {"tier": "fine"}),
        ]
        state = make_state()
        snapshots = run_command_sequence(
            state, commands, PipelineRunner, parallel=False
        )
        # 第二条命令配置未变，全部行应命中缓存直接复用
        final = snapshots[-1]
        self.assertEqual(final["ledger_summary"]["reused"], 4)
        self.assertEqual(final["ledger_summary"]["recomputed"], 0)

    def test_row_edit_then_revert_restores_identical_output(self):
        original = list(make_state().package.layers["embedding"][0])
        commands = [
            EditCommand(1, "replace_row_activation",
                        {"layer": "embedding", "row_index": 0,
                         "activation": [2.0, -2.0, 0.25]}),
            EditCommand(2, "replace_row_activation",
                        {"layer": "embedding", "row_index": 0,
                         "activation": original}),
        ]
        state = make_state()
        snapshots = run_command_sequence(
            state, commands, PipelineRunner, parallel=False
        )
        baseline = snapshots[0]
        reverted = snapshots[-1]
        self.assertEqual(
            [row["softmax"]["int"] for row in baseline["rows"]],
            [row["softmax"]["int"] for row in reverted["rows"]],
        )


if __name__ == "__main__":
    unittest.main()
