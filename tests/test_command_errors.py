from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.helpers import create_sample

import repair_studio  # noqa: F401
from repair_studio.engine import RunController


class CommandErrorTests(unittest.TestCase):
    def test_invalid_commands_are_reported_and_stream_continues(self) -> None:
        commands = [
            {"op": "set_mask_rect", "rect": {"x": 90, "y": 10, "width": 20, "height": 20}},
            {"op": "set_repair_radius", "value": 3},
            {"op": "set_quant_levels", "value": 12},
            {"op": "set_quant_levels", "value": 32},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_sample(root / "samples", "errors", width=64, height=48, commands=commands)
            report = RunController(root / "samples", root / "output", mode="sync").run()
            sample = report["samples"][0]
            records = sample["commands"]
            self.assertEqual(
                [record["business_code"] for record in records],
                ["E_MASK_OUT_OF_RANGE", "OK", "E_CONFIG_INVALID", "OK"],
            )
            self.assertEqual(len(sample["failures"]), 2)
            self.assertEqual(sample["failures"][0]["command_index"], 1)
            self.assertEqual(sample["failures"][1]["code"], "E_CONFIG_INVALID")
            self.assertEqual(sample["completed_command_cursor"], 4)
            self.assertTrue((root / "output" / "current" / "errors" / "final.png").is_file())


if __name__ == "__main__":
    unittest.main()

