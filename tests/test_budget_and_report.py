from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import repair_studio  # noqa: F401
import numpy as np
from PIL import Image

from tests.helpers import create_sample

from repair_studio.config import apply_command
from repair_studio.engine import RunController, render_full_reference
from repair_studio.imaging import load_sample


class BudgetAndReportTests(unittest.TestCase):
    def test_tiny_budget_returns_pending_then_finalizes_latest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            small = create_sample(
                root / "samples",
                "budget_pending",
                width=56,
                height=40,
                commands=[
                    {"op": "set_repair_radius", "value": 3},
                    {"op": "set_style_weights", "weights": [0.2, 0.7, 0.5]},
                    {"op": "set_quant_levels", "value": 32},
                    {"op": "set_mask_rect", "rect": {"x": 8, "y": 8, "width": 36, "height": 26}},
                ],
                config_overrides={"interaction": {"latency_budget_ms": 1}},
            )
            output = root / "output_pending"
            report = RunController(root / "samples", output, mode="interactive").run()
            sample_report = report["samples"][0]
            self.assertEqual(sample_report["status"], "ok")
            self.assertGreaterEqual(len(sample_report["commands"]), 1)
            pending = sample_report["commands"][0]
            self.assertEqual(pending["business_code"], "E_RENDER_BUDGET_PENDING")
            self.assertFalse(pending["result_is_latest"])
            self.assertLessEqual(pending["elapsed_ms"], pending["budget_ms"] + 1.0)
            self.assertIsNotNone(pending["latest_completed"])
            self.assertFalse(pending["latest_completed"]["params_match_requested"])
            self.assertIn("progress", pending)
            self.assertIn("affected_region", pending["progress"])
            self.assertIn("reused_stages", pending["latest_completed"]["audit"])
            self.assertEqual(sample_report["completed_command_cursor"], len(small.commands))

            final_path = output / "current" / "budget_pending" / "final.png"
            self.assertTrue(final_path.is_file())
            report_path = Path(report["report_path"])
            self.assertTrue(report_path.is_file())
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["summary"]["business_codes"]["E_RENDER_BUDGET_PENDING"], len(small.commands))

            params = small.config.params
            for command in small.commands:
                params = apply_command(params, small.config.width, small.config.height, command)
            reference = render_full_reference(small, root / "reference_cache", params)
            with Image.open(final_path) as opened:
                final_pixels = np.asarray(opened.convert("RGB"))
            self.assertTrue(np.array_equal(final_pixels, reference.final_pixels))

    def test_budget_large_enough_returns_ok_latest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_sample(
                root / "samples",
                "budget_ready",
                width=56,
                height=40,
                config_overrides={"interaction": {"latency_budget_ms": 5000}},
            )
            report = RunController(root / "samples", root / "output", mode="interactive").run()
            commands = report["samples"][0]["commands"]
            self.assertTrue(any(item["business_code"] == "OK" for item in commands))
            ready = next(item for item in commands if item["business_code"] == "OK")
            self.assertTrue(ready["result_is_latest"])
            self.assertTrue(ready["budget_met"])
            self.assertGreater(ready["invalidated_count"] + ready["reused_count"], 0)


if __name__ == "__main__":
    unittest.main()
