from __future__ import annotations

import importlib.util
import unittest

from memdiag.engine import run_initial
from memdiag.gui import GuiUnavailable, run_gui
from memdiag.viewmodel import build_view_model, format_layer_chain
from tests.helpers import fixture_plan, fixture_samples


class ViewModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = run_initial(fixture_plan(), fixture_samples())

    def test_view_model_has_stable_topological_order_and_blocks(self):
        view = build_view_model(self.baseline)
        self.assertEqual(
            [layer["layer_id"] for layer in view["layers"]],
            list(self.baseline.layers),
        )
        self.assertEqual(
            [block["index"] for block in view["blocks"]],
            list(range(self.baseline.pool_capacity_blocks)),
        )
        self.assertEqual(
            sorted(entry["rank"] for entry in view["error_chain"]),
            list(range(1, len(view["error_chain"]) + 1)),
        )
        for layer in view["layers"]:
            self.assertIn(" -> ", format_layer_chain(layer))

    def test_gui_module_imports_without_tk_runtime(self):
        self.assertIsNotNone(importlib.util.find_spec("memdiag.gui"))
        if importlib.util.find_spec("tkinter") is None:
            with self.assertRaises(GuiUnavailable):
                run_gui()


if __name__ == "__main__":
    unittest.main()
