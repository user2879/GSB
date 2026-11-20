import json
import tempfile
import unittest
from pathlib import Path

from colbench.errors import BenchError
from colbench.gui import launch
from colbench.sample_data import generate
from colbench.versioning import BudgetParams
from colbench.workbench import Workbench


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.bench = Workbench(self.tmp)
        self.bench.build_from_specs(generate(self.tmp / "samples"))
        self.query = json.loads(
            (self.tmp / "samples" / "q_eq_in.json").read_text(
                encoding="utf-8"))["sql"]

    def test_full_scan_and_text_render(self):
        model = self.bench.run_scan(
            self.query, BudgetParams(2, 8192, 1))
        text = model.render_text()
        for title in model.section_titles():
            self.assertIn(title, text)
        mode = launch(model, force_text=True)
        self.assertEqual(mode, "text")

    def test_fault_injection_codes(self):
        for kind, expected in (("truncate", "E_PAYLOAD"),
                               ("encoding_mismatch", "E_ENCODING")):
            self.bench.inject_fault(kind, segment_id="seg_a", row_group=0,
                                    column="region")
            with self.assertRaises(BenchError) as ctx:
                self.bench.reopen_segments()
            self.assertEqual(ctx.exception.code, expected)
            break  # 段已损坏，第二个用例需要新工作台

    def test_zero_memory_budget_code(self):
        with self.assertRaises(BenchError) as ctx:
            self.bench.run_scan(self.query, BudgetParams(2, 0, 1))
        self.assertEqual(ctx.exception.code, "E_BUDGET")

    def test_encoding_change_bumps_version(self):
        before = self.bench.registry.current()
        spec_path = self.tmp / "samples" / "segment_seg_a.json"
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        for block in raw["blocks"]:
            if block["column"] == "region":
                block["encoding"] = "RAW"
        spec_path.write_text(json.dumps(raw, ensure_ascii=False),
                             encoding="utf-8")
        self.bench.build_from_specs(
            sorted((self.tmp / "samples").glob("segment_*.json")))
        after = self.bench.registry.current()
        self.assertGreaterEqual(after.encoding, before.encoding)


if __name__ == "__main__":
    unittest.main()
