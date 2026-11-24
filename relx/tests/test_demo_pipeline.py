import json
import os
import shutil
import tempfile
import unittest

from relx.pipeline import run_demo


class DemoPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="relx_demo_")
        cls.data = os.path.join(cls.tmp, "data")
        cls.output = os.path.join(cls.tmp, "output")
        cls.summary = run_demo(cls.data, cls.output)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_summary_flags(self):
        self.assertTrue(self.summary["incremental_vs_full_identical"])
        self.assertTrue(self.summary["concurrency_identical"])
        self.assertTrue(self.summary["full_rebuild"]["ledger_consistent"])
        self.assertEqual(self.summary["full_rebuild"]["sentence_results"], 24)

    def test_output_artifacts_written(self):
        for name in ("full_rebuild.json", "incremental_final.json",
                     "consistency_check.json", "concurrency_check.json",
                     "quant_variants.json", "summary.json"):
            path = os.path.join(self.output, name)
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as handle:
                json.load(handle)

    def test_repeated_runs_are_byte_level_deterministic(self):
        import tempfile
        other_output = os.path.join(self.tmp, "output2")
        run_demo(self.data, other_output)
        for name in ("full_rebuild.json", "incremental_final.json",
                     "summary.json", "quant_variants.json"):
            with open(os.path.join(self.output, name), encoding="utf-8") as first,                  open(os.path.join(other_output, name), encoding="utf-8") as second:
                self.assertEqual(first.read(), second.read(), name)

    def test_no_entity_sentences_retained(self):
        with open(os.path.join(self.output, "full_rebuild.json"),
                  encoding="utf-8") as handle:
            report = json.load(handle)
        ids = {item["sentence_id"] for item in report["sentences"]}
        self.assertIn("S23", ids)
        self.assertIn("S24", ids)


if __name__ == "__main__":
    unittest.main()
