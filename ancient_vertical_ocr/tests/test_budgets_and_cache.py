import json
import shutil
import tempfile
import unittest
from pathlib import Path

from ancient_ocr.config import Config, write_default_config
from ancient_ocr.pipeline import Pipeline
from ancient_ocr.sample_gen import build_default_corpus


def _workspace(overrides=None):
    root = Path(tempfile.mkdtemp())
    (root / "config").mkdir()
    write_default_config(root / "config" / "config.json", overrides or {})
    cfg = Config.load(root / "config" / "config.json")
    samples = root / "samples"
    samples.mkdir()
    build_default_corpus(samples, cfg.frac_bits)
    return root, cfg


class BudgetTest(unittest.TestCase):
    def test_scan_budget_blocks_only_coarse_and_keeps_todos(self):
        root, cfg = _workspace({"scan_row_budget": 64, "compare_point_budget": 3000000})
        state = Pipeline(root / "samples", cfg, root / "snapshots").execute()
        todos = state["scheduler"]["todos"]
        self.assertTrue(todos)
        self.assertTrue(all(item["stage"] == "coarse" for item in todos))
        keys = [item["unit_key"] for item in todos]
        self.assertEqual(keys, sorted(keys))
        shutil.rmtree(root)

    def test_compare_budget_blocks_fine_stage_independently(self):
        root, cfg = _workspace({"scan_row_budget": 60000, "compare_point_budget": 100})
        state = Pipeline(root / "samples", cfg, root / "snapshots").execute()
        fine_todos = [item for item in state["scheduler"]["todos"] if item["stage"] == "fine"]
        self.assertTrue(fine_todos)
        for sample in state["samples"].values():
            ledger = sample["artifacts"]["fine"]["compare_ledger"]
            self.assertLessEqual(ledger["consumed"], cfg.compare_point_budget)
            for deduction in ledger["deductions"]:
                self.assertEqual(deduction["budget"], "compare_point")
        shutil.rmtree(root)

    def test_adjudication_budget_emits_business_code(self):
        root, cfg = _workspace({"adjudication_budget": 1})
        state = Pipeline(root / "samples", cfg, root / "snapshots").execute()
        pending = []
        for sample in state["samples"].values():
            merge = sample["artifacts"]["ordering"]["merge"]
            pending.extend(merge["pending_columns"])
            if merge["pending_columns"]:
                self.assertEqual(merge["business_code"], "E_ADJUDICATION_BUDGET")
        self.assertTrue(pending)
        shutil.rmtree(root)


class CacheInvalidationTest(unittest.TestCase):
    def test_keypoint_version_replacement_invalidates_downstream_only(self):
        root, cfg = _workspace()
        pipeline = Pipeline(root / "samples", cfg, root / "snapshots")
        pipeline.execute()
        before = {
            (e.sample_id, e.stage): e.cache_key for e in pipeline.cache.all_entries()}
        pipeline.apply_command({"op": "replace_keypoint_version",
                                "sample_id": "page_a", "version": "annot-v2"})
        state = pipeline.execute()
        after = {
            (e.sample_id, e.stage): e.cache_key for e in pipeline.cache.all_entries()}
        self.assertEqual(before[("page_a", "coarse")], after[("page_a", "coarse")])
        self.assertNotEqual(before[("page_a", "fine")], after[("page_a", "fine")])
        self.assertNotEqual(before[("page_a", "intra")], after[("page_a", "intra")])
        self.assertNotEqual(before[("page_a", "merge")], after[("page_a", "merge")])
        for other in ("page_b", "page_c"):
            for stage in ("coarse", "fine", "intra", "merge"):
                self.assertEqual(before[(other, stage)], after[(other, stage)])
        labels = set(state["invalidation"]["invalidated"])
        self.assertIn("page_a:fine", labels)
        self.assertNotIn("page_b:fine", labels)
        shutil.rmtree(root)

    def test_single_image_replacement_invalidates_that_sample_chain(self):
        root, cfg = _workspace()
        pipeline = Pipeline(root / "samples", cfg, root / "snapshots")
        pipeline.execute()
        before = {
            (e.sample_id, e.stage): e.cache_key for e in pipeline.cache.all_entries()}
        pipeline.apply_command({"op": "replace_image",
                                "sample_id": "page_b", "seed": 8088})
        pipeline.execute()
        after = {
            (e.sample_id, e.stage): e.cache_key for e in pipeline.cache.all_entries()}
        self.assertNotEqual(before[("page_b", "coarse")], after[("page_b", "coarse")])
        self.assertEqual(before[("page_a", "coarse")], after[("page_a", "coarse")])
        shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
