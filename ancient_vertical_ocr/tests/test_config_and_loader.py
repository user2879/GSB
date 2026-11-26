import json
import shutil
import tempfile
import unittest
from pathlib import Path

from ancient_ocr.config import Config, DEFAULT_CONFIG, write_default_config
from ancient_ocr.errors import E_CONFIG_INVALID, E_SAMPLE_INVALID
from ancient_ocr.loader import load_corpus
from ancient_ocr.pipeline import bootstrap_demo_workspace


class ConfigLoaderTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "config").mkdir()
        write_default_config(self.root / "config" / "config.json")
        bootstrap_demo_workspace(self.root, Config.load(
            self.root / "config" / "config.json"))

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_window_must_divide_page(self):
        bad = dict(DEFAULT_CONFIG, window_size=30)
        with self.assertRaises(Exception) as ctx:
            Config.from_dict(bad)
        self.assertEqual(ctx.exception.issue.code, E_CONFIG_INVALID)

    def test_budgets_must_be_positive_int(self):
        for key in ("scan_row_budget", "compare_point_budget", "adjudication_budget"):
            with self.assertRaises(Exception):
                Config.from_dict(dict(DEFAULT_CONFIG, **{key: 0}))

    def test_bad_samples_are_isolated(self):
        result = load_corpus(self.root / "samples")
        codes = {issue.sample_id: issue.code for issue in result.issues}
        self.assertEqual(codes["bad_oob"], E_SAMPLE_INVALID)
        self.assertEqual(codes["bad_cycle"], E_SAMPLE_INVALID)
        self.assertEqual(codes["bad_header"], E_SAMPLE_INVALID)
        self.assertEqual(set(result.samples), {"page_a", "page_b", "page_c"})
        for issue in result.issues:
            self.assertTrue(issue.location)


if __name__ == "__main__":
    unittest.main()
