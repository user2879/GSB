import json
import unittest
from pathlib import Path
import tempfile

import yaml

from mm_pipeline.config import build_config, parse_config, DEFAULT_CONFIG
from mm_pipeline.errors import (
    E_CONFIG_INVALID,
    E_IMAGE_FORMAT,
    E_ASSERTION_SCHEMA,
    E_RELATION_REF,
)
from mm_pipeline.loader import load_samples
from mm_pipeline.errors import PipelineError

from .fixtures import build_corpus, make_png


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mm-load-"))
        self.samples, self.config_path = build_corpus(self.tmp, count=3)
        self.config = parse_config(self.config_path)

    def test_happy_path_loads_all_samples(self):
        failures = []
        samples = load_samples(self.samples, self.config, failures=failures)
        self.assertEqual(len(samples), 3)
        self.assertEqual(failures, [])
        self.assertTrue(all(s.image_hash and s.assertion_hash for s in samples))

    def test_truncated_png_is_classified(self):
        bad = self.samples / "sample_000" / "image.png"
        bad.write_bytes(bad.read_bytes()[:12])
        failures = []
        samples = load_samples(self.samples, self.config, failures=failures)
        self.assertEqual(len(samples), 2)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].code, E_IMAGE_FORMAT)
        self.assertEqual(failures[0].sample, "sample_000")

    def test_wrong_size_is_image_format(self):
        target = self.samples / "sample_001"
        (target / "image.png").write_bytes(make_png(2, size=(128, 128)))
        failures = []
        load_samples(self.samples, self.config, failures=failures)
        self.assertEqual(failures[0].code, E_IMAGE_FORMAT)

    def test_missing_assertion_field(self):
        path = self.samples / "sample_000" / "assertions.json"
        doc = json.loads(path.read_text())
        del doc["assertions"][0]["value"]
        path.write_text(json.dumps(doc))
        failures = []
        load_samples(self.samples, self.config, failures=failures)
        self.assertEqual(failures[0].code, E_ASSERTION_SCHEMA)

    def test_dangling_relation_region_and_assertion(self):
        path = self.samples / "sample_000" / "relations.json"
        doc = json.loads(path.read_text())
        doc["relations"][0]["region"] = 99
        path.write_text(json.dumps(doc))
        failures = []
        load_samples(self.samples, self.config, failures=failures)
        self.assertEqual(failures[0].code, E_RELATION_REF)

        doc = json.loads(path.read_text())
        doc["relations"][0]["region"] = 0
        doc["relations"][0]["assertion_id"] = "ghost"
        path.write_text(json.dumps(doc))
        failures = []
        load_samples(self.samples, self.config, failures=failures)
        self.assertEqual(failures[0].code, E_RELATION_REF)

    def test_other_samples_continue_after_failure(self):
        bad = self.samples / "sample_000" / "image.png"
        bad.write_bytes(b"not-a-png")
        failures = []
        samples = load_samples(self.samples, self.config, failures=failures)
        self.assertEqual({s.sample_id for s in samples},
                         {"sample_001", "sample_002"})
        self.assertEqual(len(failures), 1)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mm-cfg-"))

    def _write(self, overrides):
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(overrides)
        path = self.tmp / "cfg.yaml"
        path.write_text(yaml.safe_dump(cfg))
        return path

    def test_valid_defaults(self):
        cfg = build_config(dict(DEFAULT_CONFIG))
        self.assertEqual(cfg.region_count, cfg.grid_cols * cfg.grid_rows)
        self.assertIsInstance(cfg.fingerprint(), str)

    def test_invalid_enums_and_ranges(self):
        cases = [
            {"visual_dim": 0},
            {"text_dim": -1},
            {"quant_level": 13},
            {"frac_bits": 12},
            {"quant_level": 16, "frac_bits": 8},
            {"confidence_threshold": 1.5},
            {"fusion_temperature": 0.0},
            {"fusion_temperature": float("inf")},
            {"max_workers": 0},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(PipelineError) as ctx:
                    parse_config(self._write(overrides))
                self.assertEqual(ctx.exception.code, E_CONFIG_INVALID)

    def test_nan_temperature_rejected(self):
        path = self._write({"fusion_temperature": float("nan")})
        with self.assertRaises(PipelineError) as ctx:
            parse_config(path)
        self.assertEqual(ctx.exception.code, E_CONFIG_INVALID)


if __name__ == "__main__":
    unittest.main()
