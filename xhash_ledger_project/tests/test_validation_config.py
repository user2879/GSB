"""业务要求 1：样本、配置与命令流的加载校验。"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xhash_ledger.config import parse_command, parse_config
from xhash_ledger.generator import generate_sample, write_sample
from xhash_ledger.validation import scan_samples


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.sample_dir = os.path.join(self.tmp, "samples")
        sample = generate_sample(1234, "s001")
        write_sample(self.sample_dir, "s001", sample)
        self.sample = sample

    def test_valid_sample_passes(self):
        bundles, failures = scan_samples(self.sample_dir, (64, 64))
        self.assertIn("s001", bundles)
        self.assertEqual(failures, [])

    def test_region_out_of_bounds_fails_without_stopping_others(self):
        bad = generate_sample(1235, "s002")
        bad["regions"][0]["x"] = 100
        write_sample(self.sample_dir, "s002", bad)
        bundles, failures = scan_samples(self.sample_dir, (64, 64))
        self.assertIn("s001", bundles)
        self.assertNotIn("s002", bundles)
        self.assertTrue(any(f.code == "E_SAMPLE_INVALID" and "regions" in f.field for f in failures))

    def test_region_non_integer_fails(self):
        bad = generate_sample(1236, "s003")
        bad["regions"][0]["x"] = 1.5
        write_sample(self.sample_dir, "s003", bad)
        _, failures = scan_samples(self.sample_dir, (64, 64))
        self.assertTrue(any("x" in f.field for f in failures))

    def test_controlled_vocabulary_violation(self):
        bad = generate_sample(1237, "s004")
        bad["assertion"]["sample_id"] = "s004"
        bad["assertion"]["entities"][0]["type"] = "hexagon"
        write_sample(self.sample_dir, "s004", bad)
        _, failures = scan_samples(self.sample_dir, (64, 64))
        self.assertTrue(any("entities" in f.field for f in failures))

    def test_broken_png_header(self):
        bad = generate_sample(1238, "s005")
        write_sample(self.sample_dir, "s005", bad)
        with open(os.path.join(self.sample_dir, "s005.png"), "r+b") as fh:
            fh.seek(0)
            fh.write(b"\x00" * 8)
        _, failures = scan_samples(self.sample_dir, (64, 64))
        self.assertTrue(any(f.field == "png" for f in failures))

    def test_resolution_mismatch(self):
        bundles, failures = scan_samples(self.sample_dir, (32, 32))
        self.assertNotIn("s001", bundles)
        self.assertTrue(any("resolution" in f.field for f in failures))


class ConfigTests(unittest.TestCase):
    def base(self, **overrides):
        from xhash_ledger.config import Config
        raw = Config().to_dict()
        raw.update(overrides)
        return raw

    def test_signature_bits_must_be_multiple_of_8(self):
        _, failures = parse_config(self.base(signature_bits=12))
        self.assertTrue(any(f.field == "signature_bits" for f in failures))
        self.assertTrue(all(f.code == "E_CONFIG_INVALID" for f in failures))

    def test_bucket_bits_less_than_signature_bits(self):
        _, failures = parse_config(self.base(signature_bits=16, bucket_bits=16))
        self.assertTrue(any(f.field == "bucket_bits" for f in failures))

    def test_similarity_range(self):
        _, failures = parse_config(self.base(similarity_permille=1001))
        self.assertTrue(any(f.field == "similarity_permille" for f in failures))

    def test_seed_version_declared(self):
        _, failures = parse_config(self.base(seed_version="seed-v9"))
        self.assertTrue(any(f.field == "seed_version" for f in failures))

    def test_bucket_capacity_positive(self):
        _, failures = parse_config(self.base(bucket_capacity=0))
        self.assertTrue(any(f.field == "bucket_capacity" for f in failures))

    def test_valid_config(self):
        config, failures = parse_config(self.base())
        self.assertEqual(failures, [])
        self.assertEqual(config.config_hash(), config.config_hash())

    def test_command_validation(self):
        failures = parse_command({"cmd": "explode", "sample_id": "s1"}, ("seed-v1",))
        self.assertTrue(failures)
        self.assertEqual(parse_command({"cmd": "add", "sample_id": "s1"}, ("seed-v1",)), [])


if __name__ == "__main__":
    unittest.main()
