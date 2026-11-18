"""样本生成与 PNG 编解码、加载校验测试（业务要求 1）。"""

import json
import os
import unittest

import numpy as np

from xmodal.errors import PipelineError
from xmodal.features import (extract_sample, list_samples,
                             load_sample_files)
from xmodal.png_codec import decode_png, encode_png
from xmodal.samplegen import generate_sample
from xmodal.tests._helpers import Workspace


class PngCodecTest(unittest.TestCase):
    def test_roundtrip_rgb_and_gray(self):
        rng = np.random.RandomState(7)
        rgb = rng.randint(0, 256, (20, 24, 3), dtype=np.uint8)
        gray = rng.randint(0, 256, (16, 16), dtype=np.uint8)
        decoded_rgb, size, _ = decode_png(encode_png(rgb))
        decoded_gray, size_gray, _ = decode_png(encode_png(gray))
        self.assertEqual(size, (24, 20))
        self.assertTrue(np.array_equal(decoded_rgb, rgb))
        self.assertTrue(np.array_equal(decoded_gray, gray))
        self.assertEqual(size_gray, (16, 16))

    def test_bad_signature_and_truncation(self):
        blob = encode_png(np.zeros((8, 8, 3), dtype=np.uint8))
        with self.assertRaises(PipelineError):
            decode_png(b"NOTPNG" + blob[6:])
        with self.assertRaises(PipelineError):
            decode_png(blob[: len(blob) // 2])

    def test_deterministic_bytes(self):
        arr = np.full((8, 8, 3), 120, dtype=np.uint8)
        self.assertEqual(encode_png(arr), encode_png(arr))


class SampleLoadingTest(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace(count=4)

    def tearDown(self):
        self.ws.cleanup()

    def test_list_and_extract_shapes(self):
        samples = list_samples(self.ws.input_dir)
        self.assertEqual(samples, ["sample_000", "sample_001",
                                   "sample_002", "sample_003"])
        with open(self.ws.config_path) as fh:
            cfg = json.load(fh)
        record = extract_sample(self.ws.input_dir, samples[0], cfg)
        self.assertEqual(record.sample, samples[0])
        self.assertEqual(record.image_blocks.shape, (16, 8))
        self.assertEqual(record.text_vector.shape, (32,))
        self.assertEqual(record.source_version, "raw")

    def test_missing_file_and_bad_size(self):
        sample = "sample_000"
        os.remove(os.path.join(self.ws.input_dir, sample, "text.json"))
        with self.assertRaises(PipelineError) as ctx:
            load_sample_files(self.ws.input_dir, sample, (64, 64))
        self.assertEqual(ctx.exception.code, "E_SAMPLE_INVALID")

    def test_missing_assertion_field(self):
        sample = "sample_001"
        path = os.path.join(self.ws.input_dir, sample, "text.json")
        with open(path) as fh:
            payload = json.load(fh)
        del payload["assertions"]
        with open(path, "w") as fh:
            json.dump(payload, fh)
        with open(self.ws.config_path) as fh:
            cfg = json.load(fh)
        with self.assertRaises(PipelineError) as ctx:
            extract_sample(self.ws.input_dir, sample, cfg)
        self.assertEqual(ctx.exception.code, "E_SAMPLE_INVALID")


class GeneratorDeterminismTest(unittest.TestCase):
    def test_fixed_seed_outputs_identical(self):
        img_a, regions_a, text_a = generate_sample(2, (64, 64))
        img_b, regions_b, text_b = generate_sample(2, (64, 64))
        self.assertTrue(np.array_equal(img_a, img_b))
        self.assertEqual(regions_a, regions_b)
        self.assertEqual(text_a, text_b)


if __name__ == "__main__":
    unittest.main()
