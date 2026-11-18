"""v1/v2 交换格式读写、往返一致性与损失审计测试。"""

import json
import os
import struct
import unittest

import numpy as np

from xmodal.convert import (convert_v1_to_v2, convert_v2_to_v1,
                             cross_format_deviation, sidecar_path)
from xmodal.errors import PipelineError
from xmodal.features import FeatureRecord
from xmodal.formats import (HEADER_SIZE, V1_VERSION, V2_VERSION, read_pack,
                            read_v1, read_v2, write_v1, write_v2)
from xmodal.tests._helpers import Workspace


def make_record(sample="s1", text_dim=12):
    rng = np.random.RandomState(11)
    return FeatureRecord(
        sample=sample,
        image_blocks=rng.rand(6, 7).astype(np.float32),
        text_vector=rng.rand(text_dim).astype(np.float32),
        align_weights={"mean": 0.35, "gradient": 0.25,
                       "orientation": 0.25, "text": 0.15},
        source_version=V1_VERSION,
        precision_labels={"image": "native-float64"})


class FormatRoundTripTest(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace()
        self.v1 = os.path.join(self.ws.root, "a.xmf1")
        self.v2 = os.path.join(self.ws.root, "a.json")

    def tearDown(self):
        self.ws.cleanup()

    def test_v1_roundtrip_field_by_field(self):
        record = make_record()
        write_v1(self.v1, record)
        back = read_v1(self.v1, expected_sample="s1")
        self.assertEqual(back.sample, "s1")
        np.testing.assert_array_equal(back.image_blocks, record.image_blocks)
        np.testing.assert_array_equal(back.text_vector, record.text_vector)
        self.assertEqual(back.source_version, V1_VERSION)
        for key, value in record.align_weights.items():
            self.assertAlmostEqual(back.align_weights[key], value, places=5)

    def test_v2_roundtrip_and_dispatch(self):
        record = make_record()
        write_v2(self.v2, record)
        back = read_v2(self.v2)
        np.testing.assert_allclose(back.image_blocks, record.image_blocks,
                                   rtol=1e-6)
        self.assertEqual(back.source_version, V2_VERSION)
        self.assertIsInstance(read_pack(self.v2), FeatureRecord)
        self.assertIsInstance(read_pack(self.v1) if False else
                              read_pack(self._write_v1(record)),
                              FeatureRecord)

    def _write_v1(self, record):
        path = os.path.join(self.ws.root, "b.xmf1")
        write_v1(path, record)
        return path

    def test_cross_format_same_ir_identical_blocks(self):
        record = make_record()
        write_v1(self.v1, record)
        write_v2(self.v2, record)
        a, b = read_v1(self.v1), read_v2(self.v2)
        np.testing.assert_array_equal(a.image_blocks, b.image_blocks)
        np.testing.assert_array_equal(a.text_vector, b.text_vector)
        summary = cross_format_deviation({"s1": a}, {"s1": b})
        self.assertTrue(summary[0]["name_restored"])
        self.assertEqual(summary[0]["image_max_abs"], 0.0)

    def test_conversion_v1_to_v2_labels_and_sidecar(self):
        record = make_record()
        write_v1(self.v1, record)
        sidecar = convert_v1_to_v2(self.v1, self.v2)
        self.assertEqual(sidecar["direction"], "v1_to_v2")
        self.assertEqual(sidecar["precision_labels"]["image"],
                         "float32-roundtrip")
        self.assertTrue(os.path.isfile(sidecar_path(self.v2)))
        back = read_v2(self.v2)
        self.assertEqual(back.sample, record.sample)

    def test_conversion_v2_to_v1_residual_sidecar(self):
        record = make_record()
        write_v2(self.v2, record)
        sidecar = convert_v2_to_v1(self.v2, self.v1)
        self.assertEqual(sidecar["source"], "v2")
        self.assertIn("truncation_residual", sidecar)
        self.assertLessEqual(sidecar["truncation_residual"]
                             ["image_max_abs"], 1e-6)

    def test_lossy_out_of_float32_range(self):
        record = FeatureRecord(
            sample="big",
            image_blocks=np.array([[0.1] * 7], dtype=np.float64),
            text_vector=np.array([1e40, 0.2], dtype=np.float64),
            align_weights={"mean": 1.0, "gradient": 0.0,
                           "orientation": 0.0, "text": 0.0},
            source_version=V2_VERSION)
        write_v2(self.v2, record)
        with self.assertRaises(PipelineError) as ctx:
            convert_v2_to_v1(self.v2, self.v1)
        self.assertEqual(ctx.exception.code, "E_FORMAT_LOSSY")
        self.assertIn("text.vector[0]", str(ctx.exception))

    def test_unknown_version_magic_rejected(self):
        write_v1(self.v1, make_record())
        blob = bytearray(open(self.v1, "rb").read())
        blob[8:10] = struct.pack("<H", 99)
        open(self.v1, "wb").write(blob)
        with self.assertRaises(PipelineError) as ctx:
            read_v1(self.v1)
        self.assertEqual(ctx.exception.code, "E_SAMPLE_INVALID")

    def test_offset_table_out_of_bounds(self):
        write_v1(self.v1, make_record())
        blob = bytearray(open(self.v1, "rb").read())
        blob[HEADER_SIZE:HEADER_SIZE + 4] = struct.pack(
            "<I", len(blob) + 4096)
        open(self.v1, "wb").write(blob)
        with self.assertRaises(PipelineError) as ctx:
            read_v1(self.v1)
        self.assertEqual(ctx.exception.code, "E_SAMPLE_INVALID")

    def test_non_finite_float_rejected(self):
        write_v1(self.v1, make_record())
        blob = bytearray(open(self.v1, "rb").read())
        (image_off,) = struct.unpack("<I", bytes(
            blob[HEADER_SIZE:HEADER_SIZE + 4]))
        nan = struct.unpack("<I", np.float32(np.nan).tobytes())[0]
        blob[image_off:image_off + 4] = struct.pack("<I", nan)
        open(self.v1, "wb").write(blob)
        with self.assertRaises(PipelineError) as ctx:
            read_v1(self.v1)
        self.assertEqual(ctx.exception.code, "E_SAMPLE_INVALID")


if __name__ == "__main__":
    unittest.main()
