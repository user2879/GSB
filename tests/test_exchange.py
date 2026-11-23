import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvrb.convert import downgrade_v2_to_v1, upgrade_v1_to_v2
from mvrb.errors import E_FIELD_TRUNCATED
from mvrb.exchange_v1 import parse_v1, read_v1, serialize_v1
from mvrb.exchange_v2 import parse_v2, read_v2, serialize_v2
from mvrb.utils import canonical_json

SAMPLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "samples", "demo")


class ExchangeTests(unittest.TestCase):
    def test_v1_parse_fixed_width_and_truncation(self):
        ir = read_v1(os.path.join(SAMPLE, "result_v1.txt"))
        self.assertGreater(len(ir.features), 0)
        self.assertGreater(len(ir.matches), 0)
        self.assertGreater(len(ir.voxel_points), 0)
        # 截断最后一条记录 -> E_FIELD_TRUNCATED
        text = serialize_v1(ir).splitlines()
        text[-1] = text[-1][:-3]
        broken = parse_v1("\n".join(text) + "\n")
        codes = [f.code for f in broken.failures.sorted()]
        self.assertIn(E_FIELD_TRUNCATED, codes)

    def test_v1_field_misalignment_detected(self):
        text = serialize_v1(read_v1(os.path.join(SAMPLE, "result_v1.txt")))
        lines = text.splitlines()
        # 标签错位
        lines[3] = "X|" + lines[3][2:]
        ir = parse_v1("\n".join(lines) + "\n")
        self.assertTrue(any(f.code == E_FIELD_TRUNCATED for f in ir.failures.sorted()))

    def test_v2_roundtrip_lossless(self):
        ir = read_v2(os.path.join(SAMPLE, "result_v2.json"))
        back = parse_v2(serialize_v2(ir))
        self.assertEqual(canonical_json(serialize_v2(ir)),
                         canonical_json(serialize_v2(back)))

    def test_v1_v2_roundtrip_with_sidecar_exact(self):
        ir_v2 = read_v2(os.path.join(SAMPLE, "result_v2.json"))
        ir_v1, sidecar = downgrade_v2_to_v1(ir_v2)
        # v1 文本往返
        v1_text = serialize_v1(ir_v1)
        v1_back = parse_v1(v1_text)
        restored = upgrade_v1_to_v2(v1_back, sidecar)
        self.assertEqual(canonical_json(serialize_v2(restored)),
                         canonical_json(serialize_v2(ir_v2)))

    def test_upgrade_without_sidecar_keeps_integer_exactness(self):
        ir_v2 = read_v2(os.path.join(SAMPLE, "result_v2.json"))
        ir_v1, _ = downgrade_v2_to_v1(ir_v2)
        upgraded = upgrade_v1_to_v2(parse_v1(serialize_v1(ir_v1)))
        for f in upgraded.features:
            self.assertEqual(f.x, int(f.x))
            self.assertEqual(f.y, int(f.y))
        self.assertTrue(all(m.metric in ("ncc", "ssd", "sift_ratio", "orb_hamming")
                            for m in upgraded.matches))

    def test_validation_ranges(self):
        ir = read_v2(os.path.join(SAMPLE, "result_v2.json"))
        ir.features[0].x = ir.image_size[0] + 5
        ir.features[0].y = -1
        ir.matches[0].score_permille = 1200
        fails = ir.validate()
        codes = {f.code for f in fails.sorted()}
        self.assertIn("E_COORD_OUT_OF_RANGE", codes)
        self.assertIn("E_SCORE_OUT_OF_RANGE", codes)


if __name__ == "__main__":
    unittest.main()
