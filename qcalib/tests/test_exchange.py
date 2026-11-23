import json
import tempfile
import unittest
from pathlib import Path

from tests import _bootstrap  # noqa: F401
from qcalib.enums import V1_LINE_WIDTH
from qcalib.exchange import (
    CalibEntry,
    downgrade_v2_to_v1,
    encode_v1_line,
    parse_v1,
    parse_v2,
    read_sidecar,
    upgrade_v1_to_v2,
    write_sidecar,
)
from qcalib.fingerprint import version_id
from qcalib.exchange import _entry_fp_payload


def _v1_entry(permille=250, zp=0, qbits=8, scheme="symmetric"):
    e = CalibEntry(
        group_id="g0", column="f0", scale=permille / 1000.0,
        zero_point=zp, qbits=qbits, scheme=scheme, rounding="half_up",
        precision_label="permille-v1", per_channel=True, source_version=1,
        scale_permille=permille,
    )
    e.version = version_id(1, _entry_fp_payload(e))
    return e


class ExchangeTest(unittest.TestCase):
    def test_v1_fixed_width_roundtrip(self):
        entry = _v1_entry()
        line = encode_v1_line(entry)
        self.assertEqual(len(line), V1_LINE_WIDTH + 1)  # +方案码
        parsed, failures = parse_v1(line)
        self.assertEqual(failures.codes, [])
        self.assertEqual(parsed[0].scale_permille, 250)
        self.assertAlmostEqual(parsed[0].scale, 0.25)
        self.assertEqual(parsed[0].scheme, "symmetric")

    def test_field_truncated_and_scale_invalid(self):
        _, failures = parse_v1("g0")
        self.assertEqual(failures.codes, ["E_FIELD_TRUNCATED"])
        bad_v2 = json.dumps({"entries": [{
            "group_id": "g", "column": "c", "scale": float("nan"),
            "zero_point": 0, "qbits": 8, "scheme": "symmetric",
            "rounding": "half_up"}]})
        _, f2 = parse_v2(bad_v2)
        self.assertEqual(f2.codes, ["E_SCALE_INVALID"])

    def test_upgrade_downgrade_preserves_permille(self):
        v1 = [_v1_entry(250), _v1_entry(649, zp=10, scheme="asymmetric")]
        upgraded = upgrade_v1_to_v2(v1)
        self.assertEqual({e.source_version for e in upgraded}, {2})
        downgraded, sidecar = downgrade_v2_to_v1(upgraded)
        self.assertEqual(
            [e.scale_permille for e in downgraded],
            [e.scale_permille for e in v1],
        )
        # 旁路可再次解析，且携带残差/标签/舍入/逐通道
        with tempfile.TemporaryDirectory() as d:
            sp = Path(d) / "side.json"
            write_sidecar(sp, sidecar)
            reloaded = read_sidecar(sp)
            self.assertEqual(len(reloaded["entries"]), 2)
            row = reloaded["entries"][0]
            for key in ("truncation_residual", "precision_label", "rounding", "per_channel"):
                self.assertIn(key, row)
        self.assertTrue(sidecar["version"].startswith("v2-src2-"))

    def test_v2_native_downgrade_records_residual(self):
        v2 = CalibEntry(
            group_id="g", column="c", scale=0.25037, zero_point=0, qbits=8,
            scheme="symmetric", rounding="half_even",
            precision_label="float-native-v2", per_channel=True, source_version=2,
        )
        v2.version = version_id(2, _entry_fp_payload(v2))
        down, side = downgrade_v2_to_v1([v2])
        self.assertEqual(down[0].scale_permille, 250)  # 四舍五入
        self.assertAlmostEqual(side["entries"][0]["truncation_residual"], 0.00037, places=9)


if __name__ == "__main__":
    unittest.main()
