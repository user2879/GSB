import json
import tempfile
import unittest
from pathlib import Path

from tests import _bootstrap  # noqa: F401
from qcalib.config import load_calibration_config, load_feature_table
from qcalib.errors import (
    E_CONFIG_INVALID,
    E_DUPLICATE_ROW,
    E_EMPTY_GROUP,
    E_GROUP_UNKNOWN,
    E_MISSING_COLUMN,
)


def _write(path: Path, name: str, payload) -> Path:
    target = path / name
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return target


class ConfigValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        rows = [
            {"row_id": "r1", "values": {"a": 1.0, "b": 2.0}},
            {"row_id": "r2", "values": {"a": None, "b": -1.0}},
            {"row_id": "r2", "values": {"a": 0.5, "b": 0.5}},  # 重复行
            {"row_id": "r3", "values": {"a": "nan-x", "b": 0.0}},  # 非法取值
        ]
        ft = self.root / "table.jsonl"
        ft.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        self.table, self.ft_failures = load_feature_table(ft)

    def tearDown(self):
        self.tmp.cleanup()

    def test_duplicate_row_and_bad_value_keep_loading(self):
        codes = self.ft_failures.codes
        self.assertIn(E_DUPLICATE_ROW, codes)
        self.assertIn(E_CONFIG_INVALID, codes)
        # 非法行不中止整体加载：r1/r2/r3 仍被读入（重复的第二个 r2 被丢弃）
        self.assertEqual(self.table.row_ids, ["r1", "r2", "r3"])

    def test_unknown_group_missing_column_empty_group(self):
        payload = {
            "partition": ["g_ok"],
            "groups": [
                {"group_id": "g_unknown", "columns": ["a"], "q_lo_permille": 10,
                 "q_hi_permille": 900, "qbits": 8, "fbits": 2},
                {"group_id": "g_ok", "columns": ["nope"], "q_lo_permille": 10,
                 "q_hi_permille": 900, "qbits": 8, "fbits": 2},
                {"group_id": "g_ok", "columns": [], "q_lo_permille": 10,
                 "q_hi_permille": 900, "qbits": 8, "fbits": 2},
            ],
        }
        path = _write(self.root, "cfg.json", payload)
        _, failures = load_calibration_config(path, self.table)
        codes = failures.codes
        self.assertIn(E_GROUP_UNKNOWN, codes)
        self.assertIn(E_MISSING_COLUMN, codes)
        self.assertIn(E_EMPTY_GROUP, codes)

    def test_quantile_and_bitwidth_enums(self):
        payload = {
            "partition": ["g1", "g2", "g3"],
            "groups": [
                {"group_id": "g1", "columns": ["a"], "q_lo_permille": 900,
                 "q_hi_permille": 100, "qbits": 8, "fbits": 2},
                {"group_id": "g2", "columns": ["b"], "q_lo_permille": 10,
                 "q_hi_permille": 900, "qbits": 5, "fbits": 2},
                {"group_id": "g3", "columns": ["a"], "q_lo_permille": 10,
                 "q_hi_permille": 900, "qbits": 8, "fbits": 0},
            ],
        }
        path = _write(self.root, "cfg2.json", payload)
        cfg, failures = load_calibration_config(path, self.table)
        codes = failures.codes
        self.assertEqual(codes.count(E_CONFIG_INVALID), 2)
        self.assertIn("E_ENUM_INVALID", codes)
        self.assertEqual(cfg.groups, [])


    def test_all_null_column_is_registered(self):
        rows = [
            {"row_id": "r1", "values": {"a": 1.0, "b": None}},
            {"row_id": "r2", "values": {"a": 2.0, "b": None}},
        ]
        ft = self.root / "nulls.jsonl"
        ft.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        table, _ = load_feature_table(ft)
        self.assertIn("b", table.columns)
        payload = {
            "partition": ["g"],
            "groups": [{"group_id": "g", "columns": ["a", "b"],
                        "q_lo_permille": 25, "q_hi_permille": 975,
                        "qbits": 8, "fbits": 2}],
        }
        cfg_path = _write(self.root, "null_cfg.json", payload)
        cfg, failures = load_calibration_config(cfg_path, table)
        self.assertEqual(failures.codes, [])
        self.assertEqual(cfg.groups[0].columns, ["a", "b"])
        with self.assertRaises(ValueError):
            from qcalib.calibration import calibrate_column
            calibrate_column(
                cfg.groups[0], "b", table.column_vector("b"), table.row_ids
            )


if __name__ == "__main__":
    unittest.main()
