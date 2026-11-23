import copy
import shutil
import tempfile
import unittest
from pathlib import Path

from tests import _bootstrap  # noqa: F401
from qcalib import CalibrationRunner, load_calibration_config, load_feature_table

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


class InvalidationTest(unittest.TestCase):
    def setUp(self):
        self.table, _ = load_feature_table(SAMPLES / "feature_table.jsonl")
        self.cfg, _ = load_calibration_config(SAMPLES / "groups_config.json", self.table)
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_qbits_switch_only_invalidates_changed_group(self):
        work = Path(self.tmp) / "run"
        CalibrationRunner(self.table, self.cfg, work).run(force_full=True)
        new_cfg = copy.deepcopy(self.cfg)
        target = new_cfg.group_by_id["g2"]
        target.qbits = 16
        target.params_fingerprint = "changed-qbits"
        for group in new_cfg.groups:
            group.params_fingerprint = (
                "changed-qbits" if group.group_id == "g2" else group.params_fingerprint
            )
        report = CalibrationRunner(self.table, new_cfg, work).run()
        # 仅 g2 的三列失效（9 键），g0/g1 六列复用
        self.assertEqual(len(report["invalidated"]), 9)
        self.assertEqual(len(report["reused"]), 6)

    def test_group_partition_change_invalidates_that_group(self):
        work = Path(self.tmp) / "run2"
        CalibrationRunner(self.table, self.cfg, work).run(force_full=True)
        new_cfg = copy.deepcopy(self.cfg)
        g1 = new_cfg.group_by_id["g1"]
        g1.columns = g1.columns[:-1]  # 调整组内列集合
        g1.params_fingerprint = "partition-changed"
        for group in new_cfg.groups:
            if group.group_id == "g1":
                group.params_fingerprint = "partition-changed"
        report = CalibrationRunner(self.table, new_cfg, work).run()
        # 失效包含 g1 剩余 2 列的新键与被移除列的旧键（整组失效）
        self.assertGreaterEqual(len(report["invalidated"]), 6)
        # g0(3) + g2(3) 共 6 列不受组划分调整影响，保持复用
        self.assertEqual(len(report["reused"]), 6)
        # 重算仅发生在 g1 剩余 2 列（每列 1 个标定键）
        self.assertEqual(len(report["recomputed"]), 2)
        g1_cols = {c["column"] for c in report["groups"]["g1"]}
        self.assertEqual(g1_cols, set(g1.columns))


if __name__ == "__main__":
    unittest.main()
