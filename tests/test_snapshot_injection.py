import os
import tempfile
import unittest

from fxpdiag.loader import apply_injection
from fxpdiag.errors import FailureList
from fxpdiag.models import InjectionItem
from fxpdiag.snapshots import can_resume, commit_snapshot, load_manifest
from tests.test_pipeline import make_package, make_weights


class SnapshotTests(unittest.TestCase):
    def test_atomic_commit_and_resume_decision(self):
        config = {"frac_bits": 6, "tier": "standard"}
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = os.path.join(tmp, "snap")
            commit_snapshot(
                snapshot_dir,
                {"stage1": {"value": 1}},
                {"config_version": 1, "config": config},
            )
            self.assertTrue(os.path.exists(os.path.join(snapshot_dir, "manifest.json")))
            self.assertTrue(
                can_resume(snapshot_dir, 1, config)
            )
            self.assertFalse(
                can_resume(snapshot_dir, 2, config)
            )
            self.assertFalse(
                can_resume(snapshot_dir, 1, {"frac_bits": 8, "tier": "fine"})
            )
            manifest = load_manifest(snapshot_dir)
            self.assertIn("config_hash", manifest)

    def test_commit_replaces_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = os.path.join(tmp, "snap")
            commit_snapshot(snapshot_dir, {"a": {"v": 1}},
                            {"config_version": 1, "config": {}})
            commit_snapshot(snapshot_dir, {"b": {"v": 2}},
                            {"config_version": 1, "config": {}})
            self.assertTrue(os.path.exists(os.path.join(snapshot_dir, "b.json")))
            self.assertFalse(os.path.exists(os.path.join(snapshot_dir, "a.json")))


class InjectionTests(unittest.TestCase):
    def setUp(self):
        self.package = make_package()
        self.v1, self.v2 = make_weights(self.package)

    def test_all_faults_isolated_and_coded(self):
        items = [
            InjectionItem("I1", "non_finite_scale", "embedding", 0),
            InjectionItem("I2", "zero_variance", "attention", 1),
            InjectionItem("I3", "bad_bitwidth", "embedding", 0,
                          {"bit_width": 13}),
            InjectionItem("I4", "duplicate_layer", "embedding"),
        ]
        failures = FailureList()
        reports = [
            apply_injection(self.package, self.v1, self.v2, item, failures)
            for item in items
        ]
        self.assertTrue(all(report["detected"] for report in reports))
        codes = [report["code"] for report in reports]
        self.assertIn("E_NON_FINITE", codes)
        self.assertIn("E_BITWIDTH_ENUM", codes)
        self.assertIn("E_DUPLICATE_LAYER", codes)
        self.assertIn("DEGENERATE_MARKED_AT_RUNTIME", codes)
        # 注入流程与正常权重隔离：原权重未被污染
        self.assertTrue(
            all(isinstance(row.scale, float) and row.scale == row.scale
                for row in self.v2.values())
        )
        inject_ids = [record.context["fault"] for record in failures]
        self.assertEqual(len(inject_ids), 4)


if __name__ == "__main__":
    unittest.main()
