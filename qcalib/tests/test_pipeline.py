import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tests import _bootstrap  # noqa: F401
from qcalib import CalibrationRunner, load_calibration_config, load_feature_table
from qcalib.pipeline import Pipeline
from qcalib.snapshot import version_changed
from qcalib.store import StageStore

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table, _ = load_feature_table(SAMPLES / "feature_table.jsonl")
        cls.cfg, _ = load_calibration_config(SAMPLES / "groups_config.json", cls.table)

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.work = Path(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _runner(self, **kw):
        return CalibrationRunner(self.table, self.cfg, self.work / "run", **kw)

    def test_full_then_noop_reuses_all(self):
        runner = self._runner(max_workers=4)
        runner.run(force_full=True)
        second = self._runner(max_workers=1).run()
        self.assertEqual(len(second["recomputed"]), 0)
        self.assertEqual(len(second["reused"]), len(self.cfg.groups) * 3)

    def test_single_row_edit_only_invalidates_owning_group(self):
        self._runner().run(force_full=True)
        edited, _ = load_feature_table(SAMPLES / "feature_table.jsonl")
        edited.values[edited.row_ids[2]]["f0_1"] = 3.14  # 仅属于 g0
        report = CalibrationRunner(edited, self.cfg, self.work / "run").run()
        # g0 三列的三个阶段键失效（9 个），其余 6 列复用
        self.assertEqual(len(report["invalidated"]), 9)
        self.assertEqual(len(report["reused"]), 6)
        # 失效与复用清单顺序稳定（已排序）
        self.assertEqual(report["invalidated"], sorted(report["invalidated"]))
        self.assertEqual(report["reused"], sorted(report["reused"]))

    def test_incremental_matches_fresh_full_field_by_field(self):
        self._runner().run(force_full=True)
        edited, _ = load_feature_table(SAMPLES / "feature_table.jsonl")
        edited.values[edited.row_ids[1]]["f1_0"] = -2.71
        inc = CalibrationRunner(edited, self.cfg, self.work / "run").run()
        fresh = CalibrationRunner(
            edited, self.cfg, self.work / "fresh"
        ).run(force_full=True)
        self.assertEqual(
            json.dumps(inc["groups"], sort_keys=True),
            json.dumps(fresh["groups"], sort_keys=True),
        )
        m1 = json.loads((self.work / "run/snapshot/manifest.json").read_text())
        m2 = json.loads((self.work / "fresh/snapshot/manifest.json").read_text())
        self.assertEqual(m1["content_fingerprint"], m2["content_fingerprint"])

    def test_budget_pending_is_stable(self):
        store = StageStore(self.work / "cache")
        costs = [
            sum(1 for v in self.table.column_vector(c) if v is not None)
            for g in self.cfg.groups for c in g.columns
        ]
        pipe = Pipeline(self.table, self.cfg, store,
                        element_budget=sum(sorted(costs)[:2]), max_workers=4)
        result = pipe.run(force_full=True)
        self.assertTrue(result.failures.has("E_CALIB_BUDGET_PENDING"))
        self.assertEqual(len(result.pending), len(costs) - 2)
        keys = [tuple(p["sort_key"]) for p in result.pending]
        self.assertEqual(keys, sorted(keys))

    def test_parallel_equals_serial(self):
        def run(workers):
            store = StageStore(self.work / f"w{workers}")
            return Pipeline(self.table, self.cfg, store, max_workers=workers).run(force_full=True)
        parallel = run(8)
        serial = run(1)
        payload = lambda r: json.dumps(
            {f"{g}|{c}": [r.calibration[(g, c)].to_dict(),
                          r.rescale[(g, c)].to_dict(),
                          r.errors[(g, c)].to_dict()]
             for g, c in sorted(r.calibration)}, sort_keys=True, default=str)
        self.assertEqual(payload(parallel), payload(serial))

    def test_snapshot_version_detection(self):
        runner = self._runner()
        report = runner.run(force_full=True)
        snap = self.work / "run/snapshot"
        self.assertFalse(version_changed(snap, report["snapshot_fingerprint"]))
        self.assertTrue(version_changed(snap, "deadbeefdeadbeef"))


if __name__ == "__main__":
    unittest.main()
