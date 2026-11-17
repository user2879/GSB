"""验收测试：转换保真、编辑后增量与全量一致、异常注入与快照恢复。"""
from __future__ import annotations

import json
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import formats
from app.errors import (
    E_FIELD_WIDTH,
    E_MERGE_CONFLICT,
    E_PARTITION_OOB,
    E_PRECISION_MISSING,
    E_SNAPSHOT_STALE,
    E_STEP_MAP_MISSING,
    E_STEP_UNKNOWN,
    E_VERSION_UNSUPPORTED,
)
from app.ir import Step
from app.report import write_report
from app.runner import PipelineRunner


def make_workspace() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="rp_ws_"))
    shutil.copytree(ROOT / "data", tmp / "data")
    shutil.copytree(ROOT / "config", tmp / "config")
    return tmp


def edit_sequence() -> list[dict]:
    return [
        {"op": "modify_params", "index": 0,
         "params": {"features": ["f_age", "f_click", "f_price", "f_depth"]}},
        {"op": "add_step", "index": 3,
         "step": {"name": "bucket",
                  "params": {"feature": "f_depth", "boundaries": [0.5],
                             "output": "f_depth_bucket"}}},
        {"op": "reorder_steps", "order": [0, 2, 1, 3]},
        {"op": "replace_entity", "entity": "ent_001",
         "features": {"f_age": 0.5, "f_click": 0.25, "f_price": 0.75,
                      "f_ctr": 0.4, "f_depth": 0.6}},
        {"op": "merge_entities", "source": "ent_010", "target": "ent_011"},
        {"op": "remove_step", "index": 3},
    ]


class TestParsers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = make_workspace()

    def test_three_versions_parse_to_ir(self):
        failures: list = []
        ir1 = formats.parse_v1((self.ws / "data/sample.v1.txt").read_text(), failures)
        ir2 = formats.parse_v2((self.ws / "data/sample.v2.json").read_text(), failures)
        name_map = json.loads((self.ws / "config/step_name_map.json").read_text())
        ir3 = formats.parse_v3((self.ws / "data/sample.v3.bin").read_bytes(),
                               name_map, failures)
        self.assertEqual(len(ir1.steps), 3)
        self.assertEqual([s.name for s in ir2.steps],
                         ["normalize", "cross", "bucket"])
        self.assertEqual([s.name for s in ir3.steps],
                         ["normalize", "cross", "bucket"])
        self.assertTrue(any(ir2.features.explicit_null.values()))
        self.assertTrue(any(ir3.features.missing.values()))
        self.assertEqual(failures, [])

    def test_field_width_error(self):
        text = (self.ws / "data/sample.v1.txt").read_text()
        lines = text.splitlines()
        lines[2] += "XX"
        failures: list = []
        formats.parse_v1("\n".join(lines) + "\n", failures)
        self.assertEqual(failures[0].code, E_FIELD_WIDTH)
        self.assertEqual(failures[0].stage, "parse_v1")

    def test_partition_oob_and_unknown_enum(self):
        name_map = json.loads((self.ws / "config/step_name_map.json").read_text())
        data = bytearray((self.ws / "data/sample.v3.bin").read_bytes())
        (n_steps,) = struct.unpack_from("<I", data, 4)
        struct.pack_into("<I", data, 8, 0xDEAD)
        pos = 8 + n_steps * 4
        (n_features,) = struct.unpack_from("<I", data, pos)
        pos += 4
        for _ in range(n_features):
            (flen,) = struct.unpack_from("<H", data, pos)
            pos += 2 + flen
        (n_entities,) = struct.unpack_from("<I", data, pos)
        pos += 4
        (elen,) = struct.unpack_from("<H", data, pos)
        pos += 2 + elen
        struct.pack_into("<Q", data, pos, 10 ** 9)
        failures: list = []
        ir = formats.parse_v3(bytes(data), name_map, failures)
        codes = {f.code for f in failures}
        self.assertIn(E_STEP_UNKNOWN, codes)
        self.assertIn(E_PARTITION_OOB, codes)
        self.assertEqual(len(ir.steps), n_steps)

    def test_precision_label_missing(self):
        doc = json.loads((self.ws / "data/sample.v2.json").read_text())
        entity = sorted(doc["precision"])[0]
        doc["precision"][entity] = {}
        failures: list = []
        formats.parse_v2(json.dumps(doc), failures)
        self.assertTrue(any(f.code == E_PRECISION_MISSING for f in failures))


class TestConversion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = make_workspace()

    def _runner(self) -> PipelineRunner:
        runner = PipelineRunner(self.ws, use_cache=False)
        runner.load_sources()
        return runner

    def test_v1_to_v2_exact_permille(self):
        runner = self._runner()
        runner.convert("v1")
        runner.convert("v2")
        for entity, feats in runner.ir.features.values.items():
            for value in feats.values():
                self.assertAlmostEqual(value * 1000, round(value * 1000), places=6)
            for label in runner.ir.features.precision[entity].values():
                self.assertEqual(label, "permille")

    def test_v2_to_v1_downgrade_sidecar_and_ledger(self):
        runner = self._runner()
        runner.convert("v1")
        kinds = {e["kind"] for e in runner.ledger}
        self.assertIn("permille_truncation", kinds)
        self.assertIn("step_params_sidecar", kinds)
        sidecar = json.loads((self.ws / "output/sidecar/sidecar.json").read_text())
        self.assertIn("v1_residuals", sidecar)
        self.assertIn("step_params", sidecar)
        self.assertIn("precision_labels", sidecar)

    def test_v3_step_map_missing_keeps_step(self):
        runner = self._runner()
        runner.ir.steps.append(Step("tfidf", {"k": 1}))
        runner.convert("v3")
        codes = [f.code for f in runner.failures]
        self.assertIn(E_STEP_MAP_MISSING, codes)
        self.assertIn("tfidf", [s.name for s in runner.ir.steps])
        kinds = [e["kind"] for e in runner.ledger]
        self.assertIn("step_enum_unmapped", kinds)


class TestDeterministicExecution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = make_workspace()

    def test_reproducible(self):
        results = []
        for _ in range(2):
            runner = PipelineRunner(self.ws, use_cache=False)
            runner.load_sources()
            runner.run(incremental=False)
            results.append(runner.state_digest()["final_features"])
        self.assertEqual(results[0], results[1])

    def test_bucket_boundary_equal_decision(self):
        runner = PipelineRunner(self.ws, use_cache=False)
        runner.load_sources()
        runner.ir.features.values["ent_000"]["f_ctr"] = 0.5
        result = runner.run(incremental=False)
        bucket = [s for s in result.step_results if s.name == "bucket"][0]
        self.assertTrue(any(d["type"] == "bucket_boundary_equal"
                            and d["rule"] == "lower_bucket"
                            for d in bucket.decisions))
        self.assertEqual(result.final_values["ent_000"]["f_ctr_bucket"], 1_000_000)


class TestEditConstraints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = make_workspace()

    def test_v1_rejects_new_step_type(self):
        runner = PipelineRunner(self.ws, use_cache=False)
        runner.load_sources()
        runner.convert("v1")
        ok = runner.apply_edit({"op": "add_step",
                                "step": {"name": "tfidf", "params": {}}})
        self.assertFalse(ok)
        self.assertEqual(runner.failures[-1].code, E_VERSION_UNSUPPORTED)
        self.assertNotIn("tfidf", [s.name for s in runner.ir.steps])

    def test_v2_rejects_new_explicit_null(self):
        runner = PipelineRunner(self.ws, use_cache=False)
        runner.load_sources()
        ok = runner.apply_edit({
            "op": "replace_entity", "entity": "ent_999",
            "features": {"f_age": None}, "explicit_null": True})
        self.assertFalse(ok)
        self.assertEqual(runner.failures[-1].code, E_VERSION_UNSUPPORTED)

    def test_merge_conflict_recorded(self):
        runner = PipelineRunner(self.ws, use_cache=False)
        runner.load_sources()
        ok = runner.apply_edit({"op": "merge_entities",
                                "source": "ent_000", "target": "ent_001"})
        self.assertTrue(ok)
        self.assertTrue(any(f.code == E_MERGE_CONFLICT for f in runner.failures))
        self.assertNotIn("ent_000", runner.ir.features.values)


class TestIncremental(unittest.TestCase):
    def test_param_change_invalidates_downstream_only(self):
        ws = make_workspace()
        runner = PipelineRunner(ws)
        runner.load_sources()
        first = runner.run(incremental=True)
        self.assertEqual(first.recompute, [0, 1, 2])
        runner.apply_edit({"op": "modify_params", "index": 1,
                           "params": {"pairs": [["f_age", "f_price"]]}})
        second = runner.run(incremental=True)
        self.assertEqual(second.reuse, [0])
        self.assertEqual(second.recompute, [1, 2])
        self.assertNotIn("normalize", runner.last_counter)

    def test_reorder_invalidates_from_first_changed_position(self):
        ws = make_workspace()
        runner = PipelineRunner(ws)
        runner.load_sources()
        runner.run(incremental=True)
        runner.apply_edit({"op": "reorder_steps", "order": [0, 2, 1]})
        result = runner.run(incremental=True)
        self.assertEqual(result.reuse, [0])
        self.assertEqual(result.recompute, [1, 2])

    def test_incremental_vs_full_rebuild_all_versions(self):
        for target in ("v1", "v2", "v3"):
            with self.subTest(target=target):
                ws_a = make_workspace()
                runner_a = PipelineRunner(ws_a)
                runner_a.load_sources()
                runner_a.convert(target)
                for edit in edit_sequence():
                    runner_a.apply_edit(edit)
                    runner_a.run(incremental=True)
                digest_a = runner_a.state_digest()
                ws_b = make_workspace()
                runner_b = PipelineRunner(ws_b, use_cache=False)
                runner_b.load_sources()
                runner_b.convert(target)
                for edit in edit_sequence():
                    runner_b.apply_edit(edit)
                runner_b.run(incremental=False)
                digest_b = runner_b.state_digest()
                self.assertEqual(digest_a, digest_b)


class TestSnapshot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = make_workspace()

    def test_hit_and_stale_rebuild(self):
        runner = PipelineRunner(self.ws, use_cache=False)
        runner.start()
        self.assertEqual(runner.snapshot_status, "stale_rebuild")
        runner2 = PipelineRunner(self.ws, use_cache=False)
        runner2.start()
        self.assertEqual(runner2.snapshot_status, "hit")
        ir_file = self.ws / "output/snapshot/ir.json"
        ir_file.write_text(ir_file.read_text() + "corrupted")
        runner3 = PipelineRunner(self.ws, use_cache=False)
        runner3.start()
        self.assertEqual(runner3.snapshot_status, "stale_rebuild")
        self.assertTrue(any(f.code == E_SNAPSHOT_STALE for f in runner3.failures))
        ok, _ = runner3.snapshot.verify()
        self.assertTrue(ok)

    def test_version_lag_detected(self):
        ws = make_workspace()
        runner = PipelineRunner(ws, use_cache=False)
        runner.start()
        manifest = runner.snapshot.read_manifest()
        ok, failures = runner.snapshot.verify(
            expected_min_version=manifest["global_version"] + 1)
        self.assertFalse(ok)
        self.assertEqual(failures[0].code, E_SNAPSHOT_STALE)


class TestInjectionReport(unittest.TestCase):
    def test_injection_hits_and_report(self):
        ws = make_workspace()
        runner = PipelineRunner(ws, inject_config=ROOT / "config/inject_config.json")
        runner.start()
        runner.run(incremental=True)
        runner.apply_injected_edits()
        runner.run(incremental=True)
        report = write_report(runner, ws / "output/report.json")
        self.assertEqual(report["injected_hits"],
                         ["INJ-001", "INJ-002", "INJ-003", "INJ-004", "INJ-005"])
        codes = {f["code"] for f in report["failures"]}
        self.assertIn(E_FIELD_WIDTH, codes)
        self.assertIn(E_STEP_UNKNOWN, codes)
        self.assertIn(E_PRECISION_MISSING, codes)
        self.assertIn(E_MERGE_CONFLICT, codes)
        self.assertTrue(any(d["type"] == "bucket_boundary_equal"
                            for d in report["decisions"]))
        for key in ("cross_version_deviation", "downgrade_ledger",
                    "reuse", "recompute", "failures"):
            self.assertIn(key, report)
        ent003 = [f for f in report["failures"]
                  if f["code"] == E_FIELD_WIDTH][0]
        self.assertEqual(ent003["entity_id"], "ent_003")
        self.assertTrue(any(ir.version_diffs for ir in runner.parsed.values()))


if __name__ == "__main__":
    unittest.main()
