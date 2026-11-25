import json
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from quantcalib.data.config import load_config
from quantcalib.data.samples import generate_workspace
from quantcalib.engine import EngineContext, QuantCalibEngine
from quantcalib.runner import (
    comparable_view,
    run_full_rebuild,
    run_incremental,
)
from quantcalib.schedule.commands import write_command_stream


COMMANDS = [
    {"command": "switch_receipt_version", "payload": {"version": "q1"}},
    {"command": "switch_receipt_version", "payload": {"version": "q2"}},
    {"command": "set_layer_bits",
     "payload": {"layer": "layer.fc1", "bits": 4}},
    {"command": "set_layer_granularity",
     "payload": {"layer": "layer.fc2", "granularity": 8}},
    {"command": "set_rounding_mode", "payload": {"mode": "half_up"}},
    {"command": "set_saturation_policy", "payload": {"policy": "wrap"}},
    {"command": "replace_layer_weights",
     "payload": {"layer": "layer.fc2", "seed": 99173, "scale": 1.5}},
    {"command": "set_distill_params",
     "payload": {"temperature": 3.5, "steps": 6, "soft_label_weight": 0.7}},
    {"command": "switch_receipt_version", "payload": {"version": "q3"}},
]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="qc_pipeline_"))
        generate_workspace(self.tmp)
        write_command_stream(self.tmp / "commands.json", COMMANDS)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_incremental_matches_full_rebuild_field_by_field(self):
        results = run_incremental(self.tmp, self.tmp / "commands.json")
        isolated = Path(tempfile.mkdtemp(prefix="qc_iso_")) / "snapshot"
        rebuild = run_full_rebuild(self.tmp, isolated_snapshot=isolated)
        self.assertEqual(
            comparable_view(results[-1]), comparable_view(rebuild))

    def test_serial_and_parallel_execution_match(self):
        results = run_incremental(self.tmp, self.tmp / "commands.json")
        isolated = Path(tempfile.mkdtemp(prefix="qc_par_")) / "snapshot"

        cfg_path = self.tmp / "config" / "quant.yaml"
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        original_workers = data["max_workers"]
        data["max_workers"] = 1
        cfg_path.write_text(yaml.safe_dump(data, sort_keys=True),
                            encoding="utf-8")
        serial = run_full_rebuild(self.tmp, isolated_snapshot=isolated)

        data["max_workers"] = original_workers
        cfg_path.write_text(yaml.safe_dump(data, sort_keys=True),
                            encoding="utf-8")
        isolated_par = Path(tempfile.mkdtemp(prefix="qc_par2_")) / "snapshot"
        parallel = run_full_rebuild(self.tmp, isolated_snapshot=isolated_par)
        self.assertEqual(comparable_view(serial), comparable_view(parallel))
        self.assertEqual(
            comparable_view(results[-1]), comparable_view(parallel))

    def test_snapshot_resume_keeps_hits_and_manifest(self):
        cfg, _ = load_config(self.tmp / "config" / "quant.yaml")
        engine = QuantCalibEngine(EngineContext(workdir=self.tmp), cfg)
        first = engine.run(global_version=0)
        self.assertFalse(first["resumed"])

        engine = QuantCalibEngine(EngineContext(workdir=self.tmp), cfg)
        second = engine.run(global_version=0)
        self.assertTrue(second["resumed"])
        self.assertGreater(second["reuse"]["hit_count"], 0)
        self.assertEqual(
            first["manifest"]["config_hash"],
            second["manifest"]["config_hash"],
        )

    def test_budget_pending_is_reported(self):
        cfg_path = self.tmp / "config" / "quant.yaml"
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        data["global_point_budget"] = 30
        cfg_path.write_text(yaml.safe_dump(data, sort_keys=True),
                            encoding="utf-8")
        cfg, _ = load_config(cfg_path)
        engine = QuantCalibEngine(EngineContext(workdir=self.tmp), cfg)
        result = engine.run(global_version=0)
        codes = {failure["code"] for failure in result["failures"]}
        self.assertIn("E_POINT_BUDGET_PENDING", codes)
        self.assertLessEqual(
            result["manifest"]["consumed_budget"], data["global_point_budget"])

    def test_per_layer_quota_exceeded_is_reported(self):
        cfg_path = self.tmp / "config" / "quant.yaml"
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        data["per_layer_quota"] = 20
        cfg_path.write_text(yaml.safe_dump(data, sort_keys=True),
                            encoding="utf-8")
        cfg, _ = load_config(cfg_path)
        engine = QuantCalibEngine(EngineContext(workdir=self.tmp), cfg)
        result = engine.run(global_version=0)
        codes = {failure["code"] for failure in result["failures"]}
        self.assertIn("E_QUOTA_EXCEEDED", codes)


if __name__ == "__main__":
    unittest.main()
