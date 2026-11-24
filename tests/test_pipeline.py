from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from quantcalib.errors import E_BATCH_BUDGET_PENDING, E_QUOTA_EXCEEDED
from quantcalib.pipeline import CalibrationPipeline, PipelineRequest
from quantcalib.samplegen import generate_samples


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.samples = generate_samples(self.root / "samples")
        self.receipts = (self.samples["q1"], self.samples["q2"], self.samples["q3"])

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _run(
        self,
        state: Path,
        *,
        commands: Path | None = None,
        faults: Path | None = None,
        full_rebuild: bool = False,
        config: Path | None = None,
        report: Path | None = None,
    ):
        return CalibrationPipeline(
            PipelineRequest(
                weights_path=self.samples["weights"],
                calibration_path=self.samples["calibration"],
                config_path=config or self.samples["config"],
                state_dir=state,
                receipt_paths=self.receipts,
                commands_path=commands,
                faults_path=faults,
                report_path=report,
                full_rebuild=full_rebuild,
            )
        ).run()

    def test_incremental_equals_full_and_only_changed_layer_invalidates(self) -> None:
        state = self.root / "state"
        self._run(state)
        incremental = self._run(state, commands=self.samples["commands"])
        full = self._run(self.root / "full", commands=self.samples["commands"], full_rebuild=True)
        self.assertEqual(incremental["result_digest"], full["result_digest"])
        self.assertEqual(incremental["receipt_ledger"], full["receipt_ledger"])
        self.assertEqual(
            incremental["budget"]["consumption_sequence"],
            full["budget"]["consumption_sequence"],
        )
        invalidated = incremental["invalidation"]["layers"]["retrieval.head"]
        self.assertTrue(invalidated["content_version_changed"])
        self.assertIn("receipt_ledger", invalidated["invalidated_stages"])
        for unchanged in ("embedding.proj", "encoder.block"):
            self.assertEqual(
                set(incremental["invalidation"]["layers"][unchanged]["reused_stages"]),
                {"calibration", "quantization", "error_analysis", "distillation"},
            )

    def test_second_incremental_uses_persistent_artifacts_without_recompute(self) -> None:
        state = self.root / "state"
        self._run(state, commands=self.samples["commands"])
        second = self._run(state, commands=self.samples["commands"])
        self.assertEqual(second["status"], "ok")
        for layer in second["layers"]:
            for stage in ("calibration", "quantization", "error_analysis", "distillation"):
                self.assertEqual(layer["cache"][stage]["computed"], 0)
                self.assertGreater(layer["cache"][stage]["reused"], 0)
        self.assertTrue(second["reuse"]["ledger_reused"])

    def test_calibration_artifacts_are_byte_identical_across_rebuilds(self) -> None:
        first = self._run(self.root / "state-a", full_rebuild=True)
        second = self._run(self.root / "state-b", full_rebuild=True)
        self.assertEqual(first["result_digest"], second["result_digest"])
        current_a = Path(first["artifacts"]["state_root"]) / "CURRENT"
        current_b = Path(second["artifacts"]["state_root"]) / "CURRENT"
        gen_a = json.loads(current_a.read_text())["generation"]
        gen_b = json.loads(current_b.read_text())["generation"]
        calib_a = Path(first["artifacts"]["state_root"]) / "generations" / gen_a / "stages" / "calibration"
        calib_b = Path(second["artifacts"]["state_root"]) / "generations" / gen_b / "stages" / "calibration"
        files_a = sorted(path.relative_to(calib_a) for path in calib_a.rglob("*.json"))
        files_b = sorted(path.relative_to(calib_b) for path in calib_b.rglob("*.json"))
        self.assertEqual(files_a, files_b)
        for relative in files_a:
            self.assertEqual((calib_a / relative).read_bytes(), (calib_b / relative).read_bytes())

    def test_fault_injection_ids_and_codes(self) -> None:
        report = self._run(self.root / "fault", faults=self.samples["faults"])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["fault_injections"]), 4)
        self.assertTrue(all(item["hit"] for item in report["fault_injections"]))
        self.assertEqual(
            {item["id"] for item in report["fault_injections"]},
            {"F01_NONFINITE", "F02_GROUP_INDIVISIBLE", "F03_BIT_WIDTH", "F04_CALIBRATION_DIM"},
        )
        self.assertIn("E_QUANT_BOUNDS", report["business_code_counts"])
        self.assertIn("E_CALIBRATION_INVALID", report["business_code_counts"])

    def test_budget_pending_and_quota_codes(self) -> None:
        config = yaml.safe_load(self.samples["config"].read_text())
        config["scheduling"]["global_calibration_point_budget"] = 8
        config["scheduling"]["per_layer_calibration_quota"] = 8
        config["scheduling"]["batch_quantization_point_budget"] = 4
        path = self.root / "budget-config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        report = self._run(self.root / "budget", config=path)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["budget"]["global_consumed"], 8)
        self.assertTrue(report["budget"]["pending_items"])
        self.assertTrue(
            {item["code"] for item in report["budget"]["pending_items"]}
            <= {E_BATCH_BUDGET_PENDING, E_QUOTA_EXCEEDED}
        )

    def test_report_written_and_manifest_complete(self) -> None:
        report_path = self.root / "report.json"
        report = self._run(self.root / "state", report=report_path)
        self.assertTrue(report_path.exists())
        persisted = json.loads(report_path.read_text())
        self.assertEqual(persisted["result_digest"], report["result_digest"])
        self.assertEqual(report["manifest"]["completed_layers"], report["completed_layers"])
        state_root = Path(report["artifacts"]["state_root"])
        pointer = json.loads((state_root / "CURRENT").read_text())["generation"]
        manifest = json.loads((state_root / "generations" / pointer / "manifest.json").read_text())
        for key in (
            "global_version",
            "config_hash",
            "weight_package_fingerprint",
            "calibration_fingerprint",
            "completed_layers",
            "consumed_budget",
            "stage_artifact_versions",
        ):
            self.assertIn(key, manifest)

    def test_orphan_receipt_reported(self) -> None:
        report = CalibrationPipeline(
            PipelineRequest(
                weights_path=self.samples["weights"],
                calibration_path=self.samples["calibration"],
                config_path=self.samples["config"],
                state_dir=self.root / "orphan",
                receipt_paths=(self.samples["orphan_receipts"],),
            )
        ).run()
        self.assertEqual(report["status"], "failed")
        self.assertIn("E_RECEIPT_ORPHAN", report["business_code_counts"])

    def test_delete_command_turns_existing_receipt_into_orphan(self) -> None:
        commands = self.root / "delete.jsonl"
        commands.write_text(
            json.dumps({"op": "delete_layer", "layer": "retrieval.head"}) + "\n",
            encoding="utf-8",
        )
        report = self._run(self.root / "delete-state", commands=commands)
        self.assertEqual(report["status"], "failed")
        self.assertIn("E_RECEIPT_ORPHAN", report["business_code_counts"])
        self.assertTrue(
            any(item.get("layer") == "retrieval.head" for item in report["failure_list"] if item["code"] == "E_RECEIPT_ORPHAN")
        )

    def test_worker_count_does_not_change_core_results(self) -> None:
        config = yaml.safe_load(self.samples["config"].read_text())
        config["scheduling"]["thread_pool_workers"] = 1
        one = self.root / "one.yaml"
        one.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        config["scheduling"]["thread_pool_workers"] = 8
        many = self.root / "many.yaml"
        many.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        report_one = self._run(self.root / "workers-one", config=one, full_rebuild=True)
        report_many = self._run(self.root / "workers-many", config=many, full_rebuild=True)
        self.assertEqual(report_one["layers"], report_many["layers"])
        self.assertEqual(report_one["result_digest"], report_many["result_digest"])
        self.assertEqual(
            report_one["budget"]["consumption_sequence"],
            report_many["budget"]["consumption_sequence"],
        )
        self.assertEqual(report_one["budget"]["batch_events"], report_many["budget"]["batch_events"])
        self.assertEqual(report_one["receipt_ledger"], report_many["receipt_ledger"])


if __name__ == "__main__":
    unittest.main()
