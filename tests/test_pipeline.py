from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from quantcal.errors import QuantCalError
from quantcal.injections import (
    apply_input_injection,
    load_injection_manifest,
    tamper_snapshot_version,
)
from quantcal.pipeline import run_pipeline
from quantcal.snapshot import SnapshotManager
from quantcal.util import atomic_write_json, load_json


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_data"


def _edit_json(path: Path, edit) -> None:
    document = load_json(path)
    edit(document)
    atomic_write_json(path, document)


def _edit_samples(path: Path, edit) -> None:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    edit(rows)
    path.write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input_dir = self.root / "input"
        shutil.copytree(SAMPLE, self.input_dir)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _assert_matches_full(self, incremental, tag: str, version: str) -> None:
        full = run_pipeline(
            self.input_dir,
            self.root / f"full-{tag}",
            version,
            force_rebuild=True,
        )
        for relative in ("result.json", "exchange_sidecar.json"):
            self.assertEqual(
                (incremental.output_dir / relative).read_bytes(),
                (full.output_dir / relative).read_bytes(),
                f"{tag}: {relative} differs from full rebuild",
            )
        exchange_name = "calibration_v1.txt" if version == "v1" else "calibration_v2.json"
        self.assertEqual(
            (incremental.output_dir / exchange_name).read_bytes(),
            (full.output_dir / exchange_name).read_bytes(),
            f"{tag}: exchange differs from full rebuild",
        )

    def test_incremental_edits_match_full_rebuild_and_invalidate_precisely(self) -> None:
        work = self.root / "incremental"
        initial = run_pipeline(self.input_dir, work, "v2")
        self.assertEqual(initial.exit_code, 0)
        self._assert_matches_full(initial, "initial", "v2")

        v1 = run_pipeline(self.input_dir, work, "v1")
        self._assert_matches_full(v1, "v1", "v1")
        self.assertEqual(v1.diagnostics["input_exchange_validation"]["v1_to_v2_to_v1_exact"], True)

        def change_sample(rows) -> None:
            for row in rows:
                if row["sample_id"] == "cal-0000":
                    row["features"]["attn.q.weight"][0] += 0.25
                    return

        _edit_samples(self.input_dir / "samples.jsonl", change_sample)
        sample_edit = run_pipeline(self.input_dir, work, "v1")
        self._assert_matches_full(sample_edit, "sample-edit", "v1")
        ledger = load_json(sample_edit.output_dir / "ledger.json")
        self.assertEqual(ledger["recomputed_tensors"], ["attn.q.weight", "norm.weight"])

        def bits_twelve(config) -> None:
            config["quantization_bits"] = 12

        _edit_json(self.input_dir / "config.json", bits_twelve)
        bits = run_pipeline(self.input_dir, work, "v1")
        self._assert_matches_full(bits, "bits-12", "v1")
        bits_ledger = load_json(bits.output_dir / "ledger.json")
        self.assertEqual(
            bits_ledger["stages"]["quantile"]["reused"],
            bits_ledger["tensor_order"],
        )
        self.assertEqual(bits_ledger["stages"]["quantile"]["computed"], [])

        def rounding_floor(config) -> None:
            config["rounding_mode"] = "floor"

        _edit_json(self.input_dir / "config.json", rounding_floor)
        rounding = run_pipeline(self.input_dir, work, "v1")
        self._assert_matches_full(rounding, "rounding-floor", "v1")

        def symmetric_saturation(config) -> None:
            config["saturation_strategy"] = "saturate_symmetric"

        _edit_json(self.input_dir / "config.json", symmetric_saturation)
        saturation = run_pipeline(self.input_dir, work, "v1")
        self._assert_matches_full(saturation, "saturation", "v1")

        switched = run_pipeline(self.input_dir, work, "v2")
        self._assert_matches_full(switched, "switch-v2", "v2")

    def test_budget_and_concurrency_are_order_invariant(self) -> None:
        serial = run_pipeline(
            self.input_dir,
            self.root / "serial",
            "v2",
            max_concurrency=1,
            batch_point_budget=100,
        )
        parallel = run_pipeline(
            self.input_dir,
            self.root / "parallel",
            "v2",
            max_concurrency=4,
            batch_point_budget=100,
        )
        self.assertEqual(serial.exit_code, 15)
        self.assertEqual(parallel.exit_code, 15)
        self.assertEqual(serial.result, parallel.result)
        self.assertEqual(
            serial.diagnostics["schedule"]["budget_deduction_sequence"],
            parallel.diagnostics["schedule"]["budget_deduction_sequence"],
        )
        self.assertEqual(
            [item["tensor"] for item in serial.pending],
            [item["tensor"] for item in parallel.pending],
        )
        self.assertTrue(all(item["code"] == "E_CALIB_BUDGET_PENDING" for item in serial.pending))

    def test_injection_isolation_and_snapshot_stale(self) -> None:
        manifest = load_injection_manifest(self.input_dir / "injections.json")
        nonfinite = manifest["injections"][0]
        scenario_input = self.root / "injection-input"
        hit = apply_input_injection(self.input_dir, scenario_input, nonfinite)
        result = run_pipeline(
            scenario_input,
            self.root / "injection-work",
            "v2",
            injection_hits=[hit],
        )
        self.assertEqual(result.exit_code, 2)
        self.assertEqual(result.errors[0]["code"], "E_NON_FINITE_SAMPLE")
        self.assertNotIn("probe.nonfinite", result.result["tensor_order"])
        self.assertEqual(result.diagnostics["injection_hits"][0]["injection_id"], "INJ-001")

        stale_work = self.root / "stale"
        run_pipeline(self.input_dir, stale_work, "v2")
        tamper_snapshot_version(stale_work)
        with self.assertRaises(QuantCalError) as caught:
            run_pipeline(self.input_dir, stale_work, "v2", resume=True)
        self.assertEqual(caught.exception.code, "E_SNAPSHOT_STALE")

        interrupted_work = self.root / "interrupted"
        run_pipeline(self.input_dir, interrupted_work, "v2")
        SnapshotManager(interrupted_work, "different-config-fingerprint", "v2").startup()
        with self.assertRaises(QuantCalError) as interrupted:
            run_pipeline(self.input_dir, interrupted_work, "v2")
        self.assertEqual(interrupted.exception.code, "E_SNAPSHOT_STALE")

        recoverable_work = self.root / "recoverable"
        first = run_pipeline(self.input_dir, recoverable_work, "v2")
        SnapshotManager(
            recoverable_work,
            first.diagnostics["config_fingerprint"],
            "v2",
        ).startup()
        recovered = run_pipeline(self.input_dir, recoverable_work, "v2")
        self.assertEqual(recovered.exit_code, 0)
        self.assertEqual(
            (first.output_dir / "result.json").read_bytes(),
            (recovered.output_dir / "result.json").read_bytes(),
        )

    def test_cli_returns_integer_exit_code_and_json(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "quantcal",
                "--input-dir",
                str(self.input_dir),
                "--work-dir",
                str(self.root / "cli-work"),
                "--exchange-version",
                "v2",
                "--batch-point-budget",
                "100",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 15)
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(payload["exit_code"], 15)
        self.assertTrue(payload["pending"])
        self.assertTrue(all(item["code"] == "E_CALIB_BUDGET_PENDING" for item in payload["pending"]))


if __name__ == "__main__":
    unittest.main()
