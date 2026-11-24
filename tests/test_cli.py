from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from colmv.canonical import canonical_json
from tests.helpers import ROOT, SAMPLES, sample_paths


def run_cli(arguments, *, env_extra=None, timeout=120):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "colmv", *arguments],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.paths = sample_paths(SAMPLES)

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="colmv-cli-")
        self.base = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_doctor_and_validate(self) -> None:
        doctor = run_cli(["doctor"])
        self.assertEqual(doctor.returncode, 0, doctor.stderr)
        self.assertTrue(json.loads(doctor.stdout)["numpy_ok"])
        validate = run_cli(
            ["validate", "--catalog", self.paths["catalog"], "--views", self.paths["views"]]
        )
        self.assertEqual(validate.returncode, 0, validate.stderr)
        payload = json.loads(validate.stdout)
        self.assertEqual(payload["base"]["row_count"], 24)
        self.assertEqual(len(payload["views"]), 3)

    def test_command_injections_have_exact_error_codes(self) -> None:
        truncated = run_cli(
            [
                "validate",
                "--catalog",
                self.paths["catalog"],
                "--views",
                self.paths["views"],
                "--inject",
                "truncate-page",
            ]
        )
        self.assertEqual(truncated.returncode, 5)
        self.assertEqual(json.loads(truncated.stderr)["code"], "E_INTEGRITY")

        ledger_path = self.base / "zero-budget.sqlite"
        zero_budget = run_cli(
            [
                "run",
                "--catalog",
                self.paths["catalog"],
                "--views",
                self.paths["views"],
                "--changes",
                self.paths["changes"],
                "--ledger",
                str(ledger_path),
                "--budget",
                "2",
                "--inject",
                "zero-budget",
            ]
        )
        self.assertEqual(zero_budget.returncode, 4)
        self.assertEqual(json.loads(zero_budget.stderr)["code"], "E_BUDGET")
        self.assertFalse(ledger_path.exists())

    def test_verify_paths_reports_byte_equality_and_complement(self) -> None:
        result = run_cli(
            [
                "verify-paths",
                "--catalog",
                self.paths["catalog"],
                "--views",
                self.paths["views"],
                "--changes",
                self.paths["changes"],
                "--budget",
                "1",
                "--batch-plan",
                "3,6,3,6,2",
            ]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["equal_bytes"])
        self.assertEqual(
            payload["byte_length"],
            len(canonical_json(payload["materialized"]).encode("utf-8")),
        )
        self.assertTrue(payload["strict_slot_partition"]["valid"])
        self.assertGreater(payload["resident"]["replacements"], 0)
        self.assertGreater(payload["rescan"]["rescans"], 0)

    def test_run_replay_and_hash_seed_determinism(self) -> None:
        ledger_one = self.base / "one.sqlite"
        ledger_two = self.base / "two.sqlite"
        common = [
            "run",
            "--catalog",
            self.paths["catalog"],
            "--views",
            self.paths["views"],
            "--changes",
            self.paths["changes"],
            "--budget",
            "1",
            "--batch-plan",
            "3,6,3,6,2",
        ]
        first = run_cli([*common, "--ledger", str(ledger_one)], env_extra={"PYTHONHASHSEED": "1"})
        second = run_cli([*common, "--ledger", str(ledger_two)], env_extra={"PYTHONHASHSEED": "987654"})
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        first_payload = json.loads(first.stdout)
        second_payload = json.loads(second.stdout)
        self.assertEqual(first_payload["materialized_digest"], second_payload["materialized_digest"])
        self.assertEqual(
            [batch["events"] for batch in first_payload["batches"]],
            [batch["events"] for batch in second_payload["batches"]],
        )
        self.assertEqual(first_payload["switch_history"], second_payload["switch_history"])

        replay = run_cli(
            [
                "replay",
                "--catalog",
                self.paths["catalog"],
                "--ledger",
                str(ledger_one),
                "--changes",
                self.paths["changes"],
            ]
        )
        self.assertEqual(replay.returncode, 0, replay.stderr)
        replay_payload = json.loads(replay.stdout)
        self.assertEqual(replay_payload["verified_batches"], 5)
        self.assertTrue(replay_payload["stream_complete"])
        self.assertEqual(replay_payload["materialized_digest"], first_payload["materialized_digest"])

    def test_crash_recovery_and_resume(self) -> None:
        ledger_path = self.base / "crash.sqlite"
        common = [
            "run",
            "--catalog",
            self.paths["catalog"],
            "--views",
            self.paths["views"],
            "--changes",
            self.paths["changes"],
            "--ledger",
            str(ledger_path),
            "--budget",
            "1",
            "--batch-plan",
            "3,6,3,6,2",
        ]
        crashed = run_cli([*common, "--crash-during-batch", "3"])
        self.assertEqual(crashed.returncode, 137)

        recovered = run_cli(["recover", "--ledger", str(ledger_path)])
        self.assertEqual(recovered.returncode, 0, recovered.stderr)
        recovered_payload = json.loads(recovered.stdout)
        self.assertTrue(recovered_payload["recovery"]["recovered"])
        self.assertFalse(recovered_payload["exit_report_available"])
        self.assertGreater(recovered_payload["unknown_group_count"], 0)

        resumed = run_cli(
            [
                "run",
                "--catalog",
                self.paths["catalog"],
                "--views",
                self.paths["views"],
                "--changes",
                self.paths["changes"],
                "--ledger",
                str(ledger_path),
                "--budget",
                "1",
                "--resume",
                "--batch-size",
                "4",
            ]
        )
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        payload = json.loads(resumed.stdout)
        self.assertEqual(payload["ledger"]["meta"]["cursor"], "20")
        self.assertTrue(payload["ledger"]["content_chain"]["valid"])
        replay = run_cli(
            [
                "replay",
                "--catalog",
                self.paths["catalog"],
                "--ledger",
                str(ledger_path),
                "--changes",
                self.paths["changes"],
            ]
        )
        self.assertEqual(replay.returncode, 0, replay.stderr)
        replay_payload = json.loads(replay.stdout)
        self.assertEqual(replay_payload["verified_batches"], 5)
        self.assertEqual(replay_payload["materialized_digest"], payload["materialized_digest"])

    def test_evolve_whole_replacement(self) -> None:
        ledger_path = self.base / "evolve.sqlite"
        run = run_cli(
            [
                "run",
                "--catalog",
                self.paths["catalog"],
                "--views",
                self.paths["views"],
                "--changes",
                self.paths["changes"],
                "--ledger",
                str(ledger_path),
                "--budget",
                "1",
                "--batch-plan",
                "3,6,3,6,2",
            ]
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        evolve = run_cli(
            [
                "evolve",
                "--catalog",
                self.paths["catalog"],
                "--old-views",
                self.paths["views"],
                "--new-views",
                self.paths["views_v2"],
                "--old-budget",
                "1",
                "--new-budget",
                "1",
                "--changes",
                self.paths["changes"],
                "--ledger",
                str(ledger_path),
            ]
        )
        self.assertEqual(evolve.returncode, 0, evolve.stderr)
        payload = json.loads(evolve.stdout)
        self.assertEqual(payload["ledger_application"]["old_scheme_version"], 1)
        self.assertEqual(payload["ledger_application"]["new_scheme_version"], 2)
        self.assertTrue(payload["ledger_application"]["requires_recompute"])
        self.assertFalse(payload["snapshot"]["valid"])
        self.assertTrue(payload["affected_groups"])
        self.assertTrue(payload["affected_pages"])
        self.assertEqual(payload["replacement"]["mode"], "whole_scheme_replacement")


if __name__ == "__main__":
    unittest.main()
