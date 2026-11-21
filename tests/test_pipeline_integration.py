from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from alert_pipeline.errors import PipelineError
from alert_pipeline.pipeline import execute_pipeline, recover_pipeline
from alert_pipeline.samplegen import generate_samples
from alert_pipeline.spec import load_dataset
from alert_pipeline.storage import LedgerStore
from alert_pipeline.verifier import verify_pipeline

from .helpers import cli_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sqlite_backup(source: Path, destination: Path) -> None:
    source_connection = sqlite3.connect(str(source))
    destination_connection = sqlite3.connect(str(destination))
    try:
        source_connection.backup(destination_connection)
    finally:
        destination_connection.close()
        source_connection.close()


class PipelineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.class_temporary = tempfile.TemporaryDirectory()
        cls.class_root = Path(cls.class_temporary.name)
        cls.samples = cls.class_root / "samples"
        generate_samples(cls.samples, seed=20261122)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.class_temporary.cleanup()

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _config(self, manifest: str = "manifest.json", name: str = "a"):
        return cli_config(
            self.samples / manifest,
            self.root / f"{name}.db",
            self.root / f"{name}.jsonl",
            self.root / f"{name}-out",
        )

    def test_parallel_and_serial_conclusions_are_byte_identical(self) -> None:
        parallel_config = self._config(name="parallel")
        serial_config = self._config(name="serial")
        parallel_dataset = load_dataset(parallel_config)
        serial_dataset = load_dataset(serial_config)
        parallel = execute_pipeline(
            parallel_config,
            parallel_dataset,
            mode="parallel",
            global_limit=4,
            per_service_limit=1,
        )
        serial = execute_pipeline(
            serial_config,
            serial_dataset,
            mode="serial",
            global_limit=1,
            per_service_limit=1,
        )
        parallel_bytes = (parallel_config.output_dir / "conclusions.canonical.json").read_bytes()
        serial_bytes = (serial_config.output_dir / "conclusions.canonical.json").read_bytes()
        self.assertEqual(parallel_bytes, serial_bytes)
        self.assertEqual(
            parallel.report["conclusion_digest"], serial.report["conclusion_digest"]
        )
        self.assertEqual(parallel.report["alert_count"], 40)
        self.assertEqual(parallel.report["aggregate_count"], 25)
        self.assertEqual(parallel.report["ledger_count"], 25)
        self.assertEqual(len(parallel.report["batches"]), 5)
        self.assertTrue(
            all(batch["peak_global"] <= 4 for batch in parallel.report["batches"])
        )
        self.assertTrue(
            all(
                peak <= 1
                for batch in parallel.report["batches"]
                for peak in batch["peak_by_service"].values()
            )
        )
        verification = verify_pipeline(
            parallel_config,
            load_dataset(parallel_config),
            write_report=False,
        )
        self.assertEqual(verification["business_code"], "OK")
        self.assertEqual(verification["source_format_versions"], [1, 2, 3])
        self.assertGreater(verification["suppressed_total"], 0)

    def test_version_evolution_replaces_whole_view_and_marks_old_stale(self) -> None:
        config_v1 = self._config("manifest.json", "evolve")
        execute_pipeline(
            config_v1,
            load_dataset(config_v1),
            mode="parallel",
            global_limit=4,
            per_service_limit=1,
        )
        config_v2 = replace(
            self._config("manifest_v2.json", "evolve"),
            database_path=config_v1.database_path,
        )
        outcome_v2 = execute_pipeline(
            config_v2,
            load_dataset(config_v2),
            mode="parallel",
            global_limit=4,
            per_service_limit=1,
        )
        self.assertGreater(outcome_v2.report["replacement"]["affected_count"], 0)
        self.assertTrue(outcome_v2.report["replacement"]["changed_config_fields"])
        with LedgerStore(config_v1.database_path) as store:
            generations = store.connection.execute(
                "SELECT declaration_version, status, is_current, is_stale FROM generations ORDER BY declaration_version"
            ).fetchall()
            self.assertEqual([row["declaration_version"] for row in generations], [1, 2])
            self.assertTrue(generations[0]["is_stale"])
            self.assertFalse(generations[0]["is_current"])
            self.assertEqual(generations[1]["status"], "complete")
            self.assertTrue(generations[1]["is_current"])
            current_aggregates = store.connection.execute(
                "SELECT COUNT(*) FROM current_aggregate_view"
            ).fetchone()[0]
            current_ledger = store.connection.execute(
                "SELECT COUNT(*) FROM current_ledger_view"
            ).fetchone()[0]
            self.assertEqual(current_aggregates, outcome_v2.report["aggregate_count"])
            self.assertEqual(current_ledger, outcome_v2.report["ledger_count"])
            stale = store.stale_counts()
            self.assertGreater(stale["aggregates"], 0)
            self.assertGreater(stale["ledger"], 0)
        verification = verify_pipeline(
            config_v2,
            load_dataset(config_v2),
            write_report=False,
        )
        self.assertEqual(verification["generation_id"], outcome_v2.report["generation_id"])
        self.assertEqual(verification["business_code"], "OK")

    def test_semantic_change_requires_monotonic_declaration_version(self) -> None:
        config = self._config(name="version")
        execute_pipeline(
            config,
            load_dataset(config),
            mode="parallel",
            global_limit=4,
            per_service_limit=1,
        )
        changed = replace(config, default_silence_window_ms=31000)
        with self.assertRaises(PipelineError) as caught:
            execute_pipeline(
                changed,
                load_dataset(changed),
                mode="parallel",
                global_limit=4,
                per_service_limit=1,
            )
        self.assertEqual(caught.exception.code, "E_VERSION")
        older = replace(config, declaration_version=0)
        with self.assertRaises(PipelineError) as caught:
            execute_pipeline(
                older,
                load_dataset(older),
                mode="parallel",
                global_limit=4,
                per_service_limit=1,
            )
        self.assertEqual(caught.exception.code, "E_VERSION")

    def test_forced_termination_unknown_batch_recover_and_resume(self) -> None:
        database = self.root / "crash.db"
        log = self.root / "crash.jsonl"
        output = self.root / "crash-out"
        common = [
            sys.executable,
            "-m",
            "alert_pipeline",
            "--manifest",
            str(self.samples / "manifest.json"),
            "--database",
            str(database),
            "--log",
            str(log),
            "--output-dir",
            str(output),
        ]
        crash = subprocess.run(
            common[:3] + ["run"] + common[3:] + ["--crash-before-commit-batch", "0"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(crash.returncode, 91, crash.stderr)
        connection = sqlite3.connect(str(database))
        connection.row_factory = sqlite3.Row
        attempts = connection.execute(
            "SELECT batch_index, status FROM batch_attempts ORDER BY attempt_id"
        ).fetchall()
        committed = connection.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
        connection.close()
        self.assertEqual(committed, 0)
        self.assertEqual([(row["batch_index"], row["status"]) for row in attempts], [(0, "unknown")])

        recover = subprocess.run(
            common[:3] + ["recover"] + common[3:],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(recover.returncode, 0, recover.stderr)
        recovery_report = json.loads(recover.stdout)
        self.assertTrue(recovery_report["requires_rerun"])
        self.assertFalse(recovery_report["exit_report_available"])
        self.assertEqual(recovery_report["unknown_attempts"][0]["batch_index"], 0)

        resume = subprocess.run(
            common[:3] + ["run"] + common[3:],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(resume.returncode, 0, resume.stderr)
        resume_report = json.loads(resume.stdout)
        self.assertEqual(resume_report["business_code"], "OK")
        self.assertEqual(resume_report["alert_count"], 40)
        verify = subprocess.run(
            common[:3] + ["verify"] + common[3:],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(verify.returncode, 0, verify.stderr)
        verification = json.loads(verify.stdout)
        self.assertEqual(verification["business_code"], "OK")
        connection = sqlite3.connect(str(database))
        statuses = [
            row[0]
            for row in connection.execute(
                "SELECT status FROM batch_attempts ORDER BY attempt_id"
            )
        ]
        connection.close()
        self.assertEqual(statuses[0], "superseded")
        self.assertIn("committed", statuses)

    def test_replay_tamper_reports_first_alert_and_field(self) -> None:
        config = self._config(name="replay")
        execute_pipeline(
            config,
            load_dataset(config),
            mode="parallel",
            global_limit=4,
            per_service_limit=1,
        )
        tampered = self.root / "replay-tampered.db"
        sqlite_backup(config.database_path, tampered)
        connection = sqlite3.connect(str(tampered))
        row = connection.execute(
            "SELECT generation_id,batch_index,alert_id,content_json FROM batch_aggregates ORDER BY batch_index,alert_id LIMIT 1"
        ).fetchone()
        generation_id, batch_index, alert_id, content = row
        value = json.loads(content)
        value["suppressed_count"] += 5
        connection.execute(
            "UPDATE batch_aggregates SET content_json=? WHERE generation_id=? AND batch_index=? AND alert_id=?",
            (
                json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                generation_id,
                batch_index,
                alert_id,
            ),
        )
        connection.commit()
        connection.close()
        tampered_config = replace(config, database_path=tampered)
        with self.assertRaises(PipelineError) as caught:
            recover_pipeline(
                tampered_config,
                load_dataset(tampered_config),
                declaration_version=tampered_config.declaration_version,
            )
        self.assertEqual(caught.exception.code, "E_REPLAY")
        self.assertEqual(caught.exception.details["alert_id"], alert_id)
        self.assertIn("suppressed_count", caught.exception.field)

    def test_verifier_tamper_reports_e_verify(self) -> None:
        config = self._config(name="verify")
        execute_pipeline(
            config,
            load_dataset(config),
            mode="parallel",
            global_limit=4,
            per_service_limit=1,
        )
        tampered = self.root / "verify-tampered.db"
        sqlite_backup(config.database_path, tampered)
        connection = sqlite3.connect(str(tampered))
        row = connection.execute(
            "SELECT generation_id,alert_id,content_json FROM ledger WHERE is_stale=0 ORDER BY alert_id LIMIT 1"
        ).fetchone()
        generation_id, alert_id, content = row
        value = json.loads(content)
        value["candidate_services"] = ["tampered"]
        connection.execute(
            "UPDATE ledger SET content_json=? WHERE generation_id=? AND alert_id=?",
            (
                json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                generation_id,
                alert_id,
            ),
        )
        connection.commit()
        connection.close()
        tampered_config = replace(config, database_path=tampered)
        with self.assertRaises(PipelineError) as caught:
            verify_pipeline(
                tampered_config,
                load_dataset(tampered_config),
                write_report=False,
            )
        self.assertEqual(caught.exception.code, "E_VERIFY")
        self.assertEqual(caught.exception.details["alert_id"], alert_id)
        self.assertIn("ledger", caught.exception.field)

    def test_gui_view_is_stably_sorted_and_contains_required_columns(self) -> None:
        from alert_pipeline.gui import _load_view

        config = self._config(name="gui")
        execute_pipeline(
            config,
            load_dataset(config),
            mode="parallel",
            global_limit=4,
            per_service_limit=1,
        )
        first = _load_view(config)
        second = _load_view(config)
        self.assertEqual(first["alerts"], second["alerts"])
        self.assertEqual(first["batches"], second["batches"])
        self.assertEqual(first["timeline"], second["timeline"])
        self.assertEqual(len(first["alerts"]), 25)
        self.assertTrue(all(len(row) == 8 for row in first["alerts"]))
        self.assertTrue(any("v1" in row[3] and "v3" in row[3] for row in first["alerts"]))
        self.assertEqual(len(first["batches"]), 5)


if __name__ == "__main__":
    unittest.main()
