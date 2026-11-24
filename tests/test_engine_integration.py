from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
import unittest
from pathlib import Path

from leasecache.engine import RunOptions, compare_business_snapshots, run_engine
from leasecache.errors import LeaseError
from leasecache.verifier import verify_ledger

from .helpers import free_ports, make_spec, test_tempdir, write_spec


class ParallelSchedulerTests(unittest.TestCase):
    def test_real_threads_sockets_and_quota_limits(self) -> None:
        directory = test_tempdir()
        artifacts = [
            {"key": f"art/{index}", "references": []} for index in range(6)
        ]
        stages = [{"id": "s", "concurrency": 3, "artifacts": artifacts}]
        spec_path = write_spec(
            directory,
            make_spec(
                free_ports(3),
                stages=stages,
                global_concurrency=3,
                lease_ms=100,
                seeds=[5, 5, 5],
                capacities=[10, 10, 10],
                limits=[1, 1, 1],
            ),
        )
        report = run_engine(
            spec_path,
            directory / "db.sqlite",
            directory / "workspace",
            options=RunOptions(
                lease_ms=100,
                wait_timeout_ms=6000,
                service_delay_ms=40,
            ),
        )
        scheduling = report["scheduling"]
        self.assertEqual(len(scheduling["requests"]), 6)
        self.assertGreaterEqual(scheduling["peak_global_inflight"], 2)
        self.assertLessEqual(scheduling["peak_global_inflight"], 3)
        for node_id, peak in scheduling["peak_node_inflight"].items():
            self.assertLessEqual(peak, 1, node_id)
        for stage_id, peak in scheduling["peak_stage_inflight"].items():
            self.assertLessEqual(peak, 3, stage_id)
        self.assertTrue(scheduling["queue_waits"])
        for request in scheduling["requests"]:
            self.assertIsInstance(request["ack_elapsed_mono_ms"], int)
            self.assertTrue(request["lease_id"])
        self.assertTrue(all(node["active_leases"] == 0 for node in report["nodes"]))
        self.assertTrue(all(node["used_slots"] == 0 for node in report["nodes"]))
        self.assertEqual(verify_ledger(spec_path, directory / "db.sqlite")["status"], "ok")


class IncrementalAndRecoveryTests(unittest.TestCase):
    def test_incremental_edit_matches_full_rebuild(self) -> None:
        directory = test_tempdir()
        ports = free_ports(2)
        stages = [
            {
                "id": "sources",
                "concurrency": 2,
                "artifacts": [
                    {"key": "a", "references": []},
                    {"key": "b", "references": []},
                ],
            },
            {
                "id": "package",
                "concurrency": 1,
                "artifacts": [{"key": "pkg", "references": ["a", "b"]}],
            },
        ]
        raw = make_spec(ports, stages=stages, lease_ms=80)
        original = write_spec(directory, raw, "original.json")
        run_engine(
            original,
            directory / "incremental.sqlite",
            directory / "incremental-workspace",
            options=RunOptions(lease_ms=80, wait_timeout_ms=5000),
        )
        raw["pipeline"]["stages"][1]["artifacts"][0]["references"] = ["a"]
        edited = write_spec(directory, raw, "edited.json")
        incremental = run_engine(
            edited,
            directory / "incremental.sqlite",
            directory / "incremental-workspace",
            options=RunOptions(lease_ms=80, wait_timeout_ms=5000),
        )
        full = run_engine(
            edited,
            directory / "full.sqlite",
            directory / "full-workspace",
            options=RunOptions(lease_ms=80, wait_timeout_ms=5000),
        )
        self.assertEqual(incremental["incremental"]["recomputed_keys"], ["pkg"])
        comparison = compare_business_snapshots(
            incremental["business_snapshot"], full["business_snapshot"]
        )
        self.assertTrue(comparison["equal"], comparison)

    def test_forced_termination_recovery_matches_full_rebuild(self) -> None:
        directory = test_tempdir()
        stages = [
            {
                "id": "s",
                "concurrency": 2,
                "artifacts": [
                    {"key": "a", "references": []},
                    {"key": "b", "references": []},
                    {"key": "c", "references": []},
                ],
            }
        ]
        spec_path = write_spec(
            directory,
            make_spec(
                free_ports(2),
                stages=stages,
                global_concurrency=2,
                lease_ms=100,
                seeds=[11, 11],
                capacities=[5, 5],
                limits=[2, 2],
            ),
        )
        db = directory / "recovery.sqlite"
        workspace = directory / "recovery-workspace"
        command = [
            sys.executable,
            "-m",
            "leasecache",
            "run",
            "--spec",
            str(spec_path),
            "--db",
            str(db),
            "--workspace",
            str(workspace),
            "--lease-ms",
            "100",
            "--wait-timeout-ms",
            "5000",
            "--interrupt-after-grants",
            "1",
        ]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path.cwd()) + (
            os.pathsep + environment["PYTHONPATH"]
            if environment.get("PYTHONPATH")
            else ""
        )
        process = subprocess.run(
            command,
            cwd=Path.cwd(),
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(process.returncode, 86, process.stderr)
        self.assertFalse(db.with_suffix(".report.json").exists())
        resumed = run_engine(
            spec_path,
            db,
            workspace,
            options=RunOptions(lease_ms=100, wait_timeout_ms=5000),
        )
        self.assertTrue((resumed.get("recovery") or {}).get("recovered"))
        full = run_engine(
            spec_path,
            directory / "full.sqlite",
            directory / "full-workspace",
            options=RunOptions(lease_ms=100, wait_timeout_ms=5000),
        )
        comparison = compare_business_snapshots(
            resumed["business_snapshot"], full["business_snapshot"]
        )
        self.assertTrue(comparison["equal"], comparison)

        verifier = verify_ledger(spec_path, db)
        self.assertEqual(verifier["status"], "ok")
        tampered = directory / "tampered.sqlite"
        shutil.copy2(db, tampered)
        connection = sqlite3.connect(tampered)
        try:
            connection.execute(
                "UPDATE summaries SET digest='tampered' WHERE artifact_key='a' "
                "AND node_id='node-a'"
            )
            connection.commit()
        finally:
            connection.close()
        with self.assertRaises(LeaseError) as caught:
            verify_ledger(spec_path, tampered)
        self.assertEqual(caught.exception.code, "E_VERIFY")
        self.assertEqual(caught.exception.artifact_key, "a")
        self.assertEqual(caught.exception.node_id, "node-a")
        self.assertEqual(caught.exception.field, "summaries.digest")

    def test_incremental_removal_of_node_and_artifact_matches_full(self) -> None:
        directory = test_tempdir()
        ports = free_ports(2)
        stages = [
            {
                "id": "s",
                "concurrency": 2,
                "artifacts": [
                    {"key": "a", "references": []},
                    {"key": "b", "references": []},
                    {"key": "c", "references": []},
                ],
            }
        ]
        original_raw = make_spec(
            ports,
            stages=stages,
            lease_ms=80,
            seeds=[31, 47],
            capacities=[4, 4],
            limits=[2, 2],
        )
        original = write_spec(directory, original_raw, "original.json")
        run_engine(
            original,
            directory / "incremental.sqlite",
            directory / "incremental-workspace",
            options=RunOptions(lease_ms=80, wait_timeout_ms=5000),
        )
        edited_raw = make_spec(
            [ports[0]],
            stages=[
                {
                    "id": "s",
                    "concurrency": 2,
                    "artifacts": [
                        {"key": "a", "references": []},
                        {"key": "b", "references": []},
                    ],
                }
            ],
            lease_ms=80,
            seeds=[31],
            capacities=[4],
            limits=[2],
        )
        edited = write_spec(directory, edited_raw, "edited.json")
        incremental = run_engine(
            edited,
            directory / "incremental.sqlite",
            directory / "incremental-workspace",
            options=RunOptions(lease_ms=80, wait_timeout_ms=5000),
        )
        full = run_engine(
            edited,
            directory / "full.sqlite",
            directory / "full-workspace",
            options=RunOptions(lease_ms=80, wait_timeout_ms=5000),
        )
        self.assertEqual(
            incremental["spec_reconciliation"]["removed_nodes"], ["node-b"]
        )
        self.assertEqual(
            incremental["spec_reconciliation"]["removed_artifacts"], ["c"]
        )
        comparison = compare_business_snapshots(
            incremental["business_snapshot"], full["business_snapshot"]
        )
        self.assertTrue(comparison["equal"], comparison)
        self.assertEqual(verify_ledger(edited, directory / "incremental.sqlite")["status"], "ok")


if __name__ == "__main__":
    unittest.main()
