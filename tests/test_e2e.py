"""End-to-end tests: real sockets, subprocess orchestration, ledger
persistence, crash recovery, replay verification, fault injection."""

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from helpers import ROOT, free_ports, make_requests, make_spec_dict, write_json, write_ndjson


def run_cli(*cli_args, cwd=ROOT):
    return subprocess.run(
        [sys.executable, "-m", "quota_workbench", *cli_args],
        cwd=cwd, capture_output=True, text=True, timeout=120,
    )


class E2EBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="qwb-e2e-")
        self.ports = free_ports(2, start=self._port_base())
        self.spec_path = os.path.join(self.tmp, "spec.json")
        self.req_path = os.path.join(self.tmp, "requests.ndjson")
        self.db = os.path.join(self.tmp, "ledger.db")
        self.snapshots = os.path.join(self.tmp, "snapshots")
        self.log = os.path.join(self.tmp, "run.log")
        write_json(self.spec_path, make_spec_dict(self.ports))
        write_ndjson(self.req_path, make_requests(12))

    def _port_base(self):
        return 19300

    def run_args(self, *extra):
        return (
            "run", "--spec", self.spec_path, "--requests", self.req_path,
            "--db", self.db, "--snapshot-dir", self.snapshots,
            "--rounds", "4", "--batch-size", "4", "--log", self.log, *extra,
        )


class FullRunTest(E2EBase):
    def test_run_then_replay_ok(self):
        proc = run_cli(*self.run_args())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        summary = json.loads(proc.stdout)
        self.assertEqual(summary["requests"], 12)

        replay = run_cli(
            "replay", "--spec", self.spec_path, "--requests", self.req_path,
            "--db", self.db, "--rounds", "4", "--log", self.log)
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual(json.loads(replay.stdout)["status"], "replay_ok")

    def test_two_runs_byte_identical_digest(self):
        first = run_cli(*self.run_args())
        self.assertEqual(first.returncode, 0, first.stderr)
        digest1 = json.loads(first.stdout)["content_digest"]
        db2 = self.db + ".2"
        second = run_cli(
            "run", "--spec", self.spec_path, "--requests", self.req_path,
            "--db", db2, "--snapshot-dir", self.snapshots + "2",
            "--rounds", "4", "--batch-size", "4", "--log", self.log)
        self.assertEqual(second.returncode, 0, second.stderr)
        digest2 = json.loads(second.stdout)["content_digest"]
        self.assertEqual(digest1, digest2)

    def test_outcome_distribution_aggregates(self):
        proc = run_cli(*self.run_args())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        conn = sqlite3.connect(self.db)
        rows = conn.execute(
            "SELECT COALESCE(outcome, state) FROM requests ORDER BY request_id").fetchall()
        conn.close()
        outcomes = np.array([1 if r[0] == "granted" else 0 for r in rows])
        self.assertEqual(int(outcomes.sum()) + int((1 - outcomes).sum()), 12)
        # alpha's instance is healthy -> some grants; beta's errors out -> rejects
        self.assertGreater(int(outcomes.sum()), 0)
        self.assertGreater(int((1 - outcomes).sum()), 0)

    def test_replay_detects_tampering(self):
        proc = run_cli(*self.run_args())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        conn = sqlite3.connect(self.db)
        conn.execute(
            "UPDATE requests SET outcome='rejected' WHERE request_id="
            " (SELECT request_id FROM requests WHERE outcome='granted' LIMIT 1)")
        conn.commit()
        conn.close()
        replay = run_cli(
            "replay", "--spec", self.spec_path, "--requests", self.req_path,
            "--db", self.db, "--rounds", "4", "--log", self.log)
        self.assertEqual(replay.returncode, 2)
        payload = json.loads(replay.stderr.strip().splitlines()[-1])
        self.assertEqual(payload["code"], "E_REPLAY")
        self.assertIn("request_id", payload["details"])
        self.assertIn("tenant_id", payload["details"])
        self.assertEqual(payload["details"]["field"], "outcome")


class CrashRecoveryTest(E2EBase):
    def test_forced_kill_then_recover(self):
        proc = run_cli(*self.run_args("--crash-after-batch", "1"))
        self.assertEqual(proc.returncode, 3, proc.stderr)
        conn = sqlite3.connect(self.db)
        inflight = conn.execute(
            "SELECT request_id FROM requests WHERE state='in_flight'").fetchall()
        conn.close()
        self.assertTrue(inflight, "expected committed in-flight rows after kill")

        recover = run_cli("recover", "--db", self.db, "--log", self.log)
        self.assertEqual(recover.returncode, 0, recover.stderr)
        report = json.loads(recover.stdout)
        self.assertEqual(sorted(report["unknown_requests"]),
                         sorted(r[0] for r in inflight))
        self.assertIn("no exit report", report["note"])
        conn = sqlite3.connect(self.db)
        states = conn.execute("SELECT DISTINCT state FROM requests").fetchall()
        conn.close()
        self.assertNotIn(("in_flight",), states)


class InjectionTest(E2EBase):
    def test_port_occupied(self):
        proc = run_cli(*self.run_args("--inject", "port-occupied"))
        self.assertEqual(proc.returncode, 2)
        payload = json.loads(proc.stderr.strip().splitlines()[-1])
        self.assertEqual(payload["code"], "E_PORT")
        self.assertIn(str(self.ports[0]), payload["message"])
        with open(self.log, encoding="utf-8") as fh:
            self.assertIn('"E_PORT"', fh.read())

    def test_quota_overflow(self):
        proc = run_cli(*self.run_args("--inject", "quota-overflow"))
        self.assertEqual(proc.returncode, 2)
        payload = json.loads(proc.stderr.strip().splitlines()[-1])
        self.assertEqual(payload["code"], "E_QUOTA")

    def test_snapshot_corrupt(self):
        proc = run_cli(*self.run_args("--inject", "snapshot-corrupt"))
        self.assertEqual(proc.returncode, 2)
        payload = json.loads(proc.stderr.strip().splitlines()[-1])
        self.assertEqual(payload["code"], "E_SNAPSHOT")
        self.assertIn("snapshot_digest", payload["details"]["diffs"])


class SnapshotVerifyTest(E2EBase):
    def test_verify_ok_then_version_mismatch_replaces(self):
        proc = run_cli(*self.run_args())
        self.assertEqual(proc.returncode, 0, proc.stderr)
        snapshot = os.path.join(self.snapshots, "latest.json")

        ok = run_cli("snapshot-verify", "--spec", self.spec_path,
                     "--snapshot", snapshot, "--db", self.db, "--log", self.log)
        self.assertEqual(ok.returncode, 0, ok.stderr)

        with open(self.spec_path, encoding="utf-8") as fh:
            spec = json.load(fh)
        spec["tenant_set_version"] = 2
        write_json(self.spec_path, spec)
        bad = run_cli("snapshot-verify", "--spec", self.spec_path,
                      "--snapshot", snapshot, "--db", self.db,
                      "--snapshot-dir", self.snapshots, "--replace", "--log", self.log)
        self.assertEqual(bad.returncode, 2)
        lines = bad.stdout.strip().splitlines()
        replaced = json.loads(lines[0])
        self.assertEqual(replaced["status"], "replaced")
        self.assertIn("tenant_set_version", replaced["diffs"])
        payload = json.loads(bad.stderr.strip().splitlines()[-1])
        self.assertEqual(payload["code"], "E_SNAPSHOT")
        self.assertFalse(os.path.exists(self.db), "ledger must be dropped on replacement")


if __name__ == "__main__":
    unittest.main()
