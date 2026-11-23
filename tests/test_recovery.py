import json
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest

from convd.errors import ReplayError
from convd.ledger import Ledger
from convd.replay import replay_verify, unknown_rounds
from convd.scene import load_executor_set, load_scenario

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cursor_of(db):
    conn = sqlite3.connect(db)
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key='cursor'").fetchone()
        return json.loads(row[0]) if row else 0
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


class TestForcedKillRecovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "kill.db")
        self.scenario_path = os.path.join(self.tmp.name, "scenario.json")
        self.executors_path = os.path.join(self.tmp.name, "executors.json")
        with open(self.scenario_path, "w") as fh:
            json.dump({"scenario_id": "sc-kill", "max_rounds": 8,
                       "window_ms": 400, "pulse_interval_ms": 150,
                       "lease_ms": 250, "receipt_timeout_ms": 100,
                       "budget_version": 1, "max_in_flight": 2,
                       "max_in_flight_per_executor": 1, "max_retries": 1,
                       "executor_set_version": 1}, fh)
        with open(self.executors_path, "w") as fh:
            json.dump({"executors": [
                {"executor_id": "k-a", "port": 43101,
                 "target_digest": "ta", "seed": 3},
                {"executor_id": "k-b", "port": 43102,
                 "target_digest": "tb", "seed": 4}]}, fh)

    def tearDown(self):
        subprocess.run(["pkill", "-f", "convd.executor.*sc-kill"],
                       capture_output=True)
        self.tmp.cleanup()

    def _cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "convd.cli"] + list(args),
            cwd=ROOT, capture_output=True, text=True, timeout=120)

    def test_kill_resume_replay(self):
        proc = subprocess.Popen(
            [sys.executable, "-m", "convd.cli", "run",
             "--scenario", self.scenario_path,
             "--executors", self.executors_path,
             "--db", self.db],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 60
        while cursor_of(self.db) < 3 and time.time() < deadline:
            time.sleep(0.1)
        self.assertGreaterEqual(cursor_of(self.db), 3,
                                "orchestrator did not commit rounds")
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
        subprocess.run(["pkill", "-f", "convd.executor.*sc-kill"],
                       capture_output=True)
        time.sleep(0.3)

        committed = cursor_of(self.db)
        self.assertLess(committed, 8)
        scenario = load_scenario(self.scenario_path)
        executors = load_executor_set(self.executors_path)
        ledger = Ledger(self.db)
        # Uncommitted rounds are unknown and must be re-run.
        self.assertEqual(unknown_rounds(ledger, scenario),
                         list(range(committed + 1, 9)))
        # Committed prefix replays exactly.
        result = replay_verify(scenario, executors, ledger)
        self.assertTrue(result["digest_match"])
        self.assertEqual(result["replayed_rounds"], committed)
        ledger.close()

        resumed = self._cli("resume", "--scenario", self.scenario_path,
                            "--executors", self.executors_path,
                            "--db", self.db)
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        payload = json.loads(resumed.stdout)
        self.assertEqual(payload["resumed_rounds"],
                         list(range(committed + 1, 9)))
        self.assertEqual(cursor_of(self.db), 8)

        verified = self._cli("verify", "--scenario", self.scenario_path,
                             "--executors", self.executors_path,
                             "--db", self.db)
        self.assertEqual(verified.returncode, 0, verified.stderr)
        payload = json.loads(verified.stdout)
        self.assertTrue(payload["replay"]["digest_match"])
        self.assertEqual(payload["replay"]["replayed_rounds"], 8)
        self.assertTrue(payload["aggregate"]["ok"])

    def test_tampered_ledger_fails_replay_with_first_mismatch(self):
        run = self._cli("run", "--scenario", self.scenario_path,
                        "--executors", self.executors_path,
                        "--db", self.db)
        self.assertEqual(run.returncode, 0, run.stderr)
        conn = sqlite3.connect(self.db)
        row = conn.execute(
            "SELECT round_no, executor_id, logical FROM rounds"
            " ORDER BY round_no, executor_id LIMIT 1").fetchone()
        logical = json.loads(row[2])
        logical["conclusion"] = "not_converged"
        conn.execute(
            "UPDATE rounds SET logical=? WHERE round_no=? AND"
            " executor_id=?",
            (json.dumps(logical, sort_keys=True,
                        separators=(",", ":")), row[0], row[1]))
        conn.commit()
        conn.close()
        scenario = load_scenario(self.scenario_path)
        executors = load_executor_set(self.executors_path)
        ledger = Ledger(self.db)
        with self.assertRaises(ReplayError) as ctx:
            replay_verify(scenario, executors, ledger)
        exc = ctx.exception
        self.assertEqual(exc.code, "E_REPLAY")
        self.assertEqual(exc.fields["round"], row[0])
        self.assertEqual(exc.fields["executor_id"], row[1])
        self.assertEqual(exc.fields["field"], "conclusion")
        ledger.close()


if __name__ == "__main__":
    unittest.main()
