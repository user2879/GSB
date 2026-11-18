import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
SPEC = os.path.join(ROOT, "samples", "spec.json")
SPEC_V2 = os.path.join(ROOT, "samples", "spec-v2.json")
SPEC_DRIFT = os.path.join(ROOT, "samples", "spec-drift.json")
SPEC_CORRUPT = os.path.join(ROOT, "samples", "spec-corrupt.json")
SAMPLE_BAD = os.path.join(ROOT, "samples", "bad-event-overrides.json")
SPEC_TIGHT = os.path.join(ROOT, "samples", "spec-tight-budget.json")


def run_cli(*args):
    env = dict(os.environ, PYTHONPATH=SRC)
    proc = subprocess.run(
        [sys.executable, "-m", "swa", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc


class CliAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def _paths(self, name):
        return (
            os.path.join(self.dir, f"{name}.sqlite"),
            os.path.join(self.dir, f"{name}.json"),
            os.path.join(self.dir, f"{name}.log"),
            os.path.join(self.dir, f"{name}-report.json"),
        )

    def test_results_are_identical_across_schedules_and_generation_is_stable(self):
        digests = set()
        for limit in (1, 3):
            ledger, result, log, report = self._paths(f"g{limit}")
            proc = run_cli("run", "--spec", SPEC, "--ledger", ledger, "--result", result,
                           "--log", log, "--global-limit", str(limit), "--report", report)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            digests.add(json.load(open(result))["result_digest"])
        self.assertEqual(len(digests), 1)

    def test_forced_termination_and_recovery_do_not_duplicate_windows(self):
        ledger, result, log, report = self._paths("crash")
        proc = run_cli("run", "--spec", SPEC, "--ledger", ledger, "--result", result,
                       "--log", log, "--global-limit", "3", "--crash-after-commits", "3",
                       "--report", report)
        self.assertEqual(proc.returncode, 137)
        import sqlite3

        conn = sqlite3.connect(ledger)
        pre = conn.execute("SELECT count(*) FROM window_results").fetchone()[0]
        conn.close()
        self.assertEqual(pre, 3)

        ledger2, result2, log2, report2 = self._paths("crash2")
        proc = run_cli("run", "--spec", SPEC, "--ledger", ledger, "--result", result2,
                       "--log", log2, "--global-limit", "3", "--report", report2)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        conn = sqlite3.connect(ledger)
        rows = conn.execute("SELECT count(*) FROM window_results").fetchone()[0]
        distinct = conn.execute(
            "SELECT count(*) FROM (SELECT rule_id, window_start_ms FROM window_results GROUP BY rule_id, window_start_ms)"
        ).fetchone()[0]
        conn.close()
        self.assertEqual(rows, distinct)
        clean_ledger, clean_result, _, _ = self._paths("clean")
        run_cli("run", "--spec", SPEC, "--ledger", clean_ledger, "--result", clean_result,
                "--log", os.path.join(self.dir, "clean.log"), "--global-limit", "3")
        self.assertEqual(open(result2, "rb").read(), open(clean_result, "rb").read())

    def test_replay_verifies_committed_windows(self):
        ledger, result, log, report = self._paths("replay")
        run_cli("run", "--spec", SPEC, "--ledger", ledger, "--result", result,
                "--log", log, "--global-limit", "3")
        replay_report = os.path.join(self.dir, "replay-verify.json")
        proc = run_cli("replay", "--spec", SPEC, "--ledger", ledger,
                       "--log", os.path.join(self.dir, "replay.log"),
                       "--global-limit", "3", "--report", replay_report)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.load(open(replay_report))
        self.assertEqual(payload["status"], "ok")
        self.assertGreater(payload["verified_windows"], 0)
        self.assertEqual(payload["missing_windows"], [])

    def test_rule_version_evolution_replaces_the_current_view(self):
        ledger, result, log, report = self._paths("evolve")
        run_cli("run", "--spec", SPEC, "--ledger", ledger, "--result", result,
                "--log", log, "--global-limit", "3")
        _, v2_result, v2_log, v2_report = self._paths("evolve2")
        proc = run_cli("run", "--spec", SPEC_V2, "--ledger", ledger, "--result", v2_result,
                       "--log", v2_log, "--global-limit", "3", "--report", v2_report)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.load(open(v2_report))
        self.assertTrue(payload["view_replacement"]["replaced"])
        self.assertIn("r2s-all", payload["view_replacement"]["affected_rules"])
        self.assertGreater(len(payload["view_replacement"]["affected_windows"]), 0)

    def test_same_version_parameter_drift_is_e_replay(self):
        ledger, result, log, report = self._paths("drift")
        run_cli("run", "--spec", SPEC, "--ledger", ledger, "--result", result,
                "--log", log, "--global-limit", "3")
        _, drift_result, drift_log, drift_report = self._paths("drift2")
        proc = run_cli("run", "--spec", SPEC_DRIFT, "--ledger", ledger, "--result", drift_result,
                       "--log", drift_log, "--report", drift_report)
        self.assertEqual(proc.returncode, 7)
        payload = json.load(open(drift_report))
        self.assertEqual(payload["code"], "E_REPLAY")

    def test_structured_business_error_injections(self):
        # E_SPEC via corrupt declaration.
        ledger, result, log, report = self._paths("espec")
        proc = run_cli("run", "--spec", SPEC_CORRUPT, "--ledger", ledger, "--result", result,
                       "--log", log, "--inject", "corrupt-spec", "--report", report)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(json.load(open(report))["code"], "E_SPEC")

        # E_LEDGER via unsupported format version.
        ledger, result, log, report = self._paths("eledger")
        proc = run_cli("run", "--spec", SPEC, "--ledger", ledger, "--result", result,
                       "--log", log, "--inject", "unsupported-ledger-version", "--report", report)
        self.assertEqual(proc.returncode, 5)
        self.assertEqual(json.load(open(report))["code"], "E_LEDGER")

        # E_BUDGET via sustained memory exhaustion.
        ledger, result, log, report = self._paths("ebudget")
        proc = run_cli("run", "--spec", SPEC_TIGHT, "--ledger", ledger, "--result", result,
                       "--log", log, "--inject", "budget-exhausted", "--report", report)
        self.assertEqual(proc.returncode, 6)
        payload = json.load(open(report))
        self.assertEqual(payload["code"], "E_BUDGET")
        self.assertEqual(payload["details"]["memory_budget"], 1)

        # E_EVENT is a partial-success run carrying structured rejection detail.
        ledger, result, log, report = self._paths("eevent")
        proc = run_cli("run", "--spec", SPEC, "--samples", SAMPLE_BAD, "--ledger", ledger,
                       "--result", result, "--log", log, "--report", report)
        self.assertEqual(proc.returncode, 0)
        payload = json.load(open(report))
        self.assertEqual(payload["event_report"]["rejected_events"], 1)
        self.assertEqual(payload["event_report"]["rejected"][0]["field"], "value")


if __name__ == "__main__":
    unittest.main()
