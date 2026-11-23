import os
import tempfile
import unittest

from convd.ledger import Ledger
from convd.orchestrator import Orchestrator
from convd.scene import ExecutorSet, Scenario

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def scenario(**over):
    data = {"scenario_id": "sc-orch", "max_rounds": 3, "window_ms": 400,
            "pulse_interval_ms": 30, "lease_ms": 250,
            "receipt_timeout_ms": 150, "budget_version": 1,
            "max_in_flight": 2, "max_in_flight_per_executor": 1,
            "max_retries": 1, "executor_set_version": 1}
    data.update(over)
    return Scenario(data)


def executors(port_base=42100):
    return ExecutorSet({"executors": [
        {"executor_id": "exe-a", "port": port_base, "target_digest": "ta",
         "seed": 5},
        {"executor_id": "exe-b", "port": port_base + 1,
         "target_digest": "tb", "seed": 9},
    ]})


class OrchestratorCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "t.db")

    def tearDown(self):
        self.tmp.cleanup()

    def run_orch(self, sc, ex, **kw):
        ledger = Ledger(self.db)
        orch = Orchestrator(sc, ex, ledger, **kw)
        orch.start()
        try:
            orch.run()
        finally:
            orch.stop()
        return ledger


class TestLease(OrchestratorCase):
    def test_converges_within_lease(self):
        ledger = self.run_orch(scenario(), executors(),
                               base_ms=1, jitter_ms=2)
        for _, _, entry in ledger.round_entries():
            self.assertEqual(entry["conclusion"], "converged")
            self.assertTrue(entry["in_lease"])
            self.assertLessEqual(entry["advance_ms"], 250)
        self.assertEqual(ledger.cursor(), 3)
        ledger.close()

    def test_lease_expiry_detected_with_real_timing(self):
        sc = scenario(lease_ms=2, receipt_timeout_ms=400)
        ledger = self.run_orch(sc, executors(42200), base_ms=30,
                               jitter_ms=0)
        for _, _, entry in ledger.round_entries():
            self.assertEqual(entry["conclusion"], "not_converged")
            self.assertEqual(entry["reason"], "lease_expired")
            self.assertFalse(entry["in_lease"])
        ledger.close()


class TestRetryIsolation(OrchestratorCase):
    def test_timeout_retry_booked_separately(self):
        ledger = self.run_orch(scenario(), executors(42300),
                               drop_map={"exe-a": [2]})
        retries = ledger.retry_entries()
        kinds = [(r, e, p["kind"]) for r, e, _, p in retries]
        self.assertIn((2, "exe-a", "timeout"), kinds)
        self.assertIn((2, "exe-a", "resend"), kinds)
        # Retried round still converges in its own round; retry does not
        # leak into any other round's conclusion.
        for round_no, executor_id, entry in ledger.round_entries():
            self.assertEqual(entry["conclusion"], "converged")
            if round_no == 2 and executor_id == "exe-a":
                self.assertEqual(entry["attempts"], 2)
            elif executor_id == "exe-a":
                self.assertEqual(entry["attempts"], 1)
        ledger.close()


class TestDeterminism(OrchestratorCase):
    def test_identical_digest_across_runs(self):
        ledger1 = self.run_orch(scenario(), executors(42400))
        digest1 = ledger1.content_digest()
        ledger1.close()
        os.remove(self.db)
        ledger2 = self.run_orch(scenario(), executors(42400))
        digest2 = ledger2.content_digest()
        ledger2.close()
        self.assertEqual(digest1, digest2)


class TestCrossRoundAnomalies(unittest.TestCase):
    def setUp(self):
        self.sc = scenario()
        self.ex = executors(42500)
        self.orch = Orchestrator(self.sc, self.ex, Ledger(":memory:"))
        self.spec = self.ex.specs[0]

    def tearDown(self):
        self.orch.ledger.close()

    def logical(self, round_no, version, obs, digest):
        return {"round": round_no, "executor_id": "exe-a",
                "version_before": version - 1, "version_after": version,
                "digest_before": None, "digest_after": digest,
                "obs_seq": obs, "advance_ms": 1, "attempts": 1,
                "in_lease": True, "conclusion": "converged",
                "reason": "ok"}

    def test_version_regression(self):
        self.orch._prev_logical = {"exe-a": self.logical(1, 5, 5, "x")}
        found = self.orch._cross_round_anomalies(
            self.spec, self.logical(2, 4, 6, "y"))
        kinds = [a["kind"] for a in found]
        self.assertIn("version_regression", kinds)

    def test_digest_jump(self):
        from convd.executor import state_digest_for
        good = state_digest_for("sc-orch", "exe-a", 1, "ta")
        self.orch._prev_logical = {"exe-a": self.logical(1, 1, 1, good)}
        found = self.orch._cross_round_anomalies(
            self.spec, self.logical(2, 2, 2, "tampered"))
        kinds = [a["kind"] for a in found]
        self.assertIn("digest_jump", kinds)

    def test_obs_seq_gap(self):
        from convd.executor import state_digest_for
        d1 = state_digest_for("sc-orch", "exe-a", 1, "ta")
        d3 = state_digest_for("sc-orch", "exe-a", 3, "ta")
        self.orch._prev_logical = {"exe-a": self.logical(1, 1, 1, d1)}
        found = self.orch._cross_round_anomalies(
            self.spec, self.logical(3, 3, 3, d3))
        kinds = [a["kind"] for a in found]
        self.assertIn("obs_seq_gap", kinds)
        self.assertNotIn("digest_jump", kinds)

    def test_clean_round_has_no_anomaly(self):
        from convd.executor import state_digest_for
        d1 = state_digest_for("sc-orch", "exe-a", 1, "ta")
        d2 = state_digest_for("sc-orch", "exe-a", 2, "ta")
        self.orch._prev_logical = {"exe-a": self.logical(1, 1, 1, d1)}
        found = self.orch._cross_round_anomalies(
            self.spec, self.logical(2, 2, 2, d2))
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
