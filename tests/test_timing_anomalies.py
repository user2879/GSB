import os
import tempfile
import unittest

from convd import wire
from convd.executor import state_digest_for
from convd.ledger import Ledger
from convd.orchestrator import Orchestrator, mono_ms
from convd.scene import ExecutorSet, Scenario


def scenario():
    return Scenario({"scenario_id": "sc-dup", "max_rounds": 2,
                     "window_ms": 400, "pulse_interval_ms": 20,
                     "lease_ms": 250, "receipt_timeout_ms": 150,
                     "budget_version": 1, "max_in_flight": 2,
                     "max_in_flight_per_executor": 1, "max_retries": 0,
                     "executor_set_version": 1})


def executors(port):
    return ExecutorSet({"executors": [
        {"executor_id": "d-a", "port": port, "target_digest": "ta",
         "seed": 8}]})


class TestDuplicateAndLateReceipts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "t.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_duplicate_obs_seq_keeps_first_and_registers_override(self):
        ledger = Ledger(self.db)
        orch = Orchestrator(scenario(), executors(44101), ledger,
                            base_ms=1, jitter_ms=1)
        orch.start()
        try:
            orch.run(start_round=1, up_to=1)
            # Forged replay: current round's number but an already
            # accepted observation sequence number.
            forged = {"executor_id": "d-a", "round": 2, "obs_seq": 1,
                      "obs_mono_ms": mono_ms(),
                      "state_digest": state_digest_for("sc-dup", "d-a", 1,
                                                       "ta")}
            orch.inboxes["d-a"].put(("receipt", forged, mono_ms()))
            orch.run(start_round=2, up_to=2)
        finally:
            orch.stop()
        anomalies = ledger.anomaly_entries()
        dup = [a for a in anomalies if a[2] == "duplicate_receipt"]
        self.assertEqual(len(dup), 1)
        self.assertEqual(dup[0][3]["kept"], "first")
        self.assertEqual(dup[0][3]["obs_seq"], 1)
        # Round 2 still converges on the genuine receipt (obs_seq 2).
        entries = {r: e for r, _, e in ledger.round_entries()}
        self.assertEqual(entries[2]["conclusion"], "converged")
        self.assertEqual(entries[2]["obs_seq"], 2)
        ledger.close()

    def test_late_receipt_does_not_modify_closed_round(self):
        ledger = Ledger(self.db)
        orch = Orchestrator(scenario(), executors(44102), ledger,
                            base_ms=1, jitter_ms=1)
        orch.start()
        try:
            orch.run(start_round=1, up_to=2)
            before = {r: e for r, e, _ in
                      [(r, e, x) for r, e, x in ledger.round_entries()]}
            digest_before = ledger.content_digest()
            # A receipt for closed round 1 arrives after the fact.
            late = wire.encode_receipt(
                "d-a", 1, 1, mono_ms(),
                state_digest_for("sc-dup", "d-a", 1, "ta"))
            orch.sock.sendto(late, ("127.0.0.1",
                                    orch.sock.getsockname()[1]))
            import time
            time.sleep(0.3)
            with orch.lock:
                pending = list(orch.pending_anomalies)
            self.assertTrue(any(a["kind"] == "late_receipt"
                                and a["payload"]["round"] == 1
                                for a in pending))
            # Closed rounds are untouched.
            after = {r: e for r, e, _ in
                     [(r, e, x) for r, e, x in ledger.round_entries()]}
            self.assertEqual(before, after)
            self.assertEqual(digest_before, ledger.content_digest())
        finally:
            orch.stop()
            ledger.close()


if __name__ == "__main__":
    unittest.main()
