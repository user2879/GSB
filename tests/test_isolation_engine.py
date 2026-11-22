from __future__ import annotations

import unittest

from txdiag.engine import execute
from txdiag.model import RuntimeConfig
from txdiag.spec import load_spec

from .helpers import sample


class IsolationEngineTests(unittest.TestCase):
    def test_all_isolation_visibility(self) -> None:
        spec = load_spec(sample("all_isolation"))
        result = execute(spec, RuntimeConfig(max_running=8, wait_timeout_ms=1000,
                                             wait_budget_ms=5000), "row")
        by_tx = {item["txid"]: item for item in result.isolation_conclusions}
        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(by_tx["RC1"]["outcome"], "done")
        self.assertEqual([read["value"] for read in by_tx["RC1"]["reads"][:2]], [0, 10])
        self.assertEqual([read["value"] for read in by_tx["RR1"]["reads"][:2]], [0, 0])
        self.assertTrue(by_tx["RR1"]["reads"][1]["fixed"])
        self.assertEqual(by_tx["RU1"]["reads"][0]["value"], 10)
        self.assertEqual(by_tx["RU1"]["reads"][0]["version_status"], "uncommitted")
        self.assertEqual(by_tx["RU1"]["reads"][0]["source"], "uncommitted_latest")
        self.assertEqual(by_tx["SER1"]["reads"][-1]["source"], "own_write")
        self.assertGreaterEqual(len(result.worker_threads), 4)

    def test_real_wait_and_release(self) -> None:
        spec = load_spec(sample("wait"))
        result = execute(spec, RuntimeConfig(max_running=4, wait_timeout_ms=1000,
                                             wait_budget_ms=5000), "row")
        conflict = [event for payload in result.rounds for event in payload["events"]
                    if event["event"] == "lock_conflict"]
        self.assertTrue(conflict)
        waits = [item for observed in result.observed_rounds
                 for item in observed["wait_observations"]
                 if item.get("event") == "granted"]
        self.assertTrue(waits)
        self.assertGreater(waits[0]["observed_wait_ms"], 0)
        self.assertEqual(result.result["transactions"], {"HOLDER": "done", "WAITER": "done"})
        self.assertGreater(result.result["wait_budget"]["canonical_used_ms"], 0)

    def test_deadlock_is_deterministic(self) -> None:
        spec = load_spec(sample("deadlock"))
        cfg = RuntimeConfig(max_running=4, wait_timeout_ms=1000, wait_budget_ms=5000)
        signatures = []
        for _ in range(3):
            result = execute(spec, cfg, "row")
            self.assertEqual(len(result.deadlocks), 1)
            deadlock = result.deadlocks[0]
            self.assertEqual(deadlock["cycle"], ["TX1", "TX2"])
            self.assertEqual(deadlock["victim"], "TX2")
            self.assertEqual(deadlock["basis"], "max(start_seq,txid)")
            signatures.append((deadlock["cycle"], deadlock["victim"],
                                [(p["round"], p["wait_graph"]["edges"]) for p in result.rounds]))
        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(signatures[0], signatures[2])

    def test_timeout_uses_monotonic_clock(self) -> None:
        spec = load_spec(sample("timeout"))
        result = execute(spec, RuntimeConfig(
            max_running=4, wait_timeout_ms=40, wait_budget_ms=500,
            inject=("hold-lock=SLOW:1",)), "row")
        self.assertEqual(result.status, "COMPLETED_WITH_TIMEOUTS")
        self.assertEqual(len(result.timeouts), 1)
        timeout = result.timeouts[0]
        self.assertEqual((timeout["txid"], timeout["row"], timeout["mode"]),
                         ("TIMEOUT", "slow:A", "X"))
        self.assertGreaterEqual(timeout["observed_wait_ms"], 39.0)
        self.assertEqual(result.result["transactions"]["TIMEOUT"], "aborted")


if __name__ == "__main__":
    unittest.main()
