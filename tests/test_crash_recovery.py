"""Requirement 8: forced node termination, restart recovery and unknown rerun."""
import os
import tempfile
import unittest

from shardmeta.engine import AccountingEngine
from shardmeta.ledger import Ledger
from shardmeta.models import NodeSpec, ObjectManifest, Request, ShardPlacement
from shardmeta.node import NodeOrchestrator


class CrashRecoveryTest(unittest.TestCase):
    def test_kill_restart_recovers_from_committed_ledger_only(self):
        tmp = tempfile.mkdtemp()
        nodes = {
            "n1": NodeSpec("n1", "127.0.0.1", 46001, 4, 3, 1, 1),
        }
        manifests = {
            "obj": ObjectManifest("obj", 8, 8, 3,
                                  (ShardPlacement(0, "n1", 8),)),
        }
        ledger = Ledger(os.path.join(tmp, "ledger.sqlite3"))
        orchestrator = NodeOrchestrator(nodes)
        orchestrator.start_all()
        try:
            engine = AccountingEngine(nodes, manifests, ledger,
                                      orchestrator=orchestrator, lease_ttl_ms=2000)
            engine.process_request(Request(1, "grant", "obj", 0, ttl_ms=2000))

            # Force-kill the node before the upload is committed.
            orchestrator.kill_node("n1")
            crashed_request = Request(2, "upload", "obj", 0)
            engine.mark_unknown_and_rerun(crashed_request, "node force-killed")
            unknown = ledger.fetch_unknown()
            self.assertEqual(len(unknown), 1)
            self.assertEqual(unknown[0]["reason"], "node force-killed")

            # Restart and recover only the already-committed lease.
            orchestrator.restart_node(
                "n1", recovery_leases=[lease.to_dict()
                                       for lease in engine._leases.values()])
            restored = engine.restore_state_from_ledger()
            self.assertEqual(restored["leases"], 1)
            self.assertEqual(restored["unknown_requests"][0]["seq"], 2)

            # Rerun the previously unknown request successfully.
            rerun = engine.process_request(crashed_request)
            self.assertEqual(rerun.op, "upload")
            self.assertIn("checksum", rerun.detail)
            engine.verify_against_ledger()
        finally:
            orchestrator.stop_all()
            ledger.close()


if __name__ == "__main__":
    unittest.main()
