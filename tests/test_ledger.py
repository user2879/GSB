import tempfile
import unittest
from pathlib import Path

from k8ssim.errors import ReplayError, SnapshotError
from k8ssim.session import Session
from k8ssim.storage.recovery import recover_and_replay

NODES_PATH = "samples/nodes.json"
PODS_PATH = "samples/pods.json"


class LedgerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.ledger_path = Path(self.temp.name) / "ledger.sqlite"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_format_version_and_digest_roundtrip(self) -> None:
        session = Session.create(NODES_PATH, PODS_PATH, self.ledger_path)
        session.run_full_schedule(snapshot_id="snap")
        rows = session.ledger.load_pod_results()
        self.assertEqual(len(rows), len(session.pods))
        self.assertTrue(all(row["digest"] for row in rows))
        session.close()

        reopened = Session.open(self.ledger_path)
        self.assertEqual(reopened.spec_digest(), session.spec_digest())
        result = recover_and_replay(reopened.ledger, reopened.engine)
        self.assertTrue(result.verified)
        reopened.close()

    def test_snapshot_mismatch_raises_e_snapshot_and_replaces(self) -> None:
        session = Session.create(NODES_PATH, PODS_PATH, self.ledger_path)
        session.run_full_schedule(snapshot_id="snap-v1")
        session.update_parameters(scale_factor=100)
        with self.assertRaises(SnapshotError) as ctx:
            session.ledger.verify_snapshot(
                "snap-v1", session.spec_digest(), session.params
            )
        self.assertEqual(ctx.exception.code, "E_SNAPSHOT")
        self.assertTrue(ctx.exception.mismatches)

    def test_replay_detects_first_field_mismatch(self) -> None:
        session = Session.create(NODES_PATH, PODS_PATH, self.ledger_path)
        session.run_full_schedule()
        # Tamper with one committed decision to force E_REPLAY.
        first = session.ledger.load_pod_results()[0]
        tampered = dict(first["decision"])
        tampered["bound_node"] = "node-zzz"
        import json
        session.ledger.conn.execute(
            "UPDATE pod_results SET decision_json=? WHERE seq=?",
            (json.dumps(tampered, sort_keys=True), first["seq"]),
        )
        session.ledger.conn.commit()
        with self.assertRaises(ReplayError) as ctx:
            recover_and_replay(session.ledger, session.engine)
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertEqual(ctx.exception.pod_id, first["pod_id"])
        self.assertEqual(ctx.exception.field, "bound_node")

    def test_unknown_pod_blocks_verification_and_rerun_repairs(self) -> None:
        session = Session.create(NODES_PATH, PODS_PATH, self.ledger_path)
        session.run_with_crash_marker(crash_before_seq=2)
        session.close()

        reopened = Session.open(self.ledger_path)
        recovery = reopened.recover()
        self.assertFalse(recovery.verified)
        self.assertEqual(recovery.unknown_pods, [reopened.pods[2].pod_id])
        reopened.rerun_unknown()
        verified = reopened.recover()
        self.assertTrue(verified.verified)
        reopened.close()

    def test_no_exit_report_claim_for_inflight_pod(self) -> None:
        session = Session.create(NODES_PATH, PODS_PATH, self.ledger_path)
        session.run_with_crash_marker(crash_before_seq=4)
        rows = session.ledger.load_pod_results()
        unknown = [row for row in rows if row["status"] == "UNKNOWN"]
        self.assertTrue(unknown)
        self.assertTrue(all(row["decision_json"] is None for row in unknown))
        session.close()


if __name__ == "__main__":
    unittest.main()

    def test_snapshot_restore_full_replacement_on_mismatch(self) -> None:
        from k8ssim.storage.recovery import restore_snapshot_or_replace

        session = Session.create(NODES_PATH, PODS_PATH, self.ledger_path)
        session.run_full_schedule(snapshot_id="snap-v1")
        nodes = session.nodes
        pods = session.pods
        session.engine.params
        # Mutate scale factor so the stored snapshot no longer matches.
        session.update_parameters(scale_factor=100)
        result = restore_snapshot_or_replace(
            session.ledger,
            "snap-v1",
            nodes,
            pods,
            session.params,
            session.spec_digest(),
        )
        self.assertTrue(result.replay_errors)
        # The ledger was fully replaced: no stale pod rows survive.
        self.assertEqual(session.ledger.load_pod_results(), [])
        session.close()
