from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gsbcache.content import artifact_filename
from gsbcache.converge import check_convergence
from gsbcache.errors import ConvergeError, VerifyError
from gsbcache.ledger import GENESIS_HASH, Ledger
from gsbcache.processes import NodeCluster
from gsbcache.runner import PipelineRun
from gsbcache.util import sha256_value
from gsbcache.verify import verify_ledger

from tests.helpers import make_sample


class VerifyConvergeTests(unittest.TestCase):
    def test_independent_replay_and_restart_convergence_failures_are_located(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            spec_path, spec, _ = make_sample(root)
            work = root / "work"
            run = PipelineRun(
                spec_path,
                work,
                lease_ms=180,
                queue_timeout_ms=5000,
                wait_for_expiry=True,
            ).execute()
            ledger_path = work / "ledger.sqlite3"
            nodes_root = work / "nodes"
            self.assertTrue(run["verification"]["ok"])
            self.assertTrue(run["convergence"]["ok"])
            self.assertTrue(verify_ledger(spec_path, ledger_path)["ok"])
            self.assertTrue(check_convergence(spec_path, ledger_path, nodes_root)["ok"])

            with Ledger(ledger_path, spec, read_only=True) as ledger:
                lease_rows = ledger.conn.execute(
                    "SELECT node_id,artifact_keys,digests FROM leases ORDER BY created_seq"
                ).fetchall()
                target = dict(lease_rows[0])
                target_keys = json.loads(target["artifact_keys"])
                target_key = target_keys[0]
                target_node = target["node_id"]
                expected_digest = json.loads(target["digests"])[target_key]
            artifact_path = nodes_root / target_node / "artifacts" / artifact_filename(target_key)
            artifact_path.write_bytes(artifact_path.read_bytes() + b"diverged")
            cluster = NodeCluster(spec, spec_path, nodes_root)
            try:
                nodes = cluster.start_all()
                inventory = nodes[target_node].client.inventory(target_node)
                self.assertNotEqual(inventory["digests"][target_key], expected_digest)
                self.assertNotIn(target_key, inventory["reusable_keys"])
            finally:
                cluster.stop_all()
            with self.assertRaises(ConvergeError) as ctx:
                check_convergence(spec_path, ledger_path, nodes_root)
            self.assertEqual(ctx.exception.code, "E_CONVERGE")
            self.assertEqual(ctx.exception.details["node_id"], target_node)
            self.assertEqual(ctx.exception.details["artifact_key"], target_key)
            self.assertEqual(ctx.exception.details["deviating_nodes"], [target_node])
            self.assertIn(target_key, ctx.exception.details["deviating_artifact_keys"][target_node])
            # Replay uses the immutable ledger and is unaffected by a tampered replica.
            self.assertTrue(verify_ledger(spec_path, ledger_path)["ok"])

            ledger = Ledger(ledger_path, spec)
            with ledger.transaction():
                rows = ledger.conn.execute("SELECT * FROM events ORDER BY seq").fetchall()
                previous = GENESIS_HASH
                first = True
                for row in rows:
                    item = dict(row)
                    item["artifact_keys"] = json.loads(item["artifact_keys"])
                    item["digests"] = json.loads(item["digests"])
                    item["details"] = json.loads(item["details"])
                    if first and item["event_type"] == "lease_granted":
                        item["digests"][item["artifact_keys"][0]] = "0" * 64
                        first = False
                    item["prev_hash"] = previous
                    record = {key: value for key, value in item.items() if key != "event_hash"}
                    event_hash = sha256_value(record)
                    item["event_hash"] = event_hash
                    ledger.conn.execute(
                        "UPDATE events SET digests=?,prev_hash=?,event_hash=? WHERE seq=?",
                        (
                            json.dumps(item["digests"], sort_keys=True, separators=(",", ":")),
                            item["prev_hash"],
                            event_hash,
                            item["seq"],
                        ),
                    )
                    previous = event_hash
                ledger.conn.execute(
                    "UPDATE meta SET value=? WHERE key='last_event_hash'",
                    (previous,),
                )
                ledger._refresh_digest(ledger.conn)
            ledger.close()
            with self.assertRaises(VerifyError) as ctx:
                verify_ledger(spec_path, ledger_path)
            self.assertEqual(ctx.exception.code, "E_VERIFY")
            self.assertEqual(ctx.exception.details["field"], "digests")
            self.assertIsNotNone(ctx.exception.details["artifact_key"])
            self.assertIsNotNone(ctx.exception.details["node_id"])
            self.assertIsNotNone(ctx.exception.details["event_seq"])


if __name__ == "__main__":
    unittest.main()
