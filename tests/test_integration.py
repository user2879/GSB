from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gsb.cluster import NodeCluster
from gsb.errors import LeaseError
from gsb.lease import LeaseManager
from gsb.runner import run_spec
from gsb.spec import load_spec
from gsb.verify import verify_ledger

from tests.factories import sample_spec_with_free_ports, single_node_spec, write_spec


class FullPipelineTests(unittest.TestCase):
    def test_six_shard_pipeline_and_verify(self) -> None:
        raw = sample_spec_with_free_ports()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            spec_path = write_spec(tmp_path / "spec.json", raw)
            db_path = tmp_path / "ledger.db"
            outcome = run_spec(
                spec_path,
                db_path,
                log_dir=tmp_path / "logs",
                snapshot_path=tmp_path / "snapshot.json",
            )
            self.assertEqual(outcome["result"]["status"], "completed")
            self.assertEqual(len(outcome["result"]["outcomes"]), 6)
            verify = verify_ledger(spec_path, db_path)
            self.assertEqual(verify["verified_shards"], 6)
            self.assertTrue((tmp_path / "snapshot.json").exists())

            waiting = {row["shard_id"]: row for row in outcome["result"]["wait_records"]}
            self.assertIn("global_budget", waiting["s3"]["reasons"])

    def test_node_concurrency_quota_queues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            spec_path, _ = single_node_spec(
                tmp_path,
                concurrency=1,
                lease_limit=4,
                processing_delay_ms=40,
                shards=[
                    {
                        "shard_id": f"s{i}",
                        "node_id": "n1",
                        "dependencies": [],
                        "rows": 8,
                        "resource_units": 1,
                        "priority": 10 - i,
                    }
                    for i in range(3)
                ],
            )
            outcome = run_spec(spec_path, tmp_path / "q.db", log_dir=tmp_path / "logs")
            reasons = {
                row["shard_id"]: row["reasons"] for row in outcome["result"]["wait_records"]
            }
            self.assertIn("node_concurrency_quota", reasons["s1"] + reasons["s2"])
            verify_ledger(spec_path, tmp_path / "q.db")

    def test_lease_limit_queues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            spec_path, _ = single_node_spec(
                tmp_path,
                concurrency=4,
                lease_limit=1,
                processing_delay_ms=30,
            )
            outcome = run_spec(spec_path, tmp_path / "lease.db", log_dir=tmp_path / "logs")
            reasons = [
                reason
                for record in outcome["result"]["wait_records"]
                for reason in record["reasons"]
            ]
            self.assertIn("node_lease_limit", reasons)
            verify_ledger(spec_path, tmp_path / "lease.db")


class NodeLeaseProtocolTests(unittest.TestCase):
    def test_optimistic_renewal_goes_through_tcp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            spec_path, _ = single_node_spec(tmp_path)
            spec = load_spec(spec_path)
            manager = LeaseManager(ttl_ms=1000)
            lease = manager.grant(
                shard_id="s1",
                node_id="n1",
                capacity_version=1,
                resource_units=1,
            )
            with NodeCluster(spec, log_dir=tmp_path / "logs") as cluster:
                cluster.grant_lease_on_node(lease.to_dict())
                ok = cluster.renew_lease_on_node(
                    node_id="n1",
                    lease_id=lease.lease_id,
                    expected_version=1,
                    ttl_ms=1000,
                )
                self.assertEqual(ok["lease"]["lease_version"], 2)
                with self.assertRaises(LeaseError) as ctx:
                    cluster.renew_lease_on_node(
                        node_id="n1",
                        lease_id=lease.lease_id,
                        expected_version=1,
                    )
                self.assertEqual(ctx.exception.code, "E_LEASE")


if __name__ == "__main__":
    unittest.main()
