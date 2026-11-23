import json
import tempfile
import unittest
from pathlib import Path

from diagconsole.engine import Engine, NodeKey
from diagconsole.errors import DiagError, canonical_json
from diagconsole.ledger import Ledger
from diagconsole.model import load_spec
from diagconsole.snapshot import build_snapshot, load_snapshot

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


class LedgerRollbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.spec = load_spec(SAMPLES / "clusters.json",
                              SAMPLES / "tasks.json",
                              SAMPLES / "revisions.json")
        self.ledger = Ledger(self.dir / "ledger.db")
        self.engine = Engine(self.spec, self.ledger)
        self.engine.apply_chain(None, "seed-batch")

    def tearDown(self):
        self.ledger.close()
        self.tmp.cleanup()

    def test_idempotent_rewrite(self):
        key = NodeKey(self.spec.tasks[0].task_id,
                      self.spec.tasks[0].targets[0], 1)
        node = self.engine.compute_node_locally(key)
        with self.ledger.transaction():
            first = self.ledger.upsert_node(node)
            second = self.ledger.upsert_node(node)
        self.assertEqual(first, "identical")
        self.assertEqual(second, "identical")

    def test_non_idempotent_conflict_rejected(self):
        key = NodeKey(self.spec.tasks[0].task_id,
                      self.spec.tasks[0].targets[0], 1)
        node = self.engine.compute_node_locally(key)
        node["aggregate"]["total"] += 1
        with self.assertRaises(DiagError) as ctx:
            with self.ledger.transaction():
                self.ledger.upsert_node(node)
        self.assertEqual(ctx.exception.code, "E_LEDGER")

    def test_uncommitted_transaction_invisible(self):
        other = Ledger(self.dir / "ledger.db")
        before = len(other.list_nodes())
        key = NodeKey(self.spec.tasks[0].task_id,
                      self.spec.tasks[0].targets[0], 2)
        node = self.engine.compute_node_locally(key)
        try:
            self.ledger.conn.execute("BEGIN")
            self.ledger.upsert_node(
                {**node, "version": 2, "parent": 1,
                 "task_id": key.task_id, "cluster_id": key.cluster_id})
            self.assertEqual(len(other.list_nodes()), before)
        finally:
            self.ledger.conn.rollback()
        other.close()

    def test_rollback_isolates_stale_nodes(self):
        result = self.engine.rollback(3)
        self.assertTrue(result["isolated_nodes"])
        view = self.engine.active_view(3)
        active_versions = {n["revision"] for n in view["active"]}
        self.assertEqual(active_versions, {1, 2, 3})
        isolated_versions = {n["revision"] for n in view["isolated"]}
        self.assertEqual(isolated_versions, {4, 5, 6})
        self.assertTrue(all(item["after_digest"] for item in
                            result["digest_comparison"]))

    def test_snapshot_roundtrip_and_chain_mismatch(self):
        nodes = self.ledger.list_nodes("committed")
        snapshot = build_snapshot(self.spec, 6, nodes, {"stage": "test"})
        path = self.dir / "snapshot.json"
        import diagconsole.snapshot as snap_mod
        snap_mod.save_snapshot(path, snapshot)
        loaded = load_snapshot(path, self.spec)
        self.assertEqual(loaded["root_version"], 6)
        loaded["chain"]["digest"] = "deadbeef"
        path2 = self.dir / "bad.json"
        snap_mod.save_snapshot(path2, loaded)
        with self.assertRaises(DiagError) as ctx:
            load_snapshot(path2, self.spec)
        self.assertEqual(ctx.exception.code, "E_SNAPSHOT")


if __name__ == "__main__":
    unittest.main()


class AuthorityRebuildDriftTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.spec = load_spec(SAMPLES / "clusters.json",
                              SAMPLES / "tasks.json",
                              SAMPLES / "revisions.json")
        self.ledger = Ledger(Path(self.tmp.name) / "drift.db")
        self.engine = Engine(self.spec, self.ledger)
        self.engine.apply_chain(None, "seed")

    def tearDown(self):
        self.ledger.close()
        self.tmp.cleanup()

    def test_drift_is_reported_as_e_prev_with_location(self):
        from diagconsole.errors import canonical_json
        with self.ledger.transaction() as conn:
            row = conn.execute(
                "SELECT task_id, cluster_id, version, aggregate_json "
                "FROM nodes ORDER BY version, task_id, cluster_id LIMIT 1"
            ).fetchone()
            aggregate = json.loads(row[3])
            aggregate["total"] += 1
            conn.execute(
                "UPDATE nodes SET aggregate_json=? WHERE task_id=? AND "
                "cluster_id=? AND version=?",
                (canonical_json(aggregate), row[0], row[1], row[2]))
        with self.assertRaises(DiagError) as ctx:
            self.engine.rebuild(1, use_services=False)
        self.assertEqual(ctx.exception.code, "E_PREV")
        self.assertEqual(ctx.exception.location["field"], "aggregate")
        self.assertEqual(ctx.exception.location["revision"], 1)
