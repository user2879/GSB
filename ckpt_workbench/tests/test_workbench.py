import json
import os
import shutil
import tempfile
import unittest

from workbench import app
from workbench.changeset import full_recompute
from workbench.dependencies import build_graph, propagate, validate_references
from workbench.errors import (
    ChainError,
    DependencyError,
    IncrementalError,
    LedgerError,
    SnapshotError,
)
from workbench.hashing import canon_bytes
from workbench.inject import (
    inject_dangling_parent,
    inject_shard_corruption,
    inject_snapshot_mismatch,
    run_validation_expect_failure,
)
from workbench.model import initial_state, load_declaration, plan_shards, shard_digest
from workbench.recovery import detect_inflight, replay_ledger, rollback_to_parent
from workbench.snapshot import compare_with_ledger, load_snapshot
from workbench.verifier import verify_checkpoint

DECL = {
    "seed": 20261123,
    "shard_count": 2,
    "partition_mode": "layer",
    "layer_dims": [8, 6, 4],
    "steps": 2,
}


class WorkbenchTestCase(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="ckptwb_")
        self.decl_path = os.path.join(self.root, "decl.json")
        with open(self.decl_path, "w", encoding="utf-8") as handle:
            json.dump(DECL, handle)
        self.workdir = os.path.join(self.root, "ws")
        self.decl = load_declaration(self.decl_path)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def write_change(self, name, payload):
        path = os.path.join(self.root, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path

    def test_deterministic_identical_digests(self):
        first = initial_state(DECL)
        second = initial_state(DECL)
        self.assertEqual(canon_bytes(first["rng_state"]), canon_bytes(second["rng_state"]))
        for plan in plan_shards("layer", DECL["layer_dims"]):
            self.assertEqual(
                shard_digest(plan["tensors"], first),
                shard_digest(plan["tensors"], second),
            )

    def test_dependency_graph_reference_complete_and_propagation(self):
        graph = build_graph("layer", DECL["layer_dims"])
        self.assertEqual(validate_references(graph), [])
        cursor_only = propagate(graph, ["cursor"])
        self.assertIn("content", cursor_only["affected"])
        self.assertNotIn("partition", cursor_only["affected"])
        self.assertEqual(cursor_only["affected_shards"], [])
        full = propagate(graph, ["param:W1", "momentum:W1"])
        self.assertIn("shard000", full["affected_shards"])
        self.assertIn("partition", full["affected"])
        self.assertIn("optimizer", full["affected"])
        self.assertTrue(
            any("-> shard:shard000" in e for e in full["evidence"]["shard:shard000"])
        )

    def test_incremental_unchanged_shards_byte_identical(self):
        app.initialize(self.workdir, self.decl_path)
        change = self.write_change("c.json", {
            "change_id": "cp_cursor",
            "ops": [{"op": "set_cursor", "value": 77}],
        })
        result = app.submit_change(self.workdir, self.decl_path, change)
        self.assertEqual(result["rewritten"], [])
        ledger = app.open_ledger(self.workdir)
        for shard in ledger.get_checkpoint("cp_cursor")["shards"]:
            old = ledger.get_shard("cp_baseline", shard["shard_id"])
            self.assertEqual(old["blob_path"], shard["blob_path"])
            self.assertEqual(old["shard_version"], shard["shard_version"])
            self.assertEqual(old["shard_digest"], shard["shard_digest"])
        verify_checkpoint(
            ledger, "cp_cursor",
            {"change_id": "cp_cursor", "ops": [{"op": "set_cursor", "value": 77}]},
            self.decl,
        )
        ledger.close()

    def test_partition_change_rewrites_all_shards(self):
        app.initialize(self.workdir, self.decl_path)
        change = self.write_change("p.json", {
            "change_id": "cp_unit",
            "ops": [{"op": "set_partition_mode", "mode": "unit"}],
        })
        result = app.submit_change(self.workdir, self.decl_path, change)
        ledger = app.open_ledger(self.workdir)
        entry = ledger.get_checkpoint("cp_unit")
        self.assertEqual(entry["shard_count"], 4)
        self.assertEqual(entry["partition_mode"], "unit")
        self.assertEqual(len(result["rewritten"]), 4)
        ledger.close()

    def test_e_incr_reports_first_mismatch(self):
        app.initialize(self.workdir, self.decl_path)
        ledger = app.open_ledger(self.workdir)
        ledger.begin()
        ledger.set_meta("progress_cursor", 12345)
        ledger.commit()
        ledger.close()
        ledger = app.open_ledger(self.workdir)
        bad_change = {"change_id": "cp_baseline",
                      "ops": [{"op": "set_cursor", "value": 1}]}
        with self.assertRaises(IncrementalError) as ctx:
            verify_checkpoint(ledger, "cp_baseline", bad_change, self.decl)
        self.assertEqual(ctx.exception.code, "E_INCR")
        self.assertIn("checkpoint=cp_baseline", ctx.exception.details[0])
        ledger.close()

    def test_idempotent_duplicate_submit(self):
        app.initialize(self.workdir, self.decl_path)
        path = self.write_change("t.json", {
            "change_id": "cp_train",
            "ops": [{"op": "train_steps", "count": 1}],
        })
        first = app.submit_change(self.workdir, self.decl_path, path)
        second = app.submit_change(self.workdir, self.decl_path, path)
        self.assertFalse(first.get("idempotent"))
        self.assertTrue(second["idempotent"])
        ledger = app.open_ledger(self.workdir)
        self.assertEqual(len(ledger.list_checkpoint_ids()), 2)
        ledger.close()

    def test_drop_referenced_tensor_raises_e_depend(self):
        app.initialize(self.workdir, self.decl_path)
        path = self.write_change("drop.json", {
            "change_id": "cp_drop",
            "ops": [{"op": "drop_tensor", "tensor": "W1"}],
        })
        with self.assertRaises(DependencyError) as ctx:
            app.submit_change(self.workdir, self.decl_path, path)
        self.assertEqual(ctx.exception.code, "E_DEPEND")
        self.assertTrue(any("cp_baseline" in d for d in ctx.exception.details))

    def test_e_integrity_and_e_chain(self):
        app.initialize(self.workdir, self.decl_path)
        db = os.path.join(self.workdir, "ledger.db")
        shards = os.path.join(self.workdir, "shards")
        inject_shard_corruption(db, shards, "cp_baseline", "shard000")
        self.assertEqual(
            run_validation_expect_failure(db, shards)["reported_code"], "E_INTEGRITY"
        )

        app.initialize(self.workdir, self.decl_path)
        inject_dangling_parent(db, shards, "cp_baseline")
        self.assertEqual(
            run_validation_expect_failure(db, shards)["reported_code"], "E_CHAIN"
        )
        ledger = app.open_ledger(self.workdir)
        with self.assertRaises(ChainError):
            ledger.validate()
        ledger.close()

    def test_snapshot_replace_and_e_snapshot(self):
        app.initialize(self.workdir, self.decl_path)
        snap_dir = os.path.join(self.root, "snap")
        app.snapshot_save(self.workdir, snap_dir)
        ledger = app.open_ledger(self.workdir)
        self.assertEqual(compare_with_ledger(load_snapshot(snap_dir), ledger), [])
        ledger.close()

        divergent = self.write_change("div.json", {
            "change_id": "cp_div",
            "ops": [{"op": "set_cursor", "value": 555}],
        })
        app.submit_change(self.workdir, self.decl_path, divergent)
        ledger = app.open_ledger(self.workdir)
        self.assertTrue(compare_with_ledger(load_snapshot(snap_dir), ledger))
        ledger.close()

        restore = app.snapshot_restore(self.workdir, snap_dir)
        self.assertIn("cp_baseline", restore["replaced_checkpoints"])
        self.assertTrue(os.path.isdir(restore["backup_dir"]))
        self.assertTrue(app.snapshot_status(self.workdir, snap_dir)["identical"])

        inject_snapshot_mismatch(snap_dir)
        with self.assertRaises(SnapshotError) as ctx:
            load_snapshot(snap_dir)
        self.assertEqual(ctx.exception.code, "E_SNAPSHOT")

    def test_crash_unknown_shards_and_replay(self):
        app.initialize(self.workdir, self.decl_path)
        path = self.write_change("pending.json", {
            "change_id": "cp_pending",
            "ops": [{"op": "train_steps", "count": 1}],
        })
        app.submit_change(self.workdir, self.decl_path, path, crash_point="before_commit")
        ledger = app.open_ledger(self.workdir)
        inflight = detect_inflight(ledger)
        self.assertIsNotNone(inflight)
        self.assertEqual(len(inflight["unknown_shards"]), 2)
        rollback_to_parent(ledger)
        self.assertIsNone(detect_inflight(ledger))
        report = replay_ledger(ledger, self.decl, full_recompute)
        self.assertEqual(report["status"], "OK")
        ledger.close()

    def test_ledger_rejects_version_regression(self):
        app.initialize(self.workdir, self.decl_path)
        path = self.write_change("t.json", {
            "change_id": "cp_train",
            "ops": [{"op": "train_steps", "count": 1}],
        })
        app.submit_change(self.workdir, self.decl_path, path)
        ledger = app.open_ledger(self.workdir)
        # structural check runs before digest check; equal versions violate
        # the strictly-increasing ledger_version rule.
        ledger.conn.execute(
            "UPDATE checkpoints SET ledger_version=1 WHERE checkpoint_id='cp_train'"
        )
        with self.assertRaises(LedgerError):
            ledger._validate_structure()
        ledger.close()

    def test_replay_detects_corruption(self):
        app.initialize(self.workdir, self.decl_path)
        ledger = app.open_ledger(self.workdir)
        ledger.conn.execute(
            "UPDATE checkpoints SET cursor=999 WHERE checkpoint_id='cp_baseline'"
        )
        ledger.close()
        ledger = app.open_ledger(self.workdir)
        from workbench.errors import ReplayError
        with self.assertRaises(ReplayError):
            replay_ledger(ledger, self.decl, full_recompute)
        ledger.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
