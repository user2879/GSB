import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from arbiter.engine import ArbitrationEngine
from arbiter.errors import ReplayError, StateError
from arbiter.fixedpoint import FixedPoint
from arbiter.ledger import COMMITTED, Ledger, UNKNOWN, verify_replay
from arbiter.propagation import ChangeSet
from arbiter.sample import DEFAULT_KEYS, generate_all
from arbiter.protocol import pick_free_port
from arbiter.service_manager import ServiceManager
from arbiter.spec import load_declarations

DECLS = Path(__file__).resolve().parents[1] / "decls"
PORT_OFFSET = 100


def make_engine(workdir: Path, scale: int = 2, batch_size: int = 7,
                port_offset: int = PORT_OFFSET):
    decls = load_declarations(DECLS, DEFAULT_KEYS)
    shifted_envs = []
    for env in decls.environments.values():
        object.__setattr__(env, "port", env.port + port_offset)
        shifted_envs.append(env)
    ledger = Ledger(workdir / "ledger.sqlite", session_scale=scale)
    engine = ArbitrationEngine(decls, scale, ledger, batch_size=batch_size)
    manager = ServiceManager(shifted_envs)
    return decls, engine, ledger, manager


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="wb_e2e_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run_full(self, batch_size: int, run_tag: int):
        decls, engine, ledger, manager = make_engine(
            self.tmp / f"bs{batch_size}_{run_tag}", batch_size=batch_size,
        )
        try:
            manager.start_all()
            engine.start_services_and_ingest(manager)
            outputs = engine.run_full()
            return outputs, ledger.meta_digest()
        finally:
            manager.shutdown_all()
            ledger.close()

    def test_determinism_across_batch_splits_and_ports(self):
        out_a, digest_a = self._run_full(1, 1)
        out_b, digest_b = self._run_full(13, 2)
        out_c, digest_c = self._run_full(100, 3)
        ArbitrationEngine.assert_byte_identical([out_a, out_b, out_c])
        self.assertEqual(digest_a, digest_b)
        self.assertEqual(digest_b, digest_c)

    def test_ledger_atomic_commits_and_reader_view(self):
        _, engine, ledger, manager = make_engine(self.tmp / "atomic")
        try:
            manager.start_all()
            engine.start_services_and_ingest(manager)
            outputs = engine.run_full()
            latest = ledger.latest_versions()
            self.assertEqual(
                {v["status"] for v in latest.values()}, {COMMITTED}
            )
            for view_id, item in outputs.items():
                records = ledger.load_records(view_id, item["invalidation"].new_version)
                self.assertEqual(len(records), len(DEFAULT_KEYS))
                for record in records:
                    self.assertEqual(
                        record.error_total,
                        record.error_quant + record.error_boundary + record.error_norm,
                    )
            # reader sees a consistent committed view
            ro = sqlite3.connect(ledger.db_path)
            count = ro.execute(
                "SELECT COUNT(*) FROM key_record WHERE replay_status=?", (COMMITTED,)
            ).fetchone()[0]
            ro.close()
            self.assertEqual(count, len(outputs) * len(DEFAULT_KEYS))
        finally:
            manager.shutdown_all()
            ledger.close()

    def test_incremental_preserves_unaffected_keys_byte_identical(self):
        _, engine, ledger, manager = make_engine(self.tmp / "incr")
        try:
            manager.start_all()
            engine.start_services_and_ingest(manager)
            full = engine.run_full()
            before = {
                view_id: ledger.load_records(view_id, item["invalidation"].new_version)
                for view_id, item in full.items()
            }
            invalidations = engine.propagation.propagate(
                ChangeSet(changed_policies={"view-dr-quorum": ("pool.max_size",)})
            )
            engine.run_incremental(invalidations)
            after_dr = ledger.load_records("view-dr-quorum", 2)
            after_other = ledger.load_records("view-edge-policy", 1)
            dirty = {"pool.max_size"}
            for record in after_dr:
                if record.key not in dirty:
                    prior = next(r for r in before["view-dr-quorum"] if r.key == record.key)
                    self.assertEqual(record, prior)
            # untouched view remains byte identical
            self.assertEqual(after_other, before["view-edge-policy"])
        finally:
            manager.shutdown_all()
            ledger.close()

    def test_snapshot_roundtrip_and_whole_replacement(self):
        _, engine, ledger, manager = make_engine(self.tmp / "snap")
        snapshot_path = self.tmp / "snapshot.json"
        try:
            manager.start_all()
            engine.start_services_and_ingest(manager)
            engine.run_full()
            digest = ledger.export_snapshot(snapshot_path)
        finally:
            manager.shutdown_all()
            ledger.close()

        fresh_db = self.tmp / "restored" / "ledger.sqlite"
        fresh_db.parent.mkdir(parents=True)
        ledger2 = Ledger(fresh_db, session_scale=2)
        restored_digest = ledger2.replace_with_snapshot(snapshot_path)
        self.assertEqual(restored_digest, digest)
        ledger2.close()

        # tampered digest must trigger whole-replacement failure (no incremental merge)
        tampered = self.tmp / "tampered.json"
        data = json.loads(snapshot_path.read_text())
        data["content_digest"] = "0" * 64
        tampered.write_text(json.dumps(data))
        ledger3 = Ledger(self.tmp / "tampered_db" / "ledger.sqlite", 2)
        with self.assertRaises(StateError) as ctx:
            ledger3.replace_with_snapshot(tampered)
        self.assertEqual(ctx.exception.code, "E_STATE")
        ledger3.close()

    def test_unknown_commit_requires_rerun_and_replay_mismatch_e_replay(self):
        _, engine, ledger, manager = make_engine(self.tmp / "crash")
        try:
            manager.start_all()
            engine.start_services_and_ingest(manager)
            outputs = engine.run_full()
            # simulate an interrupted transaction: only UNKNOWN marker exists
            ledger.begin_unknown("view-edge-policy", 99)
            latest = ledger.latest_versions()
            self.assertEqual(latest["view-edge-policy"]["status"], UNKNOWN)
            good_view = "view-dr-quorum"
            version = outputs[good_view]["invalidation"].new_version
            stored = ledger.load_records(good_view, version)

            # successful replay matches
            rebuilt = engine.arbitrate_view(good_view, version)
            verify_replay(good_view, stored, rebuilt)

            # corrupt one authority value -> E_REPLAY locates view/key/field
            bad = [r for r in rebuilt]
            object.__setattr__(bad[0], "authority_q", bad[0].authority_q + 1)
            with self.assertRaises(ReplayError) as ctx:
                verify_replay(good_view, stored, bad)
            self.assertEqual(ctx.exception.code, "E_REPLAY")
            self.assertEqual(ctx.exception.context.object_id, good_view)
            self.assertEqual(ctx.exception.context.field, "authority_q")
            self.assertEqual(ctx.exception.context.detail, bad[0].key)
        finally:
            manager.shutdown_all()
            ledger.close()


if __name__ == "__main__":
    unittest.main()
