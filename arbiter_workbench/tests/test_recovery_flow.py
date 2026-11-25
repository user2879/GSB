import shutil
import tempfile
import unittest
from pathlib import Path

from arbiter.engine import ArbitrationEngine
from arbiter.ledger import Ledger
from arbiter.propagation import ChangeSet
from arbiter.protocol import pick_free_port
from arbiter.recovery import recover
from arbiter.sample import DEFAULT_KEYS
from arbiter.service_manager import ServiceManager
from arbiter.spec import EnvSpec, load_declarations

DECLS = Path(__file__).resolve().parents[1] / "decls"


class IncrementalRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="wb_rec_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _engine(self, db_name: str):
        decls = load_declarations(DECLS, DEFAULT_KEYS)
        envs = [
            EnvSpec(e.env_id, e.host, pick_free_port(), e.lease_capacity, e.seed, e.weight)
            for e in decls.environments.values()
        ]
        ledger = Ledger(self.tmp / db_name, session_scale=2)
        engine = ArbitrationEngine(decls, 2, ledger)
        return decls, engine, ledger, ServiceManager(envs)

    def test_replay_after_incremental_commit_matches_committed_key_set(self):
        decls, engine, ledger, manager = self._engine("main.sqlite")
        try:
            manager.start_all()
            engine.start_services_and_ingest(manager)
            engine.run_full()
            invalidations = engine.propagation.propagate(
                ChangeSet(changed_policies={"view-dr-quorum": ("pool.max_size",)})
            )
            engine.run_incremental(invalidations)
        finally:
            manager.shutdown_all()
            ledger.close()

        def replay_fn(view_id: str, version: int):
            _, eng, ro_ledger, mgr = self._engine("replay.sqlite")
            try:
                mgr.start_all()
                eng.start_services_and_ingest(mgr)
                with Ledger(self.tmp / "main.sqlite", session_scale=2) as probe:
                    keys = tuple(r.key for r in probe.load_records(view_id, version))
                return eng.arbitrate_view(view_id, version, only_keys=keys)
            finally:
                mgr.shutdown_all()
                ro_ledger.close()

        report = recover(self.tmp / "main.sqlite", 2, replay_fn)
        self.assertEqual(report.unknown_views, ())
        self.assertIn("view-dr-quorum", report.replay_checked_views)
        self.assertEqual(len(report.replay_checked_views), 3)


if __name__ == "__main__":
    unittest.main()
