import tempfile
import unittest
from pathlib import Path

from score_pipeline.engine import Engine, make_operation
from score_pipeline.errors import (
    E_SNAPSHOT_STALE,
    E_TIMELINE_HASH,
    E_VERSION_REGRESS,
    PipelineError,
)
from score_pipeline.generator import generate_score
from score_pipeline.injection import (
    FAULT_COMMIT_CRASH_AFTER_BLOB,
    FAULT_COMMIT_CRASH_AFTER_JOURNAL,
    HardExit,
    InjectionConfig,
)
from score_pipeline.model import read_json, write_json
from score_pipeline.persistence import Store


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.score = generate_score(seed=55)
        self.engine = Engine(self.score)

    def tearDown(self):
        self.tmp.cleanup()

    def test_commit_advances_version_and_recovers(self):
        store = Store(self.root / "state")
        state = self.engine.full_rebuild(version=0)
        store.commit_initial(state)
        recovered = store.recover()
        self.assertTrue(recovered["recovered"])
        self.assertEqual(recovered["stable_version"], 0)
        self.assertEqual(recovered["orphans"], [])

    def test_version_regress_rejected(self):
        store = Store(self.root / "state")
        state = self.engine.full_rebuild(version=0)
        store.commit_initial(state)
        with self.assertRaises(PipelineError) as ctx:
            store.commit(state, None, None)
        self.assertEqual(ctx.exception.code, E_VERSION_REGRESS)

    def test_crash_after_blob_leaves_orphan_and_keeps_stable_version(self):
        injection = InjectionConfig(
            crash_at_version=1, crash_point=FAULT_COMMIT_CRASH_AFTER_BLOB
        )
        store = Store(self.root / "state", injection=injection)
        baseline = self.engine.full_rebuild(version=0)
        store.commit_initial(baseline)
        tempo = [dict(p) for p in self.score.tempo]
        tempo[1]["bpm"] = round(float(tempo[1]["bpm"]) + 10.0, 6)
        op = make_operation(1, baseline, "tempo", tempo=tempo,
                            total_beats=self.score.total_beats)
        state, _ = self.engine.incremental_apply(baseline, op)
        with self.assertRaises(HardExit):
            store.commit(state, {}, op.to_dict())

        fresh = Store(self.root / "state")
        report = fresh.recover()
        self.assertEqual(report["stable_version"], 0)
        self.assertTrue(report["missing_exit_report"])
        self.assertGreaterEqual(len(report["orphans"]), 1)
        self.assertTrue(all(item["reason"] == "dangling_blob" for item in report["orphans"]))

    def test_crash_after_journal_never_promotes_manifest(self):
        injection = InjectionConfig(
            crash_at_version=1, crash_point=FAULT_COMMIT_CRASH_AFTER_JOURNAL
        )
        store = Store(self.root / "state", injection=injection)
        baseline = self.engine.full_rebuild(version=0)
        store.commit_initial(baseline)
        dynamics = [dict(d) for d in self.score.dynamics]
        dynamics[0]["gain"] = round(float(dynamics[0]["gain"]) + 0.1, 6)
        op = make_operation(1, baseline, "dynamics", dynamics=dynamics,
                            total_beats=self.score.total_beats)
        state, _ = self.engine.incremental_apply(baseline, op)
        with self.assertRaises(HardExit):
            store.commit(state, {}, op.to_dict())
        report = Store(self.root / "state").recover()
        self.assertEqual(report["stable_version"], 0)
        self.assertTrue(any(o["reason"] == "tmp_uncommitted" for o in report["orphans"]))

    def test_resumed_state_matches_uninterrupted(self):
        store = Store(self.root / "state")
        baseline = self.engine.full_rebuild(version=0)
        store.commit_initial(baseline)
        tempo = [dict(p) for p in self.score.tempo]
        tempo[2]["bpm"] = round(float(tempo[2]["bpm"]) - 12.0, 6)
        op = make_operation(1, baseline, "tempo", tempo=tempo,
                            total_beats=self.score.total_beats)
        expected, _ = self.engine.incremental_apply(baseline, op)

        injection = InjectionConfig(
            crash_at_version=1, crash_point=FAULT_COMMIT_CRASH_AFTER_BLOB
        )
        crashing = Store(self.root / "state", injection=injection)
        state2, _ = self.engine.incremental_apply(baseline, op)
        with self.assertRaises(HardExit):
            crashing.commit(state2, {}, op.to_dict())

        recovered_store = Store(self.root / "state")
        recovered_store.recover()
        manifest = recovered_store.read_manifest()
        self.assertEqual(manifest["version"], 0)
        resumed, _ = self.engine.incremental_apply(
            self.engine.full_rebuild(version=0), op
        )
        self.assertEqual(resumed.timeline_hash, expected.timeline_hash)
        self.assertEqual(resumed.rendered_notes, expected.rendered_notes)

    def test_stale_snapshot_rejected(self):
        store = Store(self.root / "state")
        store.commit_initial(self.engine.full_rebuild(version=0))
        stale = dict(store.read_manifest())
        stale["version"] = -1
        with self.assertRaises(PipelineError) as ctx:
            store.replace_snapshot(stale)
        self.assertEqual(ctx.exception.code, E_SNAPSHOT_STALE)

    def test_corrupt_blob_detected(self):
        store = Store(self.root / "state")
        store.commit_initial(self.engine.full_rebuild(version=0))
        manifest = store.read_manifest()
        digest = sorted(manifest["shard_blobs"].values())[0]
        blob_path = store.blobs / f"{digest}.json"
        blob = read_json(blob_path)
        blob["notes"][0]["start_ms"] = 1.0
        write_json(blob_path, blob)
        with self.assertRaises(PipelineError) as ctx:
            Store(self.root / "state").recover()
        self.assertEqual(ctx.exception.code, E_TIMELINE_HASH)

    def test_controlled_close_marker_and_report(self):
        store = Store(self.root / "state")
        store.commit_initial(self.engine.full_rebuild(version=0))
        store.write_controlled_close({"status": "controlled_exit", "stable_version": 0})
        self.assertTrue(store.is_closed())
        self.assertTrue(store.exit_report_path.exists())


if __name__ == "__main__":
    unittest.main()
