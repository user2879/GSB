from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from h2burst.engine import Engine
from h2burst.errors import WorkbenchError
from h2burst.model import digest_value, load_session, parse_session
from h2burst.replay import replay_from_snapshot, replay_from_state
from h2burst.snapshot import SnapshotStore
from tests.helpers import raw_session, spec_session


ROOT = Path(__file__).resolve().parents[1]


class SnapshotReplayTests(unittest.TestCase):
    def partial(self, spec, event_count: int) -> Engine:
        engine = Engine(spec)
        while engine.event_cursor < event_count:
            engine.step()
        return engine

    def test_snapshot_round_trip_and_metadata(self) -> None:
        spec = load_session(ROOT / "samples/minimal.json")
        partial = self.partial(spec, 1)
        state_digest = partial.to_state_dict()["state_digest"]
        with tempfile.TemporaryDirectory() as temp, SnapshotStore(Path(temp) / "snap.db") as store:
            info = store.save(partial, "after-event-0")
            self.assertEqual(info.declared_version, spec.declared_version)
            self.assertEqual(info.content_digest, spec.content_digest)
            self.assertEqual(info.state_digest, state_digest)
            self.assertEqual(info.snapshot_format_version, 1)
            self.assertFalse(info.read_only)
            loaded_info, loaded_spec, loaded_state = store.load(info.snapshot_id)
            self.assertEqual(loaded_info.state_digest, state_digest)
            self.assertEqual(loaded_spec.content_digest, spec.content_digest)
            restored = Engine.from_state_dict(loaded_spec, loaded_state)
            self.assertEqual(restored.to_state_dict()["state_digest"], state_digest)

    def test_same_session_restore_is_not_whole_replacement(self) -> None:
        spec = load_session(ROOT / "samples/minimal.json")
        partial = self.partial(spec, 1)
        full = Engine(spec).run()
        with tempfile.TemporaryDirectory() as temp, SnapshotStore(Path(temp) / "snap.db") as store:
            info = store.save(partial, "checkpoint")
            result = store.restore(info.snapshot_id, spec, full)
            self.assertFalse(result.whole_replacement)
            self.assertEqual(result.engine.next_event_seq, partial.next_event_seq)
            self.assertEqual(result.engine.run_digest(), partial.run_digest())

    def test_whole_replacement_uses_snapshot_entirely_and_keeps_read_only_backups(self) -> None:
        spec_a = spec_session(
            session_id="session-a",
            declared_version=1,
            streams=[
                {"stream_id": 1, "priority": 1, "request_body_bytes": 100},
                {"stream_id": 3, "priority": 2, "request_body_bytes": 100},
            ],
        )
        engine_a = Engine(spec_a).run(drain=False)
        spec_b = spec_session(
            session_id="session-b",
            declared_version=2,
            streams=[{"stream_id": 9, "priority": 1, "request_body_bytes": 50}],
        )
        engine_b = Engine(spec_b).run(drain=False)
        with tempfile.TemporaryDirectory() as temp, SnapshotStore(Path(temp) / "snap.db") as store:
            info = store.save(engine_a, "a-checkpoint")
            result = store.restore(info.snapshot_id, spec_b, engine_b)
            self.assertTrue(result.whole_replacement)
            self.assertTrue(result.version_mismatch)
            self.assertTrue(result.content_digest_mismatch)
            self.assertEqual(result.spec.session_id, "session-a")
            self.assertEqual(set(result.engine.streams), {1, 3})
            self.assertNotIn(9, result.engine.streams)
            self.assertIsNotNone(result.backup_snapshot_id)
            self.assertIn("streams.9.priority", result.diff["removed"])
            self.assertTrue(any(item.startswith("streams.1") for item in result.diff["added"]))
            snapshots = store.list()
            self.assertGreaterEqual(len(snapshots), 2)
            self.assertTrue(all(item.read_only for item in snapshots))
            self.assertTrue(any(item.snapshot_id == result.backup_snapshot_id for item in snapshots))

    def test_tampered_state_digest_is_e_snapshot(self) -> None:
        spec = load_session(ROOT / "samples/minimal.json")
        partial = self.partial(spec, 1)
        with tempfile.TemporaryDirectory() as temp, SnapshotStore(Path(temp) / "snap.db") as store:
            info = store.save(partial, "tamper")
            row = store.connection.execute("SELECT state_json FROM snapshots WHERE id=?", (info.snapshot_id,)).fetchone()
            state = json.loads(row[0])
            state["connection_window"] += 1
            store.connection.execute(
                "UPDATE snapshots SET state_json=? WHERE id=?",
                (json.dumps(state), info.snapshot_id),
            )
            store.connection.commit()
            with self.assertRaises(WorkbenchError) as ctx:
                store.load(info.snapshot_id)
            self.assertEqual(ctx.exception.code, "E_SNAPSHOT")
            self.assertEqual(ctx.exception.field, "state_digest")

    def test_tampered_session_digest_is_e_snapshot(self) -> None:
        spec = load_session(ROOT / "samples/minimal.json")
        partial = self.partial(spec, 1)
        with tempfile.TemporaryDirectory() as temp, SnapshotStore(Path(temp) / "snap.db") as store:
            info = store.save(partial, "tamper-session")
            store.connection.execute(
                "UPDATE snapshots SET content_digest=? WHERE id=?",
                ("0" * 64, info.snapshot_id),
            )
            store.connection.commit()
            with self.assertRaises(WorkbenchError) as ctx:
                store.load(info.snapshot_id)
            self.assertEqual(ctx.exception.code, "E_SNAPSHOT")
            self.assertEqual(ctx.exception.field, "session_digest")

    def test_replay_matches_after_snapshot(self) -> None:
        spec = load_session(ROOT / "samples/minimal.json")
        partial = self.partial(spec, 1)
        full = Engine(spec).run()
        with tempfile.TemporaryDirectory() as temp, SnapshotStore(Path(temp) / "snap.db") as store:
            info = store.save(partial, "after-event-0")
            result = replay_from_snapshot(
                store,
                info.snapshot_id,
                reference_spec=spec,
                reference_engine=full,
            )
            self.assertTrue(result.match)
            self.assertGreater(result.frames_compared, 0)
            self.assertEqual(result.full_digest, result.replay_digest)

    def test_replay_reports_first_divergent_field_as_e_replay(self) -> None:
        spec = load_session(ROOT / "samples/minimal.json")
        partial = self.partial(spec, 1)
        full = Engine(spec).run()
        state = partial.to_state_dict()
        original_digest = state.pop("state_digest")
        state["connection_window"] += 1
        state["state_digest"] = digest_value(state)
        self.assertNotEqual(state["state_digest"], original_digest)
        with self.assertRaises(WorkbenchError) as ctx:
            replay_from_state(spec, state, reference_engine=full)
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertIn(ctx.exception.field, {"connection_window_before", "connection_window_after"})
        self.assertIsNotNone(ctx.exception.frame_seq)

    def test_snapshot_payload_contains_required_sections(self) -> None:
        spec = load_session(ROOT / "samples/burst_demo.json")
        engine = self.partial(spec, 5)
        state = engine.to_state_dict()
        for field in (
            "declared_version",
            "content_digest",
            "current_strategy",
            "strategy_switches",
            "connection_window",
            "streams",
            "transitions",
            "frames",
            "state_digest",
        ):
            self.assertIn(field, state)
        self.assertTrue(all("window" in stream and "state" in stream for stream in state["streams"]))

    def test_snapshot_database_persists_across_reopen(self) -> None:
        spec = load_session(ROOT / "samples/minimal.json")
        partial = self.partial(spec, 2)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "snap.db"
            store = SnapshotStore(path)
            info = store.save(partial, "persisted")
            store.close()
            reopened = SnapshotStore(path)
            try:
                loaded, _spec, _state = reopened.load(info.snapshot_id)
                self.assertEqual(loaded.label, "persisted")
            finally:
                reopened.close()


if __name__ == "__main__":
    unittest.main()
