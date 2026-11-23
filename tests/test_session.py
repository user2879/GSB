from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from config_pipeline.errors import E_PORT, E_REPLAY, PipelineError
from config_pipeline.manifest import manifest_from_dict
from config_pipeline.session import PipelineSession
from config_pipeline.simulator import SimulatorPool
from config_pipeline.util import digest_json
from tests.helpers import evolved, free_port, manifest


class SessionTests(unittest.TestCase):
    def test_full_run_evolution_replacement_and_replay(self) -> None:
        ports = (free_port(), free_port())
        original = manifest(
            ports=ports,
            budget=2,
            task_shards=2,
            task_limit=1,
            cluster_quotas=(2, 1),
        )
        with tempfile.TemporaryDirectory() as directory:
            db = str(Path(directory) / "state.db")
            snapshot_path = str(Path(directory) / "snapshot.json")
            session = PipelineSession(
                manifest_from_dict(original), db, snapshot_path=snapshot_path
            )
            try:
                first = session.run_current()
                self.assertTrue(all(shard["status"] == "success" for shard in first.shards))
                self.assertEqual(session.store.ledger_cursor, 4)
                old_generation = session.store.latest_snapshot()["generation"]
                old_digests = {
                    shard["stable_id"]: shard["compile_digest"] for shard in first.shards
                }

                unrelated = evolved(original, values={"unused": 9})
                second = session.evolve(manifest_from_dict(unrelated))
                self.assertEqual(second.recompute_ids, ())
                self.assertTrue(
                    all(shard["revision_version"] == 1 for shard in second.shards)
                )
                self.assertTrue(
                    all(
                        shard["compile_digest"] == old_digests[shard["stable_id"]]
                        for shard in second.shards
                    )
                )

                related = evolved(unrelated, version=3, values={"base": 21})
                third = session.evolve(manifest_from_dict(related))
                self.assertEqual(len(third.recompute_ids), 4)
                self.assertTrue(
                    all(shard["revision_version"] == 3 for shard in third.shards)
                )
                self.assertEqual(third.diff["added"], [])
                self.assertEqual(third.diff["removed"], [])
                self.assertEqual(len(third.diff["changed"]), 4)
                self.assertTrue(all(item["basis"] for item in third.invalidations))
                replay = session.replay()
                self.assertEqual(replay["replayed_shards"], list(replay["replayed_shards"]))
                self.assertEqual(len(replay["replayed_shards"]), 4)
                self.assertTrue(Path(snapshot_path).exists())
            finally:
                session.close()

            with sqlite3.connect(db) as connection:
                old_row = connection.execute(
                    "SELECT is_stale, is_current FROM snapshots WHERE generation = ?",
                    (old_generation,),
                ).fetchone()
                self.assertEqual(old_row, (1, 0))
                current = connection.execute(
                    "SELECT COUNT(*) FROM snapshots WHERE is_current = 1 AND is_stale = 0"
                ).fetchone()
                self.assertEqual(current[0], 1)

            reopened = PipelineSession(
                manifest_from_dict(related), db, snapshot_path=snapshot_path
            )
            try:
                recovered = reopened.recover()
                self.assertFalse(recovered.recovery.snapshot_replaced)
                self.assertEqual(len(recovered.recovery.replayed_shards), 4)
                self.assertTrue(all(shard["status"] == "success" for shard in recovered.shards))
            finally:
                reopened.close()

    def test_snapshot_file_mismatch_triggers_whole_replacement(self) -> None:
        data = manifest(ports=(free_port(),), task_shards=2, task_limit=1)
        with tempfile.TemporaryDirectory() as directory:
            db = str(Path(directory) / "state.db")
            snapshot_path = Path(directory) / "snapshot.json"
            session = PipelineSession(
                manifest_from_dict(data), db, snapshot_path=snapshot_path
            )
            try:
                session.run_current()
            finally:
                session.close()
            document = json.loads(snapshot_path.read_text(encoding="utf-8"))
            document["content_digest"] = "tampered"
            snapshot_path.write_text(json.dumps(document), encoding="utf-8")
            reopened = PipelineSession(
                manifest_from_dict(data), db, snapshot_path=snapshot_path
            )
            try:
                report = reopened.recover()
                self.assertTrue(report.recovery.snapshot_replaced)
                self.assertIn(
                    "snapshot.file:content_digest-mismatch",
                    report.recovery.replacement_reasons,
                )
                self.assertEqual(len(report.recovery.replayed_shards), 2)
            finally:
                reopened.close()

    def test_budget_version_mismatch_triggers_whole_replacement(self) -> None:
        data = manifest(ports=(free_port(),), budget=3, task_shards=1, task_limit=1)
        with tempfile.TemporaryDirectory() as directory:
            db = str(Path(directory) / "state.db")
            snapshot_path = str(Path(directory) / "snapshot.json")
            session = PipelineSession(
                manifest_from_dict(data), db, snapshot_path=snapshot_path
            )
            try:
                session.run_current()
            finally:
                session.close()
            changed = json.loads(json.dumps(data))
            changed["global_resource_budget"] = 4
            reopened = PipelineSession(
                manifest_from_dict(changed), db, snapshot_path=snapshot_path
            )
            try:
                report = reopened.recover()
                self.assertTrue(report.recovery.snapshot_replaced)
                self.assertTrue(
                    any(
                        reason.startswith("snapshot.budget_version:")
                        for reason in report.recovery.replacement_reasons
                    )
                )
                self.assertEqual(len(report.recovery.replayed_shards), 1)
                self.assertEqual(report.stats["global_budget"], 4)
            finally:
                reopened.close()

    def test_replay_mismatch_reports_first_shard_and_field(self) -> None:
        data = manifest(ports=(free_port(),), task_shards=2, task_limit=1)
        with tempfile.TemporaryDirectory() as directory:
            db = str(Path(directory) / "state.db")
            snapshot_path = str(Path(directory) / "snapshot.json")
            session = PipelineSession(
                manifest_from_dict(data), db, snapshot_path=snapshot_path
            )
            try:
                session.run_current()
            finally:
                session.close()

            original_dispatch = SimulatorPool.dispatch
            calls = {"count": 0}

            def altered_dispatch(self, shard, *, timeout=30.0):
                reply = original_dispatch(self, shard, timeout=timeout)
                calls["count"] += 1
                if calls["count"] == 1:
                    changed = json.loads(json.dumps(reply["aggregate_summary"]))
                    changed["groups"][0]["keys"]["scaled"]["sum"] += 1
                    reply = dict(reply)
                    reply["aggregate_summary"] = changed
                    reply["aggregate_digest"] = digest_json(changed)
                return reply

            reopened = PipelineSession(
                manifest_from_dict(data), db, snapshot_path=snapshot_path
            )
            try:
                with mock.patch.object(SimulatorPool, "dispatch", altered_dispatch):
                    with self.assertRaises(PipelineError) as caught:
                        reopened.recover()
                self.assertEqual(caught.exception.code, E_REPLAY)
                self.assertIsNotNone(caught.exception.shard_id)
                self.assertTrue((caught.exception.field or "").startswith("aggregate_summary."))
                self.assertIn("ledger_revision", " ".join(caught.exception.basis))
            finally:
                reopened.close()

if __name__ == "__main__":
    unittest.main()
