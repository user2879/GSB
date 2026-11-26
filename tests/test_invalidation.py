from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gsb.engine import run_session
from gsb.invalidation import audit_invalidation, full_rebuild
from gsb.spec import load_session


ROOT = Path(__file__).resolve().parent.parent


class InvalidationTests(unittest.TestCase):
    def test_changed_layer_scale_invalidates_only_dependent_shard(self) -> None:
        old = load_session(config_path=ROOT / "samples" / "session.json")
        new = load_session(config_path=ROOT / "samples" / "session_modified.json")
        first = audit_invalidation(old, new)
        second = audit_invalidation(old, new)
        self.assertEqual(first["report_digest"], second["report_digest"])
        invalid_ids = {batch["batch_id"] for batch in first["invalid_batches"]}
        self.assertEqual(
            invalid_ids,
            {"a-before-rollback", "a-after-rollback", "a-after-second-rollback"},
        )
        self.assertEqual(
            {batch["batch_id"] for batch in first["unaffected_batches"]},
            {"b-before-rollback", "b-after-rollback"},
        )
        self.assertTrue(
            all(batch["byte_identical"] for batch in first["unaffected_batches"])
        )
        self.assertEqual(
            {checkpoint["checkpoint_id"] for checkpoint in first["invalid_checkpoints"]},
            {"cp-a-base", "cp-a-v2"},
        )
        self.assertEqual(
            {checkpoint["checkpoint_id"] for checkpoint in first["unaffected_checkpoints"]},
            {"cp-b-base"},
        )
        self.assertTrue(
            all(checkpoint["byte_identical"] for checkpoint in first["unaffected_checkpoints"])
        )
        self.assertIn(
            "layer:a-embed:scale",
            first["shard_change_reasons"]["shard-a"],
        )

    def test_full_rebuild_is_stable(self) -> None:
        session = load_session(config_path=ROOT / "samples" / "session.json")
        first = full_rebuild(session)
        second = full_rebuild(session)
        self.assertEqual(first["rebuild_digest"], second["rebuild_digest"])
        self.assertEqual(first["batch_count"], 5)

    def test_ledger_matches_from_scratch_rebuild(self) -> None:
        session = load_session(config_path=ROOT / "samples" / "session.json")
        # Replace fixed sample ports with currently free ports.
        from dataclasses import replace
        from tests.helpers import free_port

        free_shards = tuple(
            replace(shard, port=free_port()) for shard in session.shards
        )
        session = replace(session, shards=free_shards)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.sqlite"
            run_session(session, ledger_path=path, startup_timeout_s=5)
            from gsb.ledger import Ledger
            from gsb.invalidation import compare_ledger_with_rebuild

            ledger = Ledger.open(path, session)
            try:
                report = compare_ledger_with_rebuild(session, ledger)
            finally:
                ledger.close()
            self.assertTrue(report["consistent"])
            self.assertEqual(report["batch_count"], 5)


if __name__ == "__main__":
    unittest.main()
