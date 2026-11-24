from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from colmv.canonical import canonical_json
from colmv.changes import load_changes
from colmv.engine import MaterializationEngine
from colmv.errors import ColmvError
from colmv.evolution import apply_evolution, build_invalidation_plan
from colmv.ledger import LedgerStore
from colmv.replay import replay_and_verify, split_uniform
from colmv.schema import load_config

from tests.helpers import SAMPLES, sample_paths


class LedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.paths = sample_paths(SAMPLES)
        cls.config = load_config(cls.paths["catalog"], cls.paths["views"])
        cls.config_v2 = load_config(cls.paths["catalog"], cls.paths["views_v2"])
        cls.changes = load_changes(cls.paths["changes"], cls.config.table)

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="colmv-ledger-")
        self.ledger_path = Path(self.temp.name) / "ledger.sqlite"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _commit_all(self, ledger: LedgerStore, *, config=None, scheme_version: int | None = None, batches=None):
        config = config or self.config
        scheme_version = scheme_version or ledger.register_scheme(config, budget=1)
        engine = MaterializationEngine(config, budget=1, path_mode="auto", scheme_version=scheme_version)
        batch_list = batches or split_uniform(self.changes, 4)
        committed = []
        for batch_no, batch in enumerate(batch_list, start=1):
            decision = engine.decide_batch(batch)
            attempt = ledger.begin_batch_attempt(
                scheme_version=scheme_version,
                batch_no=batch_no,
                start_seq=batch[0].seq,
                end_seq=batch[-1].seq,
                path=decision.path,
                planned_groups=[ref.to_dict() for ref in engine.preview_touched_groups(batch)],
            )
            result = engine.process_batch(batch, batch_no=batch_no)
            info = ledger.commit_batch(
                attempt_id=attempt,
                scheme_version=scheme_version,
                batch_result=result,
                materialized_result=engine.materialized_result(),
            )
            committed.append((result, info))
        return engine, scheme_version, committed

    def test_committed_batches_have_content_digest_and_valid_snapshot(self) -> None:
        with LedgerStore(self.ledger_path) as ledger:
            engine, scheme_version, committed = self._commit_all(ledger)
            self.assertEqual(len(committed), 5)
            self.assertEqual(ledger.last_committed_batch()["cursor"], 20)
            self.assertTrue(ledger.verify_content_chain()["valid"])
            snapshot = ledger.verify_snapshot()
            self.assertTrue(snapshot["valid"])
            self.assertEqual(snapshot["scheme_version"], scheme_version)
            self.assertEqual(ledger.snapshot()["payload"]["materialized"]["scheme"]["version"], scheme_version)

    def test_uncommitted_batch_groups_become_unknown(self) -> None:
        with LedgerStore(self.ledger_path) as ledger:
            scheme_version = ledger.register_scheme(self.config, budget=1)
            engine = MaterializationEngine(self.config, budget=1, scheme_version=scheme_version)
            first = split_uniform(self.changes, 10)[0]
            decision = engine.decide_batch(first)
            ledger.begin_batch_attempt(
                scheme_version=scheme_version,
                batch_no=1,
                start_seq=first[0].seq,
                end_seq=first[-1].seq,
                path=decision.path,
                planned_groups=[ref.to_dict() for ref in engine.preview_touched_groups(first)],
            )
        with LedgerStore(self.ledger_path) as ledger:
            recovery = ledger.recover_uncommitted()
            self.assertTrue(recovery["recovered"])
            self.assertFalse(recovery["exit_report_available"])
            self.assertGreater(len(recovery["unknown_attempts"][0]["unknown_groups"]), 0)
            unknown = ledger.unknown_groups()
            self.assertGreater(len(unknown), 0)
            self.assertTrue(all(row["status"] == "unknown" for row in unknown))
            self.assertIsNone(ledger.last_committed_batch())

    def test_snapshot_digest_mismatch_is_replaced_wholesale(self) -> None:
        with LedgerStore(self.ledger_path) as ledger:
            self._commit_all(ledger)
            with ledger.connection:
                ledger.connection.execute("UPDATE snapshot SET payload_json='{}' WHERE singleton=1")
            check = ledger.verify_snapshot()
            self.assertFalse(check["valid"])
            self.assertTrue(check["replaced"])
            self.assertTrue(check["replacement"]["available"])
            self.assertTrue(ledger.verify_snapshot()["valid"])
            self.assertGreater(len(ledger.snapshot()["payload"]["materialized"]["views"]), 0)

    def test_replay_detects_first_group_field_difference(self) -> None:
        with LedgerStore(self.ledger_path) as ledger:
            self._commit_all(ledger)
            row = ledger.connection.execute(
                "SELECT groups_json FROM batches WHERE scheme_version=1 AND batch_no=2"
            ).fetchone()
            groups = json.loads(row["groups_json"])
            groups[0]["row_count"] += 1
            with ledger.connection:
                ledger.connection.execute(
                    "UPDATE batches SET groups_json=? WHERE scheme_version=1 AND batch_no=2",
                    (json.dumps(groups, sort_keys=True, separators=(",", ":")),),
                )
            with self.assertRaises(ColmvError) as context:
                replay_and_verify(
                    ledger,
                    catalog_path=self.paths["catalog"],
                    changes=self.changes,
                )
            error = context.exception
            self.assertEqual(error.code, "E_REPLAY")
            self.assertEqual(error.details["batch_no"], 2)
            self.assertIn("row_count", error.field or "")

    def test_evolution_marks_old_traces_stale_and_clears_snapshot(self) -> None:
        with LedgerStore(self.ledger_path) as ledger:
            self._commit_all(ledger)
            plan = build_invalidation_plan(
                old_config=self.config,
                new_config=self.config_v2,
                old_budget=1,
                new_budget=1,
                changes=self.changes,
            )
            application = apply_evolution(
                ledger,
                new_config=self.config_v2,
                new_budget=1,
                plan=plan,
            )
            self.assertEqual(application["old_scheme_version"], 1)
            self.assertEqual(application["new_scheme_version"], 2)
            self.assertTrue(application["old_traces_marked_stale"])
            old_batches = ledger.connection.execute(
                "SELECT is_stale FROM batches WHERE scheme_version=1"
            ).fetchall()
            self.assertTrue(old_batches)
            self.assertTrue(all(row["is_stale"] == 1 for row in old_batches))
            self.assertIsNone(ledger.snapshot())
            self.assertEqual(ledger.current_scheme()["scheme_version"], 2)
            unique_groups = {
                (item["view_id"], json.dumps(item["group_key"], sort_keys=True))
                for item in plan["affected_groups"]
            }
            self.assertEqual(len(plan["affected_groups"]), len(unique_groups))
            self.assertEqual(
                plan["affected_groups"],
                sorted(plan["affected_groups"], key=lambda item: (item["view_id"], canonical_json(item["group_key"]))),
            )
            self.assertGreater(len(plan["affected_pages"]), 0)


if __name__ == "__main__":
    unittest.main()
