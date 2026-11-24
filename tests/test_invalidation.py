from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

from config_arbiter.engine import ArbiterEngine
from config_arbiter.ledger import Ledger
from config_arbiter.spec import KeyRule, ViewSpec, load_spec, parse_slice

from .util import ported_sample


class InvalidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.directory = Path(self.tempdir.name)
        self.spec_path, _ = ported_sample(self.directory)
        self.spec = load_spec(self.spec_path)
        self.ledger = Ledger(self.directory / "ledger.sqlite3")
        self.engine = ArbiterEngine(self.spec, self.ledger)
        self.engine.start_services()
        self.initial = self.engine.run_all()

    def tearDown(self) -> None:
        self.engine.close()
        self.tempdir.cleanup()

    def _restart_with_cn_slice(self, version: int, values: dict[str, str]):
        self.engine.close()
        cn_current = self.spec.environments["cn-prod"]
        environments = dict(self.spec.environments)
        environments["cn-prod"] = replace(
            cn_current,
            slices=cn_current.slices
            + (parse_slice(version, values, "$.environments.cn-prod.slices"),),
        )
        restarted_spec = replace(
            self.spec,
            environments=MappingProxyType(environments),
        )
        ledger = Ledger(self.directory / "ledger.sqlite3")
        engine = ArbiterEngine(restarted_spec, ledger)
        engine.start_services()
        self.ledger = ledger
        self.engine = engine
        return engine

    def test_slice_change_propagates_only_through_key_references(self) -> None:
        unaffected_digest = next(
            run.summary.summary_digest
            for run in self.initial.views
            if run.summary.view_id == "us-eu-cpu"
        )
        reports = self.engine.change_environment_slice(
            "cn-prod",
            version=3,
            values={
                "cpu_limit": "1.3001",
                "memory_limit": "7.5",
                "replicas": "4",
            },
        )
        self.assertEqual(len(reports), 1)
        report = reports[0]
        self.assertEqual(report.view_id, "global-release")
        self.assertEqual(report.previous_version, 1)
        self.assertEqual(report.new_version, 2)
        self.assertEqual(report.affected_keys, ("cpu_limit", "replicas"))
        self.assertEqual(
            [reason["kind"] for reason in report.reasons],
            ["SLICE_VALUE_CHANGED", "SLICE_VALUE_CHANGED"],
        )
        current = self.engine.current_display()["views"]
        self.assertTrue(
            all(row["view_version"] == 2 for row in current["global-release"])
        )
        self.assertTrue(
            all(row["view_version"] == 1 for row in current["us-eu-cpu"])
        )
        unaffected_after = next(
            run.summary.summary_digest
            for run in self.engine.run_all().views
            if run.summary.view_id == "us-eu-cpu"
        )
        self.assertEqual(unaffected_digest, unaffected_after)
        stale = [
            row
            for row in self.ledger.all_results()
            if row["view_id"] == "global-release" and int(row["is_stale"]) == 1
        ]
        self.assertEqual(len(stale), 3)
        self.assertEqual(self.engine.replay_current()["status"], "OK")

    def test_repeated_run_keeps_version_when_nothing_changed(self) -> None:
        self.engine.run_all()
        state = self.ledger.view_state("global-release")
        self.assertIsNotNone(state)
        self.assertEqual(state.version, 1)

    def test_participant_declaration_order_does_not_invalidate(self) -> None:
        view = self.engine.views["global-release"]
        reordered = ViewSpec(
            id=view.id,
            environments=tuple(reversed(view.environments)),
            keys=view.keys,
        )
        self.assertIsNone(self.engine.upsert_view(reordered))
        state = self.ledger.view_state("global-release")
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state.version, 1)

    def test_view_rule_change_bumps_version_and_marks_old_rows_stale(self) -> None:
        view = self.engine.views["global-release"]
        changed = ViewSpec(
            id=view.id,
            environments=view.environments,
            keys=MappingProxyType(
                {
                    "cpu_limit": KeyRule("latest", 0),
                    "memory_limit": KeyRule("interval_midpoint", 200),
                    "replicas": KeyRule("latest", 0),
                }
            ),
        )
        report = self.engine.upsert_view(changed)
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(report.new_version, 2)
        self.assertEqual(report.affected_keys, ("cpu_limit",))
        self.assertIn(
            "ARBITRATION_MODE_CHANGED",
            [reason["kind"] for reason in report.reasons],
        )
        current = self.ledger.current_results("global-release")
        self.assertTrue(all(int(row["view_version"]) == 2 for row in current))
        old_rows = [
            row
            for row in self.ledger.all_results()
            if row["view_id"] == "global-release" and int(row["view_version"]) == 1
        ]
        self.assertTrue(all(int(row["is_stale"]) == 1 for row in old_rows))
        self.assertEqual(self.engine.replay_current()["status"], "OK")

    def test_restart_with_changed_slice_values_invalidates_only_referenced_keys(self) -> None:
        engine = self._restart_with_cn_slice(
            3,
            {
                "cpu_limit": "1.3001",
                "memory_limit": "7.5",
                "replicas": "4",
                "outside-view": "9",
            },
        )
        report = engine.run_all()
        invalidations = [item for item in report.invalidations if not item.removed]
        self.assertEqual(len(invalidations), 1)
        invalidation = invalidations[0]
        self.assertEqual(invalidation.view_id, "global-release")
        self.assertEqual(invalidation.new_version, 2)
        self.assertEqual(invalidation.affected_keys, ("cpu_limit", "replicas"))
        self.assertEqual(
            [reason["kind"] for reason in invalidation.reasons],
            ["SLICE_VALUE_CHANGED", "SLICE_VALUE_CHANGED"],
        )
        self.assertTrue(
            all(
                reason.get("environment") == "cn-prod"
                for reason in invalidation.reasons
            )
        )
        self.assertEqual(engine.replay_current()["status"], "OK")

    def test_restart_with_version_only_change_keeps_view_version_and_replay(self) -> None:
        engine = self._restart_with_cn_slice(
            3,
            {
                "cpu_limit": "1.2345",
                "memory_limit": "7.5",
                "replicas": "3",
            },
        )
        report = engine.run_all()
        self.assertEqual(report.invalidations, ())
        state = engine.ledger.view_state("global-release")
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state.version, 1)
        rows = engine.ledger.current_results("global-release")
        self.assertTrue(
            all(json.loads(row["slice_versions_json"])["cn-prod"] == 3 for row in rows)
        )
        self.assertEqual(engine.replay_current()["status"], "OK")

    def test_runtime_version_only_slice_change_refreshes_without_bumping(self) -> None:
        reports = self.engine.change_environment_slice(
            "cn-prod",
            version=3,
            values={
                "cpu_limit": "1.2345",
                "memory_limit": "7.5",
                "replicas": "3",
            },
        )
        self.assertEqual(reports, [])
        state = self.ledger.view_state("global-release")
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state.version, 1)
        self.assertEqual(self.engine.replay_current()["status"], "OK")

    def test_view_removal_hides_all_current_conclusions(self) -> None:
        report = self.engine.remove_view("global-release")
        self.assertIsNotNone(report)
        assert report is not None
        self.assertTrue(report.removed)
        self.assertEqual(report.new_version, 2)
        self.assertEqual(self.ledger.current_results("global-release"), [])
        self.assertNotIn("global-release", self.engine.current_display()["views"])
        state = self.ledger.view_state("global-release")
        self.assertIsNotNone(state)
        self.assertFalse(state.active)
        self.assertEqual(self.engine.replay_current()["status"], "OK")

    def test_new_view_starts_at_version_one_and_is_current(self) -> None:
        new_view = ViewSpec(
            id="new-view",
            environments=("cn-prod", "us-prod"),
            keys=MappingProxyType({"replicas": KeyRule("latest", 0)}),
        )
        report = self.engine.upsert_view(new_view)
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(report.previous_version, 0)
        self.assertEqual(report.new_version, 1)
        self.assertIn("new-view", self.engine.current_display()["views"])
        self.assertEqual(len(self.ledger.current_results("new-view")), 1)

    def test_participant_change_affects_all_keys_and_versions_stay_monotonic(self) -> None:
        view = self.engine.views["global-release"]
        changed = ViewSpec(
            id=view.id,
            environments=("cn-prod", "eu-prod"),
            keys=view.keys,
        )
        report = self.engine.upsert_view(changed)
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(report.new_version, 2)
        self.assertEqual(report.affected_keys, ("cpu_limit", "memory_limit", "replicas"))
        self.assertIn(
            "PARTICIPANT_REMOVED",
            [reason["kind"] for reason in report.reasons],
        )
        removed = self.engine.remove_view("global-release")
        self.assertIsNotNone(removed)
        assert removed is not None
        self.assertEqual(removed.new_version, 3)
        readded = self.engine.upsert_view(changed)
        self.assertIsNotNone(readded)
        assert readded is not None
        self.assertEqual(readded.previous_version, 3)
        self.assertEqual(readded.new_version, 4)

    def test_view_missing_from_runtime_spec_is_staled_on_run_all(self) -> None:
        self.engine.close()
        restarted_spec = replace(
            self.spec,
            views=MappingProxyType({"global-release": self.spec.views["global-release"]}),
        )
        ledger = Ledger(self.directory / "ledger.sqlite3")
        engine = ArbiterEngine(restarted_spec, ledger)
        engine.start_services()
        self.ledger = ledger
        self.engine = engine
        report = engine.run_all()
        removal = next(
            item for item in report.invalidations if item.view_id == "us-eu-cpu"
        )
        self.assertTrue(removal.removed)
        self.assertEqual(removal.previous_version, 1)
        self.assertEqual(removal.new_version, 2)
        self.assertEqual(removal.reasons[0]["kind"], "VIEW_REMOVED_FROM_SPEC")
        self.assertEqual(self.ledger.current_results("us-eu-cpu"), [])
        state = self.ledger.view_state("us-eu-cpu")
        self.assertIsNotNone(state)
        assert state is not None
        self.assertFalse(state.active)


if __name__ == "__main__":
    unittest.main()
