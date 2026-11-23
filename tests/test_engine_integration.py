from __future__ import annotations

import copy
import hashlib
import json
import unittest
from dataclasses import replace

from gateway_diag.engine import DiagnosticEngine
from gateway_diag.errors import PreviewError, ReplayError, SpecError
from gateway_diag.ledger import LedgerStore, PreviewCache, PreviewSnapshot
from gateway_diag.model import load_requests, load_spec

from tests.helpers import Workspace, base_spec, reserve_ports, sample_requests


class EngineIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = Workspace()
        self.ports = reserve_ports(2)
        self.raw_spec = base_spec(self.ports)
        self.requests_raw = sample_requests(8)
        self.spec_path = self.workspace.write_spec(self.raw_spec)
        self.requests_path = self.workspace.write_requests(self.requests_raw)
        self.store = LedgerStore(self.workspace.db)
        self.spec = load_spec(self.spec_path)
        self.requests = load_requests(self.requests_path, self.spec)
        self.engine = DiagnosticEngine(
            self.spec,
            self.requests,
            self.store,
            startup_timeout=5,
            receipt_timeout=5,
        )

    def tearDown(self) -> None:
        self.store.close()
        self.workspace.close()

    def _run(self, engine: DiagnosticEngine | None = None):
        return (engine or self.engine).run()

    def test_run_preview_authority_and_replay_are_byte_consistent(self) -> None:
        report = self._run()
        self.assertEqual(report.newly_committed, report.total_batches)
        self.assertGreater(report.scheduler.max_global_observed, 0)
        self.assertTrue(report.scheduler.worker_threads)
        self.assertFalse(report.recovery.needs_rerun)
        header = self.store.ledger_header()
        self.assertIsInstance(header["format_version"], int)
        self.assertEqual(header["format_version"], 1)
        self.assertEqual(header["active_revision"], self.spec.revision)
        self.assertEqual(header["content_digest"], report.ledger_content_digest)

        preview = self.engine.preview()
        self.assertEqual(len(preview.entries), len(self.requests_raw))
        self.assertEqual(
            len({entry.batch_id for entry in preview.entries}),
            report.total_batches,
        )
        decision_by_id = {
            item.request_id: item for item in self.engine.plan().decisions
        }
        for entry in preview.entries:
            decision = decision_by_id[entry.request_id]
            self.assertEqual(entry.target, decision.target)
            self.assertEqual(entry.bucket, decision.bucket)
            self.assertEqual(entry.backend_id, decision.backend_id)
            self.assertEqual(entry.weight_version, decision.weight_version)
            self.assertEqual(len(entry.output_digest), 64)
        self.assertEqual(preview.stats["request_count"], len(self.requests_raw))
        self.assertEqual(
            preview.stats["stable_count"] + preview.stats["canary_count"],
            len(self.requests_raw),
        )
        self.assertGreaterEqual(preview.stats["total_exec_ms"], 0)

        authority = self.engine.verify_authoritative(preview)
        self.assertEqual(
            authority.canonical_digest,
            hashlib.sha256(preview.canonical_bytes()).hexdigest(),
        )
        replay = self.engine.verify_replay()
        self.assertTrue(replay.ledger_integrity_ok)
        self.assertEqual(replay.replayed_requests, len(self.requests_raw))
        self.assertFalse(self.store.recovery_report(self.engine.plan(), self.requests).needs_rerun)

    def test_second_run_skips_committed_batches(self) -> None:
        first = self._run()
        second = self._run()
        self.assertEqual(second.newly_committed, 0)
        self.assertEqual(second.committed_before, first.total_batches)
        self.assertEqual(second.scheduler.dispatched_batches, 0)
        self.assertFalse(second.recovery.needs_rerun)

    def test_preview_route_mismatch_reports_first_request_and_field(self) -> None:
        self._run()
        cache = self.engine.preview()
        first = cache.entries[0]
        tampered_entry = replace(
            first, target="stable" if first.target == "canary" else "canary"
        )
        entries = (tampered_entry,) + cache.entries[1:]
        tampered_snapshot = self.store.snapshot_from_entries(cache.revision, entries)
        with self.assertRaises(PreviewError) as caught:
            self.engine.verify_authoritative(PreviewCache(tampered_snapshot))
        self.assertEqual(caught.exception.code, "E_PREVIEW")
        self.assertEqual(caught.exception.request_id, first.request_id)
        self.assertEqual(caught.exception.field, "target")

    def test_preview_timing_statistics_mismatch_reports_e_preview(self) -> None:
        self._run()
        cache = self.engine.preview()
        stats = copy.deepcopy(dict(cache.stats))
        stats["total_exec_ms"] = int(stats["total_exec_ms"]) + 1
        tampered = PreviewCache(
            PreviewSnapshot(cache.revision, cache.entries, stats)
        )
        with self.assertRaises(PreviewError) as caught:
            self.engine.verify_authoritative(tampered)
        self.assertEqual(caught.exception.code, "E_PREVIEW")
        self.assertEqual(caught.exception.field, "stats.total_exec_ms")

    def test_preview_entry_timing_mismatch_is_checked_against_ledger(self) -> None:
        self._run()
        cache = self.engine.preview()
        first = cache.entries[0]
        tampered_entry = replace(first, wait_ms=first.wait_ms + 1)
        entries = (tampered_entry,) + cache.entries[1:]
        tampered_snapshot = self.store.snapshot_from_entries(cache.revision, entries)
        with self.assertRaises(PreviewError) as caught:
            self.engine.verify_authoritative(PreviewCache(tampered_snapshot))
        self.assertEqual(caught.exception.code, "E_PREVIEW")
        self.assertEqual(caught.exception.request_id, first.request_id)
        self.assertEqual(caught.exception.field, "wait_ms")

    def test_preview_batch_and_output_mismatches_are_localized(self) -> None:
        self._run()
        cache = self.engine.preview()
        first = cache.entries[0]
        cases = (
            (replace(first, batch_id="tampered-batch"), "batch_id"),
            (replace(first, output_digest="0" * 64), "output_digest"),
        )
        for tampered_entry, expected_field in cases:
            with self.subTest(field=expected_field):
                entries = (tampered_entry,) + cache.entries[1:]
                snapshot = self.store.snapshot_from_entries(cache.revision, entries)
                with self.assertRaises(PreviewError) as caught:
                    self.engine.verify_authoritative(PreviewCache(snapshot))
                self.assertEqual(caught.exception.request_id, first.request_id)
                self.assertEqual(caught.exception.field, expected_field)

    def test_replay_detects_ledger_output_tamper_with_request_backend_field(self) -> None:
        self._run()
        plan = self.engine.plan()
        first_decision = plan.decisions[0]
        self.store.execute_for_test(
            "UPDATE ledger_entries SET output_digest = ? WHERE revision = ? AND request_id = ?",
            ("0" * 64, self.spec.revision, first_decision.request_id),
        )
        with self.assertRaises(ReplayError) as caught:
            self.engine.verify_replay()
        self.assertEqual(caught.exception.code, "E_REPLAY")
        self.assertEqual(caught.exception.request_id, first_decision.request_id)
        self.assertEqual(caught.exception.backend_id, first_decision.backend_id)
        self.assertEqual(caught.exception.field, "output_digest")

    def test_revision_replacement_marks_old_rows_stale_and_replaces_view(self) -> None:
        self._run()
        old_revision = self.spec.revision
        old_batches = self.store.committed_batch_ids(old_revision)
        self.assertTrue(old_batches)

        changed = copy.deepcopy(self.raw_spec)
        changed["revision"] = 2
        changed["backends"][0]["seed"] += 17
        changed["rules"][0]["canary_weight_bp"] = 0
        changed["rules"][0]["version"] = 2
        changed_path = self.workspace.write_spec(changed, "spec-v2.json")
        changed_spec = load_spec(changed_path)
        changed_requests = load_requests(self.requests_path, changed_spec)
        changed_engine = DiagnosticEngine(
            changed_spec, changed_requests, self.store, startup_timeout=5, receipt_timeout=5
        )
        run = changed_engine.run()
        self.assertTrue(run.activation.changed)
        self.assertEqual(run.revision, 2)

        old_batch_rows = self.store.connection.execute(
            "SELECT stale FROM batches WHERE revision = ?", (old_revision,)
        ).fetchall()
        old_entry_rows = self.store.connection.execute(
            "SELECT stale FROM ledger_entries WHERE revision = ?", (old_revision,)
        ).fetchall()
        self.assertTrue(old_batch_rows)
        self.assertTrue(old_entry_rows)
        self.assertTrue(all(int(row[0]) == 1 for row in old_batch_rows))
        self.assertTrue(all(int(row[0]) == 1 for row in old_entry_rows))

        preview = changed_engine.preview()
        self.assertEqual(preview.revision, 2)
        self.assertTrue(all(entry.target == "stable" for entry in preview.entries))
        diff = self.store.activation_diff(2)
        self.assertTrue(diff["replaced_whole_view"])
        self.assertEqual(diff["rules"]["version_changed"], [{"id": "alpha", "old_version": 1, "new_version": 2}])
        self.assertEqual(diff["backends"]["changed"][0]["id"], "stable")
        self.assertEqual(diff["backends"]["changed"][0]["changed_fields"], ["seed"])
        with self.assertRaises(ReplayError):
            self.store.load_preview_cache(old_revision)
        self.assertEqual(
            changed_engine.verify_authoritative(preview).revision,
            2,
        )

    def test_modified_rule_requires_higher_version(self) -> None:
        self._run()
        changed = copy.deepcopy(self.raw_spec)
        changed["revision"] = 2
        changed["rules"][0]["canary_weight_bp"] = 1234
        changed["rules"][0]["version"] = 1
        path = self.workspace.write_spec(changed, "spec-bad-version.json")
        spec = load_spec(path)
        requests = load_requests(self.requests_path, spec)
        engine = DiagnosticEngine(spec, requests, self.store)
        with self.assertRaises(SpecError) as caught:
            engine.run()
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertEqual(caught.exception.entry, "rules[0]")
        self.assertEqual(caught.exception.field, "version")


if __name__ == "__main__":
    unittest.main()
