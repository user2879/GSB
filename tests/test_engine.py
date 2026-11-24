from __future__ import annotations

import json
import socketserver
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path

from receipt_diag.client import WorkerPool
from receipt_diag.engine import DiagnosticsEngine
from receipt_diag.errors import DiagnosticError
from receipt_diag.invalidation import compute_invalidation
from receipt_diag.ledger import Ledger
from receipt_diag.planning import plan_batches
from receipt_diag.specs import load_specs

from tests.helpers import run_fresh, write_sample


class EngineAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_real_worker_run_ledger_artifacts_and_replay(self) -> None:
        sample = write_sample(self.root / "sample")
        report, ledger, engine = run_fresh(sample, self.root / "run")
        try:
            self.assertTrue(report["ok"])
            self.assertEqual(report["completed_batches"], 5)
            self.assertLessEqual(report["concurrency"]["observed_global_max"], 2)
            self.assertTrue(report["concurrency"]["quota_ok"])
            self.assertEqual(
                report["concurrency"]["observed_per_worker_max"]["w-alpha"], 2
            )
            self.assertEqual(
                report["concurrency"]["observed_per_worker_max"]["w-beta"], 1
            )
            self.assertEqual(report["ledger_content_digest"], ledger.content_digest)
            self.assertTrue(Path(report["exit_report_path"]).is_file())
            for batch in report["batches"]:
                self.assertTrue(Path(batch["output_summary_path"]).is_file())
                self.assertTrue(Path(batch["param_digest_path"]).is_file())
                self.assertIsInstance(batch["wait_start_ms"], float)
                self.assertIsInstance(batch["wait_end_ms"], float)
                artifact = json.loads(Path(batch["output_summary_path"]).read_text("utf-8"))
                self.assertEqual(artifact["socket_evidence"]["host"], "127.0.0.1")
                self.assertGreater(artifact["socket_evidence"]["pid"], 0)
                self.assertEqual(artifact["socket_evidence"]["transport"], "tcp-loopback")
            status = engine.recovery_status(ledger)
            self.assertEqual(status["session_state"], "completed")
            self.assertFalse(status["prior_unclean_termination"])
            self.assertFalse(status["requires_rerun"])
            self.assertEqual(status["exit_report"]["ledger_content_digest"], ledger.content_digest)
            replay = engine.replay(ledger)
            self.assertEqual(len(replay["verified_batches"]), 5)
            versions = ledger.batch_versions()
            self.assertEqual(set(versions.values()), {1})
        finally:
            ledger.close()

    def test_three_receipt_formats_produce_identical_digests(self) -> None:
        sample = write_sample(self.root / "sample")
        summaries: dict[str, dict[str, tuple[str, str, str]]] = {}
        for fmt in ("r1", "r2", "r3"):
            report, ledger, _ = run_fresh(sample, self.root / fmt, receipt_format=fmt)
            try:
                summaries[fmt] = {
                    batch["batch_id"]: (
                        batch["output_digest"],
                        batch["receipt_digest"],
                        batch["param_digest"],
                    )
                    for batch in report["batches"]
                }
            finally:
                ledger.close()
        self.assertEqual(summaries["r1"], summaries["r2"])
        self.assertEqual(summaries["r2"], summaries["r3"])

    def test_receipt_format_corruption_and_dangling_injection(self) -> None:
        sample = write_sample(self.root / "sample")
        spec = load_specs(sample)
        for injection, expected_fragment in (
            ("receipt_corruption", "fields"),
            ("dependency_dangling", "duplicate_refs"),
        ):
            with self.subTest(injection=injection):
                ledger = Ledger(self.root / f"{injection}.db")
                try:
                    engine = DiagnosticsEngine(spec, self.root / f"{injection}-art")
                    with self.assertRaises(DiagnosticError) as caught:
                        engine.run(ledger, injection=injection)
                    self.assertEqual(caught.exception.code, "E_FIELD")
                    self.assertIn(expected_fragment, caught.exception.path)
                finally:
                    ledger.close()

    def test_ledger_digest_mismatch_injection(self) -> None:
        sample = write_sample(self.root / "sample")
        spec = load_specs(sample)
        ledger = Ledger(self.root / "digest.db")
        try:
            engine = DiagnosticsEngine(spec, self.root / "digest-art")
            with self.assertRaises(DiagnosticError) as caught:
                engine.run(ledger, injection="ledger_digest_mismatch")
            self.assertEqual(caught.exception.code, "E_LEDGER")
            self.assertIn("content_digest", caught.exception.path)
        finally:
            ledger.close()
        with self.assertRaises(DiagnosticError) as caught:
            Ledger(self.root / "digest.db").close()
        self.assertEqual(caught.exception.code, "E_LEDGER")

    def test_invalidation_reuses_unaffected_batches_and_replays(self) -> None:
        basic_dir = write_sample(self.root / "basic")
        changed_dir = write_sample(self.root / "changed", changed=True)
        old_spec = load_specs(basic_dir)
        new_spec = load_specs(changed_dir)
        old_plan = plan_batches(old_spec)
        new_plan = plan_batches(new_spec)
        invalidation = compute_invalidation(old_spec, new_spec, old_plan, new_plan)
        self.assertEqual(len(invalidation.rerun_entries), 3)
        self.assertEqual(len(invalidation.unaffected_batch_ids), 2)
        self.assertTrue(all(entry.worker_id == "w-alpha" for entry in invalidation.rerun_entries))

        ledger = Ledger(self.root / "ledger.db")
        try:
            old_engine = DiagnosticsEngine(old_spec, self.root / "artifacts")
            old_report = old_engine.run(ledger)
            before = ledger.batches_as_dict()
            new_engine = DiagnosticsEngine(new_spec, self.root / "artifacts")
            after_report = new_engine.apply_invalidation(
                ledger, old_spec, old_plan, invalidation
            )
            after = ledger.batches_as_dict()
            for batch_id in invalidation.unaffected_batch_ids:
                self.assertEqual(before[batch_id], after[batch_id])
                old_batch = next(b for b in old_report["batches"] if b["batch_id"] == batch_id)
                new_batch = next(
                    b for b in after_report["batches"] if b["batch_id"] == batch_id
                )
                self.assertEqual(old_batch["batch_version"], new_batch["batch_version"])
                self.assertEqual(old_batch["receipt_digest"], new_batch["receipt_digest"])
            for entry in invalidation.rerun_entries:
                self.assertEqual(after[entry.batch_id]["batch_version"], 2)
                self.assertEqual(after[entry.batch_id]["run_id"], new_engine.run_id("invalidation"))
            replay = new_engine.replay(ledger)
            self.assertEqual(len(replay["verified_batches"]), 5)
        finally:
            ledger.close()

    def test_snapshot_whole_replacement_and_noop(self) -> None:
        sample_basic = write_sample(self.root / "basic")
        sample_changed = write_sample(self.root / "changed", changed=True)
        report_a, ledger_a, _ = run_fresh(sample_basic, self.root / "a")
        report_b, ledger_b, _ = run_fresh(sample_changed, self.root / "b")
        snapshot_path = self.root / "a.snapshot.json"
        try:
            snapshot = ledger_a.export_snapshot(snapshot_path)
            self.assertEqual(snapshot["content_digest"], report_a["ledger_content_digest"])
            replaced = ledger_b.restore_snapshot(snapshot)
            expected = tuple(
                sorted(
                    set(batch["batch_id"] for batch in report_a["batches"])
                    | set(batch["batch_id"] for batch in report_b["batches"])
                )
            )
            self.assertEqual(replaced, expected)
            self.assertEqual(ledger_b.content_digest, report_a["ledger_content_digest"])
            self.assertEqual(ledger_b.restore_snapshot(snapshot), ())
            version_bumped = dict(snapshot)
            version_bumped["snapshot_version"] = int(snapshot["snapshot_version"]) + 1
            replaced_for_version = ledger_b.restore_snapshot(version_bumped)
            self.assertEqual(replaced_for_version, expected)
            self.assertEqual(ledger_b.snapshot_version, snapshot["snapshot_version"] + 1)
        finally:
            ledger_a.close()
            ledger_b.close()
        with Ledger(self.root / "b" / "ledger.db") as reopened:
            self.assertEqual(reopened.content_digest, report_a["ledger_content_digest"])
            self.assertEqual(len(reopened.committed_batch_ids()), 5)

    def test_replay_reports_first_field_mismatch(self) -> None:
        sample = write_sample(self.root / "sample")
        _, ledger, engine = run_fresh(sample, self.root / "run")
        try:
            first = engine.plan.sorted_batches()[0]
            ledger.connection.execute(
                "UPDATE batches SET output_digest='corrupted' WHERE batch_id=?",
                (first.batch_id,),
            )
            ledger.connection.commit()
            ledger.recompute_content_digest()
            with self.assertRaises(DiagnosticError) as caught:
                engine.replay(ledger)
            self.assertEqual(caught.exception.code, "E_REPLAY")
            self.assertEqual(
                caught.exception.path,
                f"{first.batch_id}/*/output_digest",
            )
        finally:
            ledger.close()

    def test_worker_output_is_invariant_to_request_order(self) -> None:
        sample = write_sample(self.root / "sample")
        spec = load_specs(sample)
        batch = plan_batches(spec).sorted_batches()[0]
        requests = [
            {
                "request_id": item.request_id,
                "arrival_seq": item.arrival_seq,
                "input": list(item.input_vector),
            }
            for item in batch.requests
        ]
        with WorkerPool(spec) as pool:
            forward = pool.forward_batch(batch.batch_id, batch.worker_id, requests)
            reverse = pool.forward_batch(
                batch.batch_id, batch.worker_id, list(reversed(requests))
            )
        self.assertEqual(forward["output_digest"], reverse["output_digest"])
        self.assertEqual(forward["param_digest"], reverse["param_digest"])

    def test_client_uses_real_socket_timeout(self) -> None:
        class SilentHandler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                time.sleep(0.4)

        server = socketserver.TCPServer(("127.0.0.1", 0), SilentHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        sample = write_sample(self.root / "sample")
        spec = load_specs(sample)
        worker = replace(spec.worker_by_id["w-alpha"], port=port)
        try:
            with WorkerPool(spec) as pool:
                with self.assertRaises(DiagnosticError) as caught:
                    pool._rpc(worker, {"op": "ping"}, timeout_ms=50)
            self.assertEqual(caught.exception.code, "E_WORKER")
            self.assertIn("real socket timeout", caught.exception.detail)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
