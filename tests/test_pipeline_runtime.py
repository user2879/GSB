import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from gsb.attribution import attribute_decision
from gsb.codec import encode_block
from gsb.hashing import canonical_bytes
from gsb.ledger import Ledger, batch_tx_digest
from gsb.pipeline import recover_pipeline, run_pipeline
from gsb.sample import build_changed_sample_spec, build_sample_spec
from gsb.scheduler import Scheduler, SocketTimeoutError, tcp_transact
from gsb.snapshot import open_or_replace_snapshot, write_snapshot
from gsb.spec import load_spec


ROOT = Path(__file__).resolve().parents[1]
CHILD_CRASH = """
import sys
from gsb.pipeline import run_pipeline
run_pipeline(spec_path=sys.argv[1], output_dir=sys.argv[2], session_id='crash', crash_after_commit=1)
"""


class RuntimeTestCase(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._temp.name)
        self.spec_path = self.tmp / "spec.json"
        self.spec_path.write_text(json.dumps(build_sample_spec()), encoding="utf-8")

    def tearDown(self):
        self._temp.cleanup()


class SchedulerTests(unittest.TestCase):
    def test_global_and_per_column_lease_limits_and_unique_ports(self):
        jobs = [
            {"block_group_id": f"BG-{i}", "columns": ["shared"], "row_count": i + 1, "response_delay_ms": 120}
            for i in range(4)
        ]
        scheduler = Scheduler(global_budget=2, per_column_lease=1, socket_timeout_ms=3000)
        started = time.monotonic()
        receipts = scheduler.run(jobs)
        elapsed = time.monotonic() - started
        peaks = scheduler.lease_peaks()
        self.assertEqual(len(receipts), 4)
        self.assertLessEqual(peaks["peak_global_in_flight"], 2)
        self.assertEqual(peaks["peak_global_in_flight"], 2)
        self.assertLessEqual(peaks["peak_per_column"]["shared"], 1)
        self.assertEqual(peaks["peak_per_column"]["shared"], 1)
        ports = [receipt["worker_port"] for receipt in receipts]
        self.assertEqual(len(set(ports)), 4)
        self.assertTrue(all(host_port > 1024 for host_port in ports))
        self.assertGreaterEqual(elapsed, 0.4)

    def test_single_column_lease_serializes_shared_column(self):
        scheduler = Scheduler(global_budget=4, per_column_lease={"shared": 1}, socket_timeout_ms=3000)
        jobs = [
            {"block_group_id": "BG-A", "columns": ["shared", "a"], "row_count": 1, "response_delay_ms": 80},
            {"block_group_id": "BG-B", "columns": ["shared", "b"], "row_count": 2, "response_delay_ms": 80},
            {"block_group_id": "BG-C", "columns": ["c"], "row_count": 3, "response_delay_ms": 80},
        ]
        receipts = scheduler.run(jobs)
        self.assertEqual({item["decoded_rows"] for item in receipts}, {1, 2, 3})
        self.assertEqual(scheduler.peak_per_column["shared"], 1)
        self.assertLessEqual(scheduler.peak_per_column.get("a", 1), 1)

    def test_real_socket_timeout(self):
        import socket

        blackhole = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blackhole.bind(("127.0.0.1", 0))
        blackhole.listen(1)
        port = blackhole.getsockname()[1]
        started = time.monotonic()
        try:
            with self.assertRaises(SocketTimeoutError):
                tcp_transact("127.0.0.1", port, {"block_group_id": "BG-T"}, timeout_ms=80)
            self.assertGreaterEqual(time.monotonic() - started, 0.06)
        finally:
            blackhole.close()


class VerificationAndAttributionTests(RuntimeTestCase):
    def test_faulted_sample_reports_false_skip_and_full_verification(self):
        report = run_pipeline(spec=build_sample_spec(), output_dir=self.tmp / "run", session_id="demo")
        self.assertEqual(report["summary"]["false_skips"], 1)
        issue = report["issues"][0]
        self.assertEqual(issue["code"], "E_FALSE_SKIP")
        self.assertEqual(issue["block_group_id"], "BG-A02")
        self.assertEqual(issue["missed_rows"], 4)
        self.assertEqual(issue["missed_row_indexes"], [1, 4, 5, 9])
        verification = report["verification"]
        self.assertEqual(verification["mode"], "full")
        self.assertEqual(verification["actual_decoded_block_groups"], 4)
        self.assertEqual(verification["actual_decoded_rows"], 44)
        self.assertEqual(verification["index_distorted_block_groups"], 1)
        self.assertEqual(len(report["execution"]["worker_receipts"]), 4)

    def test_compression_mismatch_is_a_distinct_attribution_source(self):
        entry = attribute_decision(
            kind="false_skip",
            query={"id": "q"},
            block_group={"id": "BG-C", "columns": [{"name": "a", "compression": "zlib-6", "encoding": "raw"}]},
            block={"column_meta": {"a": {"compression": "zlib-1", "encoding": "raw"}}},
            index_mismatches=[],
            compression_mismatches=[{"column": "a", "declared": "zlib-6", "actual": "zlib-1"}],
            encoding_mismatches=[],
        )
        sources = {item["source"]: item["ratio"] for item in entry["contributions"]}
        self.assertEqual(entry["dominant_source"], "compression_mismatch")
        self.assertEqual(sources["compression_mismatch"], 1.0)
        self.assertEqual(sources["index_distortion"], 0.0)
        self.assertEqual(sources["predicate_null_order"], 0.0)

    def test_null_count_false_skip_gets_null_semantics_contribution(self):
        predicate = {"column": "amount", "op": ">=", "value": 60}
        entry = attribute_decision(
            kind="false_skip",
            query={"id": "q-null", "predicates": [predicate]},
            block_group={"id": "BG-N", "columns": [], "skip_index": {}},
            block={"column_meta": {"amount": {"null_count": 8}}},
            group_plan={"witness_predicate": predicate, "witness_index_fields": ["null_count"]},
            index_mismatches=[],
            compression_mismatches=[],
            encoding_mismatches=[],
            used_predicate_order=[{"column": "other", "op": "is_null"}],
        )
        sources = {item["source"]: item["ratio"] for item in entry["contributions"]}
        self.assertGreater(sources["predicate_null_order"], 0.0)
        self.assertEqual(entry["dominant_source"], "predicate_null_order")


class InvalidationTests(RuntimeTestCase):
    def test_only_invalidated_group_is_decoded_and_unaffected_query_is_byte_identical(self):
        output = self.tmp / "incremental"
        first = run_pipeline(spec=build_sample_spec(), output_dir=output, session_id="demo")
        old_cache = json.loads((output / "cache.json").read_text(encoding="utf-8"))
        old_qid = canonical_bytes(old_cache["query_payloads"]["q_id_values"])

        second = run_pipeline(spec=build_changed_sample_spec(), output_dir=output, session_id="demo")
        new_cache = json.loads((output / "cache.json").read_text(encoding="utf-8"))
        self.assertEqual(second["execution"]["decode_jobs_this_run"], ["BG-A02"])
        self.assertEqual(second["summary"]["reused_block_groups"], 3)
        self.assertEqual(second["invalidation"]["invalidated_queries"], ["q_amount_high", "q_null_amount", "q_region_3"])
        self.assertEqual(second["invalidation"]["unaffected_queries"], ["q_id_values"])
        self.assertEqual(canonical_bytes(new_cache["query_payloads"]["q_id_values"]), old_qid)
        self.assertEqual(first["spec_digest"] != second["spec_digest"], True)


class LedgerTests(RuntimeTestCase):
    def test_forced_kill_marks_uncommitted_batches_unknown_and_resumes(self):
        output = self.tmp / "crash"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-c", CHILD_CRASH, str(self.spec_path), str(output)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, -9, proc.stderr)

        after_kill = recover_pipeline(output_dir=output, session_id="crash", spec_path=self.spec_path)
        recovery = after_kill["recovery"]
        self.assertEqual(recovery["status"], "requires_rerun")
        self.assertEqual(recovery["unknown_batches"], ["BG-A02", "BG-A03", "BG-A04"])
        self.assertEqual([item["block_group_id"] for item in recovery["committed_batches_replayed"]], ["BG-A01"])

        resumed = run_pipeline(spec_path=self.spec_path, output_dir=output, session_id="crash")
        self.assertEqual(resumed["execution"]["decode_jobs_this_run"], ["BG-A02", "BG-A03", "BG-A04"])
        final = resumed["ledger"]["recovery"]
        self.assertEqual(final["status"], "consistent")
        self.assertEqual(len(final["committed_batches_replayed"]), 4)
        self.assertEqual(len(final["query_results_replayed"]), 4)
        self.assertTrue(all(item["consistent"] for item in final["committed_batches_replayed"]))
        self.assertTrue(all(item["consistent"] for item in final["query_results_replayed"]))

    def test_recovery_locates_first_inconsistent_block_group_and_field(self):
        output = self.tmp / "replay"
        run_pipeline(spec=build_sample_spec(), output_dir=output, session_id="demo")
        ledger_path = output / "ledger.sqlite3"
        ledger = Ledger(ledger_path)
        row = dict(
            ledger.connection.execute("SELECT * FROM batches WHERE block_group_id='BG-A02'").fetchone()
        )
        row["content_digest"] = "0" * 64
        new_tx = batch_tx_digest(row)
        with ledger.connection:
            ledger.connection.execute(
                "UPDATE batches SET content_digest=?, tx_digest=? WHERE session_id=? AND sequence=?",
                (row["content_digest"], new_tx, row["session_id"], row["sequence"]),
            )
        ledger.close()

        spec = load_spec(build_sample_spec())
        groups = {group["id"]: group for group in spec["block_groups"]}

        def verify_batch(raw, payload):
            block_payload = payload["block"]
            rebuilt = encode_block(groups[raw["block_group_id"]], block_payload["columns"])
            if rebuilt["content_digest"] != raw["content_digest"]:
                return {"field": "content_digest", "expected": rebuilt["content_digest"], "actual": raw["content_digest"]}
            return None

        ledger = Ledger(ledger_path)
        recovery = ledger.recover("demo", [q["id"] for q in spec["queries"]], verify_batch=verify_batch)
        ledger.close()
        self.assertEqual(recovery["status"], "inconsistent")
        self.assertIsNotNone(recovery["first_inconsistency"])
        self.assertEqual(recovery["first_inconsistency"]["block_group_id"], "BG-A02")
        self.assertEqual(recovery["first_inconsistency"]["field"], "content_digest")

    def test_snapshot_mismatch_is_wholesale_replacement(self):
        path = self.tmp / "snapshot.json"
        original = {"hello": "stale", "nested": {"keep": False}, "values": [1, 2]}
        write_snapshot(path, original)
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        on_disk["snapshot_version"] = 999
        path.write_text(json.dumps(on_disk), encoding="utf-8")

        fresh = {"hello": "fresh", "replaced": True}
        snapshot, info = open_or_replace_snapshot(
            path,
            session_id="s",
            snapshot_version=2,
            spec_digest="spec-digest",
            factory=lambda: fresh,
        )
        self.assertTrue(info["existed"])
        self.assertTrue(info["replaced"])
        self.assertEqual(info["reason"], "snapshot_version")
        self.assertEqual(snapshot["hello"], "fresh")
        self.assertNotIn("nested", snapshot)
        reread = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(reread["snapshot_version"], 2)
        self.assertEqual(reread["spec_digest"], "spec-digest")


if __name__ == "__main__":
    unittest.main()
