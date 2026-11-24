import json
import socket
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from batchdiag.errors import ReplayError
from batchdiag.invalidation import recompute_all
from batchdiag.ledger import (LEDGER_FORMAT_VERSION, Ledger, open_ledger,
                              replay_check)
from batchdiag.planner import build_plan
from batchdiag.samples import generate
from batchdiag.scheduler import schedule_all
from batchdiag.spec import MergeRules, Spec, load_spec
from batchdiag.cli import start_workers, stop_workers

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "run" / "test_samples"


def free_port():
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def ephemeral_spec(base, timeout_ms=800):
    used = set()
    workers = []
    for worker in base.workers:
        port = free_port()
        while port in used:
            port = free_port()
        used.add(port)
        workers.append(_replace(worker, port=port))
    rules = MergeRules(global_concurrency=2, socket_timeout_ms=timeout_ms)
    return Spec(requests=base.requests, workers=workers, merge_rules=rules,
                model_dim=base.model_dim)


def _replace(worker, **changes):
    data = {field: getattr(worker, field)
            for field in worker.__dataclass_fields__}
    data.update(changes)
    return type(worker)(**data)


class SchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generate(SAMPLES)
        cls.base = load_spec(SAMPLES / "requests.json",
                             SAMPLES / "workers.json",
                             SAMPLES / "merge_rules.json")

    def test_real_loopback_scheduling_respects_quotas(self):
        spec = ephemeral_spec(self.base)
        plan = build_plan(spec)
        processes = start_workers(spec)
        try:
            results, trace = schedule_all(spec, plan)
        finally:
            stop_workers(processes)
        self.assertEqual(len(results), len(plan.batches))
        self.assertLessEqual(trace.max_global_in_flight,
                             spec.merge_rules.global_concurrency)
        for worker_id, peak in trace.max_per_worker_in_flight.items():
            self.assertLessEqual(
                peak, spec.worker(worker_id).quota)
        for record in trace.records:
            self.assertGreaterEqual(record.wait_ms, 0)
            self.assertLessEqual(record.wait_started_ms,
                                 record.wait_ended_ms)
            self.assertGreater(record.socket_elapsed_ms, 0)

    def test_real_socket_timeout_reports_e_timeout(self):
        spec = ephemeral_spec(self.base, timeout_ms=400)
        plan = build_plan(spec)
        processes = start_workers(spec, stall_ms=3000)
        try:
            _results, trace = schedule_all(spec, plan)
        finally:
            stop_workers(processes)
        timeouts = [r for r in trace.records
                    if r.error_code == "E_TIMEOUT"]
        self.assertTrue(timeouts)
        first = timeouts[0]
        self.assertIn("400 ms", first.error)

    def test_port_conflict_detected(self):
        from batchdiag.worker import ensure_port_free
        from batchdiag.errors import FieldError
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        try:
            with self.assertRaises(FieldError):
                ensure_port_free("127.0.0.1", port, timeout=0.5)
        finally:
            server.close()


class LedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generate(SAMPLES)
        cls.spec = load_spec(SAMPLES / "requests.json",
                             SAMPLES / "workers.json",
                             SAMPLES / "merge_rules.json")
        cls.plan = build_plan(cls.spec)

    def _build(self, path):
        results = recompute_all(self.spec, self.plan)
        ledger = Ledger(path, replace=True)
        for batch in sorted(self.plan.batches,
                            key=lambda b: (b.worker_id, b.batch_seq)):
            ledger.commit_batch(results[batch.batch_id])
        ledger.close()
        return results

    def test_format_version_and_digest_stored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "l.sqlite3")
            self._build(path)
            ledger, status = open_ledger(path)
            self.assertEqual(status.action, "opened")
            self.assertEqual(ledger.format_version,
                             LEDGER_FORMAT_VERSION)
            self.assertEqual(ledger.content_digest(),
                             ledger.stored_digest())
            self.assertEqual(len(ledger.batch_ids()),
                             len(self.plan.batches))
            ledger.close()

    def test_tampered_snapshot_is_replaced_wholesale(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "l.sqlite3")
            self._build(path)
            conn = sqlite3.connect(path)
            with conn:
                conn.execute(
                    "UPDATE batches SET output_digest='tampered' "
                    "WHERE batch_id=(SELECT batch_id FROM batches LIMIT 1)")
            conn.close()
            ledger, status = open_ledger(path)
            ledger.close()
            self.assertEqual(status.action, "replaced")
            self.assertTrue(status.differences)
            self.assertEqual(status.differences[0]["source"],
                             "schema_meta.content_digest")

    def test_format_version_mismatch_replaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "l.sqlite3")
            self._build(path)
            conn = sqlite3.connect(path)
            with conn:
                conn.execute(
                    "UPDATE schema_meta SET value='999' "
                    "WHERE key='format_version'")
            conn.close()
            _ledger, status = open_ledger(path)
            self.assertEqual(status.action, "replaced")
            self.assertIn("format_version",
                          status.differences[0]["source"])

    def test_replay_check_passes_on_clean_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "l.sqlite3")
            self._build(path)
            ledger = Ledger(path)
            results = recompute_all(self.spec, self.plan)
            self.assertTrue(replay_check(ledger, self.plan, results))
            ledger.close()

    def test_replay_check_locates_first_divergence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "l.sqlite3")
            self._build(path)
            conn = sqlite3.connect(path)
            row = conn.execute(
                "SELECT result_json FROM batches ORDER BY worker_id,"
                "batch_seq LIMIT 1").fetchone()
            data = json.loads(row[0])
            target = data["sorted_requests"][0]
            data["per_request_digest"][target] = "f" * 24
            with conn:
                conn.execute(
                    "UPDATE batches SET result_json=? WHERE batch_id=?",
                    (json.dumps(data, sort_keys=True,
                                separators=(",", ":")), data["batch_id"]))
            conn.close()
            ledger = Ledger(path)
            results = recompute_all(self.spec, self.plan)
            with self.assertRaises(ReplayError) as ctx:
                replay_check(ledger, self.plan, results)
            detail = ctx.exception.to_dict()
            self.assertEqual(detail["code"], "E_REPLAY")
            self.assertEqual(detail["batch_id"], data["batch_id"])
            self.assertEqual(detail["request_id"], target)
            self.assertEqual(detail["field"], "per_request_digest")
            ledger.close()


class CrashRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generate(SAMPLES)
        cls.spec = load_spec(SAMPLES / "requests.json",
                             SAMPLES / "workers.json",
                             SAMPLES / "merge_rules.json")

    def test_kill_and_recover_only_uses_committed_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = str(Path(tmp) / "crash.sqlite3")
            proc = subprocess.run(
                [sys.executable, "-m", "batchdiag.cli", "crash",
                 "--requests", str(SAMPLES / "requests.json"),
                 "--workers", str(SAMPLES / "workers.json"),
                 "--rules", str(SAMPLES / "merge_rules.json"),
                 "--ledger", ledger_path, "--after", "2"],
                capture_output=True, text=True)
            self.assertEqual(proc.returncode, -9)
            from batchdiag.engine import recover
            outcome = recover(self.spec, ledger_path)
            self.assertTrue(outcome["replayed_ok"])
            self.assertEqual(len(outcome["committed_batch_ids"]), 2)
            unknown = {b["batch_id"] for b in outcome["unknown_batches"]}
            plan = build_plan(self.spec)
            self.assertEqual(
                unknown,
                {b.batch_id for b in plan.batches}
                - set(outcome["committed_batch_ids"]))
            self.assertTrue(all(b["state"] == "unknown"
                                for b in outcome["unknown_batches"]))
            ledger = Ledger(ledger_path)
            self.assertEqual(len(ledger.batch_ids()), len(plan.batches))
            ledger.close()


class GuiFallbackTests(unittest.TestCase):
    def test_gui_command_degrades_without_tk(self):
        from batchdiag.cli import cmd_gui
        import argparse
        args = argparse.Namespace(
            requests=str(SAMPLES / "requests.json"),
            workers=str(SAMPLES / "workers.json"),
            rules=str(SAMPLES / "merge_rules.json"),
            ledger=":memory:")
        # Must never raise even when tkinter/Tk is unavailable.
        cmd_gui(args)


if __name__ == "__main__":
    unittest.main()
