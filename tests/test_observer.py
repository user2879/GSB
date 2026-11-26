from __future__ import annotations

import json
import socket
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path

from tcp_observer.canonical import canonical_bytes, sha256_hex
from tcp_observer.cli import execute_demo, port_and_timeout_checks
from tcp_observer.engine import ObservationWorkspace, build_observation
from tcp_observer.errors import FormatError, ObserverError, ReplayError
from tcp_observer.receipts import decode_and_normalize_file
from tcp_observer.replay import compare_items
from tcp_observer.scheduler import InFlightScheduler, ScriptedMonotonicClock, deterministic_queue_demo
from tcp_observer.scripting import validate_for_run
from tcp_observer.spec import canonical_script_digest, load_session
from tcp_observer.transport import bind_listener, run_live_exchange

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/session.json"


class SpecTests(unittest.TestCase):
    def test_valid_spec_loads(self) -> None:
        spec = load_session(SPEC)
        self.assertEqual(14, spec.event_count)
        self.assertEqual(14, len(spec.entries))
        self.assertEqual("127.0.0.1", spec.peer_host)

    def test_bad_delay_reports_field_and_line(self) -> None:
        with self.assertRaises(ObserverError) as ctx:
            load_session(ROOT / "samples/invalid/spec_bad_delay.json")
        exc = ctx.exception
        self.assertEqual("E_SPEC", exc.code)
        self.assertEqual("entries[1].delay_ms", exc.field_name)
        self.assertEqual(12, exc.line)

    def test_gap_and_dupack_codes(self) -> None:
        with self.assertRaises(ObserverError) as gap:
            validate_for_run(load_session(ROOT / "configs/session_gap.json"))
        self.assertEqual("E_SCRIPT", gap.exception.code)
        self.assertEqual(1002, gap.exception.details["gap_start_seq"])
        self.assertEqual(2, gap.exception.details["missing_count"])

        with self.assertRaises(ObserverError) as cluster:
            validate_for_run(load_session(ROOT / "configs/session_dupack_cluster.json"))
        self.assertEqual("E_DUPACK", cluster.exception.code)
        self.assertEqual(5, cluster.exception.details["count"])


class ReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        execute_demo(ROOT / "artifacts/test-shared")

    def test_three_versions_normalize_with_provenance(self) -> None:
        o1 = decode_and_normalize_file(ROOT / "samples/receipts_o1.txt")
        o2 = decode_and_normalize_file(ROOT / "samples/receipts_o2.jsonl")
        o3 = decode_and_normalize_file(ROOT / "samples/receipts_o3.jsonl")
        self.assertEqual("o1", o1[0]["source_version"])
        self.assertIn("ssthresh", o1[0]["filled_fields"])
        self.assertIn("srtt_ms", o1[0]["filled_fields"])
        self.assertIn("scale", o1[0]["filled_fields"])
        self.assertTrue(o1[0]["fill_basis"]["scale"])
        self.assertEqual("o2", o2[0]["source_version"])
        self.assertIn("scale", o2[0]["filled_fields"])
        self.assertEqual("o3", o3[0]["source_version"])
        self.assertEqual([], o3[0]["filled_fields"])

    def test_format_errors_locate_lines(self) -> None:
        cases = [
            ("samples/invalid/o1_unknown.txt", "format_version", 1),
            ("samples/invalid/o1_width.txt", "fixed_width", 1),
            ("samples/invalid/o2_missing.jsonl", "srtt_ms", 2),
            ("samples/invalid/o2_negative_duration.jsonl", "duration_ms", 2),
        ]
        for relative, field, line in cases:
            with self.subTest(relative=relative):
                with self.assertRaises(FormatError) as ctx:
                    decode_and_normalize_file(ROOT / relative)
                self.assertEqual("E_FORMAT", ctx.exception.code)
                self.assertEqual(field, ctx.exception.field_name)
                self.assertEqual(line, ctx.exception.line)


class EngineTests(unittest.TestCase):
    def test_deterministic_state_and_events(self) -> None:
        spec = load_session(SPEC)
        first = build_observation(spec)
        second = build_observation(load_session(SPEC))
        self.assertEqual(canonical_bytes(first), canonical_bytes(second))
        self.assertEqual(2, first["summary"]["totals"]["fast_retransmit_count"])
        self.assertEqual(1, first["summary"]["totals"]["timeout_count"])
        self.assertTrue(all(item["error_bounds_raw"]["rto_scaled"] > 0 for item in first["items"]))

    def test_open_window_edit_recomputes_only_suffix(self) -> None:
        spec = load_session(SPEC)
        changed_seq = 1012
        old_entry = spec.entry_for_seq(changed_seq)
        assert old_entry is not None
        new_entry = replace(old_entry, delay_ms=4)
        entries = tuple(new_entry if entry.event_seq == changed_seq else entry for entry in spec.entries)
        edited = replace(spec, entries=entries, script_digest=sha256_hex(canonical_script_digest(entries, spec.script_version)))
        workspace = ObservationWorkspace(spec)
        plan = workspace.apply_open_window_script_edit(edited, changed_seq)
        self.assertEqual(["item-001012", "item-001013"], plan["affected_items"])
        self.assertEqual(12, plan["reused_item_count"])
        self.assertTrue(plan["sealed_windows_byte_unchanged"])
        self.assertEqual([0, 1, 2], sorted(plan["sealed_window_digests_before"]))

    def test_scale_edit_affects_all_fixed_point_items(self) -> None:
        spec = load_session(SPEC)
        scaled = replace(spec, scale=2048, scale_version=2)
        old = build_observation(spec)
        new = build_observation(scaled)
        self.assertNotEqual(old["summary"]["content_digest"], new["summary"]["content_digest"])
        self.assertTrue(all(a["state_after"]["cwnd_scaled"] != b["state_after"]["cwnd_scaled"] for a, b in zip(old["items"], new["items"])))



    def test_replay_difference_is_located(self) -> None:
        import copy

        spec = load_session(SPEC)
        expected = build_observation(spec)
        bad = copy.deepcopy(expected)
        bad["items"][3]["state_after"]["cwnd_scaled"] += spec.scale
        with self.assertRaises(ReplayError) as ctx:
            compare_items(expected, bad)
        self.assertEqual("E_REPLAY", ctx.exception.code)
        self.assertEqual(expected["items"][3]["event_seq"], ctx.exception.event_seq)
        self.assertIn("state_after", ctx.exception.field_name)


class SchedulerTests(unittest.TestCase):
    def test_deterministic_double_limit_demo(self) -> None:
        demo = deterministic_queue_demo()
        self.assertEqual(2, demo["peak_in_flight"]["global"])
        waits = {ticket["batch_id"]: ticket for ticket in demo["tickets"]}
        self.assertTrue(waits["batch-a2"]["queued"])
        self.assertEqual(3, waits["batch-a2"]["wait_ms"])
        self.assertEqual(5, waits["batch-b2"]["wait_ms"])

    def test_threaded_scheduler_never_exceeds_limits(self) -> None:
        clock = ScriptedMonotonicClock()
        scheduler = InFlightScheduler(global_limit=2, connection_quota={"c-a": 1, "c-b": 1}, clock=clock)
        active = 0
        active_by_conn = {"c-a": 0, "c-b": 0}
        peak_global = 0
        peak_by_conn = {"c-a": 0, "c-b": 0}
        lock = threading.Lock()

        def worker(index: int) -> None:
            nonlocal active, peak_global
            connection = "c-a" if index % 2 == 0 else "c-b"
            ticket = scheduler.acquire(f"b{index}", connection)
            assert ticket is not None
            with lock:
                active += 1
                active_by_conn[connection] += 1
                peak_global = max(peak_global, active)
                peak_by_conn[connection] = max(peak_by_conn[connection], active_by_conn[connection])
            time.sleep(0.002)
            with lock:
                active -= 1
                active_by_conn[connection] -= 1
            scheduler.release(ticket)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(2)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertLessEqual(peak_global, 2)
        self.assertLessEqual(peak_by_conn["c-a"], 1)
        self.assertLessEqual(peak_by_conn["c-b"], 1)


class TransportTests(unittest.TestCase):
    def test_live_loopback_exchange(self) -> None:
        spec = load_session(SPEC)
        exchange = run_live_exchange(spec)
        self.assertEqual(14, len(exchange["responses"]))
        self.assertEqual([spec.initial_seq + i for i in range(14)], [r["event_seq"] for r in exchange["responses"]])

    def test_port_and_timeout_errors(self) -> None:
        spec = load_session(SPEC)
        checks = port_and_timeout_checks(spec)
        self.assertTrue(all(check["ok"] for check in checks), checks)

    def test_bind_reports_e_port(self) -> None:
        holder = bind_listener("127.0.0.1", 0)
        port = holder.getsockname()[1]
        try:
            with self.assertRaises(ObserverError) as ctx:
                bind_listener("127.0.0.1", port)
            self.assertEqual("E_PORT", ctx.exception.code)
        finally:
            holder.close()


class EndToEndRecoveryTests(unittest.TestCase):
    def test_demo_is_reproducible_and_recovers_sigkill(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            report_a = execute_demo(Path(first))
            report_b = execute_demo(Path(second))
            self.assertTrue(report_a["ok"])
            self.assertTrue(report_b["ok"])
            self.assertEqual(report_a["summary_content_digest"], report_b["summary_content_digest"])
            self.assertEqual(report_a["ledger_core_sha256"], report_b["ledger_core_sha256"])
            path_a = Path(first) / "ledger_core.jsonl"
            path_b = Path(second) / "ledger_core.jsonl"
            self.assertEqual(path_a.read_bytes(), path_b.read_bytes())
            recovery = report_a["recovery"]
            self.assertEqual(-9, recovery["forced_kill_returncode"])
            self.assertTrue(recovery["core_ledger_byte_match"])
            self.assertTrue(recovery["replay_ok"])
            self.assertEqual([3], recovery["unknown_batches_marked_and_rerun"])
            snapshot_report = json.loads((Path(first) / "snapshot_replacement.json").read_text())
            self.assertEqual("replaced_whole_snapshot", snapshot_report["replacement_action"])
            backup = Path(first) / snapshot_report["backup_path"]
            self.assertTrue(backup.exists())
            self.assertEqual(0o444, backup.stat().st_mode & 0o777)


if __name__ == "__main__":
    unittest.main(verbosity=2)
