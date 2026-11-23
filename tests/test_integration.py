from __future__ import annotations

import json
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from eventbus_diag.engine import DiagnosticEngine, FaultInjection
from eventbus_diag.spec import load_dataset

from .helpers import (
    default_channels,
    default_subscribers,
    write_events,
    write_topic_events,
    write_spec,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def wait_port_free(port: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if sys.platform != "win32":
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
            sock.close()
            return
        except OSError:
            sock.close()
            time.sleep(0.05)
    raise AssertionError(f"port {port} did not become free")


class IntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def dataset(
        self,
        *,
        version: int = 1,
        subscribers: list[dict] | None = None,
        channels: list[dict] | None = None,
        event_count: int = 6,
        handshake_timeout_ms: int = 3000,
        global_limit: int = 2,
        batch_size: int = 2,
    ):
        subscribers = subscribers or default_subscribers(2)
        channels = channels or default_channels(batch_size=batch_size, quota=2)
        spec = write_spec(
            self.root / f"spec-{version}.json",
            version=version,
            global_lease_limit=global_limit,
            handshake_timeout_ms=handshake_timeout_ms,
            subscribers=subscribers,
            channels=channels,
        )
        events = write_events(self.root / f"events-{version}.ndjson", event_count)
        return load_dataset(spec, events)

    def test_real_subprocess_dispatch_ledger_and_quota(self) -> None:
        dataset = self.dataset()
        db = self.root / "dispatch.sqlite"
        log = self.root / "dispatch.jsonl"
        engine = DiagnosticEngine(dataset, db_path=db, log_path=log)
        try:
            report = engine.run()
            self.assertEqual(report.status, "COMPLETED")
            self.assertEqual(report.batch_count, 6)
            self.assertEqual(report.confirmed_batches, 6)
            self.assertEqual(report.failed_batches, 0)
            self.assertEqual(report.verified_batches, 6)
            snapshot = engine.snapshot()
        finally:
            engine.close()

        self.assertEqual(snapshot["run"]["status"], "COMPLETED")
        self.assertEqual(snapshot["stats"]["active_leases"], 0)
        self.assertEqual(snapshot["stats"]["confirmed_batches"], 6)
        self.assertIn("in_flight_count", snapshot["run"])
        for batch in snapshot["batches"]:
            self.assertEqual(batch["state"], "CONFIRMED")
            self.assertIsNotNone(batch["ack_summary"])
            self.assertEqual(batch["quantization_version"], "eventbus-quant-v1")
            self.assertLess(batch["quota"]["global_active_after_grant"], 2 + 1)
            self.assertLessEqual(
                batch["quota"]["subscriber_active_after_grant"],
                batch["quota"]["subscriber_limit"],
            )
            self.assertLessEqual(
                batch["quota"]["channel_sends_in_window"],
                batch["quota"]["channel_rate_limit"],
            )
        with sqlite3.connect(db) as conn:
            conn.row_factory = sqlite3.Row
            info = conn.execute("SELECT format_version, content_digest FROM ledger_info").fetchone()
            transition_count = conn.execute("SELECT COUNT(*) AS n FROM transitions").fetchone()["n"]
            lease_count = conn.execute("SELECT COUNT(*) AS n FROM leases").fetchone()["n"]
            quota_count = conn.execute("SELECT COUNT(*) AS n FROM quota_usage").fetchone()["n"]
        self.assertEqual(info["format_version"], 1)
        self.assertEqual(len(info["content_digest"]), 64)
        self.assertEqual(transition_count, 12)
        self.assertEqual(lease_count, 6)
        self.assertEqual(quota_count, 6)
        log_records = [
            json.loads(line)
            for line in log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertTrue(any(record["code"] == "E_LEASE_GRANTED" for record in log_records))
        self.assertTrue(
            any(str(record.get("thread", "")).startswith("dispatch_") for record in log_records)
        )

    def test_missing_ack_field_reports_e_ack_and_retry_succeeds(self) -> None:
        subscribers = default_subscribers(1)
        dataset = self.dataset(subscribers=subscribers, event_count=2, batch_size=2)
        db = self.root / "ack.sqlite"
        log = self.root / "ack.jsonl"
        engine = DiagnosticEngine(
            dataset,
            db_path=db,
            log_path=log,
            fault_injections={
                subscribers[0]["subscriber_id"]: FaultInjection(omit_ack_field="event_ids")
            },
        )
        try:
            failed_report = engine.run()
            self.assertEqual(failed_report.status, "PARTIAL")
            self.assertEqual(failed_report.failed_batches, 1)
            self.assertEqual(failed_report.errors[0]["code"], "E_ACK")
            failed_snapshot = engine.snapshot()
            self.assertEqual(failed_snapshot["batches"][0]["failure_code"], "E_ACK")
        finally:
            engine.close()

        retry_engine = DiagnosticEngine(dataset, db_path=db, log_path=log)
        try:
            retry_report = retry_engine.run()
            self.assertEqual(retry_report.status, "COMPLETED")
            self.assertEqual(retry_report.confirmed_batches, 1)
            retry_snapshot = retry_engine.snapshot()
        finally:
            retry_engine.close()
        self.assertEqual(retry_snapshot["batches"][0]["state"], "CONFIRMED")
        log_codes = [
            json.loads(line)["code"]
            for line in log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertIn("E_ACK", log_codes)

    def test_occupied_port_reports_e_port(self) -> None:
        subscribers = default_subscribers(1)
        port = int(subscribers[0]["port"])
        dataset = self.dataset(subscribers=subscribers, event_count=1, batch_size=1)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", port))
        listener.listen(1)
        try:
            engine = DiagnosticEngine(dataset, db_path=self.root / "port.sqlite", log_path=self.root / "port.jsonl")
            try:
                report = engine.run()
            finally:
                engine.close()
        finally:
            listener.close()
        self.assertEqual(report.status, "PARTIAL")
        self.assertEqual(report.failed_batches, 1)
        self.assertEqual(report.errors[0]["code"], "E_PORT")

    def test_handshake_timeout_reports_e_timeout(self) -> None:
        subscribers = default_subscribers(1)
        dataset = self.dataset(
            subscribers=subscribers,
            event_count=1,
            batch_size=1,
            handshake_timeout_ms=1,
        )
        port = int(subscribers[0]["port"])
        engine = DiagnosticEngine(
            dataset,
            db_path=self.root / "timeout.sqlite",
            log_path=self.root / "timeout.jsonl",
        )
        try:
            report = engine.run()
        finally:
            engine.close()
        wait_port_free(port)
        self.assertEqual(report.status, "PARTIAL")
        self.assertEqual(report.failed_batches, 1)
        self.assertEqual(report.errors[0]["code"], "E_TIMEOUT")

    def test_real_lease_expiry_fails_batch_and_persists_expired_lease(self) -> None:
        subscribers = default_subscribers(1)
        subscribers[0]["processing_budget_ms"] = 60
        dataset = self.dataset(subscribers=subscribers, event_count=1, batch_size=1)
        db = self.root / "expiry.sqlite"
        engine = DiagnosticEngine(
            dataset,
            db_path=db,
            log_path=self.root / "expiry.jsonl",
            fault_injections={
                subscribers[0]["subscriber_id"]: FaultInjection(processing_delay_ms=250)
            },
        )
        try:
            report = engine.run()
            snapshot = engine.snapshot()
        finally:
            engine.close()
        self.assertEqual(report.status, "PARTIAL")
        self.assertEqual(report.failed_batches, 1)
        self.assertIn(report.errors[0]["code"], {"E_LEASE", "E_TIMEOUT"})
        self.assertEqual(snapshot["batches"][0]["state"], "FAILED")
        self.assertIn(snapshot["batches"][0]["failure_code"], {"E_LEASE", "E_TIMEOUT"})
        self.assertEqual(snapshot["leases"][0]["state"], "EXPIRED")
        expiry_logs = [
            json.loads(line)
            for line in (self.root / "expiry.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertTrue(any(record["code"] == "E_LEASE_EXPIRED" for record in expiry_logs))

    def test_slow_channel_does_not_block_other_channel_capacity(self) -> None:
        subscribers = default_subscribers(1)
        subscribers[0]["lease_limit"] = 4
        channels = [
            {
                "channel_id": "ch-a-orders",
                "topic": "orders",
                "batch_size": 1,
                "send_rate_limit": 1,
                "subscriber_quota": 1,
            },
            {
                "channel_id": "ch-b-alerts",
                "topic": "alerts",
                "batch_size": 1,
                "send_rate_limit": 10,
                "subscriber_quota": 1,
            },
        ]
        spec = write_spec(
            self.root / "fairness-spec.json",
            global_lease_limit=2,
            subscribers=subscribers,
            channels=channels,
        )
        events = write_topic_events(
            self.root / "fairness-events.ndjson",
            ["orders", "alerts", "orders", "alerts"],
        )
        dataset = load_dataset(spec, events)
        engine = DiagnosticEngine(
            dataset,
            db_path=self.root / "fairness.sqlite",
            log_path=self.root / "fairness.jsonl",
        )
        try:
            report = engine.run()
            snapshot = engine.snapshot()
        finally:
            engine.close()
        self.assertEqual(report.confirmed_batches, 4)
        waits_by_channel: dict[str, list[float]] = {}
        for batch in snapshot["batches"]:
            waits_by_channel.setdefault(batch["channel_id"], []).append(
                float(batch["wait_ms"])
            )
        self.assertLess(max(waits_by_channel["ch-b-alerts"]), 600.0)
        self.assertGreater(max(waits_by_channel["ch-a-orders"]), 800.0)

    def test_forced_termination_recovery_reruns_uncommitted_batches(self) -> None:
        subscribers = default_subscribers(1)
        dataset = self.dataset(
            subscribers=subscribers,
            event_count=3,
            batch_size=1,
            global_limit=1,
        )
        db = self.root / "crash.sqlite"
        log = self.root / "crash.jsonl"
        command = [
            sys.executable,
            "-m",
            "eventbus_diag",
            "run",
            "--spec",
            dataset.spec_path,
            "--events",
            dataset.events_path,
            "--db",
            str(db),
            "--log",
            str(log),
            "--fresh",
            "--crash-after",
            "1",
        ]
        crashed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(crashed.returncode, 75, crashed.stderr)
        with sqlite3.connect(db) as conn:
            states = dict(
                conn.execute("SELECT state, COUNT(*) FROM batches GROUP BY state").fetchall()
            )
            crashed_in_flight = conn.execute(
                "SELECT in_flight_count FROM runs WHERE stale = 0"
            ).fetchone()[0]
        self.assertEqual(states.get("CONFIRMED"), 1)
        self.assertLessEqual(crashed_in_flight, 1)

        recovered = subprocess.run(
            [
                sys.executable,
                "-m",
                "eventbus_diag",
                "recover",
                "--spec",
                dataset.spec_path,
                "--events",
                dataset.events_path,
                "--db",
                str(db),
                "--log",
                str(log),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(recovered.returncode, 0, recovered.stdout + recovered.stderr)
        payload = json.loads(recovered.stdout)
        self.assertEqual(payload["report"]["status"], "COMPLETED")
        self.assertEqual(payload["report"]["confirmed_batches"], 3)
        self.assertGreaterEqual(payload["report"]["recovery_unknown_batches"], 2)
        with sqlite3.connect(db) as conn:
            conn.row_factory = sqlite3.Row
            final_states = [
                row["state"]
                for row in conn.execute("SELECT state FROM batches ORDER BY ordinal")
            ]
            active_leases = conn.execute(
                "SELECT COUNT(*) AS n FROM leases WHERE state = 'ACTIVE'"
            ).fetchone()["n"]
            final_in_flight = conn.execute(
                "SELECT in_flight_count FROM runs WHERE stale = 0"
            ).fetchone()[0]
        self.assertEqual(final_states, ["CONFIRMED", "CONFIRMED", "CONFIRMED"])
        self.assertEqual(active_leases, 0)
        self.assertEqual(final_in_flight, 0)

    def test_e_replay_is_reported_by_recover_cli(self) -> None:
        subscribers = default_subscribers(1)
        dataset = self.dataset(
            subscribers=subscribers,
            event_count=2,
            batch_size=1,
            global_limit=1,
        )
        db = self.root / "replay.sqlite"
        log = self.root / "replay.jsonl"
        first = subprocess.run(
            [
                sys.executable,
                "-m",
                "eventbus_diag",
                "run",
                "--spec",
                dataset.spec_path,
                "--events",
                dataset.events_path,
                "--db",
                str(db),
                "--log",
                str(log),
                "--fresh",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        with sqlite3.connect(db) as conn:
            first_batch = conn.execute(
                "SELECT batch_id FROM batches ORDER BY ordinal LIMIT 1"
            ).fetchone()[0]
            conn.execute(
                "UPDATE batches SET ack_digest = 'corrupted' WHERE batch_id = ?",
                (first_batch,),
            )
        replay = subprocess.run(
            [
                sys.executable,
                "-m",
                "eventbus_diag",
                "recover",
                "--spec",
                dataset.spec_path,
                "--events",
                dataset.events_path,
                "--db",
                str(db),
                "--log",
                str(log),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(replay.returncode, 3, replay.stdout + replay.stderr)
        payload = json.loads(replay.stdout)
        self.assertEqual(payload["error"]["code"], "E_REPLAY")
        self.assertEqual(payload["error"]["details"]["batch_id"], first_batch)
        self.assertIn("mismatch_field", payload["error"]["details"])

    def test_declaration_replacement_is_whole_view_with_stable_diff(self) -> None:
        first_subscribers = default_subscribers(2)
        dataset_v1 = self.dataset(
            version=1,
            subscribers=first_subscribers,
            event_count=2,
            batch_size=2,
            global_limit=1,
        )
        engine = DiagnosticEngine(
            dataset_v1,
            db_path=self.root / "replace.sqlite",
            log_path=self.root / "replace.jsonl",
        )
        try:
            engine.run()
            second = default_subscribers(1)[0]
            second["subscriber_id"] = "s0"
            second["port"] = first_subscribers[0]["port"]
            second["processing_budget_ms"] = 250
            third = default_subscribers(1)[0]
            third["subscriber_id"] = "s-new"
            dataset_v2 = self.dataset(
                version=2,
                subscribers=[second, third],
                event_count=2,
                batch_size=3,
                global_limit=2,
            )
            diff = engine.replace_declaration(dataset_v2)
            snapshot = engine.snapshot()
        finally:
            engine.close()
        self.assertEqual(diff.added_subscribers, ("s-new",))
        self.assertEqual(diff.removed_subscribers, ("s1",))
        self.assertEqual([item["subscriber_id"] for item in diff.upgraded_subscribers], ["s0"])
        self.assertEqual(snapshot["run"]["declaration_version"], 2)
        self.assertEqual(snapshot["batches"], [])
        self.assertEqual(snapshot["stats"]["batch_count"], 0)
        with sqlite3.connect(self.root / "replace.sqlite") as conn:
            stale_runs = conn.execute("SELECT COUNT(*) FROM runs WHERE stale = 1").fetchone()[0]
        self.assertEqual(stale_runs, 1)


if __name__ == "__main__":
    unittest.main()
