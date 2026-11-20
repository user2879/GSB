from __future__ import annotations

import json
import socket
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from gateway_quota.engine import QuotaEngine
from gateway_quota.errors import ReplayMismatchError, SpecValidationError, VersionMismatchError
from gateway_quota.loader import load_events, load_parameter_snapshot, load_reconfigurations
from gateway_quota.logging import StructuredLogger
from gateway_quota.loopback import LoopbackSignalBroker, NullSignalBroker, check_loopback_port
from gateway_quota.replay import verify_replay
from gateway_quota.runner import ExperimentRunner, RunnerConfig
from gateway_quota.storage import SQLiteLedger


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


class QuotaComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot = load_parameter_snapshot(SAMPLES / "pools.json", SAMPLES / "borrows.json")
        cls.events = load_events(SAMPLES / "trace.ndjson", cls.snapshot)
        cls.reconfigs = load_reconfigurations([SAMPLES / "reconfig_v2.json"])

    def test_spec_validation_success_and_line_paths(self) -> None:
        self.assertEqual(self.snapshot.version, 1)
        self.assertEqual(len(self.snapshot.pools), 3)
        self.assertEqual(len(self.snapshot.borrow_rules), 1)
        self.assertEqual(len(self.events), 28)
        self.assertEqual(self.reconfigs[0]["snapshot"].version, 2)

        with self.assertRaises(SpecValidationError) as context:
            load_parameter_snapshot(
                SAMPLES / "pools.json",
                SAMPLES / "anomalies" / "borrow_cycle" / "borrows.json",
            )
        records = context.exception.records
        self.assertTrue(any(record.business_code == "E_BORROW" for record in records))
        self.assertTrue(all(record.code == "E_SPEC" for record in records))
        self.assertTrue(any(record.line is not None for record in records))

    def test_spec_validation_rejects_duplicate_pool_and_backwards_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            pools = json.loads((SAMPLES / "pools.json").read_text(encoding="utf-8"))
            pools["pools"].append(dict(pools["pools"][0]))
            duplicate_path = temp / "duplicate-pools.json"
            duplicate_path.write_text(json.dumps(pools), encoding="utf-8")
            with self.assertRaises(SpecValidationError) as context:
                load_parameter_snapshot(duplicate_path, SAMPLES / "borrows.json")
            self.assertTrue(any("unique" in record.message for record in context.exception.records))
            self.assertTrue(any(record.line is not None for record in context.exception.records))

            lines = (SAMPLES / "trace.ndjson").read_text(encoding="utf-8").splitlines()
            second = json.loads(lines[1])
            second["timestamp_ms"] = 999
            lines[1] = json.dumps(second, separators=(",", ":"))
            backwards_path = temp / "backwards.ndjson"
            backwards_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(SpecValidationError) as context:
                load_events(backwards_path, self.snapshot)
            self.assertTrue(
                any(record.path.endswith("timestamp_ms") for record in context.exception.records)
            )

    def _run_engine_to_transition(self) -> tuple[QuotaEngine, dict[int, object]]:
        engine = QuotaEngine(self.snapshot, NullSignalBroker())
        results = {}
        for event in self.events:
            results[event.seq] = engine.process_event(event)
            if event.seq == 22:
                engine.apply_parameters(self.reconfigs[0]["snapshot"], after_event_seq=22)
        return engine, results

    def test_lifecycle_borrow_ledger_and_overdue(self) -> None:
        engine, results = self._run_engine_to_transition()
        seq3 = results[3].audit["pools"]["alpha"]
        self.assertEqual(seq3["capacity_in_use"], 3)
        self.assertEqual(seq3["remaining_connections"], 0)
        self.assertTrue(seq3["warning"])

        seq7_codes = [issue["code"] for issue in results[7].issues]
        self.assertEqual(seq7_codes, ["E_POOLFULL"])
        self.assertEqual([issue["code"] for issue in results[6].issues], ["E_CONNLIMIT"])

        seq13 = results[13].audit
        self.assertEqual(seq13["pools"]["alpha"]["lent_out"], 1)
        self.assertEqual(seq13["pools"]["alpha"]["capacity_in_use"], 3)
        self.assertEqual(seq13["pools"]["beta"]["borrowed_in"], 1)
        self.assertEqual(seq13["pools"]["beta"]["active_connections"], 3)
        self.assertEqual(seq13["ownership"]["b3"]["origin_pool"], "beta")
        self.assertEqual(seq13["ownership"]["b3"]["current_owner_pool"], "alpha")
        self.assertEqual([issue["code"] for issue in results[14].issues], ["E_OVERDUE"])

        described = engine.describe_connection("b3")
        self.assertEqual(described["origin_pool"], "beta")
        self.assertEqual(described["current_owner_pool"], "alpha")
        self.assertEqual(described["loans"][0]["status"], "returned_late")
        self.assertTrue(all(loan["stale"] for loan in engine.loan_rows()))

        final_pools = {row["pool_id"]: row for row in engine.pool_rows()}
        self.assertTrue(all(row["active_connections"] == 0 for row in final_pools.values()))

    def test_fixed_point_waterline_rate_and_eta(self) -> None:
        engine = QuotaEngine(self.snapshot, NullSignalBroker())
        results = {}
        for event in self.events[:15]:
            results[event.seq] = engine.process_event(event)
        seq8_alpha = next(w for w in results[8].warnings if w.pool_id == "alpha")
        self.assertEqual(seq8_alpha.remaining_connections, 1)
        self.assertFalse(seq8_alpha.arrival_rate["estimable"])
        self.assertEqual(seq8_alpha.arrival_rate["reason"], "WINDOW_WARMUP_INCOMPLETE")
        self.assertFalse(seq8_alpha.exhaustion_eta["estimable"])
        self.assertIsNone(seq8_alpha.exhaustion_eta["eta_ms"])

        seq15_alpha = next(w for w in results[15].warnings if w.pool_id == "alpha")
        self.assertEqual(seq15_alpha.remaining_connections, 1)
        self.assertEqual(seq15_alpha.capacity_in_use, 2)
        self.assertEqual(seq15_alpha.waterline_percentage_fixed, 666)
        self.assertEqual(seq15_alpha.rounding, "floor_toward_negative_infinity")
        self.assertTrue(seq15_alpha.arrival_rate["estimable"])
        self.assertEqual(seq15_alpha.arrival_rate["arrivals_in_window"], 1)
        self.assertEqual(seq15_alpha.exhaustion_eta["eta_ms"], 1000)
        self.assertEqual(seq15_alpha.exhaustion_eta["estimation_window_ms"], 1000)
        self.assertEqual(seq15_alpha.exhaustion_eta["error_bound_ms"], 1000)

    def test_lifecycle_and_buffer_rejections(self) -> None:
        engine = QuotaEngine(self.snapshot, NullSignalBroker())
        for event in self.events[:15]:
            engine.process_event(event)
        closed = self.events[14]
        late_request = type(closed)(
            seq=29,
            pool="beta",
            connection="b3",
            event_type="request",
            timestamp_ms=2300,
            bytes_count=32,
            line=29,
            raw={},
        )
        result = engine.process_event(late_request)
        self.assertEqual([issue["code"] for issue in result.issues], ["E_LIFECYCLE"])

        engine = QuotaEngine(self.snapshot, NullSignalBroker())
        for event in self.events[:11]:
            engine.process_event(event)
        early_idle = type(closed)(
            seq=29,
            pool="beta",
            connection="b1",
            event_type="idle_release",
            timestamp_ms=1700,
            bytes_count=0,
            line=29,
            raw={},
        )
        result = engine.process_event(early_idle)
        self.assertEqual([issue["code"] for issue in result.issues], ["E_LIFECYCLE"])

        engine = QuotaEngine(self.snapshot, NullSignalBroker())
        for event in self.events[:20]:
            engine.process_event(event)
        too_large = type(closed)(
            seq=29,
            pool="delta",
            connection="d1",
            event_type="request",
            timestamp_ms=3400,
            bytes_count=65,
            line=29,
            raw={},
        )
        result = engine.process_event(too_large)
        self.assertEqual([issue["code"] for issue in result.issues], ["E_BUFFER"])

    def test_parameter_transition_replaces_view_and_stales_old_diagnostics(self) -> None:
        engine = QuotaEngine(self.snapshot, NullSignalBroker())
        for event in self.events[:22]:
            engine.process_event(event)
        self.assertGreater(len(engine.current_warnings()), 0)
        transition = engine.apply_parameters(self.reconfigs[0]["snapshot"], after_event_seq=22)
        self.assertEqual(transition["previous_version"], 1)
        self.assertEqual(transition["version"], 2)
        self.assertEqual(transition["affected_pools"], ["alpha", "beta", "delta"])
        self.assertEqual(
            sorted(item["pool_id"] for item in transition["cleared_alert_records"]),
            ["alpha", "beta"],
        )
        self.assertTrue(transition["summary"]["replaced_view"])
        self.assertEqual(transition["invalidated_borrow_relations"], [])
        self.assertEqual(transition["added_borrow_relations"], [])
        self.assertEqual(engine.current_warnings(), [])
        self.assertTrue(all(warning.stale for warning in engine.warnings))
        self.assertTrue(all(loan["stale"] for loan in engine.loan_rows()))

    def test_parameter_transition_lists_invalidated_borrow_relation(self) -> None:
        engine = QuotaEngine(self.snapshot, NullSignalBroker())
        for event in self.events[:13]:
            engine.process_event(event)
        changed = replace(self.snapshot, version=3, borrow_rules=())
        transition = engine.apply_parameters(changed, after_event_seq=13)
        self.assertEqual(len(transition["invalidated_borrows"]), 1)
        self.assertEqual(len(transition["invalidated_borrow_relations"]), 1)
        self.assertEqual(
            transition["invalidated_borrow_relations"][0]["borrower_pool"], "beta"
        )
        self.assertIn("b3", transition["unknown_connections"])

    def test_full_run_ledger_recovery_and_final_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "ledger.sqlite3"
            outcome = ExperimentRunner(
                RunnerConfig(ledger, self.snapshot, self.events, self.reconfigs, batch_size=5),
                signal_broker=NullSignalBroker(),
                logger=StructuredLogger(Path(directory) / "run.ndjson"),
            ).run()
            self.assertEqual(outcome["last_event_seq"], 28)
            self.assertEqual(outcome["committed_batches"], 7)
            self.assertEqual(outcome["final_replay"]["events_checked"], 28)
            self.assertTrue(outcome["final_replay"]["consistent"])

            resumed = ExperimentRunner(
                RunnerConfig(ledger, self.snapshot, self.events, self.reconfigs, batch_size=5),
                signal_broker=NullSignalBroker(),
                logger=StructuredLogger(Path(directory) / "resume.ndjson"),
            ).run()
            self.assertEqual(resumed["committed_events"], 0)
            self.assertEqual(resumed["recovery"]["recovered_through_seq"], 28)
            self.assertTrue(resumed["replay"]["consistent"])
            self.assertEqual(resumed["replay"]["parameter_versions_checked"], [1, 2])

            with SQLiteLedger(ledger) as store:
                stats = store.counts()
                self.assertEqual(stats["batches"], 7)
                self.assertEqual(stats["batch_events"], 28)
                self.assertEqual(stats["pending_batches"], 0)
                self.assertEqual(stats["parameter_transitions"], 1)
                self.assertTrue(verify_replay(store)["consistent"])

    def test_forced_termination_marks_incomplete_batch_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "crash.sqlite3"
            crash_log = Path(directory) / "crash.ndjson"
            command = [
                sys.executable,
                "-m",
                "gateway_quota.cli",
                "run",
                "--pools",
                str(SAMPLES / "pools.json"),
                "--borrows",
                str(SAMPLES / "borrows.json"),
                "--trace",
                str(SAMPLES / "trace.ndjson"),
                "--reconfig",
                str(SAMPLES / "reconfig_v2.json"),
                "--ledger",
                str(ledger),
                "--batch-size",
                "5",
                "--crash-after-seq",
                "13",
                "--no-loopback",
                "--log",
                str(crash_log),
            ]
            child = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertEqual(child.returncode, 137)
            log_records = [json.loads(line) for line in crash_log.read_text(encoding="utf-8").splitlines()]
            self.assertIn("E_SIMULATED_KILL", {record["code"] for record in log_records})
            self.assertNotIn("RUN_COMPLETE", {record["code"] for record in log_records})

            resumed = ExperimentRunner(
                RunnerConfig(ledger, self.snapshot, self.events, self.reconfigs, batch_size=5),
                signal_broker=NullSignalBroker(),
                logger=StructuredLogger(Path(directory) / "resume.ndjson"),
            ).run()
            self.assertEqual(resumed["recovery"]["recovered_through_seq"], 10)
            self.assertEqual(resumed["recovery"]["unknown_connections"], ["b1", "b2", "b3"])
            self.assertEqual(resumed["last_event_seq"], 28)
            self.assertTrue(resumed["final_replay"]["consistent"])

    def test_replay_reports_first_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "ledger.sqlite3"
            ExperimentRunner(
                RunnerConfig(ledger, self.snapshot, self.events, self.reconfigs, batch_size=5),
                signal_broker=NullSignalBroker(),
                logger=StructuredLogger(Path(directory) / "run.ndjson"),
            ).run()
            with sqlite3.connect(str(ledger)) as connection:
                row = connection.execute(
                    "SELECT audit_json FROM batch_events WHERE seq=15"
                ).fetchone()
                result = json.loads(row[0])
                result["audit"]["pools"]["alpha"]["remaining_connections"] = 999
                connection.execute(
                    "UPDATE batch_events SET audit_json=? WHERE seq=15",
                    (json.dumps(result, sort_keys=True, separators=(",", ":")),),
                )
                connection.commit()
            with SQLiteLedger(ledger) as store:
                with self.assertRaises(ReplayMismatchError) as context:
                    verify_replay(store)
            details = context.exception.records[0].details
            self.assertEqual(details["event_seq"], 15)
            self.assertEqual(details["pool_id"], "alpha")
            self.assertEqual(details["field"], "remaining_connections")
            self.assertEqual(context.exception.code, "E_REPLAY")

    def test_ledger_parameter_version_mismatch_is_e_version(self) -> None:
        anomaly = load_parameter_snapshot(
            SAMPLES / "anomalies" / "version_mismatch" / "pools.json",
            SAMPLES / "anomalies" / "version_mismatch" / "borrows.json",
        )
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "ledger.sqlite3"
            with SQLiteLedger(ledger) as store:
                store.initialize(self.snapshot)
            with SQLiteLedger(ledger) as store:
                with self.assertRaises(VersionMismatchError) as context:
                    store.initialize(anomaly)
            self.assertEqual(context.exception.code, "E_VERSION")
            self.assertEqual(context.exception.records[0].path, "pools.version")

    def test_loopback_handshake_and_port_check(self) -> None:
        port = check_loopback_port(0)
        self.assertGreater(port, 0)
        with LoopbackSignalBroker(0) as broker:
            self.assertGreater(broker.port, 0)
            established = broker.establish("loop-1")
            self.assertEqual(established["host"], "127.0.0.1")
            self.assertEqual(established["signal"], "ESTABLISHED")
            closed = broker.close("loop-1")
            self.assertEqual(closed["signal"], "CLOSED")

        occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        try:
            with self.assertRaises(OSError):
                check_loopback_port(occupied.getsockname()[1])
        finally:
            occupied.close()


if __name__ == "__main__":
    unittest.main()
