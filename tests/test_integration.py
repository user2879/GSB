from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from cluster_lease.errors import LeaseError
from cluster_lease.ledger import Ledger
from cluster_lease.loader import load_inputs
from cluster_lease.logging_utils import StructuredLogger
from cluster_lease.runtime import evolve_session, recover_session, run_session
from cluster_lease.snapshots import _content_digest, build_snapshot, load_snapshot, open_snapshot_session

from .helpers import write_bundle


class EndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        paths = write_bundle(self.root / "input")
        self.bundle = load_inputs(paths["clusters"], paths["holders"], paths["events"])
        self.paths = paths
        self.logger = StructuredLogger(str(self.root / "events.jsonl"))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_determinism_evolution_snapshot_and_replay(self) -> None:
        first_db = self.root / "first.db"
        second_db = self.root / "second.db"
        first = run_session(
            self.bundle,
            str(first_db),
            runtime_dir=str(self.root / "runtime1"),
            logger=self.logger,
            batch_size=4,
            clock_mode="logical",
        )
        second = run_session(
            self.bundle,
            str(second_db),
            runtime_dir=str(self.root / "runtime2"),
            logger=self.logger,
            batch_size=3,
            clock_mode="logical",
        )
        self.assertEqual(first["reply_sequence_digest"], second["reply_sequence_digest"])
        self.assertEqual(
            first["ledger"]["ledger_content_digest"],
            second["ledger"]["ledger_content_digest"],
        )
        self.assertEqual(14, first["processed_event_count"])
        ledger_check = Ledger(str(first_db))
        try:
            alpha_intervals = ledger_check.current_ownership_intervals("alpha")
            self.assertGreaterEqual(len(alpha_intervals), 4)
            self.assertEqual(
                len(ledger_check.current_active("alpha")),
                sum(1 for interval in alpha_intervals if interval["ended_ms"] is None),
            )
            self.assertTrue(
                all(
                    interval["end_event_seq"] is not None
                    for interval in alpha_intervals
                    if interval["ended_ms"] is not None
                )
            )
        finally:
            ledger_check.close()

        evolved_paths = write_bundle(
            self.root / "evolved",
            alpha_capacity=2,
            beta_capacity=1,
            alpha_duration=7000,
            priorities={"h-high": 8, "h-mid": 10, "h-low": 1},
        )
        evolved = load_inputs(
            evolved_paths["clusters"], evolved_paths["holders"], evolved_paths["events"]
        )
        evolution = evolve_session(
            evolved,
            str(first_db),
            runtime_dir=str(self.root / "runtime-evolve"),
            logger=self.logger,
        )
        self.assertEqual("overall_replace", evolution["replacement"])
        self.assertEqual([2, 2], [item["to_version"] for item in evolution["affected_clusters"]])
        ledger = Ledger(str(first_db))
        try:
            self.assertEqual({"alpha": 2, "beta": 2}, ledger.versions())
            self.assertTrue(ledger.stale_records(["alpha", "beta"])["decisions"])
            snapshot_path = self.root / "snapshot.json"
            snapshot = build_snapshot(ledger, evolved, str(snapshot_path))
        finally:
            ledger.close()
        opened = open_snapshot_session(
            evolved,
            str(first_db),
            str(snapshot_path),
            runtime_dir=str(self.root / "runtime-snapshot"),
            logger=self.logger,
        )
        self.assertEqual("OK", opened["code"])
        self.assertEqual("PASS", opened["replay"]["replay"])

        tampered = copy.deepcopy(snapshot)
        decision = next(
            item
            for item in tampered["clusters"]
            if item["cluster_id"] == "alpha"
        )["decisions"][0]
        decision["decision"] = "rejected"
        decision["payload"]["conclusion"] = "rejected"
        tampered["content_digest"] = _content_digest(tampered)
        tampered_path = self.root / "tampered.json"
        tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
        with self.assertRaises(LeaseError) as replay_error:
            open_snapshot_session(
                evolved,
                str(first_db),
                str(tampered_path),
                runtime_dir=str(self.root / "runtime-replay"),
                logger=self.logger,
            )
        self.assertEqual("E_REPLAY", replay_error.exception.code)
        self.assertEqual("alpha", replay_error.exception.details["cluster_id"])
        self.assertEqual(1, replay_error.exception.details["event_seq"])
        self.assertEqual("decision", replay_error.exception.field)

        with self.assertRaises(LeaseError) as digest_error:
            load_snapshot(str(snapshot_path), injection="snapshot")
        self.assertEqual("E_SNAPSHOT", digest_error.exception.code)

    def test_forced_termination_recovery(self) -> None:
        db_path = self.root / "crash.db"
        command = [
            sys.executable,
            "-m",
            "cluster_lease",
            "run",
            "--clusters",
            self.paths["clusters"],
            "--holders",
            self.paths["holders"],
            "--events",
            self.paths["events"],
            "--db",
            str(db_path),
            "--runtime-dir",
            str(self.root / "crash-runtime"),
            "--batch-size",
            "4",
            "--clock-mode",
            "logical",
            "--crash-after-inflight-batch",
            "2",
            "--log",
            str(self.root / "crash.jsonl"),
        ]
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parent.parent)
        self.assertEqual(70, completed.returncode)
        time.sleep(0.7)
        recovered = recover_session(
            self.bundle,
            str(db_path),
            runtime_dir=str(self.root / "recover-runtime"),
            logger=self.logger,
            clock_mode="logical",
        )
        self.assertFalse(recovered["exit_report_available"])
        self.assertEqual(1, len(recovered["unknown_batches"]))
        self.assertEqual("unknown", recovered["unknown_batches"][0]["status"])
        self.assertEqual(4, recovered["state"]["stream_cursor"])
        resumed = run_session(
            self.bundle,
            str(db_path),
            runtime_dir=str(self.root / "resume-runtime"),
            logger=self.logger,
            batch_size=4,
            clock_mode="logical",
        )
        self.assertEqual(list(range(5, 15)), resumed["processed_event_sequences"])
        baseline_db = self.root / "baseline.db"
        baseline = run_session(
            self.bundle,
            str(baseline_db),
            runtime_dir=str(self.root / "baseline-runtime"),
            logger=self.logger,
            batch_size=4,
            clock_mode="logical",
        )
        self.assertEqual(
            baseline["ledger"]["ledger_content_digest"],
            resumed["ledger"]["ledger_content_digest"],
        )


if __name__ == "__main__":
    unittest.main()
