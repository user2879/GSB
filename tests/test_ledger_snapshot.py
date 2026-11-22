from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from meshwork.engine import MeshRuntime
from meshwork.errors import MeshError
from meshwork.ledger import (
    AttemptCommit,
    AttemptIntent,
    FinalOutcome,
    Ledger,
)
from meshwork.logging_utils import StructuredLogger
from meshwork.model import RequestSpec
from meshwork.model import parse_spec
from meshwork.snapshot import SnapshotState, SnapshotStore, diff_snapshot_states
from tests.helpers import behavior, free_ports, instance, make_spec


def make_commit(request_id: str = "r1") -> tuple[AttemptCommit, FinalOutcome]:
    commit = AttemptCommit(
        request_id=request_id,
        request_seq=1,
        run_epoch=1,
        attempt_no=1,
        select_seq=1_000_001,
        service="svc",
        instance_id="inst-a",
        route_revision=1,
        instance_version="inst-v1",
        outcome="success",
        status="ok",
        error_code=None,
        transport_ok=True,
        started_wall_ms=1000,
        started_mono_ms=100,
        finished_mono_ms=110,
        wait_start_mono_ms=95,
        wait_end_mono_ms=100,
        budget_wait_ms=5,
        duration_ms=10,
        upstream_delay_ms=4,
        budget_used=True,
        global_inflight=1,
        service_inflight=1,
        final_attempt=True,
        detail="ok",
        breaker_after={"instance_id": "inst-a", "state": "CLOSED", "consecutive_failures": 0},
    )
    final = FinalOutcome(
        final_outcome="success",
        error_code=None,
        attempt_count=1,
        total_duration_ms=15,
        completed_at_wall_ms=1015,
    )
    return commit, final


class LedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "ledger.db"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def prepare_ledger(self) -> Ledger:
        ledger = Ledger(self.path, reset=True)
        ledger.set_meta({"global_concurrency": 2, "quota:svc": 1, "format_version": 1})
        ledger.register_request(RequestSpec("r1", "svc", {"x": 1}, 1), 1)
        return ledger

    def test_attempt_outcome_and_budget_commit_is_atomic_and_digest_changes(self) -> None:
        ledger = self.prepare_ledger()
        before = ledger.content_digest()
        ledger.begin_intent(
            AttemptIntent("r1", 1, 1, "svc", "inst-a", 1_000_001, 1000, 100)
        )
        commit, final = make_commit()
        ledger.commit_attempt(commit, final)
        after = ledger.content_digest()
        self.assertNotEqual(before, after)
        self.assertEqual(ledger.count_attempts(), 1)
        self.assertEqual(ledger.latest_outcomes()[0]["final_outcome"], "success")
        with sqlite3.connect(self.path) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM budget_usage").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM attempt_intents").fetchone()[0], 0)
        ledger.close()
        reopened = Ledger(self.path)
        reopened.verify_digest()
        reopened.close()

    def test_ledger_digest_tamper_is_rejected(self) -> None:
        ledger = self.prepare_ledger()
        commit, final = make_commit()
        ledger.commit_attempt(commit, final)
        ledger.close()
        with sqlite3.connect(self.path) as conn:
            conn.execute("UPDATE requests SET service='tampered' WHERE request_id='r1'")
            conn.commit()
        with self.assertRaises(MeshError) as ctx:
            Ledger(self.path)
        self.assertEqual(ctx.exception.code, "E_SNAPSHOT")
        self.assertEqual(ctx.exception.field, "ledger.content_digest")

    def test_recovery_marks_uncommitted_intent_unknown_and_never_claims_exit_report(self) -> None:
        ledger = self.prepare_ledger()
        ledger.begin_intent(
            AttemptIntent("r1", 1, 1, "svc", "inst-a", 1_000_001, 1000, 100)
        )
        ledger.close()
        reopened = Ledger(self.path)
        recovery = reopened.recover()
        self.assertFalse(recovery.exit_report_available)
        self.assertEqual(len(recovery.unknown_attempts), 1)
        self.assertEqual(len(recovery.requests_to_rerun), 1)
        self.assertEqual(recovery.budget_used_global, 0)
        self.assertEqual(reopened.unknown_attempts()[0]["state"], "unknown")
        reopened.close()

    def test_snapshot_digest_mismatch_and_stable_diff(self) -> None:
        store = SnapshotStore(Path(self.temp.name) / "snapshots")
        state1 = SnapshotState(
            route_revision=1,
            instance_version="v1",
            budget_used_global=2,
            budget_used_by_service={"a": 2},
            breakers={"a": {"state": "CLOSED"}},
            completed_cursor=5,
            ledger_format_version=1,
            ledger_digest="d1",
        )
        envelope = store.write(state1)
        self.assertEqual(store.load(envelope.path).digest, envelope.digest)
        raw = json.loads(Path(envelope.path).read_text(encoding="utf-8"))
        raw["payload"]["completed_cursor"] += 1
        Path(envelope.path).write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(MeshError) as ctx:
            store.load(envelope.path)
        self.assertEqual(ctx.exception.code, "E_SNAPSHOT")
        state2 = SnapshotState(
            route_revision=1,
            instance_version="v2",
            budget_used_global=3,
            budget_used_by_service={"a": 1, "b": 2},
            breakers={"a": {"state": "OPEN"}, "b": {"state": "CLOSED"}},
            completed_cursor=6,
            ledger_format_version=1,
            ledger_digest="d2",
        )
        diff = diff_snapshot_states(state1, state2)
        self.assertEqual(diff["added"], ["breakers.b", "budget_used_by_service.b"])
        self.assertEqual(diff["removed"], [])
        self.assertIn("budget_used_by_service.a", diff["changed"])
        self.assertIn("breakers.a", diff["changed"])
        self.assertEqual(diff["scalar_changed"], sorted([
            "budget_used_global",
            "completed_cursor",
            "instance_version",
            "ledger_digest",
        ]))

    def test_runtime_replaces_invalid_snapshot_as_a_whole(self) -> None:
        ports = free_ports(1)
        spec = parse_spec(
            make_spec(
                [
                    {
                        "id": "svc",
                        "quota": 1,
                        "instances": [
                            instance("a", port=ports[0], behavior_value=behavior())
                        ],
                    }
                ]
            )
        )
        ledger = Ledger(Path(self.temp.name) / "runtime.db", reset=True)
        store = SnapshotStore(Path(self.temp.name) / "runtime-snapshots")
        runtime = MeshRuntime(
            spec,
            spec.active_revision(),
            ledger,
            store,
            StructuredLogger(Path(self.temp.name) / "runtime.jsonl"),
            "snapshot-repair",
        )
        envelope = runtime.write_snapshot("before-corruption")
        raw = json.loads(Path(envelope.path).read_text(encoding="utf-8"))
        raw["payload"]["route_revision"] = 999
        Path(envelope.path).write_text(json.dumps(raw), encoding="utf-8")
        result = runtime.validate_or_replace_snapshot()
        self.assertEqual(result["status"], "replaced")
        self.assertFalse(result["restored"])
        self.assertEqual(result["error"]["code"], "E_SNAPSHOT")
        latest = store.load_latest()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.state.route_revision, 1)  # type: ignore[union-attr]
        self.assertNotEqual(latest.path, envelope.path)  # type: ignore[union-attr]
        second_path = Path(latest.path)  # type: ignore[union-attr]
        second_raw = json.loads(second_path.read_text(encoding="utf-8"))
        second_raw["payload"]["completed_cursor"] = int(
            second_raw["payload"]["completed_cursor"]
        ) + 1
        second_path.write_text(json.dumps(second_raw), encoding="utf-8")
        second_result = runtime.validate_or_replace_snapshot()
        self.assertEqual(second_result["status"], "replaced")
        self.assertIn("completed_cursor", second_result["diff"]["scalar_changed"])
        ledger.close()


if __name__ == "__main__":
    unittest.main()
