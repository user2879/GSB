from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from meshwork.errors import MeshError
from meshwork.ledger import AttemptCommit, FinalOutcome, Ledger
from meshwork.model import RequestSpec
from meshwork.replay import compare_ledgers


def commit(ledger: Ledger, request_id: str, seq: int, outcome: str = "success") -> None:
    ledger.register_request(RequestSpec(request_id, "svc", {"n": seq}, seq), seq)
    ledger.commit_attempt(
        AttemptCommit(
            request_id=request_id,
            request_seq=seq,
            run_epoch=1,
            attempt_no=1,
            select_seq=seq * 1_000_000 + 1,
            service="svc",
            instance_id="a",
            route_revision=1,
            instance_version="v1",
            outcome=outcome,
            status="ok" if outcome == "success" else "error",
            error_code=None if outcome == "success" else "E_UPSTREAM",
            transport_ok=True,
            started_wall_ms=1000 + seq,
            started_mono_ms=100 + seq,
            finished_mono_ms=110 + seq,
            wait_start_mono_ms=95 + seq,
            wait_end_mono_ms=100 + seq,
            budget_wait_ms=5,
            duration_ms=10,
            upstream_delay_ms=4,
            budget_used=True,
            global_inflight=1,
            service_inflight=1,
            final_attempt=True,
            detail="test",
            breaker_after={"state": "CLOSED"},
        ),
        FinalOutcome(
            final_outcome="success" if outcome == "success" else "exhausted",
            error_code=None if outcome == "success" else "E_EXHAUST",
            attempt_count=1,
            total_duration_ms=15,
            completed_at_wall_ms=1015 + seq,
        ),
    )


class ReplayTests(unittest.TestCase):
    def test_first_mismatch_reports_request_service_and_field(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected = Ledger(root / "expected.db", reset=True)
            actual = Ledger(root / "actual.db", reset=True)
            for index in (1, 2, 3):
                commit(expected, f"r{index}", index)
                commit(actual, f"r{index}", index)
            actual.close()
            with sqlite3.connect(root / "actual.db") as conn:
                conn.execute(
                    "UPDATE attempts SET instance_id='wrong' WHERE request_id='r2'"
                )
                conn.commit()
            tampered = Ledger(root / "actual.db", verify=False)
            with self.assertRaises(MeshError) as ctx:
                compare_ledgers(expected, tampered)
            error = ctx.exception
            self.assertEqual(error.code, "E_REPLAY")
            self.assertEqual(error.request_id, "r2")
            self.assertEqual(error.service, "svc")
            self.assertEqual(error.field, "attempts[0].instance_id")
            self.assertEqual(error.details["expected"], "a")
            self.assertEqual(error.details["actual"], "wrong")
            expected.close()
            tampered.close()


if __name__ == "__main__":
    unittest.main()
