from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from eventbus_diag.batching import build_plan
from eventbus_diag.errors import ReplayError, SpecError
from eventbus_diag.ledger import CONFIRMED, DISPATCHING, UNKNOWN, Ledger
from eventbus_diag.models import Ack, Acquisition, Lease, QuotaSnapshot, WaitRecord
from eventbus_diag.spec import load_dataset
from eventbus_diag.worker import compute_ack

from .helpers import default_channels, default_subscribers, write_events, write_spec


def acquisition_for(batch, subscriber, *, lease_id: str = "lease-test") -> Acquisition:
    return Acquisition(
        lease=Lease(
            lease_id=lease_id,
            batch_id=batch.batch_id,
            subscriber_id=batch.subscriber_id,
            channel_id=batch.channel_id,
            granted_mono_ms=10.0,
            expires_mono_ms=5000.0,
            version=1,
        ),
        wait=WaitRecord(start_mono_ms=0.0, end_mono_ms=10.0, wait_ms=10.0, reason="none"),
        quota=QuotaSnapshot(
            global_limit=3,
            subscriber_limit=subscriber.lease_limit,
            channel_rate_limit=50,
            global_active_after_grant=1,
            subscriber_active_after_grant=1,
            channel_sends_in_window=1,
        ),
    )


def ack_for(batch, subscriber) -> Ack:
    raw = compute_ack(
        subscriber_id=subscriber.subscriber_id,
        batch_id=batch.batch_id,
        events=batch.content["events"],
        seed=subscriber.seed,
        budget_ms=subscriber.processing_budget_ms,
    )
    return Ack(
        subscriber_id=raw["subscriber_id"],
        batch_id=raw["batch_id"],
        event_ids=tuple(raw["event_ids"]),
        processing_ms=raw["processing_ms"],
        ack_digest=raw["ack_digest"],
        processed_order=tuple(raw["processed_order"]),
        raw=raw,
    )


class LedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        spec = write_spec(
            self.root / "spec.json",
            subscribers=default_subscribers(1),
            channels=default_channels(batch_size=2),
        )
        events = write_events(self.root / "events.ndjson", 4)
        self.dataset = load_dataset(spec, events)
        self.plan = build_plan(self.dataset.declaration, self.dataset.events)
        self.db = self.root / "ledger.sqlite"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def prepare(self, ledger: Ledger, *, replacement=None):
        return ledger.prepare_run(
            self.dataset.declaration,
            spec_digest=self.dataset.declaration.spec_digest(),
            events_digest=self.dataset.events_digest,
            content_digest=self.dataset.content_digest,
            replacement=replacement,
        )

    def complete_first(self, ledger: Ledger, prepared) -> None:
        batch = self.plan.batches[0]
        subscriber = self.dataset.declaration.subscriber_map()[batch.subscriber_id]
        acquisition = acquisition_for(batch, subscriber)
        ledger.begin_batch(prepared.run_id, batch, acquisition)
        ledger.complete_batch(
            prepared.run_id,
            batch,
            state=CONFIRMED,
            lease=acquisition.lease,
            acquisition=acquisition,
            ack=ack_for(batch, subscriber),
        )

    def test_completed_batch_is_replay_verifiable_and_digest_stable(self) -> None:
        ledger = Ledger(self.db)
        prepared = self.prepare(ledger)
        self.complete_first(ledger, prepared)
        first_digest = ledger.snapshot()["ledger_info"]["content_digest"]
        result = ledger.verify_committed(prepared.run_id, self.plan.batches)
        self.assertEqual(result.verified_batches, (self.plan.batches[0].batch_id,))
        self.assertEqual(
            ledger.snapshot()["ledger_info"]["content_digest"], first_digest
        )
        ledger.close()

    def test_inflight_batch_becomes_unknown_then_reruns(self) -> None:
        ledger = Ledger(self.db)
        prepared = self.prepare(ledger)
        batch = self.plan.batches[0]
        subscriber = self.dataset.declaration.subscriber_map()[batch.subscriber_id]
        acquisition = acquisition_for(batch, subscriber, lease_id="lease-crashed")
        ledger.begin_batch(prepared.run_id, batch, acquisition)
        ledger.close()

        reopened = Ledger(self.db)
        resumed = self.prepare(reopened)
        self.assertTrue(resumed.resumed)
        verification = reopened.verify_and_mark_unknown(
            resumed.run_id, self.plan.batches
        )
        self.assertIn(batch.batch_id, verification.unknown_batches)
        snapshot = reopened.snapshot(resumed.run_id)
        self.assertEqual(snapshot["batches"][0]["state"], UNKNOWN)
        first_batch_transitions = [
            row for row in snapshot["transitions"] if row["batch_id"] == batch.batch_id
        ]
        self.assertEqual(
            [
                (row["from_state"], row["to_state"])
                for row in first_batch_transitions
            ],
            [("NONE", DISPATCHING), (DISPATCHING, UNKNOWN)],
        )
        lease_row = snapshot["leases"][0]
        self.assertEqual(lease_row["state"], "REVOKED")

        retry_acquisition = acquisition_for(batch, subscriber, lease_id="lease-retry")
        reopened.begin_batch(resumed.run_id, batch, retry_acquisition)
        reopened.complete_batch(
            resumed.run_id,
            batch,
            state=CONFIRMED,
            lease=retry_acquisition.lease,
            acquisition=retry_acquisition,
            ack=ack_for(batch, subscriber),
        )
        final = reopened.verify_committed(resumed.run_id, self.plan.batches)
        self.assertEqual(len(final.verified_batches), 1)
        self.assertEqual(reopened.finalize_run(resumed.run_id), "PARTIAL")
        reopened.close()

    def test_e_replay_identifies_first_corrupt_batch(self) -> None:
        ledger = Ledger(self.db)
        prepared = self.prepare(ledger)
        self.complete_first(ledger, prepared)
        with sqlite3.connect(self.db) as conn:
            conn.execute(
                "UPDATE batches SET event_ids_json = ? WHERE run_id = ? AND batch_id = ?",
                (json.dumps([999]), prepared.run_id, self.plan.batches[0].batch_id),
            )
        with self.assertRaises(ReplayError) as raised:
            ledger.verify_committed(prepared.run_id, self.plan.batches)
        error = raised.exception
        self.assertEqual(error.code, "E_REPLAY")
        self.assertEqual(error.details["batch_id"], self.plan.batches[0].batch_id)
        self.assertEqual(error.details["mismatch_field"], "event_ids")
        ledger.close()

    def test_new_declaration_version_marks_old_view_stale_and_revokes_lease(self) -> None:
        ledger = Ledger(self.db)
        prepared_v1 = self.prepare(ledger)
        batch = self.plan.batches[0]
        subscriber = self.dataset.declaration.subscriber_map()[batch.subscriber_id]
        acquisition = acquisition_for(batch, subscriber, lease_id="lease-old")
        ledger.begin_batch(prepared_v1.run_id, batch, acquisition)

        subscribers = [dict(item) for item in default_subscribers(1)]
        subscribers[0]["processing_budget_ms"] = 222
        spec2 = write_spec(
            self.root / "spec_v2.json",
            version=2,
            subscribers=subscribers,
            channels=default_channels(batch_size=2),
        )
        events2 = write_events(self.root / "events2.ndjson", 4)
        dataset2 = load_dataset(spec2, events2)
        prepared_v2 = ledger.prepare_run(
            dataset2.declaration,
            spec_digest=dataset2.declaration.spec_digest(),
            events_digest=dataset2.events_digest,
            content_digest=dataset2.content_digest,
            replacement={"old_version": 1, "new_version": 2},
        )
        self.assertFalse(prepared_v2.resumed)
        snapshot = ledger.snapshot()
        self.assertEqual(snapshot["run"]["declaration_version"], 2)
        self.assertEqual(snapshot["batches"], [])
        with sqlite3.connect(self.db) as conn:
            conn.row_factory = sqlite3.Row
            old_run = conn.execute(
                "SELECT stale FROM runs WHERE run_id = ?", (prepared_v1.run_id,)
            ).fetchone()
            old_batch = conn.execute(
                "SELECT stale FROM batches WHERE run_id = ?", (prepared_v1.run_id,)
            ).fetchone()
            lease = conn.execute(
                "SELECT state, stale FROM leases WHERE lease_id = 'lease-old'"
            ).fetchone()
        self.assertEqual(old_run["stale"], 1)
        self.assertEqual(old_batch["stale"], 1)
        self.assertEqual(lease["state"], "REVOKED")
        self.assertEqual(lease["stale"], 1)
        ledger.close()

    def test_declaration_version_cannot_go_backwards(self) -> None:
        ledger = Ledger(self.db)
        self.prepare(ledger)
        lower_declaration = replace(self.dataset.declaration, declaration_version=0)
        with self.assertRaises(SpecError):
            ledger.prepare_run(
                lower_declaration,
                spec_digest=lower_declaration.spec_digest(),
                events_digest=self.dataset.events_digest,
                content_digest=self.dataset.content_digest,
            )
        ledger.close()


if __name__ == "__main__":
    unittest.main()
