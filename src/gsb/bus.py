"""Event-bus orchestrator tying batching, leases, dispatch and adjudication."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .adoption import adjudicate, attribute_segments
from .aggregation import summary_digest
from .batching import manifest_text, plan_batches
from .clock import now_ms, sleep_ms
from .digest import sha256_hex, short_id
from .errors import LeaseError
from .lease import LeaseRegistry
from .ledger import Ledger
from .receipts import normalize
from .scheduler import Scheduler
from .spec import Spec

RENEW_FRACTION = 0.8
LEASE_TTL_FRACTION = 0.5


class Bus:
    def __init__(self, spec: Spec, run_dir: str, crash_after=None):
        self.spec = spec
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(self.run_dir / "ledger.sqlite3", spec.spec_digest)
        self.leases = LeaseRegistry()
        self.batches = plan_batches(spec)
        self.events_by_seq = {event.seq: event for event in spec.events}
        self.crash_after = crash_after
        self._processed = 0
        self.adoptions: dict[str, dict] = {}
        self.attempt_counter: dict[str, int] = {}
        self._renew_conflicts = []
        self.scheduler = Scheduler(
            spec,
            trajectory_writer=self._write_trajectory,
        )

    @property
    def manifest_path(self):
        return self.run_dir / "batch-manifest.json"

    def write_manifest(self):
        self.manifest_path.write_text(manifest_text(self.batches), encoding="utf-8")
        snapshot = {
            "schema_version": 1,
            "spec_digest": self.spec.spec_digest,
            "batch_manifest": manifest_text(self.batches),
        }
        (self.run_dir / "snapshot.json").write_text(
            json.dumps(snapshot, sort_keys=True),
            encoding="utf-8",
        )
        self.ledger.record_snapshot("session", snapshot)

    def _write_trajectory(self, round_index, payload):
        self.ledger.append_trajectory(round_index, payload)

    def run(self, pool):
        for batch in self.batches:
            self.scheduler.submit(batch, self._make_handler(pool))
        return self.scheduler.run_until_idle()

    def _dispatch_id(self, batch, attempt):
        return short_id({"batch_id": batch.batch_id, "attempt": attempt}, 24)

    def _make_handler(self, pool):
        def handler(entry):
            return self._process_batch(pool, entry)

        return handler

    def _process_batch(self, pool, entry):
        batch = entry["batch"]
        subscriber = self.spec.subscriber(batch.subscriber_id)
        client = pool.get(batch.subscriber_id)
        attempt = self.attempt_counter.get(batch.batch_id, 0) + 1
        self.attempt_counter[batch.batch_id] = attempt
        dispatch_id = self._dispatch_id(batch, attempt)
        merge_seq = attempt

        # Dispatch record is registered before receipts so x1 can be backfilled.
        dispatch = {
            "dispatch_id": dispatch_id,
            "batch_id": batch.batch_id,
            "subscriber_id": batch.subscriber_id,
            "channel_id": batch.channel_id,
            "topic": batch.topic,
            "batch_index": batch.batch_index,
            "attempt": attempt,
            "event_seqs": list(batch.event_seqs),
            "payload_summary": batch.payload_summary,
            "merge_seq": merge_seq,
            "enqueued_at_ms": entry["enqueued_at_ms"],
            "dequeued_at_ms": entry["dequeued_at_ms"],
            "granted_at_ms": None,
            "format_version": 3,
            "digest": batch.digest,
            "status": "leased",
        }
        self.ledger.register_dispatch(dispatch)

        # Lease is only granted once the batch is admitted off the queue.
        ttl_ms = max(20, int(subscriber.budget_ms * LEASE_TTL_FRACTION))
        lease = self.leases.grant(batch.subscriber_id, batch.batch_id, ttl_ms)
        client.grant_lease(lease)
        granted = now_ms()
        dispatch["granted_at_ms"] = granted
        self.ledger.register_dispatch(dispatch)

        work_ms = subscriber.budget_ms
        payloads = [self.events_by_seq[seq].payload for seq in batch.event_seqs]
        covered_refs = []
        formats = ["x1", "x2", "x3"]
        if attempt > 1:
            covered_refs = [attempt - 1]

        renew_start = granted + int(ttl_ms * RENEW_FRACTION)
        self._renew_if_due(client, lease, renew_start, work_ms, ttl_ms)

        reply = client.send_batch(
            batch,
            payloads,
            lease,
            work_ms,
            formats,
            covered_refs,
        )
        completion_ms = reply["completion_ms"]
        # Record the single processor completion instant before normalization
        # so x1 backfill reaches the same value x2/x3 carry explicitly.
        self.ledger.mark_completion(dispatch_id, completion_ms)

        # Time the format backfill segment explicitly.
        backfill_start = now_ms()
        base_processing = completion_ms - granted
        candidates = []
        for fmt in formats:
            normalized = normalize(
                reply["receipts"][fmt],
                self.ledger.lookup_dispatch,
            )
            # Each candidate's fixed-point duration includes its own format
            # normalization/backfill cost, producing an observable small delta
            # that drives the declared error-margin tiebreak.
            candidate_duration = base_processing + (now_ms() - backfill_start)
            aggregate_doc = reply["aggregate"]
            candidate = {
                "normalized": normalized,
                "merge_seq": merge_seq,
                "attempt": attempt,
                "fixed_point_duration_ms": candidate_duration,
                "summary_digest": summary_digest(
                    batch.batch_id,
                    batch.event_seqs,
                    aggregate_doc,
                    normalized["caliber_version"],
                ),
            }
            candidates.append(candidate)
        backfill_ms = max(1, now_ms() - backfill_start)

        decision = adjudicate(candidates, self.spec.error_margin_ms)
        queue_ms = max(0, entry["dequeued_at_ms"] - entry["enqueued_at_ms"])
        processing_ms = max(0, completion_ms - granted)
        decision["attribution"] = attribute_segments(
            queue_ms,
            backfill_ms,
            processing_ms,
        )

        current_lease = self.leases.for_batch(batch.batch_id)
        self.ledger.commit_batch(
            batch.batch_id,
            [c["normalized"] for c in candidates],
            {
                "winning_format": decision["winning_format"],
                "decision": decision,
                "error_margin_applied": decision["error_margin_applied"],
            },
            current_lease,
            {
                "dispatch_id": dispatch_id,
                "status": "committed",
                "dequeued_at_ms": entry["dequeued_at_ms"],
                "granted_at_ms": granted,
            },
        )
        self.adoptions[batch.batch_id] = decision

        self._processed += 1
        if self.crash_after and self._processed >= self.crash_after:
            self._force_crash()
        return decision

    def _renew_if_due(self, client, lease, renew_start, work_ms, ttl_ms):
        # Real timer-based wait; the long deterministic processing will cross
        # the renewal boundary, exercising optimistic version checks.
        while now_ms() < renew_start:
            sleep_ms(2)
        current = self.leases.for_batch(lease.batch_id)
        renewed = self.leases.renew(current.lease_id, current.version, ttl_ms)
        client.renew_lease(renewed.lease_id, current.version, renewed.expires_at_ms)

    def stale_renewal_demo(self, pool, batch_id):
        """Deliberately renew with a stale version; processor must reject."""
        lease = self.leases.for_batch(batch_id)
        client = pool.get(lease.subscriber_id)
        try:
            client.renew_lease(lease.lease_id, lease.version - 1, now_ms() + 9999)
        except LeaseError as exc:
            self._renew_conflicts.append(exc.to_dict())
            return exc.to_dict()
        raise AssertionError("stale renewal should have been rejected")

    def _force_crash(self):
        # Hard exit bypassing all cleanup: no exit report is promised.
        import os

        os._exit(137)

    def write_outputs(self):
        (self.run_dir / "adoptions.json").write_text(
            json.dumps(self.adoptions, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        trajectory_rows = [
            json.loads(row[0])
            for row in self.ledger.conn.execute(
                "SELECT payload FROM trajectory ORDER BY round_index"
            )
        ]
        (self.run_dir / "trajectory.json").write_text(
            json.dumps(trajectory_rows, sort_keys=True),
            encoding="utf-8",
        )

    def close(self):
        self.ledger.close()

    @property
    def renew_conflicts(self):
        return list(self._renew_conflicts)
