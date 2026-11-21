"""Offline consistency replay of a recovered ledger.

The replay re-drives the persisted message sequence with a virtual clock and
the deterministic consumer policy, then compares each message's final state,
attempt count, suppressed count and migration count against the ledger.  The
first divergence raises :class:`ReplayError` (``E_REPLAY``).  Versioned input
mismatches are reported as *not reproducible* instead of as a divergence.
"""

from __future__ import annotations

import heapq

from .errors import NotReproducibleError, ReplayError
from .models import (
    ACKED,
    DEAD,
    DELAYED,
    READY,
    payload_digest,
)
from .policy import planned_failures
from .queue import DelayQueue

# Virtual discrete timestamps are arbitrary monotonic units; only ordering and
# arithmetic matter because the simulator never sleeps.
_SLOT = 1
_BACKOFF_SLOT = 2
_VISIBILITY_SLOT = 3


class _ReplayMessage:
    def __init__(self, row):
        self.message_id = row["message_id"]
        self.topic = row["topic"]
        self.payload = row["payload"]
        self.idempotency_key = row["idempotency_key"]
        self.origin_offset_ms = row["origin_offset_ms"]
        self.state = "pending"
        self.attempts = 0
        self.suppressed = 0
        self.migrations = 0
        self.first_error = ""
        self.last_error = ""
        self.ready_at = self.origin_offset_ms
        self.unknown_count = 0


def check_reproducibility(queue: DelayQueue) -> None:
    versions = queue.replay_versions()
    if versions["spec_version"] != queue.spec.spec_version or \
            versions["spec_digest"] != versions["expected_spec_digest"]:
        raise NotReproducibleError(
            "topic declaration version differs from the ledger record",
            kind="spec_version",
            ledger_spec_version=versions["spec_version"],
            current_spec_version=queue.spec.spec_version,
        )
    if versions["dedup_version"] != "0":
        raise NotReproducibleError(
            "dedup table version differs; history cannot be reproduced",
            kind="dedup_version",
            ledger_dedup_version=versions["dedup_version"],
        )


def replay_ledger(queue: DelayQueue) -> dict:
    """Re-simulate the queue lifecycle and verify every persisted record."""
    check_reproducibility(queue)

    rows = queue.ledger.load_messages()
    simulated = {row["message_id"]: _ReplayMessage(row) for row in rows}
    order = [row["message_id"] for row in rows]

    ready_heap: list[tuple[int, int, str]] = []
    delayed_heap: list[tuple[int, int, str]] = []
    clock = 0
    seq = 0

    def push_ready(message_id: str, when: int) -> None:
        nonlocal seq
        heapq.heappush(ready_heap, (when, seq, message_id))
        seq += 1

    def push_delayed(message_id: str, when: int) -> None:
        nonlocal seq
        heapq.heappush(delayed_heap, (when, seq, message_id))
        seq += 1


    # Initial enqueue transitions mirror DelayQueue.load_inputs: a future
    # due time takes pending->delayed then a due transition ->ready.
    for message_id in order:
        sim = simulated[message_id]
        if sim.ready_at > 0:
            sim.state = DELAYED
            sim.migrations += 1  # pending -> delayed
            push_delayed(message_id, sim.ready_at)
        else:
            sim.state = READY
            sim.migrations += 1  # pending -> ready (enqueue_now)
            push_ready(message_id, 0)

    dedup: dict[str, str] = {}
    suppressed_counts: dict[str, int] = {}

    while ready_heap or delayed_heap:
        clock += _SLOT
        while delayed_heap and delayed_heap[0][0] <= clock:
            _, _, message_id = heapq.heappop(delayed_heap)
            sim = simulated[message_id]
            if sim.state == DELAYED:
                sim.state = READY
                sim.migrations += 1
                push_ready(message_id, clock)
        if not ready_heap:
            continue
        _, _, message_id = heapq.heappop(ready_heap)
        sim = simulated[message_id]
        if sim.state != READY:
            continue
        # claim
        sim.state = "inflight"
        sim.migrations += 1
        clock += _VISIBILITY_SLOT

        topic = queue.spec.topics[sim.topic]
        if sim.idempotency_key in dedup:
            # Duplicate delivery: suppressed, no second business effect.
            suppressed_counts[sim.idempotency_key] = \
                suppressed_counts.get(sim.idempotency_key, 0) + 1
            sim.suppressed = suppressed_counts[sim.idempotency_key]
            sim.state = ACKED
            sim.migrations += 1
            continue

        will_fail = sim.attempts < planned_failures(sim.payload)
        if not will_fail:
            dedup[sim.idempotency_key] = payload_digest(sim.payload)
            sim.state = ACKED
            sim.migrations += 1
            continue

        sim.attempts += 1
        reason = f"simulated-failure-{sim.attempts}"
        if not sim.first_error:
            sim.first_error = reason
        sim.last_error = reason
        if sim.attempts >= topic.max_deliveries:
            sim.state = DEAD
            sim.migrations += 1
        else:
            sim.state = DELAYED
            sim.migrations += 1
            backoff = topic.backoff_ms[sim.attempts - 1]
            push_delayed(message_id, clock + _BACKOFF_SLOT + backoff)

    # Compare against the ledger, preserving input order for stable reports.
    for message_id in order:
        sim = simulated[message_id]
        persisted = next(row for row in rows
                         if row["message_id"] == message_id)
        # A hard-killed lease leaves two non-replayable migrations on disk:
        # the committed claim (ready->inflight) and the recovery transition
        # (inflight->ready).  Both are recovery facts, not replayable business
        # batches, so they are removed before comparing the replay baseline.
        persisted_migrations = queue.ledger.migration_count(message_id)
        replayable_migrations = (
            persisted_migrations - 2 * persisted["unknown_count"])
        comparisons = (
            ("state", sim.state, persisted["state"]),
            ("attempts", sim.attempts, persisted["attempts"]),
            ("suppressed", sim.suppressed, persisted["suppressed"]),
            ("migration_count", sim.migrations, replayable_migrations),
        )
        for field, expected, actual in comparisons:
            if expected != actual:
                raise ReplayError(
                    "replay divergence",
                    message_id=message_id,
                    field=field,
                    expected=expected,
                    actual=actual,
                )
    unknown = sum(row["unknown_count"] for row in rows)
    return {
        "messages_verified": len(rows),
        "dead": sum(1 for sim in simulated.values() if sim.state == DEAD),
        "acked": sum(1 for sim in simulated.values() if sim.state == ACKED),
        "unknown_redeliveries": unknown,
        "status": "consistent",
    }
