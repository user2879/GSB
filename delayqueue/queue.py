"""Delay queue orchestrator: scheduling, leases, retries and recovery.

In-memory scheduling state is always backed by an atomic SQLite batch. Two
daemon threads use the real monotonic clock to (1) promote due delayed
messages and (2) reclaim inflight leases that exceed the topic visibility
timeout.
"""

from __future__ import annotations

import heapq
import threading
from dataclasses import asdict

from .errors import TimeoutConfigError
from .ledger import FORMAT_VERSION, Ledger
from .models import (
    ACKED,
    DEAD,
    DELAYED,
    INFLIGHT,
    PENDING,
    READY,
    Delivery,
    Message,
    QueueSpec,
    RecoveryReport,
    mono_ms,
    next_state,
    payload_digest,
    wall_ms,
)
from .specio import load_messages, load_queue_spec


class DelayQueue:
    def __init__(self, spec: QueueSpec, ledger: Ledger, epoch_ms: int):
        self.spec = spec
        self.ledger = ledger
        self.epoch_ms = epoch_ms
        self.messages: dict[str, Message] = {}
        self.ready: dict[str, list[Message]] = {
            topic: [] for topic in spec.topics}
        self.delayed_heap: list[tuple[int, int, Message]] = []
        self.inflight: dict[str, Message] = {}
        self.lease_started: dict[str, int] = {}
        self._counter = 0
        self._closed = False
        self._wakeup = threading.Condition()
        self._threads: list[threading.Thread] = []
        self._hydrate()

    # ------------------------------------------------------------ constructors
    @classmethod
    def open(cls, spec_path: str, ledger_path: str) -> "DelayQueue":
        spec = load_queue_spec(spec_path)
        epoch_ms = wall_ms()
        ledger = Ledger.create(ledger_path, spec, epoch_ms)
        queue = cls(spec, ledger, epoch_ms)
        queue._start_schedulers()
        return queue

    @classmethod
    def reopen(cls, ledger_path: str) -> "DelayQueue":
        """Open an existing ledger, verify it and run crash recovery."""
        ledger = Ledger.open(ledger_path)
        spec = cls._spec_from_ledger(ledger)
        epoch_ms = int(ledger.get_meta("epoch_ms"))
        queue = cls(spec, ledger, epoch_ms)
        raw_report = ledger.recover_inflight()
        queue._hydrate()
        report = RecoveryReport(
            recovered_messages=raw_report["recovered_messages"],
            unknown_messages=raw_report["unknown_messages"],
            dead_messages=raw_report["dead_messages"],
            redelivered_after_visibility=
                raw_report["redelivered_after_visibility"],
        )
        queue._last_recovery_report = report
        queue._start_schedulers()
        return queue, report

    @staticmethod
    def _spec_from_ledger(ledger: Ledger) -> QueueSpec:
        import json
        from .models import TopicSpec
        topics = {}
        for row in ledger.topic_rows():
            topic, vto, max_d, backoff, concurrency = row
            topics[topic] = TopicSpec(
                topic=topic,
                visibility_timeout_ms=vto,
                max_deliveries=max_d,
                backoff_ms=tuple(json.loads(backoff)),
                concurrency=concurrency,
            )
        return QueueSpec(
            global_concurrency=int(
                ledger.get_meta("global_concurrency", "1")),
            topics=topics,
            spec_version=ledger.get_meta("spec_version", "v1"),
        )

    def close(self) -> None:
        with self._wakeup:
            self._closed = True
            self._wakeup.notify_all()
        for thread in self._threads:
            thread.join(timeout=2)
        self.ledger.close()

    # ------------------------------------------------------------- hydration
    def _hydrate(self) -> None:
        self.messages.clear()
        self.ready = {topic: [] for topic in self.spec.topics}
        self.delayed_heap = []
        self.inflight.clear()
        self.lease_started.clear()
        for row in self.ledger.load_messages():
            message = Message(**row)
            self.messages[message.message_id] = message
            if message.state == READY:
                self.ready[message.topic].append(message)
            elif message.state == DELAYED:
                heapq.heappush(
                    self.delayed_heap,
                    (message.ready_monotonic_ms, message.seq, message),
                )
            elif message.state == INFLIGHT:
                self.inflight[message.message_id] = message
                self.lease_started[message.message_id] = mono_ms()
        for bucket in self.ready.values():
            bucket.sort(key=lambda item: (item.deliver_at_abs, item.seq))

    # ------------------------------------------------------------- ingestion
    def load_inputs(self, inputs) -> int:
        """Validate-and-enqueue already validated MessageInput records."""
        now_wall = wall_ms()
        now_mono = mono_ms()
        # Deterministic rule shared with the replay simulator: deliver_at=0
        # is immediately ready; any positive offset starts delayed.
        with self.ledger.batch() as cur:
            batch_id = self.ledger.next_batch_id(cur)
            max_seq = len(self.messages)
            for item in inputs:
                if item.message_id in self.messages:
                    continue
                max_seq += 1
                absolute = self.epoch_ms + item.deliver_at
                message = Message(
                    message_id=item.message_id,
                    topic=item.topic,
                    deliver_at_abs=absolute,
                    origin_offset_ms=item.deliver_at,
                    payload=item.payload,
                    payload_summary=payload_digest(item.payload),
                    idempotency_key=item.idempotency_key,
                    state=PENDING,
                    seq=max_seq,
                )
                if item.deliver_at > 0:
                    message.state = DELAYED
                    message.ready_monotonic_ms = (
                        now_mono + item.deliver_at)
                    reason = "enqueue"
                else:
                    message.state = READY
                    message.ready_monotonic_ms = now_mono
                    reason = "enqueue_now"
                self._insert_message(cur, message)
                cur.execute(
                    "INSERT INTO transitions(batch_id, message_id, from_state,"
                    " to_state, reason, attempts, mono_ms, wall_ms) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (batch_id, message.message_id, PENDING, message.state,
                     reason, 0, now_mono, now_wall),
                )
                self.messages[message.message_id] = message
                if message.state == READY:
                    self.ready[message.topic].append(message)
                else:
                    heapq.heappush(
                        self.delayed_heap,
                        (message.ready_monotonic_ms, message.seq, message),
                    )
            cur.execute(
                "INSERT INTO progress(batch_id, cursor_seq, committed_at_ms) "
                "VALUES(?,?,?)", (batch_id, max_seq, wall_ms()))
        with self._wakeup:
            self._wakeup.notify_all()
        return len(inputs)

    @staticmethod
    def _insert_message(cur, message: Message) -> None:
        cur.execute(
            "INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (message.message_id, message.topic, message.deliver_at_abs,
             message.origin_offset_ms, message.payload,
             message.payload_summary, message.idempotency_key, message.state,
             message.attempts, message.suppressed, message.unknown_count,
             message.ready_monotonic_ms, message.first_error,
             message.last_error, message.seq),
        )

    # ------------------------------------------------------------- schedulers
    def _start_schedulers(self) -> None:
        delayed = threading.Thread(
            target=self._delayed_loop, name="dq-delayed", daemon=True)
        leases = threading.Thread(
            target=self._lease_loop, name="dq-lease", daemon=True)
        self._threads = [delayed, leases]
        delayed.start()
        leases.start()

    def _delayed_loop(self) -> None:
        while True:
            with self._wakeup:
                if self._closed:
                    return
                self._wakeup.wait(timeout=0.02)
            now = mono_ms()
            due = []
            while self.delayed_heap and self.delayed_heap[0][0] <= now:
                _, _, message = heapq.heappop(self.delayed_heap)
                if message.state == DELAYED:
                    due.append(message)
            if due:
                with self.ledger.batch() as cur:
                    batch_id = self.ledger.next_batch_id(cur)
                    for message in due:
                        cur.execute(
                            "UPDATE messages SET state=?, "
                            "ready_monotonic_ms=? WHERE message_id=?",
                            (READY, now, message.message_id))
                        cur.execute(
                            "INSERT INTO transitions(batch_id, message_id, "
                            "from_state, to_state, reason, attempts, "
                            "mono_ms, wall_ms) VALUES(?,?,?,?,?,?,?,?)",
                            (batch_id, message.message_id, DELAYED, READY,
                             "due", message.attempts, now, wall_ms()))
                        message.state = READY
                        message.ready_monotonic_ms = now
                        self.ready[message.topic].append(message)
                    cur.execute(
                        "INSERT INTO progress(batch_id, cursor_seq, "
                        "committed_at_ms) VALUES(?,?,?)",
                        (batch_id, len(self.messages), wall_ms()))
                with self._wakeup:
                    self._wakeup.notify_all()

    def _lease_loop(self) -> None:
        while True:
            with self._wakeup:
                if self._closed:
                    return
                self._wakeup.wait(timeout=0.02)
            now = mono_ms()
            expired = []
            for message_id, message in list(self.inflight.items()):
                timeout_ms = self.spec.topics[message.topic].\
                    visibility_timeout_ms
                if now - self.lease_started[message_id] >= timeout_ms:
                    expired.append(message)
            if expired:
                with self.ledger.batch() as cur:
                    batch_id = self.ledger.next_batch_id(cur)
                    for message in expired:
                        # Special transition: attempts NOT incremented.
                        cur.execute(
                            "UPDATE messages SET state=?, "
                            "ready_monotonic_ms=? WHERE message_id=?",
                            (READY, now, message.message_id))
                        cur.execute(
                            "UPDATE visibility SET closed_mono_ms=?, "
                            "close_reason=? WHERE message_id=? "
                            "AND closed_mono_ms IS NULL",
                            (now, "visibility_timeout", message.message_id))
                        cur.execute(
                            "INSERT INTO transitions(batch_id, message_id, "
                            "from_state, to_state, reason, attempts, "
                            "mono_ms, wall_ms) VALUES(?,?,?,?,?,?,?,?)",
                            (batch_id, message.message_id, INFLIGHT, READY,
                             "visibility_timeout", message.attempts, now,
                             wall_ms()))
                        del self.inflight[message.message_id]
                        self.lease_started.pop(message.message_id, None)
                        message.state = READY
                        message.ready_monotonic_ms = now
                        self.ready[message.topic].append(message)
                    cur.execute(
                        "INSERT INTO progress(batch_id, cursor_seq, "
                        "committed_at_ms) VALUES(?,?,?)",
                        (batch_id, len(self.messages), wall_ms()))
                with self._wakeup:
                    self._wakeup.notify_all()

    # ----------------------------------------------------------------- claim
    def _topic_inflight(self, topic: str) -> int:
        return sum(1 for message in self.inflight.values()
                   if message.topic == topic)

    def claim(self, timeout: float = 0.0) -> Delivery | None:
        """Lease one ready message under global + per-topic caps.

        When every candidate is quota-blocked the call queues on the condition
        variable; no quota check is ever bypassed.
        """
        deadline = None if timeout is None else mono_ms() + int(timeout * 1000)
        with self._wakeup:
            while not self._closed:
                if len(self.inflight) < self.spec.global_concurrency:
                    candidates = []
                    for topic, bucket in self.ready.items():
                        if not bucket:
                            continue
                        if self._topic_inflight(topic) >= \
                                self.spec.topics[topic].concurrency:
                            continue
                        candidates.append(bucket[0])
                    if candidates:
                        candidates.sort(
                            key=lambda item:
                                (item.deliver_at_abs, item.seq))
                        message = candidates[0]
                        return self._commit_claim(message)
                remaining = 0.05
                if deadline is not None:
                    remaining_ms = deadline - mono_ms()
                    if remaining_ms <= 0:
                        return None
                    remaining = min(0.05, remaining_ms / 1000)
                self._wakeup.wait(timeout=remaining)
        return None

    def _commit_claim(self, message: Message) -> Delivery:
        now = mono_ms()
        with self.ledger.batch() as cur:
            batch_id = self.ledger.next_batch_id(cur)
            target = next_state(message.state, "claim")
            cur.execute(
                "UPDATE messages SET state=? WHERE message_id=?",
                (target, message.message_id))
            cur.execute(
                "INSERT INTO visibility(message_id, batch_id, lease_mono_ms,"
                " timeout_ms) VALUES(?,?,?,?)",
                (message.message_id, batch_id, now,
                 self.spec.topics[message.topic].visibility_timeout_ms))
            cur.execute(
                "INSERT INTO transitions(batch_id, message_id, from_state, "
                "to_state, reason, attempts, mono_ms, wall_ms) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (batch_id, message.message_id, message.state, INFLIGHT,
                 "claim", message.attempts, now, wall_ms()))
            cur.execute(
                "INSERT INTO progress(batch_id, cursor_seq, committed_at_ms) "
                "VALUES(?,?,?)", (batch_id, len(self.messages), wall_ms()))
            self.ready[message.topic].remove(message)
            message.state = INFLIGHT
            self.inflight[message.message_id] = message
            self.lease_started[message.message_id] = now
        delivery = Delivery(
            message_id=message.message_id,
            topic=message.topic,
            payload=message.payload,
            idempotency_key=message.idempotency_key,
            attempt=message.attempts,
            lease_monotonic_ms=now,
            visibility_timeout_ms=self.spec.topics[
                message.topic].visibility_timeout_ms,
            unknown_redelivery=message.unknown_count > 0,
        )
        with self._wakeup:
            self._wakeup.notify_all()
        return delivery

    # ------------------------------------------------------------------- ack
    def ack(self, delivery: Delivery) -> dict:
        """Confirm success; dedups on idempotency key inside one transaction."""
        message = self._require_inflight(delivery.message_id)
        now = mono_ms()
        with self.ledger.batch() as cur:
            batch_id = self.ledger.next_batch_id(cur)
            row = cur.execute(
                "SELECT message_id, suppressed_count FROM dedup WHERE "
                "idempotency_key=?", (delivery.idempotency_key,)).fetchone()
            if row is not None:
                # Already processed: no second business effect.
                cur.execute(
                    "UPDATE dedup SET suppressed_count=? WHERE "
                    "idempotency_key=?",
                    (row[1] + 1, delivery.idempotency_key))
                cur.execute(
                    "UPDATE messages SET state=?, suppressed=? WHERE "
                    "message_id=?", (ACKED, row[1] + 1, message.message_id))
                result = {"business_effect": False,
                          "suppressed": True,
                          "suppressed_count": row[1] + 1}
            else:
                cur.execute(
                    "INSERT INTO dedup VALUES(?,?,?,?)",
                    (delivery.idempotency_key, message.payload_summary, 0,
                     message.message_id))
                cur.execute(
                    "UPDATE messages SET state=? WHERE message_id=?",
                    (ACKED, message.message_id))
                result = {"business_effect": True, "suppressed": False,
                          "suppressed_count": 0}
            cur.execute(
                "UPDATE visibility SET closed_mono_ms=?, close_reason=? "
                "WHERE message_id=? AND closed_mono_ms IS NULL",
                (now, "ack", message.message_id))
            cur.execute(
                "INSERT INTO transitions(batch_id, message_id, from_state, "
                "to_state, reason, attempts, mono_ms, wall_ms) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (batch_id, message.message_id, INFLIGHT, ACKED, "ack",
                 message.attempts, now, wall_ms()))
            cur.execute(
                "INSERT INTO progress(batch_id, cursor_seq, committed_at_ms) "
                "VALUES(?,?,?)", (batch_id, len(self.messages), wall_ms()))
            del self.inflight[message.message_id]
            self.lease_started.pop(message.message_id, None)
            message.state = ACKED
            message.suppressed = result["suppressed_count"]
        with self._wakeup:
            self._wakeup.notify_all()
        return result

    # ------------------------------------------------------------------ nack
    def nack(self, delivery: Delivery, reason: str) -> dict:
        """Confirm failure: retry with backoff, or move to dead-letter."""
        self.validate_timeout(
            self.spec.topics[delivery.topic].visibility_timeout_ms)
        message = self._require_inflight(delivery.message_id)
        topic = self.spec.topics[message.topic]
        now = mono_ms()
        next_attempts = message.attempts + 1
        with self.ledger.batch() as cur:
            batch_id = self.ledger.next_batch_id(cur)
            cur.execute(
                "UPDATE visibility SET closed_mono_ms=?, close_reason=? "
                "WHERE message_id=? AND closed_mono_ms IS NULL",
                (now, "nack", message.message_id))
            if not message.first_error:
                first_error = reason
            else:
                first_error = message.first_error
            if next_attempts >= topic.max_deliveries:
                cur.execute(
                    "UPDATE messages SET state=?, attempts=?, "
                    "first_error=?, last_error=? WHERE message_id=?",
                    (DEAD, next_attempts, first_error, reason,
                     message.message_id))
                cur.execute(
                    "INSERT INTO transitions(batch_id, message_id, "
                    "from_state, to_state, reason, attempts, mono_ms, "
                    "wall_ms) VALUES(?,?,?,?,?,?,?,?)",
                    (batch_id, message.message_id, INFLIGHT, DEAD,
                     "nack_dead", next_attempts, now, wall_ms()))
                result = {"outcome": "dead", "attempts": next_attempts}
            else:
                backoff = topic.backoff_ms[next_attempts - 1]
                ready_monotonic = now + backoff
                cur.execute(
                    "UPDATE messages SET state=?, attempts=?, "
                    "ready_monotonic_ms=?, first_error=?, last_error=? "
                    "WHERE message_id=?",
                    (DELAYED, next_attempts, ready_monotonic, first_error,
                     reason, message.message_id))
                cur.execute(
                    "INSERT INTO transitions(batch_id, message_id, "
                    "from_state, to_state, reason, attempts, mono_ms, "
                    "wall_ms) VALUES(?,?,?,?,?,?,?,?)",
                    (batch_id, message.message_id, INFLIGHT, DELAYED,
                     "nack_retry", next_attempts, now, wall_ms()))
                result = {"outcome": "delayed", "attempts": next_attempts,
                          "backoff_ms": backoff}
            cur.execute(
                "INSERT INTO progress(batch_id, cursor_seq, committed_at_ms) "
                "VALUES(?,?,?)", (batch_id, len(self.messages), wall_ms()))
            del self.inflight[message.message_id]
            self.lease_started.pop(message.message_id, None)
            if result["outcome"] == "dead":
                message.state = DEAD
            else:
                message.state = DELAYED
                heapq.heappush(
                    self.delayed_heap,
                    (ready_monotonic, message.seq, message))
            message.attempts = next_attempts
            message.first_error = first_error
            message.last_error = reason
        with self._wakeup:
            self._wakeup.notify_all()
        return result

    def _require_inflight(self, message_id: str) -> Message:
        message = self.messages.get(message_id)
        if message is None or message.state != INFLIGHT:
            from .errors import IllegalTransitionError
            raise IllegalTransitionError(
                "message is not currently inflight",
                message_id=message_id,
                state=message.state if message else None)
        return message

    # ---------------------------------------------------------- timeout rules
    @staticmethod
    def validate_timeout(visibility_timeout_ms: int) -> int:
        """Runtime E_TIMEOUT guard for visibility-timeout parameters."""
        if isinstance(visibility_timeout_ms, bool) or \
                not isinstance(visibility_timeout_ms, int):
            raise TimeoutConfigError(
                "visibility timeout must be an integer",
                field="visibility_timeout_ms", value=visibility_timeout_ms)
        if visibility_timeout_ms <= 0:
            raise TimeoutConfigError(
                "visibility timeout must be a positive integer",
                field="visibility_timeout_ms", value=visibility_timeout_ms)
        return visibility_timeout_ms

    # -------------------------------------------------------------- snapshots
    def snapshot(self) -> dict:
        """Read-only, stable-ordered diagnostic view; never mutates the ledger."""
        now = mono_ms()
        topic_view = {}
        for topic in sorted(self.spec.topics):
            bucket = self.ready.get(topic, [])
            delayed_count = sum(
                1 for _, _, message in self.delayed_heap
                if message.topic == topic and message.state == DELAYED)
            inflight_ids = sorted(
                message_id for message_id, message in self.inflight.items()
                if message.topic == topic)
            topic_view[topic] = {
                "ready": len(bucket),
                "delayed": delayed_count,
                "inflight": len(inflight_ids),
                "inflight_lifetime_ms": {
                    message_id: now - self.lease_started[message_id]
                    for message_id in inflight_ids},
                "quota": self.spec.topics[topic].concurrency,
            }
        attempt_distribution: dict[int, int] = {}
        for message in self.messages.values():
            attempt_distribution[message.attempts] = \
                attempt_distribution.get(message.attempts, 0) + 1
        dead = [
            {
                "message_id": message.message_id,
                "topic": message.topic,
                "attempts": message.attempts,
                "first_error": message.first_error,
                "last_error": message.last_error,
            }
            for message in sorted(
                (item for item in self.messages.values()
                 if item.state == DEAD),
                key=lambda item: item.seq)
        ]
        state_counts = {state: 0 for state in
                        (PENDING, DELAYED, READY, INFLIGHT, ACKED, DEAD)}
        for message in self.messages.values():
            state_counts[message.state] += 1
        return {
            "format_version": FORMAT_VERSION,
            "ledger_path": self.ledger.path,
            "ledger_state": "open" if not self._closed else "closed",
            "spec_version": self.spec.spec_version,
            "dedup_version": int(self.ledger.get_meta("dedup_version", "0")),
            "global_inflight": len(self.inflight),
            "global_concurrency": self.spec.global_concurrency,
            "topics": topic_view,
            "state_counts": state_counts,
            "attempt_distribution": dict(sorted(
                attempt_distribution.items())),
            "dead": dead,
        }

    # ------------------------------------------------- replay version guards
    def replay_versions(self) -> dict:
        return {
            "spec_version": self.ledger.get_meta("spec_version", ""),
            "spec_digest": self.ledger.get_meta("spec_digest", ""),
            "dedup_version": self.ledger.get_meta("dedup_version", "0"),
            "expected_spec_digest": self.spec.canonical_digest(),
        }

    # ------------------------------------------------------------ worker pool
    def run_workers(self, handler, worker_count: int,
                    stop_when_drained: bool = True) -> list[dict]:
        """Project claims onto real worker threads; returns per-delivery logs."""
        if worker_count > self.spec.global_concurrency:
            raise TimeoutConfigError(
                "worker count cannot exceed global concurrency",
                field="worker_count", value=worker_count)
        logs: list[dict] = []
        logs_lock = threading.Lock()

        def worker_loop() -> None:
            while not self._closed:
                delivery = self.claim(timeout=0.1)
                if delivery is None:
                    if stop_when_drained and self._is_drained():
                        return
                    continue
                try:
                    handler(delivery, self)
                except Exception as exc:  # pragma: no cover - safety net
                    self.nack(delivery, f"handler-crash:{exc!r}")
                with logs_lock:
                    logs.append({"message_id": delivery.message_id,
                                 "topic": delivery.topic,
                                 "attempt": delivery.attempt})

        threads = [
            threading.Thread(target=worker_loop, name=f"dq-worker-{index}",
                             daemon=True)
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return logs

    def _is_drained(self) -> bool:
        if self.inflight or any(self.ready.values()):
            return False
        return all(message.state in (ACKED, DEAD)
                   for message in self.messages.values())

    # -------------------------------------------------------- message access
    def get_message(self, message_id: str) -> Message | None:
        return self.messages.get(message_id)

    def all_messages(self) -> list[Message]:
        return [self.messages[key] for key in sorted(
            self.messages, key=lambda item: self.messages[item].seq)]

    def wait_drained(self, timeout: float = 30.0) -> bool:
        import time
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._is_drained():
                return True
            time.sleep(0.02)
        return self._is_drained()
