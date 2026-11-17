"""Event bus: subscription routing, recovery, and replay consistency checks."""

import json
import os
import time

from .errors import BusError, E_REPLAY
from .machine import ALLOWED, SubscriberMachine


class RecoveryReport:
    def __init__(self):
        self.progress = {}
        self.final_states = {}
        self.unknown_transitions = 0
        self.torn_tail_records = 0

    def to_dict(self):
        return {
            "subscribers_recovered": len(self.progress),
            "progress": {key: self.progress[key] for key in sorted(self.progress)},
            "final_states": {key: self.final_states[key] for key in sorted(self.final_states)},
            "unknown_transitions": self.unknown_transitions,
            "torn_tail_records": self.torn_tail_records,
            "torn_tail_action": "discarded: incomplete (complete=0) rows ignored",
            "unknown_action": "marked unknown (complete=0); events require reprocessing",
        }


class EventBus:
    def __init__(self, spec, ledger, logger=None):
        self.spec = spec
        self.ledger = ledger
        self.logger = logger
        self.machines = {}
        self._clock_ms = 0
        self._event_index = {}
        for event in spec.events:
            self._event_index[(event.source_id, event.seq)] = event

    def _tick(self):
        now = time.monotonic_ns() // 1_000_000
        self._clock_ms = max(self._clock_ms + 1, now)
        return self._clock_ms

    def _log(self, code, message, **kwargs):
        if self.logger:
            self.logger.log(code, message, **kwargs)

    def recover(self):
        """Rebuild in-memory subscriber views solely from persisted ledger rows."""
        report = RecoveryReport()
        torn = self.ledger.incomplete_rows()
        report.torn_tail_records = len(torn)
        dangling = self.ledger.dangling_processing_rows()
        for row in dangling:
            self.ledger.mark_incomplete(row["id"])
            self._log("INFO", "processing migration without persisted outcome marked"
                      " unknown; event requires reprocessing",
                      subscriber_id=row["subscriber_id"], event_seq=row["event_seq"])
        report.unknown_transitions = len(dangling)
        self.machines = {}
        for subscriber in self.spec.subscribers:
            machine = SubscriberMachine(subscriber.subscriber_id)
            for row in self.ledger.transitions_for(subscriber.subscriber_id):
                machine.apply_recorded(row["from_state"], row["to_state"], row["event_seq"])
            self.machines[subscriber.subscriber_id] = machine
            progress = self.ledger.get_progress(subscriber.subscriber_id)
            report.progress[subscriber.subscriber_id] = progress if progress is not None else 0
            report.final_states[subscriber.subscriber_id] = machine.state
        self._log("INFO", "recovery completed",
                  subscribers_recovered=len(report.progress),
                  unknown_transitions=report.unknown_transitions,
                  torn_tail_records=report.torn_tail_records)
        return report

    def _subscribers_for(self, event_type):
        matched = [s for s in self.spec.subscribers if event_type in s.event_types]
        matched.sort(key=lambda s: s.subscriber_id)
        return matched

    def process_all(self, kill_after=None):
        """Route every event to subscribers; commit progress transactionally.

        ``kill_after`` simulates a forced termination: after that many commits
        the next dispatch persists its processing transition and the process
        dies immediately via ``os._exit`` (no cleanup, no exit report).
        """
        committed = self.ledger.committed_pairs()
        commits = 0
        for event in self.spec.events:
            for subscriber in self._subscribers_for(event.event_type):
                subscriber_id = subscriber.subscriber_id
                if (event.source_id, event.seq) in committed.get(subscriber_id, set()):
                    continue
                machine = self.machines[subscriber_id]
                from_state = machine.transition("processing", event.seq)
                self.ledger.record_transition(
                    subscriber_id, event.source_id, event.seq,
                    from_state, "processing", self._tick())
                if kill_after is not None and commits >= kill_after:
                    self._log("WARN", "forced termination simulated; processing"
                              " migration persisted without outcome",
                              subscriber_id=subscriber_id, event_seq=event.seq)
                    os._exit(3)
                from_state = machine.transition("committed", event.seq)
                self.ledger.commit_with_progress(
                    subscriber_id, event.source_id, event.seq, from_state, self._tick())
                committed.setdefault(subscriber_id, set()).add((event.source_id, event.seq))
                commits += 1
        return commits

    def replay_check(self):
        """Re-execute migrations from the ledger and compare field by field.

        Uses only persisted data: the subscriber set snapshot stored in the
        ledger meta and the event sequence declared by the source spec.
        """
        snapshot = json.loads(self.ledger.meta()["subscriber_snapshot"])
        subscribers = sorted(snapshot["subscribers"], key=lambda s: s["subscriber_id"])
        for raw in subscribers:
            subscriber_id = raw["subscriber_id"]
            subscribed = set(raw["event_types"])
            machine = SubscriberMachine(subscriber_id)
            committed_seen = set()
            rows = self.ledger.transitions_for(subscriber_id)
            for row in rows:
                key = (row["source_id"], row["event_seq"])
                event = self._event_index.get(key)
                if event is None:
                    raise BusError(
                        E_REPLAY, "ledger references event not present in the source"
                        " event sequence; cannot reproduce",
                        field="event_seq", subscriber_id=subscriber_id,
                        event_seq=row["event_seq"])
                if event.event_type not in subscribed:
                    raise BusError(
                        E_REPLAY, "event type %r is not subscribed by this subscriber"
                        % event.event_type,
                        field="event_type", subscriber_id=subscriber_id,
                        event_seq=row["event_seq"])
                if row["from_state"] != machine.state:
                    raise BusError(
                        E_REPLAY, "from_state %r does not match replayed state %r"
                        % (row["from_state"], machine.state),
                        field="from_state", subscriber_id=subscriber_id,
                        event_seq=row["event_seq"])
                if row["to_state"] not in ALLOWED[machine.state]:
                    raise BusError(
                        E_REPLAY, "illegal recorded transition %s -> %s"
                        % (machine.state, row["to_state"]),
                        field="to_state", subscriber_id=subscriber_id,
                        event_seq=row["event_seq"])
                if row["to_state"] == "committed":
                    if key in committed_seen:
                        raise BusError(
                            E_REPLAY, "duplicate committed migration in ledger",
                            field="event_seq", subscriber_id=subscriber_id,
                            event_seq=row["event_seq"])
                    committed_seen.add(key)
                machine.apply_recorded(row["from_state"], row["to_state"], row["event_seq"])
            if machine.transition_count != len(rows):
                raise BusError(
                    E_REPLAY, "transition count %d does not match ledger rows %d"
                    % (machine.transition_count, len(rows)),
                    field="transition_count", subscriber_id=subscriber_id)
            progress = self.ledger.get_progress(subscriber_id)
            max_committed = self.ledger.max_committed_seq(subscriber_id)
            expected = max_committed if max_committed is not None else None
            if progress != expected:
                raise BusError(
                    E_REPLAY, "progress %r does not match max committed seq %r"
                    % (progress, expected),
                    field="progress", subscriber_id=subscriber_id,
                    event_seq=progress)
        self._log("INFO", "replay check passed", subscribers_checked=len(subscribers))
