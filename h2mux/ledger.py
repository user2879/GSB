"""Append-only audit ledger for frame, state, wait, and fairness events."""

from __future__ import annotations

from typing import Any


class Ledger:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.transitions: list[dict[str, Any]] = []
        self.empty_waits: list[dict[str, Any]] = []
        self.queue_waits: list[dict[str, Any]] = []
        self.priority_events: list[dict[str, Any]] = []
        self.schedule_rounds: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.replacements: list[dict[str, Any]] = []
        self.controls: list[dict[str, Any]] = []
        self._next_seq = 1
        self._next_transition_seq = 1
        self._next_wait_id = 1
        self._next_queue_wait_id = 1
        self._next_control_seq = 1

    @property
    def next_seq(self) -> int:
        return self._next_seq

    @property
    def next_transition_seq(self) -> int:
        return self._next_transition_seq

    @property
    def next_wait_id(self) -> int:
        return self._next_wait_id

    @property
    def next_queue_wait_id(self) -> int:
        return self._next_queue_wait_id

    @property
    def next_control_seq(self) -> int:
        return self._next_control_seq

    def append_record(self, value: dict[str, Any]) -> dict[str, Any]:
        record = {"seq": self._next_seq, **value}
        self._next_seq += 1
        self.records.append(record)
        if record.get("code") not in {None, "OK"}:
            self.errors.append(
                {
                    "seq": record["seq"],
                    "frame_seq": record.get("frame_seq"),
                    "clock_ms": record.get("clock_ms"),
                    "script_index": record.get("script_index"),
                    "frame_type": record.get("frame_type"),
                    "stream_id": record.get("target_stream_id"),
                    "code": record.get("code"),
                    "reason": record.get("reason"),
                }
            )
        return record

    def append_transition(self, value: dict[str, Any]) -> dict[str, Any]:
        item = {"transition_seq": self._next_transition_seq, **value}
        self._next_transition_seq += 1
        self.transitions.append(item)
        return item

    def append_control(self, value: dict[str, Any]) -> dict[str, Any]:
        item = {"control_seq": self._next_control_seq, **value}
        self._next_control_seq += 1
        self.controls.append(item)
        return item

    def to_dict(self) -> dict[str, Any]:
        return {
            "next_seq": self._next_seq,
            "next_transition_seq": self._next_transition_seq,
            "next_wait_id": self._next_wait_id,
            "next_queue_wait_id": self._next_queue_wait_id,
            "next_control_seq": self._next_control_seq,
            "records": self.records,
            "transitions": self.transitions,
            "empty_waits": self.empty_waits,
            "queue_waits": self.queue_waits,
            "priority_events": self.priority_events,
            "schedule_rounds": self.schedule_rounds,
            "errors": self.errors,
            "replacements": self.replacements,
            "controls": self.controls,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Ledger":
        ledger = cls()
        ledger.records = list(value.get("records", []))
        ledger.transitions = list(value.get("transitions", []))
        ledger.empty_waits = list(value.get("empty_waits", []))
        ledger.queue_waits = list(value.get("queue_waits", []))
        ledger.priority_events = list(value.get("priority_events", []))
        ledger.schedule_rounds = list(value.get("schedule_rounds", []))
        ledger.errors = list(value.get("errors", []))
        ledger.replacements = list(value.get("replacements", []))
        ledger.controls = list(value.get("controls", []))
        ledger._next_seq = int(value.get("next_seq", len(ledger.records) + 1))
        ledger._next_transition_seq = int(
            value.get("next_transition_seq", len(ledger.transitions) + 1)
        )
        ledger._next_wait_id = int(value.get("next_wait_id", 1))
        ledger._next_queue_wait_id = int(value.get("next_queue_wait_id", 1))
        ledger._next_control_seq = int(value.get("next_control_seq", 1))
        if not ledger._next_wait_id:
            ledger._next_wait_id = 1 + max(
                (int(item.get("wait_id", 0)) for item in ledger.empty_waits), default=0
            )
        if not ledger._next_queue_wait_id:
            ledger._next_queue_wait_id = 1 + max(
                (int(item.get("wait_id", 0)) for item in ledger.queue_waits), default=0
            )
        return ledger

    def next_wait(self) -> int:
        result = self._next_wait_id
        self._next_wait_id += 1
        return result

    def next_queue_wait(self) -> int:
        result = self._next_queue_wait_id
        self._next_queue_wait_id += 1
        return result
