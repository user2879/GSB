"""Materialized conclusions, their event dependencies, and deterministic replay."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .errors import E_STATE, BusinessError
from .models import (
    Event,
    EventType,
    ItemResult,
    PeerScript,
    SessionConfig,
    canonical_json,
    digest_of,
)
from .state_machine import ApplyResult, SimulationState

ITEM_ORDER: tuple[str, ...] = (
    "connection_state",
    "cwnd",
    "ssthresh",
    "dup_ack_count",
    "sent_unacked",
    "rtt",
    "stats",
    "budget",
    "transitions",
    "diagnosis",
)

ITEM_LABELS: dict[str, str] = {
    "connection_state": "连接状态",
    "cwnd": "拥塞窗口 (bytes)",
    "ssthresh": "慢启动阈值 (bytes)",
    "dup_ack_count": "重复确认计数",
    "sent_unacked": "已发送未确认",
    "rtt": "RTT/RTO 定点估计",
    "stats": "会话统计",
    "budget": "会话预算",
    "transitions": "状态迁移账本",
    "diagnosis": "诊断结论",
}

ITEM_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "connection_state": (),
    "cwnd": (),
    "ssthresh": (),
    "dup_ack_count": (),
    "sent_unacked": (),
    "rtt": (),
    "stats": (),
    "budget": (),
    "transitions": (),
    "diagnosis": (
        "connection_state",
        "cwnd",
        "ssthresh",
        "dup_ack_count",
        "sent_unacked",
        "rtt",
    ),
}

ITEM_EVENT_TYPES: dict[str, frozenset[str]] = {
    "connection_state": frozenset(
        item.value
        for item in EventType
        if item
        in {
            EventType.SYN,
            EventType.SYN_ACK,
            EventType.FIN,
            EventType.FIN_ACK,
            EventType.PEER_FIN,
            EventType.CLOSE,
        }
    ),
    "cwnd": frozenset(
        {EventType.ACK.value, EventType.DUP_ACK.value, EventType.TIMEOUT.value}
    ),
    "ssthresh": frozenset({EventType.DUP_ACK.value, EventType.TIMEOUT.value}),
    "dup_ack_count": frozenset(
        {EventType.ACK.value, EventType.DUP_ACK.value, EventType.TIMEOUT.value}
    ),
    "sent_unacked": frozenset(
        {EventType.SEND.value, EventType.ACK.value, EventType.DUP_ACK.value}
    ),
    "rtt": frozenset({EventType.ACK.value, EventType.TIMEOUT.value}),
    "stats": frozenset(item.value for item in EventType),
    "budget": frozenset(item.value for item in EventType),
    "transitions": frozenset(
        item.value
        for item in {
            EventType.SYN,
            EventType.SYN_ACK,
            EventType.FIN,
            EventType.FIN_ACK,
            EventType.PEER_FIN,
            EventType.CLOSE,
        }
    ),
    "diagnosis": frozenset(item.value for item in EventType),
}


def item_value(state: SimulationState, key: str) -> Any:
    if key == "connection_state":
        return str(state.connection_state)
    if key == "cwnd":
        return state.cwnd
    if key == "ssthresh":
        return state.ssthresh
    if key == "dup_ack_count":
        return state.dup_ack_count
    if key == "sent_unacked":
        return {"bytes": state.bytes_in_flight, "segments": state.segments_in_flight}
    if key == "rtt":
        return state.rtt.as_dict()
    if key == "stats":
        return state.stats_dict()
    if key == "budget":
        return state.budget_dict()
    if key == "transitions":
        return list(state.transitions)
    if key == "diagnosis":
        return state.diagnosis()
    raise KeyError(key)


def _event_dependencies(key: str, events: Iterable[Event]) -> tuple[int, ...]:
    if key == "stats" or key == "budget":
        return tuple(event.event_seq for event in events if event.event_seq > 0)
    if key == "transitions":
        transition_events = {int(item["event_seq"]) for item in []}
        return tuple(sorted(transition_events))
    types = ITEM_EVENT_TYPES[key]
    return tuple(
        event.event_seq
        for event in events
        if event.event_seq > 0 and event.event_type in types
    )


def build_item(
    state: SimulationState,
    key: str,
    *,
    revision: int,
    basis_event_seq: int,
    events: tuple[Event, ...],
    config: SessionConfig,
    script: PeerScript,
) -> ItemResult:
    if key == "transitions":
        dependencies = tuple(
            int(transition["event_seq"])
            for transition in state.transitions
            if int(transition["event_seq"]) > 0
        )
    else:
        dependencies = _event_dependencies(key, events)
    return ItemResult(
        key=key,
        value=item_value(state, key),
        revision=revision,
        basis_event_seq=basis_event_seq,
        depends_on_items=ITEM_DEPENDENCIES[key],
        depends_on_events=dependencies,
        config_version=config.config_version,
        script_version=script.script_version,
        stale=False,
        last_affected_event_seq=basis_event_seq,
    )


def initial_items(
    state: SimulationState, config: SessionConfig, script: PeerScript
) -> dict[str, ItemResult]:
    return {
        key: build_item(
            state,
            key,
            revision=0,
            basis_event_seq=0,
            events=(),
            config=config,
            script=script,
        )
        for key in ITEM_ORDER
    }


def apply_event_to_items(
    state: SimulationState,
    event: Event,
    events: tuple[Event, ...],
    previous: dict[str, ItemResult],
    affected: ApplyResult,
    config: SessionConfig,
    script: PeerScript,
) -> tuple[dict[str, ItemResult], tuple[str, ...]]:
    changed = set(affected.changed_items)
    changed.update({"stats", "budget"})
    result = dict(previous)
    materialized: list[str] = []

    for key in ITEM_ORDER:
        if key == "diagnosis":
            continue
        if key in changed:
            old = previous.get(key)
            revision = 0 if old is None else old.revision + 1
            result[key] = build_item(
                state,
                key,
                revision=revision,
                basis_event_seq=event.event_seq,
                events=events,
                config=config,
                script=script,
            )
            materialized.append(key)

    candidate = state.diagnosis()
    old_diagnosis = previous.get("diagnosis")
    if old_diagnosis is None or canonical_json(old_diagnosis.value) != canonical_json(
        candidate
    ):
        revision = 0 if old_diagnosis is None else old_diagnosis.revision + 1
        result["diagnosis"] = build_item(
            state,
            "diagnosis",
            revision=revision,
            basis_event_seq=event.event_seq,
            events=events,
            config=config,
            script=script,
        )
        materialized.append("diagnosis")

    return result, tuple(sorted(materialized))


def normalize_events(events: Iterable[Event]) -> tuple[Event, ...]:
    source = tuple(events)
    if not source:
        return ()
    assigned = [event.event_seq for event in source]
    if all(seq == 0 for seq in assigned):
        return tuple(
            event.with_seq(index) for index, event in enumerate(source, start=1)
        )
    if any(seq <= 0 for seq in assigned):
        raise BusinessError(
            E_STATE,
            "events must be either all assigned or all unassigned",
            ["event_seq"],
        )
    if assigned != list(range(1, len(source) + 1)):
        raise BusinessError(
            E_STATE, "event sequence must be contiguous and one-based", ["event_seq"]
        )
    return source


def state_snapshot(state: SimulationState) -> dict[str, Any]:
    return {
        "connection_state": str(state.connection_state),
        "cwnd": state.cwnd,
        "ssthresh": state.ssthresh,
        "dup_ack_count": state.dup_ack_count,
        "last_ack_seq": state.last_ack_seq,
        "highest_ack_seq": state.highest_ack_seq,
        "next_send_seq": state.next_send_seq,
        "outstanding": [segment.as_dict() for segment in state.outstanding],
        "rtt": state.rtt.as_dict(),
        "stats": state.stats_dict(),
        "budget": state.budget_dict(),
        "transitions": list(state.transitions),
        "events_processed": state.events_processed,
        "last_event_mono_ms": state.last_event_mono_ms,
    }


def conclusions_bytes(items: dict[str, ItemResult]) -> str:
    return canonical_json(
        {key: items[key].as_dict() for key in ITEM_ORDER if key in items}
    )


@dataclass
class ReplayResult:
    state: SimulationState
    events: tuple[Event, ...]
    items: dict[str, ItemResult]
    checkpoints: tuple[dict[str, Any], ...]

    @property
    def conclusions_digest(self) -> str:
        return digest_of({key: self.items[key].as_dict() for key in ITEM_ORDER})


def replay(
    events: Iterable[Event],
    config: SessionConfig,
    script: PeerScript,
    *,
    capture_checkpoints: bool = True,
) -> ReplayResult:
    normalized = normalize_events(events)
    state = SimulationState(config, script)
    items = initial_items(state, config, script)
    checkpoints: list[dict[str, Any]] = []
    prefix: list[Event] = []
    if capture_checkpoints:
        checkpoints.append(
            {
                "event_seq": 0,
                "state": state_snapshot(state),
                "items": {key: items[key].as_dict() for key in ITEM_ORDER},
                "conclusions_digest": digest_of(
                    {key: items[key].as_dict() for key in ITEM_ORDER}
                ),
            }
        )

    for event in normalized:
        if event.event_seq != state.events_processed + 1:
            raise BusinessError(
                E_STATE, "event sequence must be contiguous", ["event_seq"]
            )
        prefix.append(event)
        affected = state.apply(event)
        items, _ = apply_event_to_items(
            state, event, tuple(prefix), items, affected, config, script
        )
        if capture_checkpoints:
            checkpoints.append(
                {
                    "event_seq": event.event_seq,
                    "state": state_snapshot(state),
                    "items": {key: items[key].as_dict() for key in ITEM_ORDER},
                    "conclusions_digest": digest_of(
                        {key: items[key].as_dict() for key in ITEM_ORDER}
                    ),
                }
            )
    return ReplayResult(
        state=state, events=normalized, items=items, checkpoints=tuple(checkpoints)
    )


def compare_items(
    expected: dict[str, ItemResult], actual: dict[str, ItemResult]
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    for key in sorted(set(expected) | set(actual)):
        left = expected.get(key)
        right = actual.get(key)
        if left is None or right is None:
            differences.append(
                {
                    "field": key,
                    "expected": None if left is None else left.as_dict(),
                    "actual": None if right is None else right.as_dict(),
                }
            )
            continue
        left_value = left.as_dict()
        right_value = right.as_dict()
        for field in sorted(left_value):
            if canonical_json(left_value[field]) != canonical_json(right_value[field]):
                differences.append(
                    {
                        "field": f"{key}.{field}",
                        "expected": left_value[field],
                        "actual": right_value[field],
                    }
                )
    return differences
