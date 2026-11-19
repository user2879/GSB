"""Deterministic event scripts and loopback TCP transcript-to-event runner."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .clock import ManualClock
from .errors import E_CONFIG, BusinessError
from .models import Event, EventType
from .peer import DeterministicPeer, send_batch
from .service import IngestionResult, SessionService


def load_event_script(path: str | Path) -> tuple[Event, ...]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BusinessError(
            E_CONFIG, f"cannot load event script: {exc}", ["events"]
        ) from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("events"), list):
        raise BusinessError(
            E_CONFIG, "event script requires an events list", ["events"]
        )
    events: list[Event] = []
    known = {item.value for item in EventType}
    previous_ms = -1
    for index, item in enumerate(raw["events"]):
        if not isinstance(item, dict):
            raise BusinessError(
                E_CONFIG, "event must be an object", [f"events[{index}]"]
            )
        event_type = item.get("event_type")
        if event_type not in known:
            raise BusinessError(
                E_CONFIG, "unknown event type", [f"events[{index}].event_type"]
            )
        try:
            event = Event.from_dict({**item, "event_seq": 0})
        except (TypeError, ValueError) as exc:
            raise BusinessError(
                E_CONFIG, f"invalid event: {exc}", [f"events[{index}]"]
            ) from exc
        if event.recorded_mono_ms < previous_ms:
            raise BusinessError(
                E_CONFIG,
                "event clocks must be monotonic",
                [f"events[{index}].recorded_mono_ms"],
            )
        previous_ms = event.recorded_mono_ms
        events.append(event)
    return tuple(events)


def ingest_events(
    service: SessionService, events: Iterable[Event]
) -> tuple[IngestionResult, ...]:
    results: list[IngestionResult] = []
    for event in events:
        results.append(service.ingest_event(event))
    return tuple(results)


@dataclass(frozen=True)
class LoopbackTranscript:
    handshake_events: tuple[Event, ...]
    data_events: tuple[Event, ...]
    peer_responses: tuple[dict[str, Any], ...]
    dropped_seqs: tuple[int, ...]
    rounds: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "handshake_events": [event.as_dict() for event in self.handshake_events],
            "data_events": [event.as_dict() for event in self.data_events],
            "peer_responses": list(self.peer_responses),
            "dropped_seqs": list(self.dropped_seqs),
            "rounds": self.rounds,
        }


def run_loopback_session(
    service: SessionService,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    send_segments: int = 6,
    max_rounds: int = 12,
    close_mode: str = "close_wait",
) -> tuple[LoopbackTranscript, DeterministicPeer]:
    """Drive a real loopback peer and feed its transcript into the state engine."""

    if send_segments < 1:
        raise ValueError("send_segments must be positive")
    clock = ManualClock(service.events[-1].recorded_mono_ms if service.events else 0)

    def emit(event_type: str | EventType, **kwargs: Any) -> Event:
        clock.advance_ms(1)
        return service.ingest_event(
            Event(event_type=str(event_type), recorded_mono_ms=clock.now_ms(), **kwargs)
        ).event

    peer = DeterministicPeer(service.script, host=host, port=port).start()
    handshake_events: list[Event] = []
    data_events: list[Event] = []
    peer_responses: list[dict[str, Any]] = []
    dropped: list[int] = []
    try:
        if (
            service.state.connection_state.value == "CLOSED"
            and service.state.events_processed == 0
        ):
            handshake_events.append(emit(EventType.SYN, seq=service.config.initial_seq))
            handshake_events.append(
                emit(
                    EventType.SYN_ACK,
                    seq=service.config.initial_seq,
                    ack_seq=service.config.first_data_seq,
                )
            )

        remaining = min(
            send_segments,
            service.config.budget.max_sent_bytes // service.config.mss,
            service.config.budget.max_events - service.state.events_processed,
        )
        rounds = 0
        next_seq = service.state.next_send_seq
        while remaining > 0 and rounds < max_rounds:
            rounds += 1
            available_bytes = max(
                service.config.mss, service.state.cwnd - service.state.bytes_in_flight
            )
            available_bytes = min(available_bytes, service.config.window_limit)
            slots = max(1, available_bytes // service.config.mss)
            count = min(remaining, slots)
            packets = [
                {
                    "seq": next_seq + index * service.config.mss,
                    "payload_bytes": service.config.mss,
                    "client_mono_ms": clock.now_ms(),
                }
                for index in range(count)
            ]
            for packet in packets:
                data_events.append(
                    emit(
                        EventType.SEND,
                        seq=packet["seq"],
                        payload_bytes=packet["payload_bytes"],
                    )
                )
            result = send_batch(host, peer.port, packets)
            peer_responses.extend(result.as_dict()["responses"])
            dropped.extend(result.dropped_seqs)
            for response in result.responses:
                if response["action"] == "ack":
                    data_events.append(
                        emit(
                            EventType.ACK,
                            ack_seq=response["ack_seq"],
                            rtt_sample_ms=max(0, int(response.get("delay_ms", 0))),
                            note=f"arrival_index={response.get('arrival_index')}",
                        )
                    )
                elif response["action"] == "dupack":
                    data_events.append(
                        emit(
                            EventType.DUP_ACK,
                            ack_seq=response["ack_seq"],
                            note=f"reordered={response.get('reordered', False)}",
                        )
                    )
                else:
                    raise BusinessError(
                        E_CONFIG,
                        f"unsupported peer action: {response['action']}",
                        ["peer_script.action"],
                    )
            for seq in result.dropped_seqs:
                data_events.append(emit(EventType.TIMEOUT, seq=seq))
            next_seq += count * service.config.mss
            remaining -= count
            if not result.responses and result.dropped_seqs:
                # A deterministic dropped batch cannot advance without a rerun; leave the
                # timeout state visible rather than fabricating an acknowledgement.
                break

        if service.state.connection_state.value == "ESTABLISHED":
            fin_seq = service.state.next_send_seq
            handshake_events.append(emit(EventType.FIN, seq=fin_seq))
            if close_mode == "close_wait":
                handshake_events.append(emit(EventType.PEER_FIN, seq=fin_seq))
                handshake_events.append(emit(EventType.CLOSE, seq=fin_seq))
            elif close_mode == "fin_ack":
                handshake_events.append(
                    emit(EventType.FIN_ACK, seq=fin_seq, ack_seq=fin_seq)
                )
            else:
                raise ValueError("close_mode must be close_wait or fin_ack")
    finally:
        peer.stop()

    return (
        LoopbackTranscript(
            handshake_events=tuple(handshake_events),
            data_events=tuple(data_events),
            peer_responses=tuple(peer_responses),
            dropped_seqs=tuple(dropped),
            rounds=rounds,
        ),
        peer,
    )
