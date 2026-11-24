from __future__ import annotations

from copy import deepcopy
from typing import Any

from h2burst.model import SessionSpec, parse_session


def raw_session(
    *,
    streams: list[dict[str, Any]] | None = None,
    events: list[dict[str, Any]] | None = None,
    initial_window: int = 65535,
    initial_stream_window: int = 65535,
    max_concurrent_streams: int = 16,
    max_frame_size: int = 16384,
    scheduling_quantum: int = 3,
    session_id: str = "test-session",
    declared_version: int = 1,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "session_id": session_id,
        "declared_version": declared_version,
        "connection": {
            "initial_window": initial_window,
            "initial_stream_window": initial_stream_window,
            "max_concurrent_streams": max_concurrent_streams,
            "max_frame_size": max_frame_size,
            "scheduling_quantum": scheduling_quantum,
        },
        "streams": deepcopy(
            streams
            if streams is not None
            else [{"stream_id": 1, "priority": 1, "request_body_bytes": 40000}]
        ),
        "events": deepcopy(events if events is not None else []),
    }


def spec_session(**kwargs: Any) -> SessionSpec:
    return parse_session(raw_session(**kwargs))


def wu(seq: int, increment: int, *, stream_id: int | None = None, arrival_ms: int | None = None) -> dict[str, Any]:
    event: dict[str, Any] = {
        "seq": seq,
        "arrival_ms": seq if arrival_ms is None else arrival_ms,
        "type": "window_update",
        "increment": increment,
    }
    if stream_id is None:
        event["scope"] = "connection"
    else:
        event["scope"] = "stream"
        event["stream_id"] = stream_id
    return event


def rst(seq: int, stream_id: int, error_code: int = 8) -> dict[str, Any]:
    return {
        "seq": seq,
        "arrival_ms": seq,
        "type": "rst_stream",
        "stream_id": stream_id,
        "error_code": error_code,
    }


def goaway(seq: int, last_stream_id: int, error_code: int = 0) -> dict[str, Any]:
    return {
        "seq": seq,
        "arrival_ms": seq,
        "type": "goaway",
        "last_stream_id": last_stream_id,
        "error_code": error_code,
    }


def priority(seq: int, stream_id: int, value: int) -> dict[str, Any]:
    return {
        "seq": seq,
        "arrival_ms": seq,
        "type": "priority_change",
        "stream_id": stream_id,
        "priority": value,
    }


def strategy(seq: int, value: str) -> dict[str, Any]:
    return {
        "seq": seq,
        "arrival_ms": seq,
        "type": "strategy_change",
        "strategy": value,
    }
