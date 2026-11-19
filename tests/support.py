from __future__ import annotations

import copy
import json
from typing import Any

from srebudget.clock import LogicalClock
from srebudget.engine import DiagnosticEngine
from srebudget.loader import load_chain_data, load_trace_lines
from srebudget.models import ChainSpec, TraceStream


def service(
    service_id: str,
    upstream: list[str],
    *,
    timeout_ms: int = 1000,
    window_hours: int = 1,
    ratio_bp: int = 5000,
    min_samples: int = 1,
    failure_threshold: int = 3,
    cooldown_ms: int = 1000,
    half_open_probes: int = 1,
) -> dict[str, Any]:
    return {
        "id": service_id,
        "upstream": upstream,
        "timeout_ms": timeout_ms,
        "budget": {
            "window_hours": window_hours,
            "ratio_bp": ratio_bp,
            "min_samples": min_samples,
        },
        "circuit": {
            "failure_threshold": failure_threshold,
            "cooldown_ms": cooldown_ms,
            "half_open_probes": half_open_probes,
        },
    }


def chain_data(
    services: list[dict[str, Any]], version: str = "test-chain-v1"
) -> dict[str, Any]:
    return {"version": version, "services": services}


def chain(services: list[dict[str, Any]], version: str = "test-chain-v1") -> ChainSpec:
    return load_chain_data(chain_data(services, version))


def trace(
    records: list[dict[str, Any]], chain_spec: ChainSpec
) -> TraceStream:
    lines = [json.dumps(record, separators=(",", ":")) + "\n" for record in records]
    return load_trace_lines(lines, chain_spec)


def call(
    timestamp_ms: int,
    path: list[str],
    outcome: str = "success",
    duration_ms: int = 10,
) -> dict[str, Any]:
    return {
        "timestamp_ms": timestamp_ms,
        "path": path,
        "outcome": outcome,
        "duration_ms": duration_ms,
    }


def drive(
    chain_spec: ChainSpec,
    trace_stream: TraceStream,
    start: int = 1,
    end: int | None = None,
    engine: DiagnosticEngine | None = None,
    clock: LogicalClock | None = None,
) -> tuple[DiagnosticEngine, LogicalClock, list[Any]]:
    clock = clock or LogicalClock(0)
    engine = engine or DiagnosticEngine(chain_spec, clock)
    decisions = []
    stop = len(trace_stream) + 1 if end is None else end
    for item in trace_stream.calls[start - 1 : stop - 1]:
        clock.follow(item.timestamp_ms)
        decisions.append(engine.process_call(item))
    return engine, clock, decisions


def clone(value: Any) -> Any:
    return copy.deepcopy(value)
