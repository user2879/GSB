"""Generate the deterministic local demonstration sessions.

Run from the repository root:

    python tools/generate_samples.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def write_json(name: str, payload: dict) -> None:
    path = SAMPLES / name
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def minimal() -> dict:
    return {
        "format_version": 1,
        "session_id": "minimal-two-stream",
        "declared_version": 1,
        "notes": "Small deterministic round-robin, state, and flow-ledger sample.",
        "connection": {
            "initial_window": 65535,
            "initial_stream_window": 65535,
            "max_concurrent_streams": 10,
            "max_frame_size": 16384,
            "scheduling_quantum": 3,
        },
        "streams": [
            {"stream_id": 1, "priority": 1, "request_body_bytes": 50000},
            {"stream_id": 3, "priority": 2, "request_body_bytes": 22000},
        ],
        "events": [
            {"seq": 0, "arrival_ms": 0, "type": "window_update", "scope": "connection", "increment": 10000},
            {"seq": 1, "arrival_ms": 4, "type": "window_update", "scope": "stream", "stream_id": 1, "increment": 8000},
            {"seq": 2, "arrival_ms": 8, "type": "window_update", "scope": "stream", "stream_id": 3, "increment": 6000},
            {"seq": 3, "arrival_ms": 12, "type": "strategy_change", "strategy": "proportional"},
            {"seq": 4, "arrival_ms": 16, "type": "priority_change", "stream_id": 3, "priority": 1},
            {"seq": 5, "arrival_ms": 20, "type": "rst_stream", "stream_id": 3, "error_code": 8},
        ],
    }


def burst_demo() -> dict:
    rng = np.random.default_rng(20261124)
    stream_ids = [2 * index + 1 for index in range(28)]
    priorities = rng.integers(1, 9, size=len(stream_ids))
    bodies = rng.integers(30_000, 90_001, size=len(stream_ids))
    streams = [
        {
            "stream_id": stream_id,
            "priority": int(priority),
            "request_body_bytes": int(body),
        }
        for stream_id, priority, body in zip(stream_ids, priorities, bodies)
    ]

    script: list[dict] = []

    def add(arrival_ms: int, event_type: str, **fields: object) -> None:
        script.append({"seq": len(script), "arrival_ms": arrival_ms, "type": event_type, **fields})

    add(0, "window_update", scope="connection", increment=393_216)
    add(2, "window_update", scope="stream", stream_id=1, increment=65_536)
    add(4, "window_update", scope="stream", stream_id=3, increment=98_304)
    add(6, "strategy_change", strategy="proportional")
    add(8, "window_update", scope="stream", stream_id=5, increment=131_072)
    add(10, "priority_change", stream_id=5, priority=1)
    add(12, "priority_change", stream_id=7, priority=2)
    add(14, "window_update", scope="connection", increment=393_216)
    add(16, "window_update", scope="stream", stream_id=9, increment=65_536)
    add(18, "strategy_change", strategy="round_robin")
    add(20, "window_update", scope="stream", stream_id=11, increment=196_608)
    add(22, "window_update", scope="stream", stream_id=13, increment=65_536)
    add(24, "rst_stream", stream_id=27, error_code=8)
    add(26, "window_update", scope="connection", increment=524_288)
    add(28, "strategy_change", strategy="proportional")
    add(30, "window_update", scope="stream", stream_id=15, increment=65_536)
    add(32, "window_update", scope="stream", stream_id=17, increment=131_072)
    add(34, "priority_change", stream_id=19, priority=1)
    add(36, "strategy_change", strategy="round_robin")
    add(38, "window_update", scope="connection", increment=524_288)
    add(40, "window_update", scope="stream", stream_id=21, increment=65_536)
    add(42, "window_update", scope="stream", stream_id=23, increment=131_072)
    add(44, "window_update", scope="stream", stream_id=25, increment=65_536)
    add(46, "strategy_change", strategy="proportional")
    add(48, "priority_change", stream_id=3, priority=1)
    add(50, "window_update", scope="stream", stream_id=7, increment=262_144)
    add(52, "window_update", scope="connection", increment=524_288)
    add(54, "window_update", scope="stream", stream_id=29, increment=131_072)
    add(56, "strategy_change", strategy="round_robin")
    add(58, "window_update", scope="stream", stream_id=31, increment=65_536)
    add(60, "window_update", scope="connection", increment=524_288)
    add(62, "window_update", scope="stream", stream_id=33, increment=131_072)
    add(64, "priority_change", stream_id=9, priority=1)
    add(66, "strategy_change", strategy="proportional")
    add(68, "window_update", scope="connection", increment=524_288)
    add(70, "goaway", last_stream_id=45, error_code=0)

    return {
        "format_version": 1,
        "session_id": "burst-demo-28-streams",
        "declared_version": 1,
        "notes": "Fixed-seed burst windows, strategy changes, RST_STREAM, priority changes, and GOAWAY.",
        "connection": {
            "initial_window": 65535,
            "initial_stream_window": 65535,
            "max_concurrent_streams": 64,
            "max_frame_size": 16384,
            "scheduling_quantum": 5,
        },
        "streams": streams,
        "events": script,
    }


def scale_1000() -> dict:
    rng = np.random.default_rng(10002026)
    stream_ids = [2 * index + 1 for index in range(1000)]
    priorities = rng.integers(1, 17, size=len(stream_ids))
    bodies = rng.integers(20_000, 120_001, size=len(stream_ids))
    streams = [
        {
            "stream_id": stream_id,
            "priority": int(priority),
            "request_body_bytes": int(body),
        }
        for stream_id, priority, body in zip(stream_ids, priorities, bodies)
    ]
    script: list[dict] = []

    def add(event_type: str, **fields: object) -> None:
        seq = len(script)
        arrival = seq * 3 + int(rng.integers(0, 3))
        script.append({"seq": seq, "arrival_ms": arrival, "type": event_type, **fields})

    add("window_update", scope="connection", increment=1_048_576)
    burst_streams = [1, 3, 5, 9, 15, 21, 33, 45, 61, 77, 93, 111, 129, 147, 165, 181]
    event_cursor = 0
    while len(script) < 108:
        stage = len(script)
        if stage % 6 == 0:
            add("window_update", scope="connection", increment=262_144)
        elif stage % 11 == 0:
            stream_id = stream_ids[(event_cursor * 37 + 101) % len(stream_ids)]
            add("priority_change", stream_id=stream_id, priority=int(rng.integers(1, 17)))
        elif stage % 17 == 0 and stage > 40:
            stream_id = stream_ids[-1 - (event_cursor % 20)]
            add("rst_stream", stream_id=stream_id, error_code=8)
        else:
            stream_id = burst_streams[event_cursor % len(burst_streams)]
            increment = int(rng.choice(np.array([65_536, 98_304, 131_072, 196_608, 262_144])))
            add("window_update", scope="stream", stream_id=stream_id, increment=increment)
        event_cursor += 1
    add("goaway", last_stream_id=1901, error_code=0)

    # The loop above targets 108 events; append/truncate deterministically so the
    # final GOAWAY is exactly sequence 108.
    if len(script) > 109:
        script = script[:108]
        add("goaway", last_stream_id=1901, error_code=0)
    if len(script) < 109:
        while len(script) < 108:
            add("window_update", scope="connection", increment=131_072)
        add("goaway", last_stream_id=1901, error_code=0)
    for index, event in enumerate(script):
        event["seq"] = index

    return {
        "format_version": 1,
        "session_id": "scale-1000-streams",
        "declared_version": 1,
        "notes": "Fixed NumPy seed; 1000 streams exercise scale selection and starvation diagnostics.",
        "connection": {
            "initial_window": 65535,
            "initial_stream_window": 65535,
            "max_concurrent_streams": 1000,
            "max_frame_size": 16384,
            "scheduling_quantum": 12,
        },
        "streams": streams,
        "events": script,
    }


def main() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    write_json("minimal.json", minimal())
    write_json("burst_demo.json", burst_demo())
    write_json("scale_1000.json", scale_1000())
    print("generated", ", ".join(sorted(path.name for path in SAMPLES.glob("*.json"))))


if __name__ == "__main__":
    main()
