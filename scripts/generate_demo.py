"""Generate the deterministic local demonstration inputs.

Run from the repository root:
    python scripts/generate_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def function(
    function_id: str,
    priority: int,
    memory_mib: int,
    instance_concurrency: int,
    lease_ms: int,
    minimum: int,
    maximum: int,
) -> dict[str, object]:
    return {
        "function_id": function_id,
        "priority": priority,
        "memory_mib": memory_mib,
        "instance_concurrency": instance_concurrency,
        "lease_ms": lease_ms,
        "min_concurrency": minimum,
        "max_concurrency": maximum,
    }


V1 = {
    "schema_version": 1,
    "declaration_version": 1,
    "global_concurrency": 12,
    "preemption_checkpoint_ms": 100,
    "description": "Initial demo declaration: report service owns spare quota.",
    "functions": [
        function("api", 30, 1024, 2, 700, 2, 4),
        function("batch", 20, 512, 1, 900, 1, 4),
        function("report", 10, 256, 1, 1200, 2, 6),
    ],
    "preemptions": [
        {"higher": "api", "lower": "report", "return_deadline_ms": 400},
        {"higher": "batch", "lower": "report", "return_deadline_ms": 500},
    ],
}


V2 = {
    "schema_version": 1,
    "declaration_version": 2,
    "global_concurrency": 12,
    "preemption_checkpoint_ms": 150,
    "description": "Report quota and api borrowing deadline evolved.",
    "functions": [
        function("api", 30, 1024, 2, 700, 2, 4),
        function("batch", 20, 512, 1, 900, 1, 4),
        function("report", 10, 256, 1, 1000, 2, 5),
    ],
    "preemptions": [
        {"higher": "api", "lower": "report", "return_deadline_ms": 600},
        {"higher": "batch", "lower": "report", "return_deadline_ms": 500},
    ],
}


EVENTS = [
    {"event_seq": 1, "arrival_ms": 0, "function_id": "api", "expected_occupancy_ms": 800, "concurrency_demand": 4},
    {"event_seq": 2, "arrival_ms": 0, "function_id": "report", "expected_occupancy_ms": 1000, "concurrency_demand": 3},
    {"event_seq": 3, "arrival_ms": 0, "function_id": "report", "expected_occupancy_ms": 900, "concurrency_demand": 3},
    {"event_seq": 4, "arrival_ms": 100, "function_id": "api", "expected_occupancy_ms": 400, "concurrency_demand": 2},
    {"event_seq": 5, "arrival_ms": 300, "function_id": "batch", "expected_occupancy_ms": 500, "concurrency_demand": 3},
    {"event_seq": 6, "arrival_ms": 350, "function_id": "batch", "expected_occupancy_ms": 400, "concurrency_demand": 2},
    {"event_seq": 7, "arrival_ms": 600, "function_id": "report", "expected_occupancy_ms": 500, "concurrency_demand": 2},
    {"event_seq": 8, "arrival_ms": 700, "function_id": "api", "expected_occupancy_ms": 300, "concurrency_demand": 3},
    {"event_seq": 9, "arrival_ms": 900, "function_id": "batch", "expected_occupancy_ms": 400, "concurrency_demand": 4},
    {"event_seq": 10, "arrival_ms": 1200, "function_id": "report", "expected_occupancy_ms": 600, "concurrency_demand": 2},
    {"event_seq": 11, "arrival_ms": 1500, "function_id": "api", "expected_occupancy_ms": 500, "concurrency_demand": 2},
]


def write() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    (SAMPLES / "declarations_v1.json").write_text(
        json.dumps(V1, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (SAMPLES / "declarations_v2.json").write_text(
        json.dumps(V2, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (SAMPLES / "events.ndjson").open("w", encoding="utf-8", newline="\n") as handle:
        for event in EVENTS:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    session = {
        "watermark_scale": 1000,
        "watermark_rounding": "FLOOR_TOWARD_ZERO",
        "borrow_return_error_ms": 50,
    }
    (SAMPLES / "session.json").write_text(
        json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    write()
    print(f"wrote deterministic samples to {SAMPLES}")

