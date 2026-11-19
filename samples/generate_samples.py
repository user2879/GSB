"""Generate deterministic local sample specifications (no external resources)."""

from __future__ import annotations

import json
from pathlib import Path

SAMPLES = {
    "sample_valid.json": {
        "width": 16,
        "height": 14,
        "seed": 20261119,
        "blocked": [
            [0, 0], [1, 0], [0, 1],
            [15, 13], [14, 13], [15, 12],
            [7, 6], [8, 6], [7, 7],
        ],
        "concurrency": 4,
        "memory_budget": 8,
        "max_rollback": 12,
        "alternative_limit": 32,
        "constraints": [
            {
                "id": "rooms",
                "type": "min_rooms",
                "params": {
                    "count": 5,
                    "min_room_w": 2,
                    "min_room_h": 2,
                    "max_room_w": 4,
                    "max_room_h": 4,
                },
            },
            {"id": "spacing", "type": "room_spacing", "params": {"min_distance": 1}},
            {"id": "region_cap", "type": "max_region_area", "params": {"max_area": 120}},
            {"id": "all_connected", "type": "connectivity", "params": {}},
            {
                "id": "start_to_end",
                "type": "reachability",
                "params": {"start": [1, 1], "end": [14, 12]},
            },
        ],
    },
    "sample_conflict.json": {
        "width": 10,
        "height": 10,
        "seed": 7,
        "blocked": [],
        "concurrency": 2,
        "memory_budget": 4,
        "max_rollback": 2,
        "alternative_limit": 16,
        "constraints": [
            {
                "id": "many_rooms",
                "type": "min_rooms",
                "params": {
                    "count": 4,
                    "min_room_w": 3,
                    "min_room_h": 3,
                    "max_room_w": 4,
                    "max_room_h": 4,
                },
            },
            # Connecting everything forces one large region; the tight cap can
            # make the specification unsatisfiable so E_CONFLICT is exercised.
            {"id": "tiny_regions", "type": "max_region_area", "params": {"max_area": 8}},
            {"id": "joined", "type": "connectivity", "params": {}},
        ],
    },
    "sample_reach.json": {
        "width": 12,
        "height": 10,
        "seed": 99,
        "blocked": [[5, 4], [5, 5], [5, 6], [6, 4], [6, 5], [6, 6]],
        "concurrency": 3,
        "memory_budget": 6,
        "max_rollback": 8,
        "alternative_limit": 24,
        "constraints": [
            {
                "id": "rooms",
                "type": "min_rooms",
                "params": {
                    "count": 3,
                    "min_room_w": 2,
                    "min_room_h": 2,
                    "max_room_w": 3,
                    "max_room_h": 3,
                },
            },
            {"id": "spacing", "type": "room_spacing", "params": {"min_distance": 1}},
            {
                "id": "path",
                "type": "reachability",
                "params": {"start": [1, 1], "end": [10, 8]},
            },
        ],
    },
    "sample_invalid.json": {
        "width": 6,
        "height": -2,
        "seed": 3,
        "blocked": [[99, 0]],
        "constraints": [
            {"id": "bad", "type": "unknown_type", "params": {}},
        ],
    },
}


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    for name, payload in SAMPLES.items():
        target = out_dir / name
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
