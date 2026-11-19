"""Deterministically generate a local haptic manifest (no real devices)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def build_manifest(seed: int, strict_sync: bool = False) -> dict:
    rng = random.Random(seed)
    devices = [
        {
            "device_id": "pad-left",
            "sample_rate_hz": 1000,
            "frame_duration_ms": 10,
            "min_intensity": 0.0,
            "max_intensity": 1.0,
            "levels": 16,
            "max_commands_per_frame": 3,
        },
        {
            "device_id": "glove-a",
            "sample_rate_hz": 800,
            "frame_duration_ms": 8,
            "min_intensity": -1.0,
            "max_intensity": 1.0,
            "levels": 32,
            "max_commands_per_frame": 2,
        },
        {
            "device_id": "vest-core",
            "sample_rate_hz": 500,
            "frame_duration_ms": 12,
            "min_intensity": 0.0,
            "max_intensity": 100.0,
            "levels": 8,
            "max_commands_per_frame": 4,
        },
        {
            "device_id": "trigger-r",
            "sample_rate_hz": 2000,
            "frame_duration_ms": 5,
            "min_intensity": 0.0,
            "max_intensity": 1.0,
            "levels": 64,
            "max_commands_per_frame": 5,
        },
    ]
    device_map = {item["device_id"]: item for item in devices}
    event_types = ["envelope", "texture", "tap"]
    events = []
    # Guarantee every device spans both anchor times with anchor-centered runs.
    for device_index, device in enumerate(devices):
        low = device["min_intensity"]
        high = device["max_intensity"]
        for anchor_index, anchor_ms in enumerate((80, 200)):
            start = max(0, anchor_ms - 12)
            end = anchor_ms + 12
            events.append(
                {
                    "event_id": f"evt-anchor-{device_index}-{anchor_index}",
                    "event_type": event_types[anchor_index % len(event_types)],
                    "start_ms": start,
                    "duration_ms": end - start,
                    "target_device": device["device_id"],
                    "curve": [
                        {"time_ms": start, "intensity": low},
                        {"time_ms": anchor_ms, "intensity": low + (high - low) * 0.7},
                        {"time_ms": end, "intensity": low},
                    ],
                }
            )
    cursor = 0
    for index in range(26):
        device = devices[rng.randrange(len(devices))]
        start = cursor + rng.randrange(0, 15)
        duration = rng.choice([6, 10, 14, 20, 28, 40])
        midpoint = start + duration // 2
        low = device["min_intensity"]
        high = device["max_intensity"]
        span = high - low
        peak = low + span * rng.choice([0.4, 0.6, 0.8, 1.0])
        curve = [
            {"time_ms": start, "intensity": low},
            {"time_ms": midpoint, "intensity": peak},
            {"time_ms": start + duration, "intensity": low},
        ]
        events.append(
            {
                "event_id": f"evt-{index:03d}",
                "event_type": event_types[index % len(event_types)],
                "start_ms": start,
                "duration_ms": duration,
                "target_device": device["device_id"],
                "curve": curve,
            }
        )
        cursor = start

    # Two overlapping events on the same device to exercise additive blending.
    target = devices[0]
    events.extend(
        [
            {
                "event_id": "evt-overlap-a",
                "event_type": "envelope",
                "start_ms": 200,
                "duration_ms": 60,
                "target_device": target["device_id"],
                "curve": [
                    {"time_ms": 200, "intensity": 0.0},
                    {"time_ms": 230, "intensity": 0.5},
                    {"time_ms": 260, "intensity": 0.0},
                ],
            },
            {
                "event_id": "evt-overlap-b",
                "event_type": "texture",
                "start_ms": 220,
                "duration_ms": 40,
                "target_device": target["device_id"],
                "curve": [
                    {"time_ms": 220, "intensity": 0.0},
                    {"time_ms": 240, "intensity": 0.4},
                    {"time_ms": 260, "intensity": 0.0},
                ],
            },
        ]
    )
    anchor_devices = [item["device_id"] for item in devices]
    sync_anchors = [
        {
            "anchor_id": "sync-0",
            "time_ms": 80,
            "device_ids": anchor_devices,
            "reference_device": "pad-left",
            "allowed_samples": 1 if strict_sync else 2,
        },
        {
            "anchor_id": "sync-1",
            "time_ms": 200,
            "device_ids": anchor_devices,
            "reference_device": "pad-left",
            "allowed_samples": 1 if strict_sync else 2,
        },
    ]
    return {
        "devices": devices,
        "events": events,
        "sync_anchors": sync_anchors,
        "global": {"concurrency": 3, "inflight_limit": 8},
        "state_dir": "out/state",
        "output_file": "out/commands.json",
        "_note": f"deterministic sample seed={seed} strict_sync={strict_sync}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a deterministic haptic manifest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-sync", action="store_true")
    parser.add_argument("--output", default="samples/manifest.json")
    args = parser.parse_args()
    manifest = build_manifest(args.seed, strict_sync=args.strict_sync)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output} (seed={args.seed}, strict_sync={args.strict_sync})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
