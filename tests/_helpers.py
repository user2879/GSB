"""Shared deterministic manifest construction helpers for the test suite."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from haptic_engine.parser import parse_manifest  # noqa: E402


def base_manifest(tmp_root: str) -> dict:
    return {
        "devices": [
            {
                "device_id": "d-a",
                "sample_rate_hz": 1000,
                "frame_duration_ms": 10,
                "min_intensity": 0.0,
                "max_intensity": 1.0,
                "levels": 16,
                "max_commands_per_frame": 2,
            },
            {
                "device_id": "d-b",
                "sample_rate_hz": 800,
                "frame_duration_ms": 5,
                "min_intensity": -1.0,
                "max_intensity": 1.0,
                "levels": 8,
                "max_commands_per_frame": 3,
            },
        ],
        "events": [
            {
                "event_id": "e1",
                "event_type": "envelope",
                "start_ms": 0,
                "duration_ms": 100,
                "target_device": "d-a",
                "curve": [
                    {"time_ms": 0, "intensity": 0.0},
                    {"time_ms": 50, "intensity": 1.0},
                    {"time_ms": 100, "intensity": 0.0},
                ],
            },
            {
                "event_id": "e2",
                "event_type": "tap",
                "start_ms": 20,
                "duration_ms": 40,
                "target_device": "d-b",
                "curve": [
                    {"time_ms": 20, "intensity": 0.0},
                    {"time_ms": 40, "intensity": 1.0},
                    {"time_ms": 60, "intensity": 0.0},
                ],
            },
        ],
        "sync_anchors": [
            {
                "anchor_id": "a1",
                "time_ms": 50,
                "device_ids": ["d-a", "d-b"],
                "reference_device": "d-a",
                "allowed_samples": 2,
            }
        ],
        "global": {"concurrency": 2, "inflight_limit": 4},
        "state_dir": f"{tmp_root}/state",
        "output_file": f"{tmp_root}/commands.json",
    }


def write_manifest(tmp_root: Path, payload: dict | None = None) -> Path:
    tmp_root.mkdir(parents=True, exist_ok=True)
    payload = payload or base_manifest(str(tmp_root))
    path = tmp_root / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def parsed(tmp_root: Path, payload: dict | None = None):
    path = write_manifest(tmp_root, payload)
    return parse_manifest(json.loads(path.read_text(encoding="utf-8")))


def mutate(payload: dict) -> dict:
    return copy.deepcopy(payload)
