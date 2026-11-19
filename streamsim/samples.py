"""Deterministic local sample generation."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .canon import atomic_write_json


DEFAULT_SEED = 20261119
FRAGMENT_COUNT = 20
FRAGMENT_DURATION_MS = 2000
BITRATE_LADDER = (400_000, 700_000, 1_100_000, 1_600_000, 2_300_000)


def generate_samples(destination: str | Path, *, seed: int = DEFAULT_SEED) -> dict:
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    trace_bandwidths = [1_800_000, 1_250_000, 2_400_000, 950_000, 2_900_000, 3_600_000]
    trace_rtts = [24, 38, 18, 55, 22, 12]
    network = {
        "segments": [
            {"bandwidth_bps": int(bandwidth), "rtt_ms": int(rtt)}
            for bandwidth, rtt in zip(trace_bandwidths, trace_rtts)
        ]
    }

    fragments = []
    for segment_index in range(FRAGMENT_COUNT):
        representations = []
        for rep_index, bitrate in enumerate(BITRATE_LADDER):
            base_size = bitrate * FRAGMENT_DURATION_MS // 8000
            jitter_percent = int(rng.integers(-4, 5))
            size = max(1, base_size * (100 + jitter_percent) // 100)
            representations.append(
                {"bitrate_bps": bitrate, "size_bytes": size}
            )
        fragments.append(
            {
                "index": segment_index,
                "duration_ms": FRAGMENT_DURATION_MS,
                "representations": representations,
            }
        )
    media = {"fragments": fragments}

    events = {
        "events": [
            {"at_ms": 8_000, "action": "pause", "order": 0},
            {"at_ms": 10_000, "action": "resume", "order": 1},
            {"at_ms": 14_000, "action": "trace_switch", "segment_index": 3, "order": 2},
            {"at_ms": 22_000, "action": "jump", "target_ms": 6_000, "order": 3},
            {"at_ms": 30_000, "action": "trace_switch", "segment_index": 4, "order": 4},
            {"at_ms": 36_000, "action": "jump", "target_ms": 24_000, "order": 5},
        ]
    }
    interrupt_events = {
        "events": list(events["events"])
        + [{"at_ms": 15_000, "action": "terminate", "order": 6}]
    }
    interrupt_events["events"].sort(key=lambda item: (item["at_ms"], item["order"]))

    atomic_write_json(root / "network.json", network)
    atomic_write_json(root / "media.json", media)
    atomic_write_json(root / "events.json", events)
    atomic_write_json(root / "events_interrupt.json", interrupt_events)

    base_client = {
        "startup_buffer_ms": 4_000,
        "max_buffer_ms": 10_000,
        "safety_margin_ms": 1_200,
        "strategy": "hybrid",
        "switch_penalty": 0.25,
        "concurrency": 2,
        "memory_budget_bytes": 12_000_000,
        "throughput_safety_ratio": 0.88,
    }

    def write_session(
        name: str,
        event_file: str,
        state_path: str,
        snapshot_path: str,
        client: dict,
    ) -> None:
        atomic_write_json(
            root / name,
            {
                "network_trace": "network.json",
                "media_manifest": "media.json",
                "event_script": event_file,
                "client": client,
                "state_dir": state_path,
                "snapshot_dir": snapshot_path,
            },
        )

    write_session(
        "session.json",
        "events.json",
        "sample_runs/full/state",
        "sample_runs/full/snapshots",
        base_client,
    )
    write_session(
        "session_interrupt.json",
        "events_interrupt.json",
        "sample_runs/interrupt/state",
        "sample_runs/interrupt/snapshots",
        base_client,
    )
    c1_client = dict(base_client)
    c1_client["concurrency"] = 1
    write_session(
        "session_concurrency1.json",
        "events.json",
        "sample_runs/concurrency1/state",
        "sample_runs/concurrency1/snapshots",
        c1_client,
    )

    manifest = {
        "seed": seed,
        "root": str(root),
        "sessions": {
            "continuous": str(root / "session.json"),
            "interrupt": str(root / "session_interrupt.json"),
            "concurrency1": str(root / "session_concurrency1.json"),
        },
    }
    atomic_write_json(root / "samples_manifest.json", manifest)
    return manifest


def clean_sample_runs(destination: str | Path) -> None:
    target = Path(destination).resolve() / "sample_runs"
    if target.exists():
        shutil.rmtree(target)
