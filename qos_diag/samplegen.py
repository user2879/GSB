from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import atomic_write_json
from .model import STAGES


DEFAULT_CONFIG: dict[str, Any] = {
    "format": "qos-diag/config@1",
    "scale": 2,
    "window_ms": 40,
    "slide_ms": 20,
    "global_max_inflight": 5,
    "pipeline_max_inflight": 2,
    "call_timeout_ms": 1000,
    "batch_size": 6,
    "selection_deadband_work": 8,
}


def build_deterministic_bundle_data(seed: int = 287) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    pipeline_count = 3
    events_per_pipeline = 12
    pipelines: list[dict[str, Any]] = []
    shapers: list[dict[str, Any]] = []
    port = 43000 + (seed % 100) * 10
    for p_index in range(1, pipeline_count + 1):
        pipeline_id = f"p-{p_index:02d}"
        ids = {
            stage: f"sh-{p_index:02d}-{stage}"
            for stage in STAGES
        }
        pipelines.append({"pipeline_id": pipeline_id, "stage_shaper_ids": ids})
        for stage_index, stage in enumerate(STAGES):
            port += 1
            shapers.append(
                {
                    "shaper_id": ids[stage],
                    "pipeline_id": pipeline_id,
                    "stage": stage,
                    "host": "127.0.0.1",
                    "port": port,
                    "refill_rate_per_ms": int(rng.integers(1, 4)),
                    "burst_capacity_tokens": int(rng.integers(8, 18)),
                    "service_delay_ms": 0,
                }
            )

    events: list[dict[str, Any]] = []
    arrival_ms = 0
    total = pipeline_count * events_per_pipeline
    for seq in range(total):
        if seq:
            arrival_ms += int(rng.integers(1, 7))
        pipeline_index = seq % pipeline_count
        events.append(
            {
                "event_id": f"ev-{seq:03d}",
                "pipeline_id": f"p-{pipeline_index + 1:02d}",
                "arrival_ms": arrival_ms,
                "packet_length": int(rng.integers(6, 17)),
            }
        )
    return {
        "config": copy.deepcopy(DEFAULT_CONFIG),
        "pipelines_raw": {"format": "qos-diag/pipelines@1", "pipelines": pipelines},
        "shapers_raw": {"format": "qos-diag/shapers@1", "shapers": shapers},
        "events_raw": {"format": "qos-diag/events@1", "events": events},
    }


def write_sample_dir(target: str | Path, data: dict[str, Any]) -> Path:
    root = Path(target)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "config.json", data["config"])
    atomic_write_json(root / "pipelines.json", data["pipelines_raw"])
    atomic_write_json(root / "shapers.json", data["shapers_raw"])
    atomic_write_json(root / "events.json", data["events_raw"])
    return root


def generate_all_samples(workspace: str | Path, seed: int = 287) -> dict[str, Path]:
    workspace = Path(workspace)
    base_dir = workspace / "samples" / "base"
    data = build_deterministic_bundle_data(seed)
    write_sample_dir(base_dir, data)

    param_data = copy.deepcopy(data)
    p02_relay = next(
        item
        for item in param_data["shapers_raw"]["shapers"]
        if item["pipeline_id"] == "p-02" and item["stage"] == "relay"
    )
    old_rate = p02_relay["refill_rate_per_ms"]
    p02_relay["refill_rate_per_ms"] = old_rate + 2
    param_dir = workspace / "samples" / "edited_param"
    write_sample_dir(param_dir, param_data)
    atomic_write_json(
        param_dir / "edit.json",
        {
            "edit_type": "shaper_rate",
            "pipeline_id": "p-02",
            "stage": "relay",
            "shaper_id": p02_relay["shaper_id"],
            "field": "refill_rate_per_ms",
            "old_value": old_rate,
            "new_value": p02_relay["refill_rate_per_ms"],
        },
    )

    arrival_data = copy.deepcopy(data)
    events = arrival_data["events_raw"]["events"]
    target_seq = None
    for seq in range(4, len(events) - 2):
        if events[seq + 1]["arrival_ms"] - events[seq]["arrival_ms"] >= 3:
            target_seq = seq
            break
    if target_seq is None:
        target_seq = 8
    old_arrival = events[target_seq]["arrival_ms"]
    events[target_seq]["arrival_ms"] = old_arrival + 2
    arrival_dir = workspace / "samples" / "edited_arrival"
    write_sample_dir(arrival_dir, arrival_data)
    atomic_write_json(
        arrival_dir / "edit.json",
        {
            "edit_type": "event_arrival",
            "event_seq": target_seq,
            "event_id": events[target_seq]["event_id"],
            "field": "arrival_ms",
            "old_value": old_arrival,
            "new_value": events[target_seq]["arrival_ms"],
            "delta_ms": 2,
        },
    )

    burst_data = copy.deepcopy(data)
    p03_egress = next(
        item
        for item in burst_data["shapers_raw"]["shapers"]
        if item["pipeline_id"] == "p-03" and item["stage"] == "egress"
    )
    old_burst = p03_egress["burst_capacity_tokens"]
    p03_egress["burst_capacity_tokens"] = old_burst + 4
    burst_dir = workspace / "samples" / "edited_burst"
    write_sample_dir(burst_dir, burst_data)
    atomic_write_json(
        burst_dir / "edit.json",
        {
            "edit_type": "shaper_burst",
            "pipeline_id": "p-03",
            "stage": "egress",
            "shaper_id": p03_egress["shaper_id"],
            "field": "burst_capacity_tokens",
            "old_value": old_burst,
            "new_value": p03_egress["burst_capacity_tokens"],
        },
    )
    return {"base": base_dir, "edited_param": param_dir, "edited_arrival": arrival_dir, "edited_burst": burst_dir}
