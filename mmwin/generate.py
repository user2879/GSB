from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import atomic_write_json, canonical_json
from .png import write_png_rgb


DEFAULT_SEED = 20261124


def _atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")
    temp.replace(path)


def _delay(seed: int, event_id: str) -> int:
    digest = hashlib.sha256(f"delay:{seed}:{event_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 1400


def _permutation(seed: int, event_ids: list[str]) -> list[str]:
    return sorted(
        event_ids,
        key=lambda event_id: hashlib.sha256(
            f"arrival:{seed}:{event_id}".encode("utf-8")
        ).digest(),
    )


def _frame_image(seed: int, index: int) -> np.ndarray:
    rng = np.random.default_rng(seed + index * 7919)
    height, width = 96, 128
    yy, xx = np.mgrid[0:height, 0:width]
    background = np.stack(
        [
            (24 + xx * 0.55 + index * 3) % 180,
            (46 + yy * 1.2) % 190,
            (70 + (xx + yy) * 0.3) % 170,
        ],
        axis=2,
    )
    image = background.astype(np.float64)
    palette = [
        (235, 74, 58),
        (42, 157, 143),
        (244, 211, 94),
        (91, 104, 210),
        (230, 126, 34),
    ]
    for shape in range(4):
        colour = np.array(palette[(index + shape) % len(palette)], dtype=np.float64)
        x0 = int(rng.integers(0, width - 18))
        y0 = int(rng.integers(0, height - 15))
        w = int(rng.integers(10, 38))
        h = int(rng.integers(8, 28))
        if shape % 2 == 0:
            image[y0 : min(height, y0 + h), x0 : min(width, x0 + w)] = colour
        else:
            cx = x0 + w / 2
            cy = y0 + h / 2
            radius = max(4.0, min(w, h) / 2)
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
            image[mask] = colour
    for _ in range(2):
        x0 = int(rng.integers(0, width))
        y0 = int(rng.integers(0, height))
        x1 = int(rng.integers(0, width))
        y1 = int(rng.integers(0, height))
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        xs = np.linspace(x0, x1, steps + 1).astype(int)
        ys = np.linspace(y0, y1, steps + 1).astype(int)
        image[ys, xs] = (245, 245, 245)
    noise = rng.normal(0.0, 7.0, size=image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def generate_samples(sample_dir: str | Path, *, seed: int = DEFAULT_SEED, clean: bool = True) -> dict[str, Any]:
    sample_dir = Path(sample_dir)
    if clean and sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = sample_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    frame_times = [
        60,
        330,
        760,
        1120,
        1510,
        1960,
        2410,
        2870,
        3360,
        3890,
        4140,
        6010,
        6500,
        6980,
        7440,
        7860,
    ]
    frame_records: list[dict[str, Any]] = []
    for index, timestamp in enumerate(frame_times):
        image = _frame_image(seed, index)
        filename = f"frame_{index:04d}_t{timestamp:08d}.png"
        relative = Path("frames") / filename
        write_png_rgb(sample_dir / relative, image)
        frame_records.append(
            {
                "event_id": f"img-{index:04d}",
                "timestamp_ms": timestamp,
                "frame_index": index,
                "path": relative.as_posix(),
            }
        )

    text_attributes = [
        ("vehicle", "lane_state", "clear"),
        ("vehicle", "brake_state", "active"),
        ("pedestrian", "zone_state", "occupied"),
        ("signal", "phase", "green"),
        ("weather", "visibility", "good"),
        ("road", "surface", "dry"),
    ]
    text_values = ["clear", "active", "occupied", "green", "degraded", "blocked"]
    text_records: list[dict[str, Any]] = []
    for index in range(34):
        timestamp = int(90 + index * 235 + rng.integers(0, 150))
        entity_type, attribute, default_value = text_attributes[index % len(text_attributes)]
        value = default_value if index % 4 else text_values[(index + 2) % len(text_values)]
        polarity = float(np.clip(rng.normal(0.25, 0.65), -1.0, 1.0))
        text_records.append(
            {
                "event_id": f"txt-{index:04d}",
                "timestamp_ms": timestamp,
                "entity": {
                    "id": f"{entity_type}-{index % 7:02d}",
                    "type": entity_type,
                },
                "attribute": attribute,
                "value": value,
                "confidence": round(float(rng.uniform(0.55, 1.0)), 6),
                "polarity": round(polarity, 6),
            }
        )

    channels = {
        "temperature": (-20.0, 120.0),
        "vibration": (0.0, 10.0),
        "pressure": (0.0, 200.0),
    }
    sensor_records: list[dict[str, Any]] = []
    for index in range(40):
        timestamp = int(70 + index * 198 + rng.integers(0, 120))
        channel = list(channels)[index % len(channels)]
        low, high = channels[channel]
        if channel == "temperature":
            value = 18.0 + rng.normal(0, 8) + (index % 5) * 3
        elif channel == "vibration":
            value = 1.2 + abs(rng.normal(0, 1.1)) + (index % 3) * 0.2
        else:
            value = 85.0 + rng.normal(0, 15) + (index % 4) * 2
        value = float(np.clip(value, low + 0.001, high - 0.001))
        sensor_records.append(
            {
                "event_id": f"sen-{index:04d}",
                "timestamp_ms": timestamp,
                "channel": channel,
                "value": round(value, 6),
            }
        )

    all_records = frame_records + text_records + sensor_records
    all_events = {
        record["event_id"]: record for record in all_records
    }
    base_order = _permutation(seed, list(all_events))
    # Move the oldest events to the end so that the already advanced watermark
    # closes their windows before they arrive.  This is a genuine late-arrival
    # scenario, not a timestamp pre-sort.
    low_ids = sorted(
        all_events,
        key=lambda event_id: (all_events[event_id]["timestamp_ms"], event_id),
    )[:9]
    low_set = set(low_ids)
    ordered = [event_id for event_id in base_order if event_id not in low_set]
    latest_id = max(
        ordered,
        key=lambda event_id: (all_events[event_id]["timestamp_ms"], event_id),
    )
    ordered.remove(latest_id)
    ordered.append(latest_id)
    ordered.extend(low_ids)
    arrivals = []
    for sequence, event_id in enumerate(ordered):
        arrivals.append(
            {
                "arrival_seq": sequence,
                "event_id": event_id,
                "delay_ms": _delay(seed, event_id),
            }
        )

    config = {
        "schema_version": 1,
        "window_length_ms": 1200,
        "window_step_ms": 600,
        "allowed_lateness_ms": 400,
        "watermark_rule": "max_seen_minus_lateness",
        "late_policy": "resettle",
        "visual_feature_dim": 12,
        "text_feature_dim": 16,
        "visual_block_rows": 4,
        "visual_block_cols": 3,
        "visual_frames_per_window": 3,
        "quantization_level": "Q8",
        "fractional_bits": 8,
        "quantization_levels": {"Q6": 6, "Q8": 8, "Q10": 10, "Q12": 12},
        "fusion_temperature": 0.75,
        "thread_pool_size": 4,
        "concurrency_limit": 8,
        "max_events_per_window": 64,
        "checkpoint_commit_interval_windows": 2,
        "batch_event_budget": 7,
        "sensor_channels": {
            name: {"min": low, "max": high}
            for name, (low, high) in channels.items()
        },
        "frame_width": 128,
        "frame_height": 96,
    }
    plan = {
        "schema_version": 1,
        "seed": seed,
        "arrivals": arrivals,
    }
    duplicate_plan = json.loads(json.dumps(plan))
    duplicate_plan["arrivals"].append(
        {
            "arrival_seq": len(duplicate_plan["arrivals"]),
            "event_id": high_timestamp_event_id(all_events),
            "delay_ms": 0,
        }
    )
    drop_config = json.loads(json.dumps(config))
    drop_config["late_policy"] = "drop"
    q12_config = json.loads(json.dumps(config))
    q12_config["quantization_level"] = "Q12"
    q12_config["fractional_bits"] = 12

    _atomic_write_jsonl(sample_dir / "frames.jsonl", frame_records)
    _atomic_write_jsonl(sample_dir / "text_assertions.jsonl", text_records)
    _atomic_write_jsonl(sample_dir / "sensor_readings.jsonl", sensor_records)
    atomic_write_json(sample_dir / "window_config.json", config)
    atomic_write_json(sample_dir / "injection_plan.json", plan)
    atomic_write_json(sample_dir / "injection_plan_duplicate.json", duplicate_plan)
    atomic_write_json(sample_dir / "window_config_drop.json", drop_config)
    atomic_write_json(sample_dir / "window_config_q12.json", q12_config)
    summary = {
        "seed": seed,
        "counts": {
            "image": len(frame_records),
            "text": len(text_records),
            "sensor": len(sensor_records),
            "arrivals": len(arrivals),
        },
        "files": {
            "frames_jsonl": "frames.jsonl",
            "text_jsonl": "text_assertions.jsonl",
            "sensor_jsonl": "sensor_readings.jsonl",
            "config": "window_config.json",
            "plan": "injection_plan.json",
        },
    }
    atomic_write_json(sample_dir / "manifest.json", summary)
    return summary


def high_timestamp_event_id(events: dict[str, dict[str, Any]]) -> str:
    return max(
        (record for record in events.values() if record["event_id"].startswith("txt-")),
        key=lambda record: (record["timestamp_ms"], record["event_id"]),
    )["event_id"]
