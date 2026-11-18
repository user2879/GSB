"""Deterministic local sample generator.

A single seeded NumPy default_rng drives every choice, so regenerating into an
empty directory reproduces all four inputs byte-for-byte. The generator emits
valid data only; faults are described separately in the injection manifest and
applied at runtime by the injection layer.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np

from .config import config_from_dict

SEED = 20261118


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, sort_keys=True, ensure_ascii=False, indent=2)


def generate(
    out_dir: str,
    *,
    num_users: int = 120,
    requests_per_user: int = 4,
    candidates_per_request: int = 6,
    decimal_width: int = 6,
    num_buckets: int = 18,
    seed: int = SEED,
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    groups = ["control", "treatment_a", "treatment_b"]

    config_dict = {
        "experiment_id": "exp-rec-demo",
        "groups": groups,
        "control_group": "control",
        "group_weights": {"control": 1, "treatment_a": 1, "treatment_b": 1},
        "hash_seed": 777,
        "num_buckets": num_buckets,
        "primary_metric": "ctr",
        "guardrail_metrics": ["cvr"],
        "metric_direction": "higher_is_better",
        "min_sample_size": 200,
        "min_lift": "0.005",
        "attribution_tolerance": "0.01",
        "decimal_width": decimal_width,
        "quantization_step": None,
        "concurrency": 4,
        "batch_size": 256,
        "snapshot_every_batches": 2,
        "z_quantile": 1.959963984540054,
        "confidence_level": 0.95,
    }
    config = config_from_dict(config_dict)
    from .hashing import assign_users

    user_ids = [f"u{idx:05d}" for idx in range(num_users)]
    assignments, _ = assign_users(user_ids, config)
    user_group = {uid: assignments[uid].group for uid in user_ids}

    channels = ["organic", "paid", "social"]
    tiers = ["new", "mature", "dormant"]
    attributes: Dict[str, Dict[str, str]] = {}
    for idx, uid in enumerate(user_ids):
        attributes[uid] = {
            "channel": channels[int(rng.integers(0, len(channels)))],
            "tier": tiers[int(rng.integers(0, len(tiers)))],
        }

    base_click = {"control": 0.18, "treatment_a": 0.22, "treatment_b": 0.17}
    base_conv = {"control": 0.05, "treatment_a": 0.06, "treatment_b": 0.05}
    channel_lift = {"organic": 0.03, "paid": 0.01, "social": -0.01}

    log_path = os.path.join(out_dir, "exposures.jsonl")
    request_counter = 0
    with open(log_path, "w", encoding="utf-8") as handle:
        for uid in user_ids:
            group = user_group[uid]
            channel = attributes[uid]["channel"]
            for _ in range(requests_per_user):
                request_id = f"r{request_counter:06d}"
                request_counter += 1
                ts = int(rng.integers(0, 1_000))
                item_ids = [
                    f"i{int(rng.integers(0, 5000)):05d}" for _ in range(candidates_per_request)
                ]
                seen: set = set()
                unique_items = []
                for item in item_ids:
                    while item in seen:
                        item = f"i{int(rng.integers(0, 5000)):05d}"
                    seen.add(item)
                    unique_items.append(item)
                for position, item in enumerate(unique_items, start=1):
                    score = float(rng.random())
                    p_click = min(
                        0.95,
                        max(0.01, base_click[group] + channel_lift[channel] + 0.02 * score),
                    )
                    clicked = int(rng.random() < p_click)
                    p_conv = base_conv[group] + (0.01 if clicked else 0.0)
                    converted = int(rng.random() < min(0.9, p_conv))
                    record = {
                        "request_id": request_id,
                        "group": group,
                        "user_id": uid,
                        "item_id": item,
                        "position": position,
                        "score": round(score, 6),
                        "clicked": clicked,
                        "converted": converted,
                        "event_ts_ms": ts,
                    }
                    ts += int(rng.integers(1, 50))
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    config_path = os.path.join(out_dir, "experiment_config.json")
    attrs_path = os.path.join(out_dir, "user_attributes.json")
    injections_path = os.path.join(out_dir, "injections.json")
    _write_json(config_path, config.to_dict())
    _write_json(attrs_path, attributes)

    first_request = "r000000"
    injections = {
        "injections": [
            {
                "id": "INJ-001",
                "fault": "non_finite_score",
                "stage": "load",
                "request_id": first_request,
            },
            {
                "id": "INJ-002",
                "fault": "group_out_of_range",
                "stage": "load",
                "request_id": "r000001",
            },
            {
                "id": "INJ-003",
                "fault": "bad_click_flag",
                "stage": "load",
                "request_id": "r000002",
            },
            {
                "id": "INJ-004",
                "fault": "stale_snapshot_reuse",
                "stage": "snapshot",
                "group": "global",
            },
        ]
    }
    _write_json(injections_path, injections)
    return {
        "log": log_path,
        "config": config_path,
        "attributes": attrs_path,
        "injections": injections_path,
    }
