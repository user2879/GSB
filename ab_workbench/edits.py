"""Deterministic edit operations over the local sample inputs.

These helpers power the incremental-vs-full equivalence harness. Every write is
byte-deterministic (sorted keys, fixed separators) so content fingerprints change
only when the underlying logical content actually changes.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from .generator import generate


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, sort_keys=True, ensure_ascii=False, indent=2)


def append_batch(log_path: str, rows: List[Dict[str, Any]]) -> None:
    existing = read_jsonl(log_path)
    write_jsonl(log_path, existing + rows)


def correct_one_record(
    log_path: str, request_id: str, item_id: str, patch: Dict[str, Any]
) -> bool:
    rows = read_jsonl(log_path)
    changed = False
    for row in rows:
        if row["request_id"] == request_id and row["item_id"] == item_id:
            row.update(patch)
            changed = True
    write_jsonl(log_path, rows)
    return changed


def update_config(config_path: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    config = read_json(config_path)
    config.update(patch)
    write_json(config_path, config)
    return config


def regenerate_sample(
    out_dir: str, *, decimal_width: int, num_buckets: int, seed: int
) -> Dict[str, str]:
    return generate(
        out_dir,
        decimal_width=decimal_width,
        num_buckets=num_buckets,
        seed=seed,
    )


def synthesize_new_batch(
    start_request_index: int,
    count_requests: int,
    candidates: int,
    group: str,
    user_ids: List[str],
    rng,
) -> List[Dict[str, Any]]:
    import numpy as np

    rows: List[Dict[str, Any]] = []
    for offset in range(count_requests):
        request_id = f"r{start_request_index + offset:06d}"
        uid = user_ids[int(rng.integers(0, len(user_ids)))]
        ts = int(rng.integers(1_000_000, 2_000_000))
        items = sorted(
            {f"i{int(rng.integers(5000, 9000)):05d}" for _ in range(candidates)}
        )
        for position, item in enumerate(items, start=1):
            score = float(rng.random())
            rows.append(
                {
                    "request_id": request_id,
                    "group": group,
                    "user_id": uid,
                    "item_id": item,
                    "position": position,
                    "score": round(score, 6),
                    "clicked": int(rng.random() < 0.2),
                    "converted": int(rng.random() < 0.05),
                    "event_ts_ms": ts,
                }
            )
            ts += int(rng.integers(1, 20))
    return rows
