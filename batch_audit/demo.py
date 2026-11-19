"""Deterministic demo workspace construction, fully local and offline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .constants import DEFAULT_OUTPUT_DIM, DEFAULT_SUPPORTED_SLOTS
from .dataset import GeneratedAssets, generate_assets, save_assets


def build_demo_feed(
    sample_ids: Sequence[str],
    *,
    supported_slots: Sequence[int] = DEFAULT_SUPPORTED_SLOTS,
    seed: int = 0,
) -> dict[str, Any]:
    """Build a shuffled feed with retransmissions, rejected entries, and an empty slot."""

    if not sample_ids:
        raise ValueError("demo feed requires at least one sample")
    slots = tuple(sorted(int(slot) for slot in supported_slots))
    if not slots:
        raise ValueError("demo feed requires at least one slot")
    content_slots = slots[:-1] if len(slots) > 1 else slots
    failure_count = 2 if len(sample_ids) >= 4 else 0
    success_count = len(sample_ids) - failure_count
    entries: list[dict[str, Any]] = []
    arrival = 1
    for index in range(success_count):
        entries.append(
            {
                "sample_id": sample_ids[index],
                "arrival_seq": arrival,
                "batch_slot": content_slots[index % len(content_slots)],
                "admitted": True,
            }
        )
        arrival += 1

    duplicate_count = min(2, success_count)
    for index in range(duplicate_count):
        entries.append(
            {
                "sample_id": sample_ids[index],
                "arrival_seq": arrival,
                "batch_slot": content_slots[index % len(content_slots)],
                "admitted": True,
            }
        )
        arrival += 1

    reasons = ("policy_denied", "precheck_missing_field")
    for index in range(failure_count):
        sample_index = success_count + index
        entries.append(
            {
                "sample_id": sample_ids[sample_index],
                "arrival_seq": arrival,
                "batch_slot": slots[index % len(slots)],
                "admitted": False,
                "reject_reason": reasons[index % len(reasons)],
            }
        )
        arrival += 1

    rng = np.random.RandomState((int(seed) & 0xFFFFFFFF) ^ 0x5EED5EED)
    order = rng.permutation(len(entries)).tolist()
    shuffled = [entries[index] for index in order]
    return {"schema": "batch-feed-v1", "entries": shuffled}


def create_demo_workspace(
    workspace: Path,
    *,
    seed: int = 20261120,
    sample_count: int = 12,
    input_dim: int = 6,
    hidden_dim: int = 8,
    output_dim: int = DEFAULT_OUTPUT_DIM,
    supported_slots: Sequence[int] = DEFAULT_SUPPORTED_SLOTS,
) -> dict[str, Any]:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    assets = generate_assets(
        seed=seed,
        sample_count=sample_count,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )
    dataset_path = workspace / "dataset.npz"
    model_path = workspace / "model.npz"
    feed_path = workspace / "feed.json"
    save_assets(assets, dataset_path, model_path)
    feed = build_demo_feed(assets.sample_ids, supported_slots=supported_slots, seed=seed)
    feed_path.write_text(
        json.dumps(feed, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "schema": "batch-audit-workspace-v1",
        "seed": int(seed),
        "sample_count": int(sample_count),
        "input_dim": int(input_dim),
        "hidden_dim": int(hidden_dim),
        "output_dim": int(output_dim),
        "supported_slots": list(sorted(int(slot) for slot in supported_slots)),
        "files": {
            "feed": "feed.json",
            "dataset": "dataset.npz",
            "model": "model.npz",
            "ledger": "ledger.sqlite3",
            "snapshot": "snapshot.json",
            "report": "report.json",
            "structured_log": "structured.jsonl",
        },
    }
    (workspace / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {"assets": assets, "feed": feed, "manifest": manifest, "workspace": workspace}
