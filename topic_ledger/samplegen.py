from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .canonical import canonical_json


DEFAULT_SEED = 398_039_801


def build_sample_bundle(seed: int = DEFAULT_SEED) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    vocab_size = 12
    vocab = [{"term_id": index, "term": f"term_{index:02d}"} for index in range(vocab_size)]
    shard_ids = [f"shard-{index:02d}" for index in range(4)]
    doc_count = 12
    weights = rng.random(vocab_size) + 0.35
    weights /= weights.sum()
    documents: list[dict[str, Any]] = []
    for doc_index in range(doc_count):
        length = 10 + int(rng.integers(0, 9))
        chosen = rng.choice(vocab_size, size=length, replace=True, p=weights)
        shard_id = shard_ids[doc_index % len(shard_ids)]
        documents.append(
            {
                "doc_id": f"doc-{doc_index:03d}",
                "shard_id": shard_id,
                "revision": 1,
                "term_ids": [int(value) for value in chosen],
            }
        )

    topology = {
        "global_concurrency": 3,
        "nodes": [
            {
                "node_id": "node-a",
                "host": "127.0.0.1",
                "port": 48101,
                "concurrency_quota": 2,
                "lease_limit": 2,
            },
            {
                "node_id": "node-b",
                "host": "127.0.0.1",
                "port": 48102,
                "concurrency_quota": 2,
                "lease_limit": 2,
            },
            {
                "node_id": "node-c",
                "host": "127.0.0.1",
                "port": 48103,
                "concurrency_quota": 1,
                "lease_limit": 1,
            },
        ],
        "shards": shard_ids,
        "routing": {
            "shard-00": "node-a",
            "shard-01": "node-b",
            "shard-02": "node-c",
            "shard-03": "node-a",
        },
    }
    inference = {
        "topic_count": 4,
        "sample_rounds": 2,
        "max_rounds_per_shard": 3,
        "sample_point_budget": 100_000,
        "lease_ttl_ms": 250,
        "interaction_budget_ms": 500,
        "topic_parameter_version": 17,
        "topic_parameter_seed": 20261127,
        "fixed_point": {"bit_width_tier": "I64", "scale": 1_000_000},
    }
    commands = [
        {"command_id": "cmd-01", "op": "run_sampling", "batch_id": "batch-initial"},
        {
            "command_id": "cmd-02",
            "op": "upsert_document",
            "document": {
                "doc_id": "doc-000",
                "shard_id": "shard-00",
                "revision": 2,
                "term_ids": [0, 1, 1, 2, 3, 5, 8, 11, 0, 4, 7, 9],
            },
        },
        {"command_id": "cmd-03", "op": "run_sampling", "batch_id": "batch-after-replace"},
        {
            "command_id": "cmd-04",
            "op": "set_config",
            "patch": {"topic_count": 5, "topic_parameter_version": 18},
        },
        {"command_id": "cmd-05", "op": "run_sampling", "batch_id": "batch-after-topics"},
        {"command_id": "cmd-06", "op": "snapshot", "label": "final"},
    ]
    return {
        "seed": seed,
        "vocab": vocab,
        "documents": documents,
        "topology": topology,
        "inference": inference,
        "commands": commands,
    }


def write_sample_bundle(root: str | Path, bundle: dict[str, Any] | None = None) -> list[Path]:
    root = Path(root)
    bundle = bundle or build_sample_bundle()
    config_dir = root / "config"
    sample_dir = root / "samples"
    config_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    files = {
        config_dir / "nodes.json": bundle["topology"],
        config_dir / "inference.json": bundle["inference"],
        config_dir / "commands.json": bundle["commands"],
        sample_dir / "documents.json": {
            "seed": bundle["seed"],
            "vocab": bundle["vocab"],
            "documents": bundle["documents"],
        },
    }
    written: list[Path] = []
    for path, value in files.items():
        path.write_text(canonical_json(value) + "\n", encoding="utf-8")
        written.append(path)
    return written
