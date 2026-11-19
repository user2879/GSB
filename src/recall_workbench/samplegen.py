"""Generate deterministic local fixtures and an independent high-precision truth set."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .fusion import run_fusion_for_user
from .models import Dataset, Interaction, Item, TruthEntry
from .recall import RecallContext, run_recall_for_user
from .snapshot import atomic_write_json, atomic_write_text
from .validation import load_config, load_dataset, sha256_bytes, stable_json_bytes


DEFAULT_SEED = 20261120


def default_config(seed: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "seed": seed,
        "event_types": ["view", "click", "purchase"],
        "event_weights": {"view": 1.0, "click": 2.5, "purchase": 5.0},
        "channels": {
            "lexical": {"enabled": True, "weight": 0.44, "candidate_limit": 14},
            "collaborative": {"enabled": True, "weight": 0.36, "candidate_limit": 14},
            "popularity": {"enabled": True, "weight": 0.20, "candidate_limit": 14},
        },
        "fusion": {
            "strategy": "weighted_sum",
            "temperature": 1.0,
            "normalization": "max",
            "zero_channel_policy": "redistribute",
        },
        "ranking_features": [
            {"name": "lexical_norm", "weight": 0.17},
            {"name": "collaborative_norm", "weight": 0.13},
            {"name": "freshness", "weight": 0.11},
            {"name": "popularity_prior", "weight": 0.09},
        ],
        "fixed_point": {
            "primary_bits": 5,
            "reference_bits": 9,
            "tolerance": 0.01,
            "rounding": "half_even",
        },
        "score_clip": {"min": 0.0, "max": 1.0},
        "retrieval": {
            "half_life_ms": 12 * 24 * 60 * 60 * 1000,
            "popularity_floor": 0.0,
        },
        "execution": {
            "max_workers": 4,
            "snapshot_commit_interval": 5,
            "top_k": 8,
        },
    }


def _catalog(rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    categories = {
        "audio": ["audio", "headphone", "speaker", "sound", "wireless"],
        "camera": ["camera", "lens", "photo", "sensor", "portable"],
        "kitchen": ["kitchen", "coffee", "blender", "steel", "compact"],
        "outdoor": ["outdoor", "camp", "hiking", "weather", "light"],
        "office": ["office", "desk", "keyboard", "monitor", "ergonomic"],
        "sports": ["sports", "running", "training", "fitness", "track"],
    }
    adjectives = ["mini", "pro", "classic", "smart", "eco", "rapid", "quiet", "rugged", "daily", "travel"]
    items: list[dict[str, Any]] = []
    terms_by_item: dict[str, list[str]] = {}
    category_names = sorted(categories)
    for index in range(72):
        item_id = f"i{index:03d}"
        category = category_names[index % len(category_names)]
        terms = [category, adjectives[(index // len(category_names)) % len(adjectives)]]
        terms.extend(categories[category][1 + (index // 3) % 3 : 2 + (index // 3) % 3])
        if index >= 52:
            terms.append("longtail")
        if index >= 68:
            terms.append("seasonal")
        if index % 7 == 0:
            terms.append("bestseller")
        rng.shuffle(terms)
        deduplicated = []
        for term in terms:
            if term not in deduplicated:
                deduplicated.append(term)
        items.append({"item_id": item_id, "terms": deduplicated})
        terms_by_item[item_id] = deduplicated
    return items, terms_by_item


def _interactions(rng: random.Random, terms_by_item: dict[str, list[str]]) -> tuple[list[dict[str, Any]], list[str]]:
    user_count = 48
    cold_start_count = 3
    regular_users = [f"u{index:04d}" for index in range(user_count - cold_start_count)]
    cold_users = [f"u{index:04d}" for index in range(user_count - cold_start_count, user_count)]
    category_items: dict[str, list[str]] = {}
    for item_id, terms in terms_by_item.items():
        category_items.setdefault(terms[0] if terms else "unknown", []).append(item_id)
    categories = sorted(category_items)
    base_ms = 1_700_000_000_000
    day_ms = 24 * 60 * 60 * 1000
    records: list[dict[str, Any]] = []

    for user_index, user_id in enumerate(regular_users):
        interests = rng.sample(categories, k=2)
        event_count = rng.randint(5, 16)
        for event_index in range(event_count):
            draw = rng.random()
            if draw < 0.62:
                item_id = rng.choice(category_items[rng.choice(interests)])
            elif draw < 0.82:
                item_id = f"i{rng.randint(52, 67):03d}"
            elif draw < 0.92:
                item_id = f"i{rng.randint(68, 71):03d}"
            else:
                item_id = f"i{rng.randint(0, 71):03d}"
            day = rng.randint(0, 59)
            minute = rng.randint(0, 1439)
            event_ms = base_ms + day * day_ms + minute * 60_000 + user_index * 13 + event_index
            event_draw = rng.random()
            if event_draw < 0.58:
                event_type = "view"
            elif event_draw < 0.88:
                event_type = "click"
            else:
                event_type = "purchase"
            records.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "event_ms": event_ms,
                    "event_type": event_type,
                }
            )
    # Add explicit fixed-period bursts for the seasonal items so the dataset
    # carries a visible periodic heat pattern, not just random temporal noise.
    for cycle in range(4):
        burst_day = 59 - cycle * 14
        for item_offset, item_id in enumerate(("i068", "i069", "i070", "i071")):
            user_offsets = (
                (cycle * 7 + item_offset * 3) % len(regular_users),
                (cycle * 5 + item_offset + 11) % len(regular_users),
            )
            for offset in user_offsets:
                event_ms = base_ms + burst_day * day_ms + item_offset * 60_000 + offset
                records.append(
                    {
                        "user_id": regular_users[offset],
                        "item_id": item_id,
                        "event_ms": event_ms,
                        "event_type": "click" if cycle % 2 == 0 else "view",
                    }
                )
    records.sort(key=lambda record: (record["user_id"], record["event_ms"], record["item_id"], record["event_type"]))
    return records, regular_users + cold_users


def _invalid_lines(items: list[dict[str, Any]], interactions: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    invalid_items = [
        {"item_id": "", "terms": ["bad"]},
        {"item_id": "i900", "terms": []},
        {"item_id": "i901", "terms": ["ok", ""]},
        {"item_id": "i000", "terms": ["duplicate"]},
        "{not valid json",
    ]
    invalid_interactions = [
        {"user_id": "", "item_id": "i000", "event_ms": 1700000000000, "event_type": "view"},
        {"user_id": "u0000", "item_id": "i999", "event_ms": 1700000000000, "event_type": "view"},
        {"user_id": "u0001", "item_id": "i001", "event_ms": "not-a-time", "event_type": "view"},
        {"user_id": "u0002", "item_id": "i002", "event_ms": 1700000000000, "event_type": "share"},
        {"user_id": "u0003", "item_id": "", "event_ms": 1700000000000, "event_type": "click"},
        "[]",
    ]
    item_lines = [json.dumps(item, ensure_ascii=False) for item in items]
    item_lines.extend(line if isinstance(line, str) else json.dumps(line, ensure_ascii=False) for line in invalid_items)
    interaction_lines = [json.dumps(record, ensure_ascii=False) for record in interactions]
    interaction_lines.extend(
        line if isinstance(line, str) else json.dumps(line, ensure_ascii=False) for line in invalid_interactions
    )
    return item_lines, interaction_lines


def generate_samples(output_dir: Path, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    config = default_config(seed)
    atomic_write_json(output_dir / "config.json", config)

    items, terms_by_item = _catalog(rng)
    interactions, user_ids = _interactions(rng, terms_by_item)
    item_lines, interaction_lines = _invalid_lines(items, interactions)
    atomic_write_text(output_dir / "items.jsonl", "\n".join(item_lines) + "\n")
    atomic_write_text(output_dir / "interactions.jsonl", "\n".join(interaction_lines) + "\n")

    loaded_config, failures = load_config(output_dir / "config.json")
    if loaded_config is None:
        raise RuntimeError(f"generated config is invalid: {[failure.to_dict() for failure in failures]}")
    valid_dataset = Dataset(
        interactions=[
            Interaction(
                user_id=record["user_id"],
                item_id=record["item_id"],
                event_ms=int(record["event_ms"]),
                event_type=record["event_type"],
                line=index,
            )
            for index, record in enumerate(interactions, start=1)
        ],
        items={
            item["item_id"]: Item(item_id=item["item_id"], terms=tuple(item["terms"]), line=index)
            for index, item in enumerate(items, start=1)
        },
        truth={user_id: TruthEntry(candidates=(), ranking=()) for user_id in user_ids},
    )
    context = RecallContext.build(valid_dataset.items, valid_dataset.histories, loaded_config)
    truth_users: dict[str, Any] = {}
    for user_id in sorted(user_ids):
        recall_record = run_recall_for_user(user_id, context, loaded_config)
        oracle = run_fusion_for_user(recall_record, loaded_config, bits=24)
        truth_users[user_id] = {
            "candidates": oracle["final_order"][: loaded_config.top_k],
            "ranking": oracle["final_order"],
        }
    truth = {
        "schema_version": 1,
        "generator_seed": seed,
        "source_config_hash": loaded_config.config_hash,
        "oracle": {"fractional_bits": 24, "rounding": "half_even", "mode": "full_fixed_point_chain"},
        "top_k": loaded_config.top_k,
        "users": truth_users,
    }
    atomic_write_json(output_dir / "truth.json", truth)

    # Loading here verifies that the on-disk truth and fixtures agree with the
    # strict loader and records the intentional invalid fixtures as failures.
    dataset = load_dataset(output_dir, loaded_config)
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "config_hash": loaded_config.config_hash,
        "counts": {
            "catalog_valid": len(items),
            "interactions_valid": len(interactions),
            "users_total": len(user_ids),
            "cold_start_users": 3,
            "truth_users": len(truth_users),
            "input_failures": len(dataset.failures),
        },
        "failure_codes": sorted({failure.code for failure in dataset.failures}),
        "files": {
            name: {"sha256": digest}
            for name, digest in sorted(dataset.file_hashes.items())
        },
        "deterministic_digest": hashlib.sha256(
            stable_json_bytes({"truth": truth, "items": items, "interactions": interactions})
        ).hexdigest(),
    }
    atomic_write_json(output_dir / "sample_manifest.json", manifest)
    return manifest
