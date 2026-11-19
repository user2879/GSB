#!/usr/bin/env python3
"""Generate deterministic nested-document samples and change-batch templates."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

COLLECTION = "orders"
PRIMARY_KEY = "id"
INDEX_PATH = "metrics.amount"
COLLECTION_VERSION = 7
REGIONS = ("east", "north", "south", "west")


def dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def make_document(number: int, amount: int, *, prefix: str = "doc") -> dict[str, Any]:
    return {
        "id": f"{prefix}-{number:06d}",
        "customer": {
            "region": REGIONS[number % len(REGIONS)],
            "segment": number % 7,
        },
        "metrics": {
            "amount": amount,
            "latency_ms": 10 + (number * 13) % 240,
        },
        "attributes": {
            "priority": number % 5,
            "active": number % 4 != 0,
        },
        "tags": [f"t{number % 3}", f"bucket-{number % 5}"],
    }


def generate(scale: int, seed: int, capacity: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if scale <= 0:
        raise SystemExit("scale must be a positive integer")
    if capacity <= 0:
        raise SystemExit("capacity must be a positive integer")
    rng = random.Random(seed)
    documents = []
    for number in range(1, scale + 1):
        amount = rng.randrange(0, max(20, scale * 12))
        documents.append(make_document(number, amount))
    documents.sort(key=lambda item: item["id"])

    def op_set(number: int, value: int) -> dict[str, Any]:
        return {
            "op": "set_path",
            "pk": f"doc-{number:06d}",
            "path": INDEX_PATH,
            "value": value,
        }

    def op_delete(number: int) -> dict[str, Any]:
        return {"op": "delete", "pk": f"doc-{number:06d}"}

    def op_insert(number: int, value: int) -> dict[str, Any]:
        return {
            "op": "insert",
            "document": make_document(number, value, prefix="ins"),
        }

    batches = [
        {
            "id": "incremental",
            "description": "mixed key migrations, deletes and inserts in one atomic batch",
            "changes": [
                op_set(1, scale * 20 + 11),
                op_set(7, scale * 20 + 12),
                op_set(13, -7),
                op_set(21, scale * 20 + 13),
                op_delete(2),
                op_delete(30),
                op_insert(1, scale * 20 + 14),
                op_insert(2, scale * 20 + 15),
            ],
        },
        {
            "id": "stale",
            "description": "additional migrations and a delete for ledger/recovery checks",
            "changes": [
                op_set(3, scale * 30 + 1),
                op_set(9, 3),
                op_set(33, scale * 30 + 2),
                op_delete(40),
            ],
        },
        {
            "id": "isolation",
            "description": "apply with --retain-stale to exercise query-time stale isolation",
            "changes": [
                op_set(4, scale * 40 + 1),
                op_set(10, scale * 40 + 2),
                op_delete(44),
                op_insert(3, scale * 40 + 3),
            ],
        },
    ]
    collection = {
        "name": COLLECTION,
        "version": COLLECTION_VERSION,
        "primary_key": PRIMARY_KEY,
        "documents": documents,
        "indexes": [
            {
                "path": INDEX_PATH,
                "segment_capacity": capacity,
                "allow_null": False,
            }
        ],
    }
    changes = {
        "collection": COLLECTION,
        "primary_key": PRIMARY_KEY,
        "index_path": INDEX_PATH,
        "batches": batches,
    }
    plan = {
        "seed": seed,
        "scale": scale,
        "segment_capacity": capacity,
        "collection_version": COLLECTION_VERSION,
        "collection": COLLECTION,
        "primary_key": PRIMARY_KEY,
        "index_path": INDEX_PATH,
        "document_count": len(documents),
        "expected_initial_segments": (len(documents) + capacity - 1) // capacity,
        "batches": [
            {
                "id": batch["id"],
                "description": batch["description"],
                "change_count": len(batch["changes"]),
                "command": (
                    f"python -m docdb_index apply --session sessions/demo "
                    f"--input samples/changes.json --batch-id {batch['id']}"
                ),
            }
            for batch in batches
        ],
        "command_templates": {
            "init": "python -m docdb_index init --session sessions/demo --input samples/collections.json --reset",
            "normal_incremental": "python -m docdb_index apply --session sessions/demo --input samples/changes.json --batch-id incremental --batch-size 2",
            "stale_history": "python -m docdb_index apply --session sessions/demo --input samples/changes.json --batch-id stale --batch-size 1",
            "stale_window": "python -m docdb_index apply --session sessions/demo --input samples/changes.json --batch-id isolation --retain-stale",
            "reclaim": "python -m docdb_index reclaim --session sessions/demo --collection orders",
            "query_range": "python -m docdb_index query --session sessions/demo --collection orders --index metrics.amount --range 0 100 --offset 0 --limit 10",
            "query_equality": "python -m docdb_index query --session sessions/demo --collection orders --index metrics.amount --eq 42",
            "diagnose": "python -m docdb_index diagnose --session sessions/demo --collection orders --index metrics.amount",
            "snapshot_create": "python -m docdb_index snapshot create --session sessions/demo --label snap-000001",
            "snapshot_restore": "python -m docdb_index snapshot restore --session sessions/demo --snapshot snap-000001",
            "snapshot_list": "python -m docdb_index snapshot list --session sessions/demo",
            "segment_fault": "python -m docdb_index inject --session sessions/demo --collection orders --index metrics.amount --type segment",
            "bound_fault": "python -m docdb_index inject --session sessions/demo --collection orders --index metrics.amount --type bound",
            "ledger_fault": "python -m docdb_index inject --session sessions/demo --collection orders --index metrics.amount --type ledger",
            "verify": "python -m docdb_index verify --session sessions/demo",
            "replay": "python -m docdb_index replay --session sessions/demo",
            "gui": "python -m docdb_index gui --session sessions/demo --collection orders --index metrics.amount",
        },
    }
    return collection, changes, plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--capacity", type=int, default=5)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    collection, changes, plan = generate(args.scale, args.seed, args.capacity)
    dump(output / "collections.json", {"collections": [collection]})
    dump(output / "changes.json", changes)
    dump(output / "plan.json", plan)
    print(
        json.dumps(
            {
                "ok": True,
                "collections": str((output / "collections.json").resolve()),
                "changes": str((output / "changes.json").resolve()),
                "plan": str((output / "plan.json").resolve()),
                "scale": args.scale,
                "seed": args.seed,
                "capacity": args.capacity,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
