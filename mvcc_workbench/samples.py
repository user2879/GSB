"""Deterministic local sample generation (JSON, shuffled JSON, and NDJSON)."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any


def _value_digest(row_key: str, tx_id: str, version: int) -> str:
    material = f"{row_key}|{tx_id}|{version}".encode("utf-8")
    return "sha256:" + hashlib.sha256(material).hexdigest()


def _build_spec(
    *,
    name: str,
    row_count: int,
    query_count: int,
    chain_pattern: tuple[int, ...],
    snapshots: tuple[int, ...],
    budget: dict[str, int],
    notes: str,
) -> dict[str, Any]:
    rows = [f"row:{index:03d}" for index in range(1, row_count + 1)]
    transactions: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    record_by_pair: dict[tuple[str, int], dict[str, Any]] = {}
    version_index = 0

    for row_index, row_key in enumerate(rows):
        chain_length = chain_pattern[row_index % len(chain_pattern)]
        prev_version: int | None = None
        for version in range(1, chain_length + 1):
            tx_id = f"T{version_index + 1:04d}"
            start_seq = 10 + version_index * 4
            rolled_back = version_index > 0 and (
                version_index % 11 == 0 or (version == 2 and row_index % 5 == 0)
            )
            if rolled_back:
                transaction = {
                    "id": tx_id,
                    "start_seq": start_seq,
                    "outcome": "rolledback",
                    "commit_seq": None,
                }
            else:
                delay = 2 + (version_index * 3 + row_index) % 9
                transaction = {
                    "id": tx_id,
                    "start_seq": start_seq,
                    "outcome": "committed",
                    "commit_seq": start_seq + delay,
                }
            transactions.append(transaction)
            record = {
                "row_key": row_key,
                "tx_id": tx_id,
                "version": version,
                "value_digest": _value_digest(row_key, tx_id, version),
                "prev_version": prev_version,
            }
            records.append(record)
            record_by_pair[(row_key, version)] = record
            prev_version = version
            version_index += 1

    committed_tx_ids = [
        transaction["id"]
        for transaction in transactions
        if transaction["outcome"] == "committed"
    ]
    reader_tx_id = committed_tx_ids[0]
    read_points = [
        {
            "id": f"RP{index + 1:03d}",
            "snapshot_seq": snapshot,
            "tx_id": reader_tx_id,
        }
        for index, snapshot in enumerate(snapshots)
    ]

    query_rows = rows[: min(4, len(rows))]
    pairs = [
        (read_point["id"], row_key)
        for read_point in read_points
        for row_key in query_rows
    ]
    queries: list[dict[str, Any]] = []
    for index in range(query_count):
        read_point_id, row_key = pairs[index % len(pairs)]
        predicate_selector = index % 4
        if predicate_selector == 0:
            predicate = {"op": "exists"}
        elif predicate_selector == 1:
            predicate = {"op": "version_eq", "value": 1 + (index % 3)}
        elif predicate_selector == 2:
            record = record_by_pair[(row_key, 1 + (index % 2))]
            predicate = {"op": "digest_eq", "value": record["value_digest"]}
        else:
            record = record_by_pair[(row_key, 1 + (index % 3))]
            predicate = {"op": "tx_eq", "value": record["tx_id"]}
        queries.append(
            {
                "id": f"Q{index + 1:04d}",
                "read_point_id": read_point_id,
                "row_key": row_key,
                "predicate": predicate,
            }
        )

    return {
        "format": "mvcc-diagnostic-spec",
        "format_version": 1,
        "name": name,
        "transactions": transactions,
        "records": records,
        "read_points": read_points,
        "queries": queries,
        "budget": budget,
        "notes": notes,
    }


def small_sample() -> dict[str, Any]:
    return _build_spec(
        name="mvcc-small-per-query",
        row_count=4,
        query_count=6,
        chain_pattern=(2,),
        snapshots=(18, 34),
        budget={"global_concurrency": 2, "per_read_point_concurrency": 1},
        notes="小规模样本：低查询数、低行键数、平均链长 2，默认逐查询回溯。",
    )


def large_sample() -> dict[str, Any]:
    return _build_spec(
        name="mvcc-large-materialized",
        row_count=20,
        query_count=60,
        chain_pattern=(4, 5),
        snapshots=(55, 95, 135, 175, 215, 255),
        budget={"global_concurrency": 4, "per_read_point_concurrency": 2},
        notes="较大样本：20 行、90 版本、平均链长 4.5、60 查询、24 个唯一读点/行组合。",
    )


def shuffled_copy(value: dict[str, Any], seed: int = 20261124) -> dict[str, Any]:
    cloned = json.loads(json.dumps(value, ensure_ascii=False))
    randomizer = random.Random(seed)
    for key in ("transactions", "records", "read_points", "queries"):
        randomizer.shuffle(cloned[key])
    return cloned


def _ndjson_text(value: dict[str, Any]) -> str:
    lines: list[str] = []
    meta = {
        "kind": "meta",
        "format": value["format"],
        "format_version": value["format_version"],
        "name": value["name"],
        "budget": value["budget"],
        "notes": value.get("notes"),
    }
    lines.append(json.dumps(meta, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    for kind, key in (
        ("transaction", "transactions"),
        ("record", "records"),
        ("read_point", "read_points"),
        ("query", "queries"),
    ):
        for item in value[key]:
            payload = {"kind": kind, **item}
            lines.append(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_samples(destination: str | Path = "data") -> dict[str, str]:
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    small = small_sample()
    large = large_sample()
    shuffled = shuffled_copy(large)
    paths = {
        "small_json": root / "demo_small.json",
        "large_json": root / "demo_large.json",
        "large_shuffled_json": root / "demo_large_shuffled.json",
        "large_ndjson": root / "demo_large.ndjson",
    }
    _write_json(paths["small_json"], small)
    _write_json(paths["large_json"], large)
    _write_json(paths["large_shuffled_json"], shuffled)
    paths["large_ndjson"].write_text(
        _ndjson_text(shuffled), encoding="utf-8", newline="\n"
    )
    return {key: str(path) for key, path in paths.items()}
