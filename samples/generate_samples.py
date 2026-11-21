#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvcc_workbench.value_ops import version_value_digest  # noqa: E402


def transaction(
    txn_id: str,
    start_seq: int,
    outcome: str,
    commit_seq: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "txn_id": txn_id,
        "start_seq": start_seq,
        "outcome": outcome,
    }
    if commit_seq is not None:
        result["commit_seq"] = commit_seq
    return result


def record(row_key: str, version_no: int, txn_id: str, value: int, prev_version: int | None) -> dict[str, Any]:
    summary = {"balance": value, "currency": "CNY"}
    return {
        "row_key": row_key,
        "version_no": version_no,
        "txn_id": txn_id,
        "value_summary": summary,
        "value_digest": version_value_digest(summary),
        "prev_version": prev_version,
    }


def condition(column: str, op: str, value: int) -> dict[str, Any]:
    return {"column": column, "op": op, "value": value}


def base_dataset(seed: int) -> dict[str, Any]:
    jitter = seed % 11
    transactions = [
        transaction("T1", 1, "committed", 3),
        transaction("T2", 2, "committed", 5),
        transaction("T3", 4, "committed", 8),
        transaction("T4", 6, "committed", 10),
        transaction("T5", 7, "rolledback"),
        transaction("T6", 9, "committed", 12),
        transaction("T7", 11, "committed", 14),
        transaction("T8", 13, "rolledback"),
        transaction("T9", 15, "committed", 18),
        transaction("T10", 16, "committed", 20),
    ]
    records = [
        record("acct:1", 1, "T1", 100 + jitter, None),
        record("acct:1", 2, "T2", 120 + jitter, 1),
        record("acct:1", 3, "T3", 135 + jitter, 2),
        record("acct:1", 4, "T5", 999 + jitter, 3),
        record("acct:1", 5, "T6", 150 + jitter, 4),
        record("acct:2", 1, "T2", 200 + jitter, None),
        record("acct:2", 2, "T4", 260 + jitter, 1),
        record("acct:2", 3, "T4", 280 + jitter, 2),
        record("acct:2", 4, "T5", 5, 3),
        record("acct:2", 5, "T7", 320 + jitter, 4),
        record("acct:2", 6, "T9", 400 + jitter, 5),
        record("acct:3", 1, "T3", 125 + jitter, None),
        record("acct:3", 2, "T6", 75, 1),
        record("acct:3", 3, "T8", 88, 2),
        record("acct:3", 4, "T10", 110, 3),
    ]
    read_points = [
        {"read_point_id": "RP01", "snapshot_seq": 9, "reader_txn_id": "T3"},
        {"read_point_id": "RP02", "snapshot_seq": 11, "reader_txn_id": "T4"},
        {"read_point_id": "RP03", "snapshot_seq": 16, "reader_txn_id": "T6"},
        {"read_point_id": "RP04", "snapshot_seq": 7, "reader_txn_id": "T3"},
    ]
    queries = [
        {
            "query_id": "Q01",
            "read_point_id": "RP01",
            "row_key": "acct:1",
            "predicate": {"mode": "all", "conditions": [condition("balance", ">=", 130 + jitter)]},
            "priority": 10,
        },
        {
            "query_id": "Q02",
            "read_point_id": "RP01",
            "row_key": "acct:2",
            "predicate": {"mode": "all", "conditions": [condition("balance", ">=", 250 + jitter)]},
            "priority": 20,
        },
        {
            "query_id": "Q03",
            "read_point_id": "RP02",
            "row_key": "acct:2",
            "predicate": {"mode": "all", "conditions": [condition("balance", ">=", 250 + jitter)]},
            "priority": 10,
        },
        {
            "query_id": "Q04",
            "read_point_id": "RP02",
            "row_key": "acct:1",
            "predicate": {"mode": "all", "conditions": [condition("balance", ">=", 140 + jitter)]},
            "priority": 20,
        },
        {
            "query_id": "Q05",
            "read_point_id": "RP03",
            "row_key": "acct:1",
            "predicate": {"mode": "all", "conditions": [condition("balance", ">=", 150 + jitter)]},
            "priority": 10,
        },
        {
            "query_id": "Q06",
            "read_point_id": "RP03",
            "row_key": "acct:2",
            "predicate": {"mode": "all", "conditions": [condition("balance", ">=", 350 + jitter)]},
            "priority": 20,
        },
        {
            "query_id": "Q07",
            "read_point_id": "RP04",
            "row_key": "acct:3",
            "predicate": {"mode": "all", "conditions": [condition("balance", ">=", 120 + jitter)]},
            "priority": 10,
        },
        {
            "query_id": "Q08",
            "read_point_id": "RP02",
            "row_key": "acct:3",
            "predicate": {"mode": "all", "conditions": [condition("balance", ">=", 70 + jitter)]},
            "priority": 20,
        },
    ]
    return {
        "schema_version": 1,
        "manifest": {
            "schema_version": 1,
            "dataset_id": f"local-mvcc-seed-{seed}",
            "txn_set_version": 1,
            "chain_version": 1,
            "budget_version": 1,
            "read_point_set_version": 1,
        },
        "transactions": transactions,
        "records": records,
        "read_points": read_points,
        "queries": queries,
        "budget": {
            "global_cost_budget": 35,
            "round_cost_budget": 14,
            "global_concurrency": 3,
            "per_read_point_concurrency": 1,
            "task_timeout_ms": 1000,
        },
        "visibility_policy": {
            "own_uncommitted_visible": True,
            "own_write_ignores_snapshot": True,
            "declaration": (
                "Read-your-own-writes is explicit: a committed version written by the reader "
                "transaction may ignore snapshot_seq when own_write_ignores_snapshot is true. "
                "Rolled-back versions are never visible."
            ),
        },
    }


def evolved_dataset(seed: int) -> dict[str, Any]:
    result = base_dataset(seed)
    jitter = seed % 11
    result["manifest"]["txn_set_version"] = 2
    result["manifest"]["chain_version"] = 2
    for txn in result["transactions"]:
        if txn["txn_id"] == "T5":
            txn["outcome"] = "committed"
            txn["commit_seq"] = 9
    result["transactions"].append(transaction("T11", 21, "committed", 23))
    result["records"].extend(
        [
            record("acct:1", 6, "T11", 160 + jitter, 5),
            record("acct:2", 7, "T11", 450 + jitter, 6),
        ]
    )
    result["transactions"].sort(key=lambda item: item["txn_id"])
    result["records"].sort(key=lambda item: (item["row_key"], item["version_no"]))
    result["manifest"]["dataset_id"] = f"local-mvcc-seed-{seed}"
    return result


def invalid_cycle(seed: int) -> dict[str, Any]:
    result = copy.deepcopy(base_dataset(seed))
    for item in result["records"]:
        if item["row_key"] == "acct:3" and item["version_no"] == 1:
            item["prev_version"] = 2
        if item["row_key"] == "acct:3" and item["version_no"] == 2:
            item["prev_version"] = 1
    result["manifest"]["dataset_id"] = f"invalid-chain-cycle-seed-{seed}"
    return result


def invalid_txn_regression(seed: int) -> dict[str, Any]:
    result = copy.deepcopy(base_dataset(seed))
    for txn in result["transactions"]:
        if txn["txn_id"] == "T4":
            txn["commit_seq"] = 5
    result["manifest"]["dataset_id"] = f"invalid-txn-regression-seed-{seed}"
    return result


def invalid_snapshot_digest(seed: int) -> dict[str, Any]:
    result = base_dataset(seed)
    result["manifest"]["dataset_id"] = f"invalid-snapshot-digest-seed-{seed}"
    result["injection"] = {
        "kind": "snapshot_digest",
        "snapshot_content_digest": "sha256:" + "0" * 64,
        "expected_code": "E_SNAPSHOT",
    }
    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic local MVCC samples")
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    samples = {
        "base.json": base_dataset(args.seed),
        "evolved.json": evolved_dataset(args.seed),
        "invalid_chain_cycle.json": invalid_cycle(args.seed),
        "invalid_txn_commit_regression.json": invalid_txn_regression(args.seed),
        "invalid_snapshot_digest.json": invalid_snapshot_digest(args.seed),
    }
    for name, payload in samples.items():
        write_json(args.out / name, payload)
        print(f"wrote {args.out / name}")
    print(f"seed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
