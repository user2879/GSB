from __future__ import annotations

from pathlib import Path
from typing import Any

from rts_pipeline.canonical import hash_value, hash_without
from rts_pipeline.generator import generate_workspace
from rts_pipeline.rules import faction_hash


def make_workspace(root: Path, **overrides: Any) -> dict[str, Any]:
    options: dict[str, Any] = {
        "seed": 41,
        "ticks": 14,
        "scenario": "normal",
        "faction_count": 3,
        "chunk_ticks": 4,
    }
    options.update(overrides)
    return generate_workspace(root, **options)


def rehash_ledger(ledger: dict[str, Any]) -> None:
    business_payload = {
        "schema_version": ledger["schema_version"],
        "game": {
            key: value
            for key, value in ledger["game"].items()
            if key != "chunk_ticks"
        },
        "rules_hash": ledger["rules_hash"],
        "events_hash": ledger["events_hash"],
        "initial_snapshot_hash": ledger["initial_snapshot_hash"],
        "exit_snapshot_hash": ledger["exit_snapshot_hash"],
        "tick_records": ledger["tick_records"],
        "event_records": ledger["event_records"],
        "rng_records": ledger["rng_records"],
        "violations": ledger["violations"],
        "budget_report": ledger["budget_report"],
        "result_codes": ledger["result_codes"],
    }
    ledger["business_hash"] = hash_value(business_payload)
    ledger["ledger_hash"] = hash_without(ledger, "ledger_hash")


def mutate_tick_state(ledger: dict[str, Any], tick: int, faction: str) -> None:
    record = next(item for item in ledger["tick_records"] if item["tick"] == tick)
    state = record["snapshot"]["factions"][faction]
    state["resources"] += 1
    record["faction_hashes"][faction] = faction_hash(state)
    record["state_hash"] = hash_value(record["snapshot"])
    rehash_ledger(ledger)
