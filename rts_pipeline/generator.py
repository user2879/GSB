"""Deterministic scenario, rules, and event-stream generation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .budget import compute_budget_report
from .canonical import atomic_write_json, hash_without
from .constants import ENGINE_VERSION, PIPELINE_VERSION, SCHEMA_VERSION, UNIT_TYPES
from .engine import synthesize_game
from .errors import PipelineError, require
from .events import events_digest, game_document, normalize_events, write_events
from .rng import derive_u64
from .rules import initial_state, rules_hash, validate_rules


class StableEntropy:
    def __init__(self, seed: int, label: str) -> None:
        self.key = hashlib.sha256(
            f"rts-generator-v1|{seed}|{label}".encode()
        ).digest()
        self.counter = 0

    def next_u64(self) -> int:
        digest = hashlib.sha256(self.key + self.counter.to_bytes(8, "big")).digest()
        self.counter += 1
        return int.from_bytes(digest[:8], "big")

    def randint(self, low: int, high: int) -> int:
        if high <= low:
            return low
        return low + self.next_u64() % (high - low + 1)

    def chance(self, numerator: int, denominator: int) -> bool:
        return self.next_u64() % denominator < numerator

    def choice(self, values: list[Any]) -> Any:
        return values[self.next_u64() % len(values)]


def make_rules(factions: list[str], seed: int) -> dict[str, Any]:
    require(bool(factions), "E_SCHEMA", "at least one faction is required")
    body: dict[str, Any] = {}
    for index, faction in enumerate(factions):
        offset = (seed + index * 17) % 4
        body[faction] = {
            "resource_cap": 260 + offset * 20,
            "collect_rate": 5 + (index + seed) % 3,
            "queue_capacity": 10 + (index + offset) % 3,
            "unit_costs": {
                "infantry": 12 + index % 3,
                "scout": 19 + offset,
                "tank": 35 + 2 * (index % 3),
            },
            "build_ticks": {
                "infantry": 1,
                "scout": 1,
                "tank": 2,
            },
        }
    return validate_rules({"schema_version": SCHEMA_VERSION, "factions": body})


def _faction_sim(rules: dict[str, Any], faction: str) -> dict[str, Any]:
    return {
        "resources": 0,
        "queue": [],
        "next_local_seq": 0,
        "rule": rules["factions"][faction],
    }


def _advance(queue: list[dict[str, Any]]) -> None:
    for item in queue:
        if item["remaining"] > 0:
            item["remaining"] -= 1


def _new_event(
    tick: int,
    faction: str,
    action: str,
    target: str,
    amount: int,
    entropy: StableEntropy,
    sim: dict[str, Any],
) -> dict[str, Any]:
    event = {
        "tick": tick,
        "actor": faction,
        "action": action,
        "target": target,
        "amount": amount,
        "rng_seed_delta": entropy.randint(0, 2_147_483_647),
        "local_seq": sim["next_local_seq"],
    }
    sim["next_local_seq"] += 1
    return event


def generate_events(
    rules: dict[str, Any],
    seed: int,
    ticks: int,
    scenario: str,
    overflow_at: tuple[int, str] | None = None,
) -> list[dict[str, Any]]:
    entropy = StableEntropy(seed, f"{scenario}|{ticks}|{','.join(sorted(rules['factions']))}")
    factions = sorted(rules["factions"])
    simulations = {faction: _faction_sim(rules, faction) for faction in factions}
    events: list[dict[str, Any]] = []

    for tick in range(ticks):
        for faction in factions:
            sim = simulations[faction]
            body = sim["rule"]
            _advance(sim["queue"])

            if overflow_at == (tick, faction):
                delta = entropy.randint(0, 2_147_483_647)
                roll = derive_u64(seed, tick, delta) % 10000
                bonus = roll % body["collect_rate"]
                amount = (body["resource_cap"] - sim["resources"]) // body["collect_rate"] + 1
                amount = max(1, amount)
                events.append(
                    {
                        "tick": tick,
                        "actor": faction,
                        "action": "collect",
                        "target": "injection_depot",
                        "amount": amount,
                        "rng_seed_delta": delta,
                        "local_seq": sim["next_local_seq"],
                    }
                )
                sim["next_local_seq"] += 1
                if sim["resources"] + amount * body["collect_rate"] + bonus <= body["resource_cap"]:
                    raise PipelineError("E_INTERNAL", "overflow injection did not cross the resource cap")

            collect_attempts = entropy.randint(1, 2)
            if scenario == "contention":
                collect_attempts += entropy.randint(1, 2)
            for _ in range(collect_attempts):
                amount = entropy.randint(1, 4)
                delta = entropy.randint(0, 2_147_483_647)
                roll = derive_u64(seed, tick, delta) % 10000
                bonus = roll % body["collect_rate"]
                gain = amount * body["collect_rate"] + bonus
                while amount > 0 and sim["resources"] + gain > body["resource_cap"]:
                    amount -= 1
                    gain = amount * body["collect_rate"] + bonus
                if amount <= 0:
                    continue
                events.append(
                    {
                        "tick": tick,
                        "actor": faction,
                        "action": "collect",
                        "target": entropy.choice(["resource_field", "salvage_yard", "relay_cache"]),
                        "amount": amount,
                        "rng_seed_delta": delta,
                        "local_seq": sim["next_local_seq"],
                    }
                )
                sim["next_local_seq"] += 1
                sim["resources"] += gain

            produce_attempts = entropy.randint(0, 2)
            if scenario == "contention":
                produce_attempts += entropy.randint(0, 2)
            for _ in range(produce_attempts):
                candidates = [
                    unit
                    for unit in UNIT_TYPES
                    if body["unit_costs"][unit] <= sim["resources"]
                    and len(sim["queue"]) + 1 <= body["queue_capacity"]
                ]
                if not candidates:
                    break
                unit = entropy.choice(candidates)
                cost = body["unit_costs"][unit]
                events.append(
                    _new_event(tick, faction, "produce", unit, 1, entropy, sim)
                )
                sim["resources"] -= cost
                sim["queue"].append(
                    {"unit": unit, "remaining": body["build_ticks"][unit]}
                )

            ready_units = sorted(
                {item["unit"] for item in sim["queue"] if item["remaining"] == 0}
            )
            if ready_units and entropy.chance(3, 4):
                unit = entropy.choice(ready_units)
                ready_indices = [
                    index
                    for index, item in enumerate(sim["queue"])
                    if item["unit"] == unit and item["remaining"] == 0
                ]
                amount = 1
                if scenario == "contention" and len(ready_indices) >= 2:
                    amount = entropy.randint(1, min(2, len(ready_indices)))
                events.append(
                    _new_event(tick, faction, "dispatch", unit, amount, entropy, sim)
                )
                remove = set(ready_indices[:amount])
                sim["queue"] = [
                    item
                    for index, item in enumerate(sim["queue"])
                    if index not in remove
                ]

    return normalize_events(events, rules)


def assign_event_indices(
    event_records: list[dict[str, Any]], violations: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(
        event_records, key=lambda item: (item["tick"], item["actor"], item["local_seq"])
    )
    index_by_key: dict[tuple[int, str, int], int] = {}
    indexed_events: list[dict[str, Any]] = []
    rng_records: list[dict[str, Any]] = []
    for index, record in enumerate(ordered, start=1):
        item = dict(record)
        item["event_index"] = index
        indexed_events.append(item)
        key = (item["tick"], item["actor"], item["local_seq"])
        index_by_key[key] = index
        rng = dict(item["rng"])
        rng["event_index"] = index
        rng_records.append(rng)
    indexed_violations: list[dict[str, Any]] = []
    for violation in violations:
        key = (violation["tick"], violation["actor"], violation["local_seq"])
        item = dict(violation)
        item["event_index"] = index_by_key[key]
        indexed_violations.append(item)
    indexed_violations.sort(key=lambda item: item["event_index"])
    return indexed_events, indexed_violations, rng_records


def generate_workspace(
    root: Path,
    seed: int,
    ticks: int,
    scenario: str,
    faction_count: int,
    chunk_ticks: int,
    overflow_at: tuple[int, str] | None = None,
) -> dict[str, Any]:
    require(ticks > 0, "E_SCHEMA", "ticks must be positive")
    require(chunk_ticks > 0, "E_SCHEMA", "chunk_ticks must be positive")
    require(0 < faction_count <= 26, "E_SCHEMA", "faction_count must be between 1 and 26")
    factions = [chr(ord("A") + index) for index in range(faction_count)]
    rules = make_rules(factions, seed)
    events = generate_events(rules, seed, ticks, scenario, overflow_at=overflow_at)
    root = Path(root)
    input_dir = root / "input"
    truth_dir = root / "truth"
    report_dir = root / "report"
    input_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(input_dir / "rules.json", rules)
    write_events(input_dir / "events.jsonl", events)
    game = game_document(
        seed=seed,
        tick_start=0,
        tick_end=ticks - 1,
        chunk_ticks=chunk_ticks,
        scenario=scenario,
        events=events,
        rules=rules,
    )
    atomic_write_json(input_dir / "game.json", game)

    result = synthesize_game(
        events=events,
        rules=rules,
        seed=seed,
        tick_start=0,
        tick_end=ticks - 1,
        initial_snapshot=initial_state(rules),
    )
    event_records, violations, _ = assign_event_indices(
        result["event_records"], result["violations"]
    )
    budget_report = compute_budget_report(event_records, result["tick_records"], rules, violations)
    truth_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "engine_version": ENGINE_VERSION,
        "document_kind": "truth_ledger",
        "game": game,
        "rules_hash": rules_hash(rules),
        "events_hash": events_digest(events),
        "tick_records": result["tick_records"],
        "event_records": event_records,
        "rng_records": [dict(record["rng"], event_index=record["event_index"]) for record in event_records],
        "violations": violations,
        "budget_report": budget_report,
        "result_codes": result["result_codes"],
        "exit_snapshot": result["exit_snapshot"],
        "exit_snapshot_hash": result["exit_snapshot_hash"],
    }
    truth_payload["truth_hash"] = hash_without(truth_payload, "truth_hash")
    atomic_write_json(truth_dir / "ledger.json", truth_payload)
    factions_truth_dir = truth_dir / "factions"
    factions_truth_dir.mkdir(parents=True, exist_ok=True)
    faction_truth_files: dict[str, str] = {}
    for faction in sorted(rules["factions"]):
        faction_ticks = [
            {
                "tick": tick_record["tick"],
                "state": tick_record["snapshot"]["factions"][faction],
                "resource_queue_hash": tick_record["faction_hashes"][faction],
                "event_keys": [
                    key
                    for key in tick_record["event_keys"]
                    if key["actor"] == faction
                ],
            }
            for tick_record in result["tick_records"]
        ]
        faction_events = [
            record for record in event_records if record["actor"] == faction
        ]
        faction_truth = {
            "schema_version": SCHEMA_VERSION,
            "faction": faction,
            "rules": rules["factions"][faction],
            "ticks": faction_ticks,
            "event_records": faction_events,
            "budget_report": budget_report["factions"][faction],
            "final_state": result["exit_snapshot"]["factions"][faction],
        }
        faction_truth["faction_truth_hash"] = hash_without(
            faction_truth, "faction_truth_hash"
        )
        relative = f"truth/factions/{faction}.json"
        atomic_write_json(root / relative, faction_truth)
        faction_truth_files[faction] = relative

    report = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "scenario": scenario,
        "seed": seed,
        "event_count": len(events),
        "tick_start": 0,
        "tick_end": ticks - 1,
        "tick_span": ticks,
        "chunk_ticks": chunk_ticks,
        "faction_event_counts": game["faction_event_counts"],
        "events_hash": events_digest(events),
        "rules_hash": rules_hash(rules),
        "truth_hash": truth_payload["truth_hash"],
        "faction_truth_files": faction_truth_files,
        "truth_result_codes": truth_payload["result_codes"],
        "event_stream_bytes": len((input_dir / "events.jsonl").read_bytes()),
        "deterministic_schema": "canonical-json-sha256-v1",
    }
    report["generation_hash"] = hash_without(report, "generation_hash")
    atomic_write_json(report_dir / "generation.json", report)
    return {
        "root": str(root),
        "game": game,
        "rules": rules,
        "events": events,
        "truth": truth_payload,
        "generation_report": report,
    }
