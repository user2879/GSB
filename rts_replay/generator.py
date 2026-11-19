"""Deterministic event-stream and rule-configuration generator.

A single :class:`numpy.random.Generator` seeded with the fixed match seed drives
all structural choices (action types, amounts, targets) through a fixed nested
loop, so the same seed always produces the exact same byte stream. Per-event
``rng_seed_delta`` values are assigned from a deterministic global ordinal and
must not be confused with the structural RNG: the synthesis RNG substreams are
re-derived independently in :mod:`rts_replay.rng`.
"""
from __future__ import annotations

import json

from .rules import rules_from_dict
import os
from dataclasses import dataclass

import numpy as np

from .events import (
    ACTION_COLLECT,
    ACTION_MOVE,
    ACTION_PRODUCE,
    Event,
)
from .hashing import canonical_dumps, sha256_bytes
from .rng import RNG_CHAIN_VERSION
from .rules import FactionRules, RuleConfig

COLLECT_TARGETS = ("mine_a", "mine_b", "mine_c")
ORDER_TARGETS = ("base", "front", "rally")


@dataclass(frozen=True)
class ScenarioProfile:
    name: str
    seed: int
    ticks: int
    queue_ticks: int
    faction_specs: tuple[tuple[str, int, int, int, int], ...]
    kind: str = "normal"


PROFILES: tuple[ScenarioProfile, ...] = (
    ScenarioProfile(
        name="normal_match",
        seed=20261119,
        ticks=60,
        queue_ticks=3,
        faction_specs=(
            ("red", 1000, 120, 80, 3),
            ("blue", 1000, 100, 90, 3),
            ("green", 1200, 140, 70, 2),
        ),
    ),
    ScenarioProfile(
        name="resource_overflow_match",
        seed=20261120,
        ticks=30,
        queue_ticks=3,
        faction_specs=(
            ("red", 120, 200, 80, 3),
            ("blue", 150, 220, 90, 2),
        ),
        kind="overflow",
    ),
    ScenarioProfile(
        name="contention_match",
        seed=20261121,
        ticks=30,
        queue_ticks=2,
        faction_specs=(
            ("red", 2000, 300, 40, 1),
            ("blue", 2000, 300, 40, 2),
        ),
        kind="contention",
    ),
    ScenarioProfile(
        name="fast_jump_match",
        seed=20261122,
        ticks=120,
        queue_ticks=2,
        faction_specs=(
            ("red", 1500, 150, 100, 4),
            ("blue", 1500, 160, 110, 4),
            ("green", 1500, 140, 90, 3),
        ),
        kind="fast",
    ),
    ScenarioProfile(
        name="seq_gap_match",
        seed=20261123,
        ticks=20,
        queue_ticks=3,
        faction_specs=(
            ("red", 1000, 120, 80, 3),
            ("blue", 1000, 100, 90, 3),
        ),
        kind="seq_gap",
    ),
)


def build_rules(profile: ScenarioProfile) -> RuleConfig:
    factions = tuple(
        FactionRules(
            faction=name,
            resource_cap=resource_cap,
            collect_rate=collect_rate,
            unit_cost=unit_cost,
            max_concurrent_commands=max_concurrent_commands,
        )
        for name, resource_cap, collect_rate, unit_cost, max_concurrent_commands in profile.faction_specs
    )
    config = RuleConfig(
        match_seed=profile.seed,
        rng_chain_version=RNG_CHAIN_VERSION,
        queue_ticks=profile.queue_ticks,
        factions=factions,
    )
    config.validate()
    return config


def _choose_action(rng: np.random.Generator) -> str:
    return str(rng.choice(
        [ACTION_COLLECT, ACTION_PRODUCE, ACTION_MOVE], p=(0.5, 0.3, 0.2)
    ))


def generate_events(profile: ScenarioProfile, config: RuleConfig) -> list[Event]:
    rng = np.random.default_rng(profile.seed)
    events: list[Event] = []
    next_seq = {name: 0 for name in config.faction_names()}
    delta_counter = 0

    for tick in range(1, profile.ticks + 1):
        for faction in config.factions:
            if profile.kind == "contention" and faction.faction == "red":
                event_count = int(rng.integers(2, 5))
                forced = [ACTION_PRODUCE] * event_count
            else:
                event_count = int(rng.integers(1, 4))
                forced = [_choose_action(rng) for _ in range(event_count)]

            for action in forced:
                rules = config.get(faction.faction)
                if action == ACTION_COLLECT:
                    amount = int(rng.integers(1, rules.collect_rate * 2 + 1))
                    target = str(rng.choice(COLLECT_TARGETS))
                elif action == ACTION_PRODUCE:
                    amount = int(rng.integers(1, 4))
                    target = str(rng.choice(ORDER_TARGETS))
                else:
                    amount = int(rng.integers(1, 6))
                    target = str(rng.choice(ORDER_TARGETS))

                delta_counter += 1
                events.append(
                    Event(
                        tick=tick,
                        actor=faction.faction,
                        action=action,
                        target=target,
                        amount=amount,
                        rng_seed_delta=delta_counter,
                        seq=next_seq[faction.faction],
                    )
                )
                next_seq[faction.faction] += 1

    if profile.kind == "seq_gap":
        events = _inject_seq_gap(events)
    return events


def _inject_seq_gap(events: list[Event]) -> list[Event]:
    """Drop one red-faction event to create an actor-local sequence gap."""
    red_events = [idx for idx, event in enumerate(events) if event.actor == "red"]
    victim = red_events[len(red_events) // 2]
    return [event for idx, event in enumerate(events) if idx != victim]


def write_scenario(workspace_dir: str, profile: ScenarioProfile) -> dict:
    sample_dir = os.path.join(workspace_dir, "samples", profile.name)
    os.makedirs(sample_dir, exist_ok=True)
    config = build_rules(profile)
    events = generate_events(profile, config)

    events_path = os.path.join(sample_dir, "events.jsonl")
    payload = b"\n".join(canonical_dumps(event.to_dict()) for event in events) + b"\n"
    with open(events_path, "wb") as handle:
        handle.write(payload)

    rules_path = os.path.join(sample_dir, "rules.json")
    rules_payload = canonical_dumps(config.to_dict())
    with open(rules_path, "wb") as handle:
        handle.write(rules_payload)

    ticks = sorted({event.tick for event in events})
    manifest = {
        "scenario": profile.name,
        "kind": profile.kind,
        "match_seed": profile.seed,
        "rng_chain_version": config.rng_chain_version,
        "event_count": len(events),
        "tick_count": len(ticks),
        "first_tick": ticks[0],
        "last_tick": ticks[-1],
        "factions": list(config.faction_names()),
        "events_file": "events.jsonl",
        "rules_file": "rules.json",
        "events_sha256": sha256_bytes(payload),
        "rules_sha256": sha256_bytes(rules_payload),
    }
    manifest_path = os.path.join(sample_dir, "manifest.json")
    with open(manifest_path, "wb") as handle:
        handle.write(canonical_dumps(manifest))
    return manifest


def generate_all(workspace_dir: str) -> list[dict]:
    os.makedirs(os.path.join(workspace_dir, "samples"), exist_ok=True)
    manifests = [write_scenario(workspace_dir, profile) for profile in PROFILES]
    index_path = os.path.join(workspace_dir, "samples", "scenario_index.json")
    with open(index_path, "wb") as handle:
        handle.write(
            canonical_dumps(
                {"scenarios": [manifest["scenario"] for manifest in manifests]}
            )
        )
    return manifests


def load_rules(path: str) -> RuleConfig:
    with open(path, "r", encoding="utf-8") as handle:
        return rules_from_dict(json.loads(handle.read()))
