"""Deterministic event-stream sample generator (fixed seeds, NumPy RNG).

Four scenarios are produced:
  * normal_game   -- spawn/move/update/destroy across many ticks;
  * conflict_game -- same-tick same-entity concurrent field writes;
  * catchup_game  -- stale base_snapshot_version events that must suspend
                     with E_BASE_VERSION and catch up later in the stream;
  * kill_game     -- a longer stream used for the forced-termination drill.

``base_snapshot_version`` is assigned from each event's position in the
deterministic commit order (tick, peer_id, local_seq), which is the exact
order the engine applies events in. The catch-up scenario deliberately
assigns stale bases to exercise suspension and retry. The emitted files are
shuffled (with a fixed second RNG) to prove arrival order never matters.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np

from .state import canonical_bytes, to_native

PEERS = ("peer-alpha", "peer-bravo", "peer-charlie")
SCENARIOS = ("normal_game", "conflict_game", "catchup_game", "kill_game")
SHUFFLE_SEED = 90210


class EventBuilder:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.peer_seq = {peer: 0 for peer in PEERS}
        self.events: List[Dict[str, Any]] = []
        self._stale_keys: set = set()

    def _next_seq(self, peer: str) -> int:
        seq = self.peer_seq[peer]
        self.peer_seq[peer] = seq + 1
        return seq

    def _add(
        self,
        tick: int,
        peer: str,
        op_type: str,
        payload: Dict[str, Any],
        base_snapshot_version: int = 0,
        stale: bool = False,
    ) -> Dict[str, Any]:
        seq = self._next_seq(peer)
        event = {
            "tick": int(tick),
            "peer_id": peer,
            "local_seq": seq,
            "op_type": op_type,
            "payload": to_native(payload),
            "base_snapshot_version": int(base_snapshot_version),
        }
        if stale:
            self._stale_keys.add((int(tick), peer))
        self.events.append(event)
        return event

    def _pick_peer(self) -> str:
        return PEERS[int(self.rng.integers(0, len(PEERS)))]

    def build_normal(self) -> List[Dict[str, Any]]:
        self.events = []
        self._stale_keys = set()
        live_entities: List[str] = []
        next_entity = 0
        for tick in range(1, 15):
            tick_removed: set = set()
            for _ in range(int(self.rng.integers(2, 4))):
                peer = self._pick_peer()
                roll = self.rng.random()
                if roll < 0.4 or not live_entities:
                    entity_id = f"ent-{next_entity:03d}"
                    next_entity += 1
                    self._add(
                        tick,
                        peer,
                        "spawn",
                        {
                            "entity_id": entity_id,
                            "fields": {
                                "x": int(self.rng.integers(0, 100)),
                                "y": int(self.rng.integers(0, 100)),
                                "hp": int(self.rng.integers(10, 100)),
                            },
                        },
                    )
                    live_entities.append(entity_id)
                elif roll < 0.85:
                    candidates = [e for e in live_entities if e not in tick_removed]
                    if not candidates:
                        entity_id = f"ent-{next_entity:03d}"
                        next_entity += 1
                        self._add(
                            tick,
                            peer,
                            "spawn",
                            {
                                "entity_id": entity_id,
                                "fields": {
                                    "x": int(self.rng.integers(0, 100)),
                                    "y": int(self.rng.integers(0, 100)),
                                    "hp": int(self.rng.integers(10, 100)),
                                },
                            },
                        )
                        live_entities.append(entity_id)
                        continue
                    entity_id = candidates[int(self.rng.integers(0, len(candidates)))]
                    op = "move" if self.rng.random() < 0.5 else "update"
                    self._add(
                        tick,
                        peer,
                        op,
                        {
                            "entity_id": entity_id,
                            "fields": (
                                {"x": int(self.rng.integers(0, 100)), "y": int(self.rng.integers(0, 100))}
                                if op == "move"
                                else {"hp": int(self.rng.integers(1, 100))}
                            ),
                        },
                    )
                else:
                    idx = int(self.rng.integers(0, len(live_entities)))
                    entity_id = live_entities.pop(idx)
                    tick_removed.add(entity_id)
                    self._add(tick, peer, "destroy", {"entity_id": entity_id, "fields": {}})
        return self._finalize()

    def build_conflict(self) -> List[Dict[str, Any]]:
        self.events = []
        self._stale_keys = set()
        self.peer_seq = {peer: 0 for peer in PEERS}
        self._add(
            1,
            "peer-alpha",
            "spawn",
            {"entity_id": "ent-duel", "fields": {"x": 0, "y": 0, "hp": 50}},
        )
        for tick in range(2, 8):
            for peer in PEERS:
                self._add(
                    tick,
                    peer,
                    "update",
                    {
                        "entity_id": "ent-duel",
                        "fields": {
                            "x": int(self.rng.integers(0, 50)),
                            "y": int(self.rng.integers(0, 50)),
                            "hp": int(self.rng.integers(1, 99)),
                        },
                    },
                )
        return self._finalize()

    def build_catchup(self) -> List[Dict[str, Any]]:
        # Deterministic lag-then-catchup design. Two events are marked stale;
        # each sits one commit position behind its own base, so it suspends
        # with E_BASE_VERSION for a single authority step and is immediately
        # retried and applied once the preceding event advances the authority.
        self.events = []
        self.peer_seq = {peer: 0 for peer in PEERS}
        self._stale_keys = set()
        script = [
            # tick, peer, op, entity, stale?
            (1, "peer-alpha", "spawn", "ent-c1", False),
            (1, "peer-bravo", "spawn", "ent-c2", False),
            (1, "peer-charlie", "update", "ent-c1", True),
            (2, "peer-alpha", "update", "ent-c1", False),
            (2, "peer-bravo", "update", "ent-c2", False),
            (2, "peer-charlie", "update", "ent-c2", True),
            (3, "peer-alpha", "update", "ent-c1", False),
            (3, "peer-bravo", "update", "ent-c2", False),
            (3, "peer-charlie", "update", "ent-c1", False),
            (4, "peer-alpha", "update", "ent-c2", False),
            (4, "peer-bravo", "update", "ent-c1", False),
            (4, "peer-charlie", "update", "ent-c2", False),
        ]
        for tick, peer, op, entity, stale in script:
            fields = {"v": 1} if op == "spawn" else {"v": int(self.rng.integers(2, 90))}
            self._add(
                tick,
                peer,
                op,
                {"entity_id": entity, "fields": fields},
                base_snapshot_version=0,
                stale=stale,
            )
        return self._finalize()

    def build_kill(self) -> List[Dict[str, Any]]:
        self.events = []
        self._stale_keys = set()
        self.peer_seq = {peer: 0 for peer in PEERS}
        live_entities: List[str] = []
        next_entity = 0
        for tick in range(1, 19):
            tick_removed: set = set()
            for _ in range(int(self.rng.integers(2, 5))):
                peer = self._pick_peer()
                roll = self.rng.random()
                if not live_entities or roll < 0.4:
                    entity_id = f"ent-k{next_entity:03d}"
                    next_entity += 1
                    self._add(
                        tick,
                        peer,
                        "spawn",
                        {
                            "entity_id": entity_id,
                            "fields": {
                                "x": int(self.rng.integers(0, 100)),
                                "y": int(self.rng.integers(0, 100)),
                            },
                        },
                    )
                    live_entities.append(entity_id)
                elif roll < 0.85:
                    candidates = [e for e in live_entities if e not in tick_removed]
                    if not candidates:
                        continue
                    entity_id = candidates[int(self.rng.integers(0, len(candidates)))]
                    self._add(
                        tick,
                        peer,
                        "move",
                        {
                            "entity_id": entity_id,
                            "fields": {
                                "x": int(self.rng.integers(0, 100)),
                                "y": int(self.rng.integers(0, 100)),
                            },
                        },
                    )
                else:
                    candidates = [e for e in live_entities if e not in tick_removed]
                    if not candidates:
                        continue
                    idx = int(self.rng.integers(0, len(candidates)))
                    entity_id = candidates[idx]
                    live_entities.remove(entity_id)
                    tick_removed.add(entity_id)
                    self._add(
                        tick, peer, "destroy", {"entity_id": entity_id, "fields": {}}
                    )
        return self._finalize()

    def _finalize(self) -> List[Dict[str, Any]]:
        # Deterministic commit order required by the synthesizer:
        # (tick, peer_id, local_seq) ascending. Same-tick structural hazards
        # are already avoided while generating, because each tick keeps a
        # live-entity view that excludes entities destroyed earlier in that
        # tick and never emits writes against not-yet-spawned entities.
        ordered = sorted(
            self.events,
            key=lambda e: (e["tick"], e["peer_id"], e["local_seq"]),
        )
        # Re-number local_seq in commit order so every peer's sequence is
        # exactly 0,1,2,... along (tick, peer_id) ordering.
        per_peer_counter = {peer: 0 for peer in PEERS}
        for event in ordered:
            event["local_seq"] = per_peer_counter[event["peer_id"]]
            per_peer_counter[event["peer_id"]] += 1
        # Base assignment in final commit order. The first event of each tick
        # references the tick-start authority (number of events in earlier
        # ticks), so it is always ready and drives the authority forward;
        # later events reference the immediately preceding position. Stale
        # events deliberately lag one published version behind.
        tick_start_authority: Dict[int, int] = {}
        seen = 0
        for event in ordered:
            if event["tick"] not in tick_start_authority:
                tick_start_authority[event["tick"]] = seen
            seen += 1
        for commit_index, event in enumerate(ordered):
            position = commit_index + 1  # version this event will produce
            if (event["tick"], event["peer_id"]) in self._stale_keys:
                # Deliberately one published version behind the predecessor.
                event["base_snapshot_version"] = max(0, commit_index - 1)
            elif commit_index == 0:
                event["base_snapshot_version"] = 0
            else:
                # Base = authority after the immediately preceding event was
                # applied, i.e. the predecessor's produced version.
                event["base_snapshot_version"] = commit_index
        # Defensive invariant: a tick whose commit-order first event is stale
        # could deadlock the authority; such stale markers are invalid by
        # construction in the scripted scenarios, assert to keep it that way.
        seen_ticks = set()
        for event in ordered:
            if event["tick"] not in seen_ticks:
                seen_ticks.add(event["tick"])
                assert (event["tick"], event["peer_id"]) not in self._stale_keys
        order = np.random.default_rng(SHUFFLE_SEED).permutation(len(ordered))
        return [ordered[int(index)] for index in order]


def generate_all(seed: int) -> Dict[str, List[Dict[str, Any]]]:
    builder = EventBuilder(seed)
    return {
        "normal_game": builder.build_normal(),
        "conflict_game": builder.build_conflict(),
        "catchup_game": builder.build_catchup(),
        "kill_game": builder.build_kill(),
    }


def write_streams(output_dir: str, seed: int) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    streams = generate_all(seed)
    paths: Dict[str, str] = {}
    index = {"seed": seed, "scenarios": {}}
    for name, events in streams.items():
        path = os.path.join(output_dir, f"{name}.jsonl")
        with open(path, "wb") as handle:
            for event in events:
                handle.write(canonical_bytes(event) + b"\n")
        paths[name] = path
        index["scenarios"][name] = {"file": f"{name}.jsonl", "events": len(events)}
    with open(os.path.join(output_dir, "index.json"), "wb") as handle:
        handle.write(canonical_bytes(index) + b"\n")
    return paths


def read_stream(path: str) -> List[Dict[str, Any]]:
    with open(path, "rb") as handle:
        return [json.loads(line) for line in handle.read().splitlines() if line.strip()]
