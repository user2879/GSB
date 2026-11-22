"""Deterministic local sample generation (fixed seed, no network).

Running this module (``python -m dialog_state_pipeline.sample_data``) writes a
self-contained sample directory containing:

* ``events.jsonl``          - append / replace / delete batches
* ``events_base.jsonl``     - the pre-edit stream used by incremental tests
* ``fsm.yaml``              - state machine + slot schema
* ``pipeline_config.json``  - incremental run configuration
* ``full_replay_config.json``- byte-identical policy, separate snapshot dir
* ``injection_config.json`` - independent fault injection file
"""

import json
import random
from pathlib import Path

import yaml

SEED = 20261122

FSM_YAML = {
    "version": 1,
    "states": ["greeting", "collecting", "confirming", "closed"],
    "initial_state": "greeting",
    "slot_types": {
        "city": {"type": "string", "max_length": 16,
                 "enum": ["beijing", "shanghai", "hangzhou", "shenzhen"]},
        "score": {"type": "int", "min": 0, "max": 100},
        "temperature": {"type": "float", "min": -50.0, "max": 50.0},
    },
    "slots": {
        "city": {"type": "city", "strategy": "overwrite"},
        "visit_count": {"type": "int", "strategy": "larger"},
        "comfort": {"type": "temperature", "strategy": "weighted"},
        "member_tier": {"type": "string", "strategy": "reject"},
    },
    "transitions": [
        {"id": "R1", "from": "greeting", "action": "start", "to": "collecting"},
        {"id": "R2", "from": "collecting", "action": "provide", "to": "collecting"},
        {"id": "R3", "from": "collecting", "action": "confirm", "to": "confirming"},
        {"id": "R4", "from": "confirming", "action": "confirm", "to": "closed"},
        {"id": "R5", "from": "confirming", "action": "revise", "to": "collecting"},
    ],
}

# A path that stays legal against the FSM rule table:
# greeting -> collecting (R1), stays collecting via provide (R2), loops
# confirming <-> collecting with confirm/revise (R3/R5) and finally closes.
ACTIONS = ["start", "provide", "provide", "provide",
           "confirm", "revise", "provide", "confirm", "confirm"]
CITIES = ["beijing", "shanghai", "hangzhou", "shenzhen"]
TIERS = ["silver", "gold", "platinum"]


def _slot_payload(rng, turn_index):
    updates = [
        {"slot": "city", "value": rng.choice(CITIES),
         "confidence": rng.choice([7000, 8000, 9000, 10000])},
        {"slot": "visit_count", "value": rng.randint(1, 40),
         "confidence": rng.choice([6000, 7500, 9000])},
        {"slot": "comfort", "value": round(rng.uniform(10.0, 30.0), 2),
         "confidence": rng.choice([5000, 8000, 9500])},
    ]
    if turn_index % 3 == 0:
        updates.append({"slot": "member_tier", "value": rng.choice(TIERS),
                        "confidence": 9000})
    return updates


def generate_events(num_sessions=4, turns_per_session=8, seed=SEED):
    """Return ``(base_lines, edited_lines)`` as JSON-serialisable dicts."""
    rng = random.Random(seed)
    base = []
    sessions = [f"S{idx:03d}" for idx in range(1, num_sessions + 1)]
    for sid in sessions:
        for turn_no in range(1, turns_per_session + 1):
            action = ACTIONS[(turn_no - 1) % len(ACTIONS)]
            base.append({
                "session_id": sid,
                "op": "append",
                "turn": turn_no,
                "batch_id": f"B{((turn_no - 1) // 3) + 1}",
                "utterance": f"{sid}-utterance-{turn_no}",
                "action": action,
                "slot_updates": _slot_payload(rng, turn_no),
            })

    edited = [dict(item) for item in base]
    # Edit batch 4: replace one utterance per session (same turn position).
    edit_batch = "B4"
    for sid in sessions:
        edited.append({
            "session_id": sid, "op": "replace", "turn": 4,
            "batch_id": edit_batch,
            "utterance": f"{sid}-utterance-4-REVISED",
            "action": "provide",
            "slot_updates": [
                {"slot": "city", "value": rng.choice(CITIES), "confidence": 9999},
                {"slot": "comfort", "value": round(rng.uniform(15.0, 25.0), 2),
                 "confidence": 8888},
            ],
        })
    # Edit batch 5: delete turn 2 of S001 and append a fresh turn to S002.
    edited.append({"session_id": "S001", "op": "delete", "turn": 2,
                   "batch_id": "B5"})
    edited.append({
        "session_id": "S002", "op": "append",
        "batch_id": "B5",
        "utterance": "S002-utterance-9-APPENDED", "action": "confirm",
        "slot_updates": [
            {"slot": "city", "value": "hangzhou", "confidence": 9500},
            {"slot": "visit_count", "value": 42, "confidence": 9500},
        ],
    })
    return base, edited


def _write_jsonl(path, lines):
    with open(path, "w", encoding="utf-8") as handle:
        for item in lines:
            handle.write(json.dumps(item, sort_keys=True, ensure_ascii=False))
            handle.write("\n")


def write_sample_dir(target):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    base, edited = generate_events()
    _write_jsonl(target / "events_base.jsonl", base)
    _write_jsonl(target / "events.jsonl", edited)
    with open(target / "fsm.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(FSM_YAML, handle, sort_keys=True, allow_unicode=True)

    common = {
        "version": 1,
        "max_workers": 3,
        "global_budget": 64,
        "default_quota": 12,
        "session_quotas": {"S001": 12, "S002": 14},
        "confidence_max": 10000,
        "fixed_point_decimals": 6,
    }
    incremental = dict(common, name="incremental",
                       snapshot_dir=str(target / "snapshot_inc"))
    full = dict(common, name="full_replay",
                snapshot_dir=str(target / "snapshot_full"))
    budget_tight = dict(common, name="budget_demo",
                        global_budget=10,
                        snapshot_dir=str(target / "snapshot_budget"))
    with open(target / "pipeline_config.json", "w", encoding="utf-8") as handle:
        json.dump(incremental, handle, indent=2, sort_keys=True)
    with open(target / "full_replay_config.json", "w", encoding="utf-8") as handle:
        json.dump(full, handle, indent=2, sort_keys=True)
    with open(target / "budget_config.json", "w", encoding="utf-8") as handle:
        json.dump(budget_tight, handle, indent=2, sort_keys=True)

    injection = {
        "injections": [
            {
                "injection_id": "INJ-BAD-SLOT-01",
                "kind": "bad_slot_value",
                "session_id": "S003",
                "batch_id": "B2",
                "slot": "comfort",
                "bad_value": "not-a-number",
                "confidence": 9000,
            },
            {
                "injection_id": "INJ-TURN-JUMP-01",
                "kind": "turn_jump",
                "session_id": "S004",
                "batch_id": "B3",
                "gap": 5,
            },
            {
                "injection_id": "INJ-UNKNOWN-TX-01",
                "kind": "unknown_transition",
                "session_id": "S002",
                "batch_id": "B3",
                "action": "fly_to_mars",
            },
            {
                "injection_id": "INJ-QUOTA-01",
                "kind": "quota_override",
                "session_id": "S001",
                "quota": 2,
            },
        ]
    }
    with open(target / "injection_config.json", "w", encoding="utf-8") as handle:
        json.dump(injection, handle, indent=2, sort_keys=True)
    return target


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "samples"
    path = write_sample_dir(out)
    print(f"sample directory written to {path}")
