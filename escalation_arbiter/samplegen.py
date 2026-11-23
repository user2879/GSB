"""Deterministic sample generator (no external randomness source)."""

import json
import os
import random

TYPES = ["LOGIN", "PAYMENT", "COMPLAINT", "DEVICE", "ACCESS", "REFUND"]
ROLES = ["customer", "agent", "system", "auditor", "partner"]
TERM_POOL = ["login", "payment", "refund", "device", "risk", "complaint",
             "location", "amount", "token", "session", "fraud", "verify"]


def _write(path, payload):
    with open(path, "w", encoding="utf-8") as fh:
        if path.endswith(".jsonl"):
            for row in payload:
                fh.write(json.dumps(row, ensure_ascii=False,
                                    sort_keys=True) + "\n")
        else:
            json.dump(payload, fh, ensure_ascii=False, indent=2,
                      sort_keys=True)


def generate(samples_dir, ticket_count=12, seed=20261123):
    os.makedirs(samples_dir, exist_ok=True)
    rng = random.Random(seed)

    type_weights = {etype: 300 + rng.randint(0, 700) for etype in TYPES}
    role_cred = {role: 450 + rng.randint(0, 500) for role in ROLES}
    mutex = [
        {"rid": "1", "kind": "mutex", "left": "PAYMENT",
         "right": "REFUND"},
        {"rid": "2", "kind": "mutex", "left": "LOGIN",
         "right": "DEVICE"},
        {"rid": "3", "kind": "mutex", "left": "COMPLAINT",
         "right": "ACCESS"},
    ]
    support = [
        {"rid": "1", "kind": "support", "left": "LOGIN",
         "right": "DEVICE"},
        {"rid": "2", "kind": "support", "left": "PAYMENT",
         "right": "COMPLAINT"},
    ]
    priority_tiers = [
        {"limit": 600, "level": 1, "label": "LOW"},
        {"limit": 1200, "level": 2, "label": "MEDIUM"},
        {"limit": 2000, "level": 3, "label": "HIGH"},
    ]
    reason_codes = [
        {"code": "MX1", "name": "payment_refund_conflict"},
        {"code": "MX2", "name": "login_device_conflict"},
        {"code": "MX3", "name": "complaint_access_conflict"},
        {"code": "SP1", "name": "login_device_support"},
        {"code": "SP2", "name": "payment_complaint_support"},
        {"code": "MX9", "name": "rare_conflict_slot"},
        {"code": "SP9", "name": "rare_support_slot"},
    ]
    rules = {
        "type_weights": type_weights,
        "role_credibility": role_cred,
        "mutex": mutex,
        "support": support,
        "priority_tiers": priority_tiers,
        "reason_codes": reason_codes,
    }
    _write(os.path.join(samples_dir, "rules.json"), rules)

    config = {
        "decimals": 3,
        "quantum": 5,
        "half_life_ms": 90000,
        "deduct_permille": 700,
        "bonus_permille": 1200,
        "fusion": "sum",
        "zero_floor": 0,
        "budget_ms": 500,
        "concurrency": 4,
        "snapshot_interval": 3,
        "conf_min_permille": 100,
        "conf_max_permille": 1000,
    }
    _write(os.path.join(samples_dir, "config.json"), config)

    evidence_rows = []
    for ticket_index in range(ticket_count):
        tid = f"T{ticket_index:04d}"
        base = rng.randint(0, 5000)
        count = rng.randint(3, 7)
        current = base
        for seq in range(1, count + 1):
            current += rng.randint(0, 120000)
            evidence_rows.append({
                "ticket_id": tid,
                "seq": seq,
                "etype": rng.choice(TYPES),
                "role": rng.choice(ROLES),
                "ts": current,
                "terms": rng.sample(TERM_POOL, k=rng.randint(2, 4)),
                "confidence": rng.randint(200, 1000),
            })
    _write(os.path.join(samples_dir, "evidence.jsonl"), evidence_rows)

    commands = [
        {"op": "set_version", "version": "v1"},
        {"op": "set_version", "version": "v2"},
        {"op": "set_type_weight", "etype": "PAYMENT", "weight": 920},
        {"op": "set_role_cred", "role": "system", "credibility": 980},
        {"op": "add_relation", "rid": "4", "kind": "mutex",
         "left": "DEVICE", "right": "REFUND"},
        {"op": "add_reason_code", "code": "MX4",
         "name": "device_refund_conflict"},
        {"op": "set_half_life", "half_life_ms": 45000},
        {"op": "replace_evidence", "evidence": evidence_rows[0]},
        {"op": "remove_relation", "rid": "4", "kind": "mutex"},
        {"op": "remove_reason_code", "code": "MX4"},
        {"op": "set_tier_limit", "level": 2, "limit": 1100},
        {"op": "set_version", "version": "v3"},
        {"op": "set_decimals", "decimals": 4},
        {"op": "set_decimals", "decimals": 3},
        {"op": "set_version", "version": "v2"},
    ]
    _write(os.path.join(samples_dir, "commands.json"),
           {"commands": commands})

    first_tid = f"T{0:04d}"
    first_seq = evidence_rows[0]["seq"]
    injections = [
        {"inj_id": "INJ-001", "kind": "timestamp_reorder",
         "tid": f"T{ticket_count - 1:04d}", "stage": "decay"},
        {"inj_id": "INJ-002", "kind": "relation_self_reference",
         "tid": f"T{ticket_count - 2:04d}", "stage": "relation",
         "target_seq": 1},
        {"inj_id": "INJ-003", "kind": "evidence_type_unknown",
         "tid": f"T{ticket_count - 3:04d}", "stage": "decay",
         "target_seq": first_seq},
        {"inj_id": "INJ-004", "kind": "fixed_width_overflow",
         "tid": f"T{ticket_count - 4:04d}", "stage": "arbitration"},
    ]
    _write(os.path.join(samples_dir, "injections.json"),
           {"injections": injections})
    return {
        "rules": os.path.join(samples_dir, "rules.json"),
        "config": os.path.join(samples_dir, "config.json"),
        "evidence": os.path.join(samples_dir, "evidence.jsonl"),
        "commands": os.path.join(samples_dir, "commands.json"),
        "injections": os.path.join(samples_dir, "injections.json"),
    }
