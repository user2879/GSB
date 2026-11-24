"""Deterministic synthetic sample generation."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any


DEFAULT_SEED = 20261124
HALF_LIFE_MS = 86_400_000

TYPE_PLANS = {
    "WO-0001": ["login_anomaly", "device_risk", "abuse_report"],
    "WO-0002": ["payment_fraud", "appeal_context"],
    "WO-0003": ["abuse_report", "device_risk", "refund_request"],
    "WO-0004": ["login_anomaly", "payment_fraud", "appeal_context", "device_risk"],
    "WO-0005": ["login_anomaly", "device_risk", "appeal_context"],
    "WO-0006": ["payment_fraud", "device_risk", "refund_request", "abuse_report"],
    "WO-0007": ["appeal_context", "refund_request", "login_anomaly"],
    "WO-0008": ["baseline_signal"],
}

ROLE_PLAN = {
    "WO-0001": ["system", "agent", "customer"],
    "WO-0002": ["system", "customer"],
    "WO-0003": ["customer", "system", "agent"],
    "WO-0004": ["system", "agent", "customer", "reviewer"],
    "WO-0005": ["agent", "system", "customer"],
    "WO-0006": ["system", "system", "customer", "agent"],
    "WO-0007": ["customer", "agent", "reviewer"],
    "WO-0008": ["system"],
}

WEIGHTS = {
    "login_anomaly": 1.2,
    "payment_fraud": 1.45,
    "abuse_report": 1.1,
    "device_risk": 0.95,
    "appeal_context": 0.55,
    "refund_request": 0.7,
    "baseline_signal": 0.5,
}

ROLE_CREDIBILITY = {
    "system": 1.0,
    "agent": 0.85,
    "customer": 0.7,
    "reviewer": 0.9,
}

TERMS = [
    "refund",
    "chargeback",
    "identity",
    "device",
    "urgent",
    "repeat",
    "verified",
    "dispute",
    "login",
    "risk",
]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _confidence(rng: random.Random) -> float:
    return round(rng.uniform(0.55, 0.99), 4)


def generate_sample(out_dir: str | Path, *, seed: int = DEFAULT_SEED, overwrite: bool = True) -> dict[str, Any]:
    out_dir = Path(out_dir)
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    evidence_rows: list[dict[str, Any]] = []
    base_times: dict[str, int] = {}
    for workorder_index, (workorder_id, types) in enumerate(TYPE_PLANS.items()):
        base_ms = 1_700_000_000_000 + workorder_index * 10_000_000
        base_times[workorder_id] = base_ms
        roles = ROLE_PLAN[workorder_id]
        for index, evidence_type in enumerate(types):
            if index == 0:
                offset = 0
            elif index == 1:
                offset = HALF_LIFE_MS
            elif index == 2:
                offset = 2 * HALF_LIFE_MS
            else:
                offset = rng.randrange(HALF_LIFE_MS, 3 * HALF_LIFE_MS)
            term_count = 2 + rng.randrange(3)
            terms = rng.sample(TERMS, term_count)
            confidence = 0.5 if workorder_id == "WO-0008" else _confidence(rng)
            evidence_rows.append(
                {
                    "workorder_id": workorder_id,
                    "evidence_seq": index + 1,
                    "evidence_type": evidence_type,
                    "source_role": roles[index],
                    "event_ms": base_ms + offset,
                    "terms": terms,
                    "confidence": confidence,
                }
            )
    evidence_rows.sort(key=lambda row: (row["workorder_id"], row["evidence_seq"]))
    with (out_dir / "evidence.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in evidence_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

    rules = {
        "schema_version": 1,
        "evidence_type_weights": WEIGHTS,
        "source_role_credibility": ROLE_CREDIBILITY,
        "mutual_exclusions": [
            {"id": "mx-001", "left_type": "payment_fraud", "right_type": "appeal_context"},
            {"id": "mx-002", "left_type": "abuse_report", "right_type": "device_risk"},
        ],
        "supports": [
            {"id": "sup-001", "left_type": "login_anomaly", "right_type": "device_risk"},
            {"id": "sup-002", "left_type": "payment_fraud", "right_type": "device_risk"},
            {"id": "sup-003", "left_type": "refund_request", "right_type": "appeal_context"},
        ],
        "half_life_ms": HALF_LIFE_MS,
        "priority_thresholds": [
            {"priority": "P1", "min_score": 0.0},
            {"priority": "P2", "min_score": 0.25},
            {"priority": "P3", "min_score": 1.0},
        ],
    }
    _write_json(out_dir / "rules.json", rules)

    config = {
        "schema_version": 1,
        "concurrency": 4,
        "batch_evidence_budget": 100,
        "workorder_evidence_quota": 10,
        "fixed_decimal_width": 4,
        "quantization_level": "Q5",
        "snapshot_commit_interval": 2,
        "exclusion_strength": 0.08,
        "support_strength": 0.05,
        "zero_lower_bound": 0.0,
        "fusion_mode": "SUM",
    }
    _write_json(out_dir / "config.json", config)
    _write_json(out_dir / "input_meta.json", {"version": 0})

    edits = [
        {
            "op": "replace_evidence",
            "workorder_id": "WO-0002",
            "evidence_seq": 2,
            "evidence": {
                "workorder_id": "WO-0002",
                "evidence_seq": 2,
                "evidence_type": "refund_request",
                "source_role": "reviewer",
                "event_ms": base_times["WO-0002"] + 2 * HALF_LIFE_MS,
                "terms": ["refund", "verified", "urgent"],
                "confidence": 0.83,
            },
        },
        {
            "op": "set_evidence_type_weight",
            "evidence_type": "payment_fraud",
            "value": 0.9,
        },
        {
            "op": "set_role_credibility",
            "source_role": "customer",
            "value": 0.6,
        },
        {
            "op": "add_relation",
            "kind": "support",
            "relation": {
                "id": "sup-004",
                "left_type": "abuse_report",
                "right_type": "login_anomaly",
                "strength": 0.03,
            },
        },
        {
            "op": "delete_relation",
            "relation_id": "mx-002",
        },
        {
            "op": "delete_evidence",
            "workorder_id": "WO-0004",
            "evidence_seq": 1,
        },
        {
            "op": "set_half_life_ms",
            "value": 2 * HALF_LIFE_MS,
        },
        {
            "op": "set_priority_threshold",
            "priority": "P2",
            "min_score": 0.3,
        },
    ]
    _write_json(out_dir / "edits.json", {"commands": edits})

    injections = {
        "injections": [
            {
                "id": "inj-001",
                "workorder_id": "WO-0001",
                "stage": "load",
                "fault": "unknown_evidence_type",
                "evidence_seq": 1,
            },
            {
                "id": "inj-002",
                "workorder_id": "WO-0002",
                "stage": "load",
                "fault": "confidence_out_of_range",
                "evidence_seq": 2,
            },
            {
                "id": "inj-003",
                "workorder_id": "WO-0003",
                "stage": "load",
                "fault": "negative_timestamp",
                "evidence_seq": 1,
            },
            {
                "id": "inj-004",
                "workorder_id": "WO-0004",
                "stage": "decay",
                "fault": "zero_half_life",
                "evidence_seq": 1,
            },
            {
                "id": "inj-005",
                "workorder_id": "WO-0005",
                "stage": "relations",
                "fault": "relation_unknown_type",
                "evidence_seq": 1,
            },
        ]
    }
    _write_json(out_dir / "injections.json", injections)
    return {
        "seed": seed,
        "workorders": len(TYPE_PLANS),
        "evidence_count": len(evidence_rows),
        "edits": len(edits),
        "injections": len(injections["injections"]),
        "path": str(out_dir),
    }
