from __future__ import annotations

import argparse
import json
import random
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_workbench.canonical import decimal_text  # noqa: E402


USERS = [f"u-{index:03d}" for index in range(1, 9)]
ITEMS = [f"i-{index:03d}" for index in range(1, 7)]
BEHAVIORS = ["view", "click", "favorite", "cart", "purchase", "refund"]


def shift_precision(definition: dict[str, Any], shift: int) -> dict[str, Any]:
    if shift == 0:
        return definition
    result = dict(definition)
    old_places = int(result["decimal_places"])
    new_places = max(0, min(18, old_places + shift))
    quantum = Decimal(str(result["quantization_step"]))
    if shift > 0:
        quantum = quantum.scaleb(-shift)
    elif shift < 0:
        quantum = quantum.scaleb(-shift)
    result["decimal_places"] = new_places
    result["quantization_step"] = decimal_text(quantum)
    return result


def build_definitions(shift: int = 0) -> list[dict[str, Any]]:
    definitions = [
        {
            "name": "user_event_count_5m",
            "side": "user",
            "source_field": None,
            "window_ms": 300_000,
            "aggregate": "count",
            "decimal_places": 0,
            "quantization_step": "1",
            "upstream_features": [],
        },
        {
            "name": "user_amount_sum_10m",
            "side": "user",
            "source_field": "amount",
            "window_ms": 600_000,
            "aggregate": "sum",
            "decimal_places": 4,
            "quantization_step": "0.0001",
            "upstream_features": [],
        },
        {
            "name": "user_risk_latest_5m",
            "side": "user",
            "source_field": "risk_score",
            "window_ms": 300_000,
            "aggregate": "latest",
            "decimal_places": 4,
            "quantization_step": "0.0001",
            "upstream_features": [],
        },
        {
            "name": "user_value_composite_10m",
            "side": "user",
            "source_field": None,
            "window_ms": 600_000,
            "aggregate": "sum",
            "decimal_places": 6,
            "quantization_step": "0.000001",
            "upstream_features": ["user_amount_sum_10m", "user_risk_latest_5m"],
        },
        {
            "name": "item_event_count_5m",
            "side": "item",
            "source_field": None,
            "window_ms": 300_000,
            "aggregate": "count",
            "decimal_places": 0,
            "quantization_step": "1",
            "upstream_features": [],
        },
        {
            "name": "item_amount_sum_10m",
            "side": "item",
            "source_field": "amount",
            "window_ms": 600_000,
            "aggregate": "sum",
            "decimal_places": 2,
            "quantization_step": "0.01",
            "upstream_features": [],
        },
        {
            "name": "item_risk_mean_10m",
            "side": "item",
            "source_field": "risk_score",
            "window_ms": 600_000,
            "aggregate": "mean",
            "decimal_places": 4,
            "quantization_step": "0.0001",
            "upstream_features": [],
        },
        {
            "name": "item_value_composite_10m",
            "side": "item",
            "source_field": None,
            "window_ms": 600_000,
            "aggregate": "sum",
            "decimal_places": 4,
            "quantization_step": "0.0001",
            "upstream_features": ["item_amount_sum_10m", "item_risk_mean_10m"],
        },
    ]
    return [shift_precision(item, shift) for item in definitions]


def make_event(
    sequence: int,
    timestamp_ms: int,
    user_id: str,
    item_id: str,
    behavior: str,
    amount: float,
    dwell_ms: int,
    risk_score: float,
    injection_id: str | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "event_id": f"evt-{sequence:06d}",
        "user_id": user_id,
        "item_id": item_id,
        "behavior": behavior,
        "timestamp_ms": timestamp_ms,
        "context": {
            "amount": round(amount, 6),
            "dwell_ms": dwell_ms,
            "risk_score": round(risk_score, 6),
            "session_depth": max(1, dwell_ms // 1000),
        },
    }
    if injection_id:
        event["_injection_id"] = injection_id
    return event


def build_events(count: int = 42, timestamp_start: int = 1_700_000_000_000) -> list[dict[str, Any]]:
    rng = random.Random(188)
    events: list[dict[str, Any]] = []
    timestamp = timestamp_start
    for sequence in range(1, count + 1):
        timestamp += rng.randint(1_500, 18_000)
        amount = rng.uniform(2.0, 800.0)
        dwell_ms = rng.randint(500, 90_000)
        risk_score = rng.uniform(0.0, 1.0)
        events.append(
            make_event(
                sequence,
                timestamp,
                rng.choice(USERS),
                rng.choice(ITEMS),
                rng.choice(BEHAVIORS),
                amount,
                dwell_ms,
                risk_score,
            )
        )
    return events


def append_events(
    events: list[dict[str, Any]], start_sequence: int, count: int, seed: int
) -> tuple[list[dict[str, Any]], int]:
    rng = random.Random(seed)
    timestamp = events[-1]["timestamp_ms"]
    appended: list[dict[str, Any]] = []
    sequence = start_sequence
    for offset in range(count):
        timestamp += rng.randint(2_000, 12_000)
        event = make_event(
            sequence,
            timestamp,
            rng.choice(USERS),
            rng.choice(ITEMS),
            rng.choice(BEHAVIORS),
            rng.uniform(5.0, 600.0),
            rng.randint(800, 70_000),
            rng.uniform(0.0, 1.0),
        )
        appended.append(event)
        sequence += 1
    return appended, sequence


def build_plan(
    events: list[dict[str, Any]], definitions: list[dict[str, Any]], shift: int
) -> dict[str, Any]:
    first_appends, sequence = append_events(events, len(events) + 1, 4, seed=1881)
    second_appends, sequence = append_events(
        events + first_appends, sequence, 3, seed=1885
    )
    correction_id = events[-4]["event_id"]
    correction = next(item for item in events if item["event_id"] == correction_id)
    by_name = {item["name"]: item for item in definitions}
    composite = dict(by_name["user_value_composite_10m"])
    composite["window_ms"] = 900_000
    if shift == 0:
        composite["decimal_places"] = 8
        composite["quantization_step"] = "0.00000001"
    else:
        composite = shift_precision(composite, shift)
    count_definition = dict(by_name["user_event_count_5m"])
    count_definition["window_ms"] = 600_000
    return {
        "seed": 188,
        "batches": [
            {
                "batch_id": "append-4",
                "appends": first_appends,
                "backfills": [],
                "definition_changes": [],
            },
            {
                "batch_id": "backfill-amount",
                "appends": [],
                "backfills": [
                    {
                        "event_id": correction["event_id"],
                        "patch": {
                            "context": {
                                "amount": round(float(correction["context"]["amount"]) + 7.25, 6),
                                "risk_score": round(
                                    min(1.0, float(correction["context"]["risk_score"]) + 0.05), 6
                                ),
                            }
                        },
                    }
                ],
                "definition_changes": [],
            },
            {
                "batch_id": "change-composite-precision",
                "appends": [],
                "backfills": [],
                "definition_changes": [
                    {"op": "upsert", "definition": composite},
                ],
            },
            {
                "batch_id": "append-3-and-count-window",
                "appends": second_appends,
                "backfills": [],
                "definition_changes": [
                    {"op": "upsert", "definition": count_definition},
                ],
            },
        ],
    }


def build_config(concurrency: int = 4, point_budget: int = 5000) -> dict[str, Any]:
    return {
        "version": 1,
        "weights": {
            "user_event_count_5m": 0.20,
            "user_amount_sum_10m": 0.50,
            "user_risk_latest_5m": 0.30,
            "user_value_composite_10m": 0.40,
            "item_event_count_5m": 0.15,
            "item_amount_sum_10m": 0.45,
            "item_risk_mean_10m": -0.25,
            "item_value_composite_10m": 0.35,
        },
        "fusion_temperature": 2.0,
        "fusion_bias": 0.05,
        "clamp_interval": {"min": -50.0, "max": 50.0},
        "concurrency": concurrency,
        "point_budget_per_batch": point_budget,
        "snapshot_commit_interval": 1,
        "score_decimal_places": 4,
        "score_quantization_step": "0.0001",
        "allowed_behaviors": BEHAVIORS,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_events(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def build_injections(
    definitions: list[dict[str, Any]], events: list[dict[str, Any]], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    invalid_definitions = list(definitions)
    invalid_definitions.extend(
        [
            {
                "name": "cycle_alpha",
                "side": "user",
                "source_field": None,
                "window_ms": 60_000,
                "aggregate": "count",
                "decimal_places": 0,
                "quantization_step": "1",
                "upstream_features": ["cycle_beta"],
                "_injection_id": "INJ-004",
            },
            {
                "name": "cycle_beta",
                "side": "user",
                "source_field": None,
                "window_ms": 60_000,
                "aggregate": "count",
                "decimal_places": 0,
                "quantization_step": "1",
                "upstream_features": ["cycle_alpha"],
                "_injection_id": "INJ-004",
            },
            {
                "name": "bad_quantum",
                "side": "item",
                "source_field": "amount",
                "window_ms": 60_000,
                "aggregate": "sum",
                "decimal_places": 2,
                "quantization_step": "0.0000001",
                "upstream_features": [],
                "_injection_id": "INJ-005",
            },
        ]
    )
    write_json(output_dir / "definitions.json", {"features": invalid_definitions})

    injection_events = [dict(item) for item in events[:18]]
    cursor = injection_events[-1]["timestamp_ms"]
    nonfinite = make_event(
        9001,
        cursor + 1_000,
        USERS[0],
        ITEMS[0],
        "purchase",
        1.0,
        5_000,
        0.2,
        injection_id="INJ-001",
    )
    nonfinite["context"]["risk_score"] = float("nan")
    unknown = make_event(
        9002,
        cursor + 2_000,
        USERS[1],
        ITEMS[1],
        "teleport",
        3.0,
        4_000,
        0.3,
        injection_id="INJ-002",
    )
    continuation = make_event(
        9003,
        cursor + 3_000,
        USERS[2],
        ITEMS[2],
        "view",
        4.0,
        2_000,
        0.1,
    )
    injection_events.extend([nonfinite, unknown, continuation])
    write_events(output_dir / "events.jsonl", injection_events)

    late_event = make_event(
        9100,
        max(1, cursor - 5_000),
        USERS[0],
        ITEMS[0],
        "click",
        2.0,
        1_500,
        0.15,
        injection_id="INJ-003",
    )
    write_json(
        output_dir / "edits.json",
        {
            "seed": 188,
            "batches": [
                {
                    "batch_id": "inject-late-backfill",
                    "appends": [],
                    "backfills": [
                        {"_injection_id": "INJ-003", "late_event": late_event}
                    ],
                    "definition_changes": [],
                }
            ],
        },
    )
    write_json(
        output_dir / "injections.json",
        {
            "injections": [
                {
                    "id": "INJ-001",
                    "kind": "non_finite_context",
                    "business_code": "E_EVENT_INVALID",
                },
                {
                    "id": "INJ-002",
                    "kind": "unknown_behavior",
                    "business_code": "E_EVENT_INVALID",
                },
                {
                    "id": "INJ-003",
                    "kind": "late_backfill_timestamp",
                    "business_code": "E_EVENT_INVALID",
                },
                {
                    "id": "INJ-004",
                    "kind": "dependency_cycle",
                    "business_code": "E_DEFINITION_INVALID",
                },
                {
                    "id": "INJ-005",
                    "kind": "quantization_out_of_range",
                    "business_code": "E_DEFINITION_INVALID",
                },
                {
                    "id": "INJ-006",
                    "kind": "feature_exception",
                    "business_code": "E_FEATURE_COMPUTE",
                },
                {
                    "id": "INJ-007",
                    "kind": "score_exception",
                    "business_code": "E_SCORE_COMPUTE",
                },
            ],
            "faults": [
                {
                    "id": "INJ-006",
                    "stage": "feature",
                    "feature": "user_amount_sum_10m",
                    "exception_type": "InjectedFeatureFault",
                },
                {
                    "id": "INJ-007",
                    "stage": "score",
                    "entity_type": "item",
                    "entity_id": "i-001",
                    "feature": "item_risk_mean_10m",
                    "exception_type": "InjectedScoreFault",
                },
            ],
        },
    )


def generate(output_dir: Path, seed: int = 188, decimal_shift: int = 0) -> None:
    _ = seed  # The deterministic generator currently pins its seed to 188 by contract.
    output_dir.mkdir(parents=True, exist_ok=True)
    definitions = build_definitions(decimal_shift)
    events = build_events()
    plan = build_plan(events, definitions, decimal_shift)
    write_json(output_dir / "definitions.json", {"features": definitions})
    write_events(output_dir / "events.jsonl", events)
    config = build_config()
    (output_dir / "scoring.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    low_budget = build_config(point_budget=20)
    (output_dir / "scoring_low_budget.yaml").write_text(
        yaml.safe_dump(low_budget, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    write_json(output_dir / "edits.json", plan)
    build_injections(definitions, events, output_dir / "injections")
    write_json(
        output_dir / "manifest.json",
        {
            "seed": 188,
            "files": [
                "definitions.json",
                "events.jsonl",
                "scoring.yaml",
                "edits.json",
                "injections/injections.json",
            ],
            "event_count": len(events),
            "batch_count": len(plan["batches"]),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic local workbench samples")
    parser.add_argument("--output-dir", type=Path, default=Path("samples"))
    parser.add_argument("--seed", type=int, default=188)
    parser.add_argument("--decimal-shift", type=int, default=0)
    args = parser.parse_args()
    generate(args.output_dir, args.seed, args.decimal_shift)
    print(json.dumps({"output_dir": str(args.output_dir), "seed": 188}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
