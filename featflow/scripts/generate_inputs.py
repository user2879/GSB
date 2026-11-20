"""Deterministic synthetic input generator.

Everything is produced from fixed seeds and integer arithmetic; no network,
no real accounts.  Scale tiers:

* smoke  : 24 events  (fast, budget pressure)
* small  : 120 events (default acceptance run)
* medium : 600 events
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from featflow.common import write_json  # noqa: E402

BEHAVIORS = ["view", "cart", "buy"]
CONTEXT_FIELDS = ["price", "duration"]
TIERS = {"smoke": 24, "small": 120, "medium": 600}


class DeterministicRng:
    """Integer LCG so values are identical on every platform/Python build."""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0x7FFFFFFF

    def next_int(self, low: int, high: int) -> int:
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return low + self.state % (high - low + 1)

    def pick(self, options: list[Any]) -> Any:
        return options[self.next_int(0, len(options) - 1)]


def gen_events(n: int, seed: int) -> list[dict[str, Any]]:
    rng = DeterministicRng(seed)
    rows: list[dict[str, Any]] = []
    ts = 1_000
    for idx in range(n):
        ts += rng.next_int(120, 480)
        n_users = max(3, n // 8)
        n_items = max(4, n // 4)
        rows.append({
            "event_id": f"E{idx:06d}",
            "user_id": f"U{rng.next_int(1, n_users):03d}",
            "item_id": f"I{rng.next_int(1, n_items):04d}",
            "behavior": rng.pick(BEHAVIORS),
            "ts_ms": ts,
            "context": {
                "price": rng.next_int(10, 500) + rng.next_int(0, 3) / 4,
                "duration": rng.next_int(1, 90),
            },
        })
    return rows


def feature_defs() -> list[dict[str, Any]]:
    # Wide 3-level DAG: with cache_block_limit=3 the level-1/2 features must
    # reference evicted level-0 blocks, forcing deterministic recomputation.
    return [
        # level 0 - base features (6 blocks per point)
        {"name": "u_view_price_sum", "side": "user", "window_ms": 5000,
         "agg": "sum", "source_field": "price", "upstream": [],
         "empty_value": 0},
        {"name": "u_event_count", "side": "user", "window_ms": 8000,
         "agg": "count", "source_field": "duration", "upstream": [],
         "empty_value": 0},
        {"name": "u_duration_mean", "side": "user", "window_ms": 6000,
         "agg": "mean", "source_field": "duration", "upstream": [],
         "empty_value": 0},
        {"name": "u_last_price", "side": "user", "window_ms": 10000,
         "agg": "last", "source_field": "price", "upstream": [],
         "empty_value": 0},
        {"name": "i_buy_count", "side": "item", "window_ms": 7000,
         "agg": "count", "source_field": "price", "upstream": [],
         "empty_value": 0},
        {"name": "i_price_mean", "side": "item", "window_ms": 9000,
         "agg": "mean", "source_field": "price", "upstream": [],
         "empty_value": 0},
        # level 1 - derived
        {"name": "u_recent_score", "side": "user", "window_ms": 9000,
         "agg": "mean", "source_field": None,
         "upstream": ["u_view_price_sum", "u_last_price"], "empty_value": 0},
        {"name": "i_trend", "side": "item", "window_ms": 11000,
         "agg": "last", "source_field": None,
         "upstream": ["i_price_mean"], "empty_value": 0},
        # level 2 - derived over level 0 + level 1 (forces evicted re-reference)
        {"name": "u_engagement", "side": "user", "window_ms": 12000,
         "agg": "sum", "source_field": None,
         "upstream": ["u_event_count", "u_duration_mean", "u_recent_score"],
         "empty_value": 0},
    ]


def label_defs() -> list[dict[str, Any]]:
    return [
        {"name": "high_engagement_buy",
         "trigger_behaviors": ["buy"],
         "decision_feature": "u_engagement",
         "threshold": 120,
         "positive_op": "ge",
         "observed_features": ["u_engagement", "u_recent_score", "u_event_count"],
         "delay_ms": 1000},
        {"name": "item_hot_click",
         "trigger_behaviors": ["view", "cart"],
         "decision_feature": "i_buy_count",
         "threshold": 2,
         "positive_op": "ge",
         "observed_features": ["i_buy_count", "i_trend", "i_price_mean"],
         "delay_ms": 500},
    ]


def config(tier: str) -> dict[str, Any]:
    budgets = {"smoke": 260, "small": 900, "medium": 4500}
    quotas = {"smoke": 60, "small": 140, "medium": 700}
    return {
        "thread_concurrency": 4,
        "global_event_budget": budgets[tier],
        "feature_point_quota": quotas[tier],
        "cache_block_limit": 3,
        "snapshot_commit_every": 25,
        "fixed_point_scale": 4,
        "quant_step": "0.25",
        "default_empty_value": "0",
        "behaviors": BEHAVIORS,
        "context_fields": CONTEXT_FIELDS,
        "business_codes": {
            "E_EVENT_INVALID": "4001",
            "E_DEF_CYCLE": "4002",
            "E_DEF_PARAM": "4003",
            "E_BACKFILL_ORDER": "4004",
            "E_SAMPLE_PENDING": "4005",
            "E_FEAT_BUDGET_EXHAUSTED": "5001",
            "E_GLOBAL_BUDGET_EXHAUSTED": "5002",
            "E_INCREMENT_MISMATCH": "5003",
            "E_SNAPSHOT_STALE": "5004",
            "E_INJECTION_HIT": "6001",
        },
    }


def injections(n: int) -> dict[str, Any]:
    return {"rules": [
        {"injection_id": "INJ-001", "event_id": f"E{min(3, n - 1):06d}",
         "stage": "load", "kind": "non_finite_context",
         "context_field": "price", "bad_ts_ms": -1},
        {"injection_id": "INJ-002", "event_id": f"E{min(8, n - 1):06d}",
         "stage": "load", "kind": "unknown_behavior",
         "context_field": "", "bad_ts_ms": -1},
        {"injection_id": "INJ-003", "event_id": f"E{min(12, n - 1):06d}",
         "stage": "load", "kind": "backfill_order",
         "context_field": "", "bad_ts_ms": 1},
    ]}


def edit_sequence(n: int) -> dict[str, Any]:
    return {"edits": [
        {"edit_id": "EDIT-000", "type": "correct_event",
         "event_id": f"E{min(2, n - 1):06d}",
         "bad_ts_ms": 1,
         "context": {"price": 10.0, "duration": 5},
         "expect_reject": "E_BACKFILL_ORDER"},
        {"edit_id": "EDIT-001", "type": "correct_event",
         "event_id": f"E{min(5, n - 1):06d}",
         "context": {"price": 199.5}},
        {"edit_id": "EDIT-002", "type": "window_change",
         "feature": "u_view_price_sum", "window_ms": 6500},
        {"edit_id": "EDIT-003", "type": "agg_change",
         "feature": "u_last_price", "agg": "sum"},
        {"edit_id": "EDIT-004", "type": "label_delay_change",
         "label": "high_engagement_buy", "delay_ms": 1500},
        {"edit_id": "EDIT-005", "type": "cache_budget_change",
         "cache_block_limit": 4},
    ]}


def query_points(n: int) -> list[dict[str, Any]]:
    # Target an entity that has no events near the anchor -> empty-window output.
    return [
        {"ts_ms": 500, "user_id": "U999", "item_id": "I9999"},
        {"ts_ms": 2_000_000_000, "user_id": "U001", "item_id": "I0001"},
    ]


def write_yaml(path: str, payload: dict[str, Any]) -> None:
    import yaml

    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(payload, fh, sort_keys=True, allow_unicode=True,
                       default_flow_style=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--tier", choices=list(TIERS), default="small")
    ap.add_argument("--seed", type=int, default=20261121)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    n = TIERS[args.tier]
    events = gen_events(n, args.seed)
    with open(os.path.join(out, "events.jsonl"), "w", encoding="utf-8", newline="\n") as fh:
        for row in events:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")))
            fh.write("\n")
    write_json(os.path.join(out, "feature_defs.json"), feature_defs())
    write_json(os.path.join(out, "label_defs.json"), label_defs())
    write_json(os.path.join(out, "injections.json"), injections(n))
    write_json(os.path.join(out, "edits.json"), edit_sequence(n))
    write_json(os.path.join(out, "query_points.json"), query_points(n))
    write_yaml(os.path.join(out, "pipeline.yaml"), config(args.tier))
    write_json(os.path.join(out, "manifest.json"), {
        "seed": args.seed, "tier": args.tier, "event_count": n,
        "generator": "scripts/generate_inputs.py",
    })
    print(f"generated {n} events and all definitions in {out} (tier={args.tier})")


if __name__ == "__main__":
    main()
