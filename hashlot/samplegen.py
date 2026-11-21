"""本地确定性列式样本生成（不访问网络）。

使用 NumPy ``default_rng(seed)`` 固定随机序列，并显式注入倾斜分组键：
少量热点键占据大量行，用于触发热点子分区再分发与借用路径。
"""

import argparse
import json
import os

import numpy as np


def _bitmap(n, null_rate, rng):
    bitmap = (rng.random(n) >= null_rate).astype(int)
    return [int(x) for x in bitmap]


def generate(seed=20261122, n_events=480, n_sessions=24, scale=1000):
    rng = np.random.default_rng(seed)
    session_id = np.arange(n_sessions, dtype=np.int64)
    sess_region = rng.integers(0, 4, size=n_sessions, dtype=np.int64)
    sess_bitmap = [1] * n_sessions
    sess_score = rng.normal(50, 10, size=n_sessions)
    sess_score_bitmap = [1] * n_sessions

    hot_sessions = [int(session_id[0]), int(session_id[1]), int(session_id[2])]
    hot_weights = np.array([0.28, 0.18, 0.12])
    other_weight = np.full(n_sessions - 3, (1.0 - hot_weights.sum()) / (n_sessions - 3))
    weights = np.concatenate([hot_weights, other_weight])
    weights = weights / weights.sum()
    chosen = rng.choice(session_id, size=n_events, p=weights)
    order = np.argsort(chosen, kind="stable")
    chosen = chosen[order]
    event_session = [int(x) for x in chosen]

    amount = rng.normal(100, 25, size=n_events).round(3)
    amount[:60] += rng.normal(40, 5, size=60)
    latency = np.abs(rng.normal(20, 8, size=n_events)).round(3)
    amount_bitmap = _bitmap(n_events, 0.03, rng)
    latency_bitmap = _bitmap(n_events, 0.05, rng)
    region_bitmap = [1] * n_events

    doc = {
        "scale": scale,
        "batch_size": 48,
        "slot_params_version": 1,
        "tables": [
            {
                "name": "events",
                "row_count": n_events,
                "columns": [
                    {"name": "event_id", "dtype": "int64",
                     "null_bitmap": [1] * n_events,
                     "values": [int(x) for x in np.arange(1, n_events + 1)]},
                    {"name": "session_id", "dtype": "int64",
                     "null_bitmap": [1] * n_events,
                     "values": event_session},
                    {"name": "amount", "dtype": "float64",
                     "null_bitmap": amount_bitmap,
                     "values": [None if b == 0 else float(v)
                                for b, v in zip(amount_bitmap, amount)]},
                    {"name": "latency_ms", "dtype": "float64",
                     "null_bitmap": latency_bitmap,
                     "values": [None if b == 0 else float(v)
                                for b, v in zip(latency_bitmap, latency)]},
                    {"name": "region", "dtype": "int64",
                     "null_bitmap": region_bitmap,
                     "values": [int(sess_region[s]) for s in event_session]},
                ],
            },
            {
                "name": "sessions",
                "row_count": n_sessions,
                "columns": [
                    {"name": "session_id", "dtype": "int64",
                     "null_bitmap": sess_bitmap,
                     "values": [int(x) for x in session_id]},
                    {"name": "region", "dtype": "int64",
                     "null_bitmap": sess_bitmap,
                     "values": [int(x) for x in sess_region]},
                    {"name": "risk_score", "dtype": "float64",
                     "null_bitmap": sess_score_bitmap,
                     "values": [float(round(float(x), 3)) for x in sess_score]},
                ],
            },
        ],
        "query": {
            "source_table": "events",
            "filters": [
                {"table": "events", "column": "region", "op": "ge", "value": 0}
            ],
            "group_keys": ["session_id"],
            "aggregates": [
                {"func": "count", "column": "*", "alias": "cnt"},
                {"func": "sum", "column": "amount", "alias": "amount_sum"},
                {"func": "avg", "column": "latency_ms", "alias": "latency_avg"},
            ],
        },
        "topology": {
            "partition_count": 8,
            "global_max_connections": 4,
            "partitions": [
                dict(
                    partition_id=i,
                    max_connections=(0 if i in (0, 1) else 1),
                    key_cap_per_connection=3,
                    idle_recycle_ms=20,
                )
                for i in range(8)
            ],
            "borrows": [
                {"lender": 6, "borrower": 0, "borrowed_key_cap": 2,
                 "deadline_ms": 500},
                {"lender": 7, "borrower": 1, "borrowed_key_cap": 2,
                 "deadline_ms": 500},
            ],
        },
        "hotspot": {
            "enabled": True,
            "row_threshold": 90,
            "subpartition_count": 4,
        },
        "release_on_key_cap": False,
        "skew_injection": {
            "seed": seed,
            "hot_sessions": hot_sessions,
            "hot_weights": [float(x) for x in hot_weights],
            "note": "前三个会话为倾斜键，合计承载 58% 事件行",
        },
    }
    return doc


def main():
    parser = argparse.ArgumentParser(description="生成本地确定性倾斜样本")
    parser.add_argument("--out", default="samples/skew_sample.json")
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--rows", type=int, default=480)
    parser.add_argument("--scale", type=int, default=1000)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    doc = generate(args.seed, args.rows, scale=args.scale)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {args.out} rows={args.rows}")


if __name__ == "__main__":
    main()
